package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/fleet"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

func TestHandlerRejectsRuntimeOverrides(t *testing.T) {
	handler := newTestHandler(t)

	tests := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{
			name:   "load endpoint rejects runtime overrides",
			method: http.MethodPost,
			path:   "/v1/models/alpha/load",
			body:   `{"environment":{"MAX_MODEL_LEN":"1"}}`,
		},
		{
			name:   "unload endpoint accepts no body",
			method: http.MethodPost,
			path:   "/v1/models/alpha/unload",
			body:   `{"native_args":["--gpu-memory-utilization","1"]}`,
		},
		{
			name:   "switch endpoint rejects unknown override field",
			method: http.MethodPost,
			path:   "/v1/deployments/switch",
			body:   `{"model_id":"alpha","environment":{"MAX_MODEL_LEN":"1"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(tt.method, tt.path, bytes.NewBufferString(tt.body))
			handler.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
			}
			var payload map[string]map[string]string
			if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if payload["error"]["code"] == "" {
				t.Fatalf("missing structured error: %s", recorder.Body.String())
			}
		})
	}
}

func TestHandlerLoadAcceptsStrictInstanceCount(t *testing.T) {
	handler := newTestHandler(t)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/models/alpha/load", bytes.NewBufferString(`{"instances":1}`))
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusAccepted {
		t.Fatalf("load status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var payload struct {
		Operation fleet.Operation `json:"operation"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Operation.TargetInstances != 1 {
		t.Fatalf("operation instances = %d, want 1", payload.Operation.TargetInstances)
	}

	for _, body := range []string{`{"instances":0}`, `{"instances":65}`, `{"instances":1,"image":"x"}`, `{}`} {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, "/v1/models/alpha/load", bytes.NewBufferString(body))
		handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("load body %s status = %d, want 400; response=%s", body, recorder.Code, recorder.Body.String())
		}
	}
}

func TestHandlerSwitchAndDeploymentStatus(t *testing.T) {
	handler := newTestHandler(t)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/deployments/switch", bytes.NewBufferString(`{"model_id":"alpha"}`))
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusAccepted {
		t.Fatalf("switch status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if recorder.Header().Get("Location") == "" {
		t.Fatal("switch response missing Location header")
	}

	recorder = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodGet, "/v1/deployments/alpha", nil)
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("deployment status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var status fleet.Status
	if err := json.Unmarshal(recorder.Body.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if status.ModelID != "alpha" {
		t.Fatalf("model id = %q", status.ModelID)
	}
}

func TestHandlerListsModelsWithDocumentedShape(t *testing.T) {
	handler := newTestHandler(t)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("list status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var payload struct {
		Models []struct {
			Model  fleet.ModelInfo `json:"model"`
			Status fleet.Status    `json:"status"`
		} `json:"models"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(payload.Models) != 1 {
		t.Fatalf("models len = %d", len(payload.Models))
	}
	if payload.Models[0].Model.ID != "alpha" || payload.Models[0].Status.ModelID != "alpha" {
		t.Fatalf("unexpected model payload: %+v", payload.Models[0])
	}
	var raw map[string][]map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &raw); err != nil {
		t.Fatalf("decode raw list: %v", err)
	}
	if _, leaked := raw["models"][0]["image"]; leaked {
		t.Fatalf("model fields leaked at top level: %s", recorder.Body.String())
	}
}

func TestBearerAuthenticationProtectsEverythingExceptHealth(t *testing.T) {
	handler := RequireBearerToken(newTestHandler(t), "test-token")

	for _, tt := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"canonical models list", http.MethodGet, "/v1/models", ""},
		{"double slash models list", http.MethodGet, "//v1/models", ""},
		{"encoded leading slash models list", http.MethodGet, "/%2fv1/models", ""},
		{"double slash mutation", http.MethodPost, "//v1/models/alpha/load", ""},
		{"encoded slash mutation", http.MethodPost, "/v1%2fmodels/alpha/load", ""},
		{"unknown route", http.MethodGet, "/not-found", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			request := httptest.NewRequest(tt.method, tt.path, bytes.NewBufferString(tt.body))
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusUnauthorized {
				t.Fatalf("unauthenticated status = %d, body = %s", recorder.Code, recorder.Body.String())
			}
			if recorder.Header().Get("WWW-Authenticate") == "" {
				t.Fatal("unauthenticated response missing WWW-Authenticate")
			}
		})
	}

	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Authorization", "Bearer test-token")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("authenticated status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/healthz", nil)
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("health status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/readyz", nil)
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code == http.StatusUnauthorized {
		t.Fatalf("ready endpoint should remain public, body = %s", recorder.Body.String())
	}
}

func newTestHandler(t *testing.T) http.Handler {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fleet.json")
	body := `{
		"version": 1,
		"runtime": {
			"poll_interval": "1ms",
			"operation_timeout": "100ms",
			"readiness_timeout": "1s",
			"remove_on_unload": true
		},
		"models": [{"id": "alpha", "image": "alpha-image"}]
	}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	cfg, err := manifest.Load(path)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	driver := &apiDriver{states: map[string]deployment.ContainerState{}}
	manager := fleet.NewManager(cfg, driver, slog.Default())
	return NewHandler(manager, slog.Default())
}

type apiDriver struct {
	mu     sync.Mutex
	states map[string]deployment.ContainerState
}

func (d *apiDriver) Start(_ context.Context, model manifest.Model, assigned []int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.states[model.ID] = deployment.ContainerState{Exists: true, Running: true, Status: "running", AssignedGPUs: assigned, Port: model.InstancePort}
	return nil
}

func (d *apiDriver) Stop(_ context.Context, model manifest.Model, _ bool) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	delete(d.states, model.ID)
	return nil
}

func (d *apiDriver) Inspect(_ context.Context, model manifest.Model) (deployment.ContainerState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	state, ok := d.states[model.ID]
	if !ok {
		return deployment.ContainerState{}, deployment.ErrNotFound
	}
	state.Exists = true
	return state, nil
}

func (d *apiDriver) List(context.Context) ([]deployment.ContainerState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var result []deployment.ContainerState
	for id, state := range d.states {
		state.Exists, state.ModelID, state.InstanceID, state.InstanceIndex = true, id, id, 1
		result = append(result, state)
	}
	return result, nil
}

// TestHandlerMatchesLLMConduitFleetContract pins the fields llmconduit's
// dashboard and mesh worker deserialize as required.
func TestHandlerMatchesLLMConduitFleetContract(t *testing.T) {
	handler := newTestHandler(t)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/models/alpha/load", bytes.NewBufferString(`{"instances":1}`)))
	if recorder.Code != http.StatusAccepted {
		t.Fatalf("load status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var lifecycle map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &lifecycle); err != nil {
		t.Fatal(err)
	}
	operation, _ := lifecycle["operation"].(map[string]any)
	for _, field := range []string{"id", "kind", "model_id", "state", "created_at"} {
		if _, ok := operation[field]; !ok || lifecycle["changed"] != true {
			t.Fatalf("lifecycle response missing %q: %s", field, recorder.Body.String())
		}
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		recorder = httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
		var payload struct {
			Models []struct {
				Model  map[string]any `json:"model"`
				Status map[string]any `json:"status"`
			} `json:"models"`
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
			t.Fatal(err)
		}
		model, status := payload.Models[0].Model, payload.Models[0].Status
		instances, _ := status["instances"].([]any)
		if len(instances) == 1 && status["phase"] == "ready" {
			for _, field := range []string{"id", "image", "max_instances"} {
				if _, ok := model[field]; !ok {
					t.Fatalf("model missing %q: %v", field, model)
				}
			}
			for _, field := range []string{"model_id", "phase", "desired_state"} {
				if _, ok := status[field]; !ok {
					t.Fatalf("status missing %q: %v", field, status)
				}
			}
			instance := instances[0].(map[string]any)
			// The test profile has no port; llmconduit still requires the key.
			for _, field := range []string{"instance_id", "index", "port", "phase"} {
				if _, ok := instance[field]; !ok {
					t.Fatalf("instance missing %q: %v", field, instance)
				}
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("model never became ready: %s", recorder.Body.String())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestLifecycleErrorsMapToClientStatusCodes(t *testing.T) {
	h := &Handler{}
	for _, tc := range []struct {
		err  error
		want int
	}{
		{&fleet.InvalidRequestError{Err: errors.New("instances must be between 1 and 64")}, http.StatusBadRequest},
		{&fleet.InsufficientResourcesError{ModelID: "alpha", Err: errors.New("no GPUs")}, http.StatusConflict},
		{&fleet.ConflictError{}, http.StatusConflict},
		{errors.New(`unknown model "x"`), http.StatusNotFound},
	} {
		recorder := httptest.NewRecorder()
		h.writeLifecycleResult(recorder, fleet.Operation{}, false, tc.err)
		if recorder.Code != tc.want {
			t.Fatalf("%T status = %d, want %d", tc.err, recorder.Code, tc.want)
		}
	}
}
