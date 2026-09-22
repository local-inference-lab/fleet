package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

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
			name:   "load endpoint accepts no body",
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
	d.states[model.ID] = deployment.ContainerState{Exists: true, Running: true, Status: "running", AssignedGPUs: assigned}
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
