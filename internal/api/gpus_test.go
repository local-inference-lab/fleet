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
	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

type apiGPUProvider struct {
	mu      sync.Mutex
	devices []gpu.Device
	err     error
}

func (p *apiGPUProvider) Snapshot(context.Context) ([]gpu.Device, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.devices, p.err
}

func utilization(value int) *int { return &value }

// newPlacementTestHandler serves a topology manifest: groups {0,1},{2,3};
// alpha is TP2 and gamma TP1 with placement, beta pins GPU 3 statically.
// nvidia-smi additionally reports ungrouped GPU 4; GPU 3 is over the
// max_used_memory_mib threshold.
func newPlacementTestHandler(t *testing.T, provider *apiGPUProvider) (http.Handler, *fleet.Manager) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fleet.json")
	body := `{
		"version": 1,
		"runtime": {
			"poll_interval": "1ms",
			"operation_timeout": "1s",
			"readiness_timeout": "2s",
			"remove_on_unload": true,
			"concurrent_deployments": true,
			"gpu_topology": {"groups": [[0, 1], [2, 3]], "max_used_memory_mib": 1024}
		},
		"models": [
			{"id": "alpha", "image": "alpha-image", "placement": {"gpu_count": 2}},
			{"id": "beta", "image": "beta-image", "gpus": "device=3"},
			{"id": "gamma", "image": "gamma-image", "placement": {"gpu_count": 1}}
		]
	}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	cfg, err := manifest.Load(path)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if provider.devices == nil && provider.err == nil {
		name := "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
		provider.devices = []gpu.Device{
			{Index: 0, Name: name, MemoryTotalMiB: 97887, MemoryUsedMiB: 0, MemoryFreeMiB: 97000, UtilizationPercent: utilization(0)},
			{Index: 1, Name: name, MemoryTotalMiB: 97887, MemoryUsedMiB: 0, MemoryFreeMiB: 97000, UtilizationPercent: utilization(3)},
			{Index: 2, Name: name, MemoryTotalMiB: 97887, MemoryUsedMiB: 0, MemoryFreeMiB: 97000},
			{Index: 3, Name: name, MemoryTotalMiB: 97887, MemoryUsedMiB: 4096, MemoryFreeMiB: 93000},
			{Index: 4, Name: name, MemoryTotalMiB: 97887, MemoryUsedMiB: 0, MemoryFreeMiB: 97000},
		}
	}
	driver := &apiDriver{states: map[string]deployment.ContainerState{}}
	manager := fleet.NewManagerWithGPUProvider(cfg, driver, provider, slog.Default())
	return NewHandler(manager, slog.Default()), manager
}

func serve(handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(method, path, bytes.NewBufferString(body)))
	return recorder
}

func errorCode(t *testing.T, recorder *httptest.ResponseRecorder) string {
	t.Helper()
	var payload struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode error body %s: %v", recorder.Body.String(), err)
	}
	if payload.Error.Message == "" {
		t.Fatalf("error body missing message: %s", recorder.Body.String())
	}
	return payload.Error.Code
}

func waitHandlerOperation(t *testing.T, manager *fleet.Manager, id string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if op, _ := manager.Operation(id); op.State == "succeeded" {
			return
		} else if op.State == "failed" {
			t.Fatalf("operation failed: %+v", op)
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("operation %s did not succeed", id)
}

func TestHandlerGPUInventory(t *testing.T) {
	handler, manager := newPlacementTestHandler(t, &apiGPUProvider{})
	recorder := serve(handler, http.MethodPost, "/v1/models/alpha/load", `{"instances":1}`)
	var accepted struct {
		Operation fleet.Operation `json:"operation"`
	}
	_ = json.Unmarshal(recorder.Body.Bytes(), &accepted)
	waitHandlerOperation(t, manager, accepted.Operation.ID)

	recorder = serve(handler, http.MethodGet, "/v1/gpus", "")
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var raw struct {
		SampledAt string           `json:"sampled_at"`
		Groups    [][]int          `json:"groups"`
		GPUs      []map[string]any `json:"gpus"`
		Stale     *bool            `json:"stale"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, err := time.Parse(time.RFC3339, raw.SampledAt); err != nil {
		t.Fatalf("sampled_at %q is not RFC3339: %v", raw.SampledAt, err)
	}
	if raw.Stale != nil || len(raw.Groups) != 2 || len(raw.GPUs) != 5 {
		t.Fatalf("inventory = %s", recorder.Body.String())
	}
	for _, field := range []string{"index", "name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_percent", "group", "assigned"} {
		for _, g := range raw.GPUs {
			if _, ok := g[field]; !ok {
				t.Fatalf("GPU missing %q: %v", field, g)
			}
		}
	}
	gpu0, gpu2, gpu4 := raw.GPUs[0], raw.GPUs[2], raw.GPUs[4]
	if assigned := gpu0["assigned"].([]any); len(assigned) != 1 ||
		assigned[0].(map[string]any)["model_id"] != "alpha" || assigned[0].(map[string]any)["instance_id"] != "alpha" {
		t.Fatalf("GPU 0 assigned = %v", gpu0["assigned"])
	}
	if assigned, ok := gpu2["assigned"].([]any); !ok || len(assigned) != 0 {
		t.Fatalf("free GPU assigned must be [], got %v", gpu2["assigned"])
	}
	if gpu2["utilization_percent"] != nil || gpu0["utilization_percent"] != float64(0) {
		t.Fatalf("utilization = %v / %v", gpu0["utilization_percent"], gpu2["utilization_percent"])
	}
	if gpu0["group"] != float64(0) || gpu2["group"] != float64(1) || gpu4["group"] != nil {
		t.Fatalf("groups = %v %v %v", gpu0["group"], gpu2["group"], gpu4["group"])
	}

	// gpu_count reaches the models list for placed and static profiles.
	recorder = serve(handler, http.MethodGet, "/v1/models", "")
	var models struct {
		Models []struct {
			Model fleet.ModelInfo `json:"model"`
		} `json:"models"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &models); err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, entry := range models.Models {
		counts[entry.Model.ID] = entry.Model.GPUCount
	}
	if counts["alpha"] != 2 || counts["beta"] != 1 || counts["gamma"] != 1 {
		t.Fatalf("gpu_count = %v", counts)
	}
}

func TestHandlerGPUInventoryFailureAndEmptyTopology(t *testing.T) {
	handler, _ := newPlacementTestHandler(t, &apiGPUProvider{err: errors.New("nvidia-smi: not found")})
	recorder := serve(handler, http.MethodGet, "/v1/gpus", "")
	if recorder.Code != http.StatusServiceUnavailable || errorCode(t, recorder) != "gpu_query_failed" {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	// Without a topology, groups is [] (not null) and every group is null.
	path := filepath.Join(t.TempDir(), "fleet.json")
	body := `{"version": 1, "runtime": {"poll_interval": "1ms", "operation_timeout": "1s", "readiness_timeout": "1s", "remove_on_unload": true},
		"models": [{"id": "alpha", "image": "alpha-image"}]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := manifest.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	provider := &apiGPUProvider{devices: []gpu.Device{{Index: 0, Name: "GPU", MemoryTotalMiB: 10, MemoryFreeMiB: 10}}}
	plain := NewHandler(fleet.NewManagerWithGPUProvider(cfg, &apiDriver{states: map[string]deployment.ContainerState{}}, provider, slog.Default()), slog.Default())
	recorder = serve(plain, http.MethodGet, "/v1/gpus", "")
	var raw map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	groups, ok := raw["groups"].([]any)
	gpus := raw["gpus"].([]any)
	if recorder.Code != http.StatusOK || !ok || len(groups) != 0 || gpus[0].(map[string]any)["group"] != nil {
		t.Fatalf("no-topology inventory = %d %s", recorder.Code, recorder.Body.String())
	}

	// A bearer token guards the route like every other /v1 endpoint.
	recorder = serve(RequireBearerToken(plain, "token"), http.MethodGet, "/v1/gpus", "")
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated /v1/gpus = %d", recorder.Code)
	}
}
