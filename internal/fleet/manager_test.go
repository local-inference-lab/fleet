package fleet

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

func TestActivateUnloadsOtherModelsBeforeStartingTarget(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	manager := newTestManager(t, driver)

	op, noOp, err := manager.Activate("beta")
	if err != nil {
		t.Fatalf("Activate() error = %v", err)
	}
	if noOp {
		t.Fatal("Activate() returned no-op for switch")
	}
	waitOperation(t, manager, op.ID, "succeeded")

	driver.mu.Lock()
	calls := append([]string(nil), driver.calls...)
	driver.mu.Unlock()
	want := []string{"inspect:alpha", "inspect:beta", "stop:alpha:true", "start:beta", "inspect:beta"}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("driver calls = %#v, want %#v", calls, want)
	}
	if status, _ := manager.Status("alpha"); status.Phase != PhaseUnloaded {
		t.Fatalf("alpha phase = %s", status.Phase)
	}
	if status, _ := manager.Status("beta"); status.Phase != PhaseReady || status.Desired != "ready" {
		t.Fatalf("beta status = %+v", status)
	}
}

func TestActivateIsIdempotentAndConflictsWhileOperationRuns(t *testing.T) {
	driver := newFakeDriver()
	driver.blockStart = make(chan struct{})
	manager := newTestManager(t, driver)

	op, noOp, err := manager.Activate("alpha")
	if err != nil {
		t.Fatalf("Activate(alpha) error = %v", err)
	}
	if noOp {
		t.Fatal("first Activate(alpha) returned no-op")
	}
	same, sameNoOp, err := manager.Activate("alpha")
	if err != nil {
		t.Fatalf("second Activate(alpha) error = %v", err)
	}
	if !sameNoOp || same.ID != op.ID {
		t.Fatalf("second Activate(alpha) = (%+v, %v), want same operation no-op", same, sameNoOp)
	}
	_, _, err = manager.Activate("beta")
	var conflict *ConflictError
	if !errors.As(err, &conflict) {
		t.Fatalf("Activate(beta) error = %v, want ConflictError", err)
	}
	if conflict.Operation.ID != op.ID {
		t.Fatalf("conflict operation = %s, want %s", conflict.Operation.ID, op.ID)
	}

	close(driver.blockStart)
	waitOperation(t, manager, op.ID, "succeeded")
}

func TestRefreshMonitorsHealthAndReadiness(t *testing.T) {
	readyStatus := http.StatusServiceUnavailable
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := "ready"
		if readyStatus != http.StatusNoContent {
			body = "warming"
		}
		return &http.Response{
			StatusCode: readyStatus,
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})

	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Health: "unhealthy"}
	manager := newTestManager(t, driver)
	manager.client = &http.Client{Transport: transport}
	status, _ := manager.Status("alpha")
	if status.Phase != PhaseUnhealthy || status.Health != "unhealthy" {
		t.Fatalf("unhealthy Docker state status = %+v", status)
	}

	driver.mu.Lock()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	driver.mu.Unlock()
	manager.manifest.Models[0].Readiness = &manifest.Readiness{URL: "http://127.0.0.1/ready", SuccessStatus: http.StatusNoContent}
	manager.refresh(context.Background())
	status, _ = manager.Status("alpha")
	if status.Phase != PhaseUnhealthy || status.LastError == "" {
		t.Fatalf("warming readiness status = %+v", status)
	}

	readyStatus = http.StatusNoContent
	manager.refresh(context.Background())
	status, _ = manager.Status("alpha")
	if status.Phase != PhaseReady || status.LastError != "" {
		t.Fatalf("ready status = %+v", status)
	}
}

func TestConcurrentPlacementAllocatesAroundRunningDeployments(t *testing.T) {
	driver := newFakeDriver()
	manager := newTestManager(t, driver)
	manager.manifest.Runtime.ConcurrentDeployments = true
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}}
	manager.manifest.Runtime.GPUTopology.MaxUsedMemoryMiB = 1024
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 4, Strategy: "pcie_affinity"}
	manager.manifest.Models[1].Placement = &manifest.Placement{GPUCount: 2, Strategy: "pcie_affinity"}
	manager.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0, MemoryUsedMiB: 0}, {Index: 1, MemoryUsedMiB: 0},
		{Index: 2, MemoryUsedMiB: 0}, {Index: 3, MemoryUsedMiB: 0},
		{Index: 4, MemoryUsedMiB: 0}, {Index: 5, MemoryUsedMiB: 0},
		{Index: 6, MemoryUsedMiB: 0}, {Index: 7, MemoryUsedMiB: 0},
	}}

	op, noOp, err := manager.Activate("alpha")
	if err != nil || noOp {
		t.Fatalf("Activate(alpha) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	op, noOp, err = manager.Activate("beta")
	if err != nil || noOp {
		t.Fatalf("Activate(beta) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")

	alpha, _ := manager.Status("alpha")
	beta, _ := manager.Status("beta")
	if !reflect.DeepEqual(alpha.AssignedGPUs, []int{0, 1, 6, 7}) {
		t.Fatalf("alpha GPUs = %v", alpha.AssignedGPUs)
	}
	if !reflect.DeepEqual(beta.AssignedGPUs, []int{2, 3}) {
		t.Fatalf("beta GPUs = %v", beta.AssignedGPUs)
	}
}

func TestReloadManifestUpdatesConfiguredModels(t *testing.T) {
	driver := newFakeDriver()
	manager := newTestManager(t, driver)
	next := loadFleetManifestForTest(t)
	next.Models = []manifest.Model{
		next.Models[1],
		{ID: "gamma", Image: "gamma-image"},
	}

	if err := manager.ReloadManifest(next); err != nil {
		t.Fatalf("ReloadManifest() error = %v", err)
	}
	models := manager.Models()
	if got := []string{models[0].ID, models[1].ID}; !reflect.DeepEqual(got, []string{"beta", "gamma"}) {
		t.Fatalf("models = %v, want beta and gamma", got)
	}
	if _, ok := manager.Status("alpha"); ok {
		t.Fatal("removed unloaded model alpha still has a status")
	}
	status, ok := manager.Status("gamma")
	if !ok || status.Phase != PhaseUnknown || status.Desired != "unloaded" {
		t.Fatalf("new model status = (%+v, %v)", status, ok)
	}
}

func TestReloadManifestRejectsChangesToActiveModel(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	manager := newTestManager(t, driver)
	next := loadFleetManifestForTest(t)
	next.Models[0].Image = "replacement-image"

	err := manager.ReloadManifest(next)
	if err == nil || !strings.Contains(err.Error(), "cannot be changed") {
		t.Fatalf("ReloadManifest() error = %v, want active-model rejection", err)
	}
	if got := manager.Models()[0].Image; got != "alpha-image" {
		t.Fatalf("active manifest image = %q, want retained alpha-image", got)
	}
}

func newTestManager(t *testing.T, driver *fakeDriver) *Manager {
	t.Helper()
	cfg := loadFleetManifestForTest(t)
	m := NewManager(cfg, driver, slog.Default())
	m.refresh(context.Background())
	return m
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

func loadFleetManifestForTest(t *testing.T) *manifest.Manifest {
	t.Helper()
	path := filepath.Join(t.TempDir(), "lil-fleet-manager-test.json")
	body := `{
		"version": 1,
		"runtime": {
			"poll_interval": "1ms",
			"operation_timeout": "100ms",
			"readiness_timeout": "1s",
			"remove_on_unload": true
		},
		"models": [
			{"id": "alpha", "image": "alpha-image"},
			{"id": "beta", "image": "beta-image"}
		]
	}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	cfg, err := manifest.Load(path)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	return cfg
}

func waitOperation(t *testing.T, manager *Manager, id, state string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		op, ok := manager.Operation(id)
		if !ok {
			t.Fatalf("operation %s missing", id)
		}
		if op.State == state {
			return
		}
		if op.State == "failed" {
			t.Fatalf("operation failed: %+v", op)
		}
		time.Sleep(time.Millisecond)
	}
	op, _ := manager.Operation(id)
	t.Fatalf("operation %s state = %s, want %s", id, op.State, state)
}

type fakeDriver struct {
	mu         sync.Mutex
	states     map[string]deployment.ContainerState
	calls      []string
	blockStart chan struct{}
}

func newFakeDriver() *fakeDriver {
	return &fakeDriver{states: make(map[string]deployment.ContainerState)}
}

func (d *fakeDriver) Start(ctx context.Context, model manifest.Model, assigned []int) error {
	if d.blockStart != nil {
		select {
		case <-d.blockStart:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "start:"+model.ID)
	d.states[model.ID] = deployment.ContainerState{Exists: true, Running: true, Status: "running", AssignedGPUs: assigned}
	return nil
}

func (d *fakeDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "stop:"+model.ID+":"+strconv.FormatBool(remove))
	delete(d.states, model.ID)
	return nil
}

type fakeGPUProvider struct {
	devices []gpu.Device
}

func (p fakeGPUProvider) Snapshot(context.Context) ([]gpu.Device, error) {
	return p.devices, nil
}

func (d *fakeDriver) Inspect(ctx context.Context, model manifest.Model) (deployment.ContainerState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "inspect:"+model.ID)
	state, ok := d.states[model.ID]
	if !ok {
		return deployment.ContainerState{}, deployment.ErrNotFound
	}
	state.Exists = true
	return state, nil
}
