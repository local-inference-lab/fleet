package fleet

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

// readinessSwitch serves 200 for URLs whose host:port is marked ready and 503
// otherwise, optionally delaying every probe.
type readinessSwitch struct {
	mu    sync.Mutex
	ready map[string]bool
	delay time.Duration
	calls atomic.Int64
}

func (r *readinessSwitch) set(hostPort string, ready bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ready[hostPort] = ready
}

func (r *readinessSwitch) client() *http.Client {
	return &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		r.calls.Add(1)
		if r.delay > 0 {
			select {
			case <-time.After(r.delay):
			case <-request.Context().Done():
				return nil, request.Context().Err()
			}
		}
		r.mu.Lock()
		ready := r.ready[request.URL.Host]
		r.mu.Unlock()
		status := http.StatusServiceUnavailable
		if ready {
			status = http.StatusOK
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header), Request: request}, nil
	})}
}

func newConcurrentTestManager(t *testing.T, driver *fakeDriver) (*Manager, *readinessSwitch) {
	t.Helper()
	manager := newTestManager(t, driver)
	manager.manifest.Runtime.ConcurrentDeployments = true
	manager.manifest.Models[0].Readiness = &manifest.Readiness{URL: "http://127.0.0.1:9101/health", SuccessStatus: 200}
	manager.manifest.Models[1].Readiness = &manifest.Readiness{URL: "http://127.0.0.1:9102/health", SuccessStatus: 200}
	readiness := &readinessSwitch{ready: map[string]bool{}}
	manager.client = readiness.client()
	return manager, readiness
}

func TestReadinessWaitDoesNotBlockOtherModelsWithConcurrentDeployments(t *testing.T) {
	driver := newFakeDriver()
	manager, readiness := newConcurrentTestManager(t, driver)
	manager.manifest.Runtime = withReadinessTimeout(t, manager.manifest.Runtime, 10*time.Second)

	alpha, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	waitInstancePhase(t, manager, "alpha", PhaseLoading)

	// alpha stays in its readiness wait; an unrelated model still loads.
	readiness.set("127.0.0.1:9102", true)
	beta, noOp, err := manager.Activate("beta")
	if err != nil || noOp {
		t.Fatalf("Activate(beta) while alpha loads = (%+v, %v, %v), want a new operation", beta, noOp, err)
	}
	waitOperation(t, manager, beta.ID, "succeeded")

	// Per-model admission: a duplicate joins alpha's operation and a
	// conflicting request for alpha is rejected.
	same, sameNoOp, err := manager.Activate("alpha")
	if err != nil || !sameNoOp || same.ID != alpha.ID {
		t.Fatalf("duplicate Activate(alpha) = (%+v, %v, %v), want operation %s", same, sameNoOp, err, alpha.ID)
	}
	_, _, err = manager.Unload("alpha")
	var conflict *ConflictError
	if !errors.As(err, &conflict) || conflict.Operation.ID != alpha.ID {
		t.Fatalf("Unload(alpha) during load error = %v, want conflict with %s", err, alpha.ID)
	}
	if op, _ := manager.Operation(alpha.ID); op.State != "running" {
		t.Fatalf("alpha operation = %+v, want still running", op)
	}

	readiness.set("127.0.0.1:9101", true)
	waitOperation(t, manager, alpha.ID, "succeeded")
}

func TestExclusiveModeStillSerializesAllOperations(t *testing.T) {
	driver := newFakeDriver()
	manager, readiness := newConcurrentTestManager(t, driver)
	manager.manifest.Runtime = withReadinessTimeout(t, manager.manifest.Runtime, 10*time.Second)
	manager.manifest.Runtime.ConcurrentDeployments = false

	alpha, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	waitInstancePhase(t, manager, "alpha", PhaseLoading)
	for _, call := range []func() (Operation, bool, error){
		func() (Operation, bool, error) { return manager.Activate("beta") },
		func() (Operation, bool, error) { return manager.Unload("beta") },
	} {
		_, _, err := call()
		var conflict *ConflictError
		if !errors.As(err, &conflict) || conflict.Operation.ID != alpha.ID {
			t.Fatalf("request during exclusive load error = %v, want conflict with %s", err, alpha.ID)
		}
	}
	readiness.set("127.0.0.1:9101", true)
	waitOperation(t, manager, alpha.ID, "succeeded")
}

type blockingGPUProvider struct {
	devices  []gpu.Device
	entered  chan struct{}
	release  chan struct{}
	deadline atomic.Bool
}

func (p *blockingGPUProvider) Snapshot(ctx context.Context) ([]gpu.Device, error) {
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= gpuSnapshotTimeout {
		p.deadline.Store(true)
	}
	p.entered <- struct{}{}
	<-p.release
	return p.devices, nil
}

func TestGPUSnapshotRunsOutsideManagerLockWithTimeoutAndRechecksState(t *testing.T) {
	driver := newFakeDriver()
	manager, readiness := newConcurrentTestManager(t, driver)
	readiness.set("127.0.0.1:9101", true)
	readiness.set("127.0.0.1:9102", true)
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0, 1}}
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	manager.manifest.Models[1].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	provider := &blockingGPUProvider{
		devices: []gpu.Device{{Index: 0}, {Index: 1}},
		entered: make(chan struct{}, 2), release: make(chan struct{}),
	}
	manager.gpus = provider

	type result struct {
		op  Operation
		err error
	}
	results := make(chan result, 2)
	for _, id := range []string{"alpha", "beta"} {
		go func() {
			op, _, err := manager.Activate(id)
			results <- result{op, err}
		}()
	}
	<-provider.entered
	<-provider.entered

	// Both activations are inside nvidia-smi at once; status reads must not
	// wait for them.
	done := make(chan struct{})
	go func() {
		manager.Statuses()
		manager.Models()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("status reads blocked behind the GPU snapshot")
	}
	if !provider.deadline.Load() {
		t.Fatal("GPU snapshot context has no bounded deadline")
	}

	close(provider.release)
	var assigned [][]int
	for range 2 {
		r := <-results
		if r.err != nil {
			t.Fatalf("Activate error = %v", r.err)
		}
		waitOperation(t, manager, r.op.ID, "succeeded")
		status, _ := manager.Status(r.op.ModelID)
		assigned = append(assigned, status.AssignedGPUs)
	}
	if len(assigned[0]) != 1 || len(assigned[1]) != 1 || intersects(assigned[0], assigned[1]) {
		t.Fatalf("concurrent activations shared GPUs: %v", assigned)
	}
}

type blockingListDriver struct {
	*fakeDriver
	block   chan struct{}
	entered chan struct{}
}

func (d *blockingListDriver) List(ctx context.Context) ([]deployment.ContainerState, error) {
	states, err := d.fakeDriver.List(ctx)
	if d.block != nil {
		d.entered <- struct{}{}
		<-d.block
	}
	return states, err
}

func TestRefreshDropsObservationsMadeStaleByLifecycleWrites(t *testing.T) {
	fake := newFakeDriver()
	fake.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	manager := newTestManager(t, fake)
	driver := &blockingListDriver{fakeDriver: fake, block: make(chan struct{}), entered: make(chan struct{}, 1)}
	manager.driver = driver

	finished := make(chan struct{})
	go func() {
		manager.refresh(context.Background())
		close(finished)
	}()
	<-driver.entered
	// A lifecycle write lands while refresh holds an observation of the
	// running container.
	manager.updateTransition("alpha", PhaseFailed, "unloaded")
	close(driver.block)
	<-finished

	status, _ := manager.Status("alpha")
	if status.Phase != PhaseFailed || status.Desired != "unloaded" {
		t.Fatalf("stale refresh overwrote lifecycle state: %+v", status)
	}
	driver.block = nil
	manager.refresh(context.Background())
	if status, _ := manager.Status("alpha"); status.Phase != PhaseReady {
		t.Fatalf("fresh refresh status = %+v, want ready", status)
	}
}

func TestRefreshDoesNotOverrideStatusOwnedByRunningOperation(t *testing.T) {
	driver := newFakeDriver()
	driver.blockStart = make(chan struct{})
	manager := newTestManager(t, driver)
	op, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	waitInstancePhase(t, manager, "alpha", PhaseLoading)
	manager.refresh(context.Background())
	if status, _ := manager.Status("alpha"); status.Phase != PhaseLoading || status.Desired != "ready" {
		t.Fatalf("refresh replaced in-flight status: %+v", status)
	}
	close(driver.blockStart)
	waitOperation(t, manager, op.ID, "succeeded")
}

func TestRequestedStopExitCodesAreUnloadedNotFailed(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   deployment.ContainerState
		desired string
		want    Phase
	}{
		{"sigkill after requested stop", deployment.ContainerState{ExitCode: 137}, "unloaded", PhaseUnloaded},
		{"sigterm after requested stop", deployment.ContainerState{ExitCode: 143}, "unloaded", PhaseUnloaded},
		{"oom kill is a failure", deployment.ContainerState{ExitCode: 137, OOMKilled: true}, "unloaded", PhaseFailed},
		{"sigkill while serving", deployment.ContainerState{ExitCode: 137}, "ready", PhaseFailed},
		{"crash after requested stop", deployment.ContainerState{ExitCode: 1}, "unloaded", PhaseFailed},
		{"clean exit", deployment.ContainerState{}, "ready", PhaseUnloaded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.state.Exists, tc.state.Status = true, "exited"
			got, probe := instanceFromState(manifest.Model{ID: "alpha"}, tc.state, tc.desired, time.Now())
			if got.Phase != tc.want || probe {
				t.Fatalf("phase = %s (probe %v), want %s", got.Phase, probe, tc.want)
			}
		})
	}

	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: false, Status: "exited", ExitCode: 137}
	manager := newTestManager(t, driver)
	if status, _ := manager.Status("alpha"); status.Phase != PhaseUnloaded {
		t.Fatalf("stopped container after restart = %+v, want unloaded", status)
	}
}

func TestRefreshListsOnceAndProbesInParallel(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	readiness := &readinessSwitch{ready: map[string]bool{}, delay: 300 * time.Millisecond}
	for _, port := range []string{"9001", "9002", "9003"} {
		readiness.set("127.0.0.1:"+port, true)
		driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Port: 9001}
	}
	driver.states["alpha--2"] = deployment.ContainerState{Exists: true, Running: true, Port: 9002}
	driver.states["alpha--3"] = deployment.ContainerState{Exists: true, Running: true, Port: 9003}
	manager.client = readiness.client()
	driver.mu.Lock()
	driver.calls = nil
	driver.mu.Unlock()

	started := time.Now()
	manager.refresh(context.Background())
	elapsed := time.Since(started)

	status, _ := manager.Status("alpha")
	if status.ReadyInstances != 3 {
		t.Fatalf("ready instances = %d, want 3 (%+v)", status.ReadyInstances, status)
	}
	if elapsed >= 800*time.Millisecond {
		t.Fatalf("three 300ms probes took %s, want parallel", elapsed)
	}
	driver.mu.Lock()
	calls := append([]string(nil), driver.calls...)
	driver.mu.Unlock()
	if len(calls) != 1 || calls[0] != "list" {
		t.Fatalf("driver calls per refresh = %v, want one list", calls)
	}
}

func TestActivationPollsReadinessFasterThanReconcileInterval(t *testing.T) {
	driver := newFakeDriver()
	manager, readiness := newConcurrentTestManager(t, driver)
	manager.manifest.Runtime = withRuntime(t, `"poll_interval": "1m", "readiness_timeout": "10s"`)

	op, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	waitInstancePhase(t, manager, "alpha", PhaseLoading)
	readiness.set("127.0.0.1:9101", true)
	deadline := time.Now().Add(2500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if current, _ := manager.Operation(op.ID); current.State == "succeeded" {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("activation did not notice readiness within about a second despite a one-minute poll interval")
}

type blockingStopDriver struct {
	*fakeDriver
	stopping chan string
	release  chan struct{}
}

func (d *blockingStopDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	d.stopping <- model.ID
	<-d.release
	return d.fakeDriver.Stop(ctx, model, remove)
}

func TestExclusiveSwitchStopsOnlyLoadedModelsInParallel(t *testing.T) {
	fake := newFakeDriver()
	fake.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	fake.states["beta"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	cfg := loadFleetManifestForTest(t)
	cfg.Models = append(cfg.Models, manifest.Model{ID: "gamma", Image: "gamma-image"}, manifest.Model{ID: "delta", Image: "delta-image"})
	driver := &blockingStopDriver{fakeDriver: fake, stopping: make(chan string, 4), release: make(chan struct{})}
	manager := NewManager(cfg, driver, newDiscardLogger())
	manager.refresh(context.Background())

	op, _, err := manager.Activate("gamma")
	if err != nil {
		t.Fatal(err)
	}
	stopped := map[string]bool{}
	for range 2 {
		select {
		case id := <-driver.stopping:
			stopped[id] = true
		case <-time.After(time.Second):
			t.Fatalf("stops ran sequentially or not at all; started %v", stopped)
		}
	}
	close(driver.release)
	waitOperation(t, manager, op.ID, "succeeded")
	if !stopped["alpha"] || !stopped["beta"] || len(driver.stopping) != 0 {
		t.Fatalf("stopped %v (extra %d), want exactly alpha and beta", stopped, len(driver.stopping))
	}
}

type pullingDriver struct {
	*fakeDriver
	pullErr error
}

func (d *pullingDriver) EnsureImage(ctx context.Context, image string) error {
	if _, ok := ctx.Deadline(); !ok {
		return errors.New("pull has no deadline")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "pull:"+image)
	return d.pullErr
}

func TestActivationEnsuresImageBeforeCreate(t *testing.T) {
	fake := newFakeDriver()
	driver := &pullingDriver{fakeDriver: fake}
	manager := NewManager(loadFleetManifestForTest(t), driver, newDiscardLogger())
	manager.refresh(context.Background())
	op, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	fake.mu.Lock()
	calls := strings.Join(fake.calls, ",")
	fake.mu.Unlock()
	if !strings.Contains(calls, "pull:alpha-image,start:alpha") {
		t.Fatalf("calls = %s, want pull before start", calls)
	}

	driver.pullErr = errors.New("manifest unknown")
	op, _, err = manager.Activate("beta")
	if err != nil {
		t.Fatal(err)
	}
	waitOperationState(t, manager, op.ID, "failed")
	if failed, _ := manager.Operation(op.ID); !strings.Contains(failed.Error, "manifest unknown") {
		t.Fatalf("pull failure = %+v", failed)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if strings.Contains(strings.Join(fake.calls, ","), "start:beta") {
		t.Fatal("container started after a failed pull")
	}
}

func TestFinishedOperationsAreEvictedByAgeAndCount(t *testing.T) {
	manager := newTestManager(t, newFakeDriver())
	clock := time.Date(2026, 9, 30, 0, 0, 0, 0, time.UTC)
	manager.now = func() time.Time { return clock }
	finished := func(id string, at time.Time) Operation {
		return Operation{ID: id, Kind: "activate", ModelID: "alpha", State: "succeeded", CreatedAt: at, FinishedAt: &at}
	}
	manager.mu.Lock()
	manager.operations["old"] = finished("old", clock.Add(-2*time.Hour))
	manager.operations["running"] = Operation{ID: "running", Kind: "activate", ModelID: "beta", State: "running", CreatedAt: clock.Add(-3 * time.Hour)}
	for i := range maxOperations + 10 {
		id := "recent-" + strings.Repeat("x", i%3) + time.Duration(i).String()
		manager.operations[id] = finished(id, clock.Add(-time.Duration(maxOperations+10-i)*time.Second))
	}
	manager.evictOperationsLocked()
	count := len(manager.operations)
	_, hasOld := manager.operations["old"]
	_, hasRunning := manager.operations["running"]
	_, hasOldestRecent := manager.operations["recent-0s"]
	_, hasNewest := manager.operations["recent-"+strings.Repeat("x", (maxOperations+9)%3)+time.Duration(maxOperations+9).String()]
	manager.mu.Unlock()

	if hasOld || !hasRunning || hasOldestRecent || !hasNewest {
		t.Fatalf("eviction kept old=%v running=%v oldestRecent=%v newest=%v", hasOld, hasRunning, hasOldestRecent, hasNewest)
	}
	if count > maxOperations {
		t.Fatalf("retained %d operations, want at most %d", count, maxOperations)
	}
}

func waitInstancePhase(t *testing.T, manager *Manager, modelID string, phase Phase) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if status, _ := manager.Status(modelID); status.Phase == phase {
			return
		}
		time.Sleep(time.Millisecond)
	}
	status, _ := manager.Status(modelID)
	t.Fatalf("%s phase = %s, want %s", modelID, status.Phase, phase)
}

func waitOperationState(t *testing.T, manager *Manager, id, state string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if op, _ := manager.Operation(id); op.State == state {
			return
		}
		time.Sleep(time.Millisecond)
	}
	op, _ := manager.Operation(id)
	t.Fatalf("operation %s = %+v, want %s", id, op, state)
}

func withReadinessTimeout(t *testing.T, _ manifest.RuntimeConfig, timeout time.Duration) manifest.RuntimeConfig {
	t.Helper()
	return withRuntime(t, `"poll_interval": "1ms", "readiness_timeout": "`+timeout.String()+`"`)
}

// withRuntime builds a validated runtime block (durations are unexported).
func withRuntime(t *testing.T, fields string) manifest.RuntimeConfig {
	t.Helper()
	path := t.TempDir() + "/runtime.json"
	body := `{"version": 1, "runtime": {"operation_timeout": "1s", "remove_on_unload": true, "concurrent_deployments": true, ` + fields + `},
		"models": [{"id": "alpha", "image": "alpha-image"}]}`
	if err := writeFile(path, body); err != nil {
		t.Fatal(err)
	}
	cfg, err := manifest.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return cfg.Runtime
}
