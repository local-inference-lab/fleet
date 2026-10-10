package fleet

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

func ptr(value int) *int { return &value }

func wantGPUSelectionError(t *testing.T, err error, code string) {
	t.Helper()
	var selection *GPUSelectionError
	if !errors.As(err, &selection) || selection.Code != code {
		t.Fatalf("error = %v (%T), want GPUSelectionError %s", err, err, code)
	}
}

// newExplicitTestManager is the replica manager (groups {0,1},{2,3}) with
// alpha placed at gpu_count 2 and beta at gpu_count 1.
func newExplicitTestManager(t *testing.T) (*Manager, *fakeDriver) {
	t.Helper()
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 2, Strategy: "pcie_affinity"}
	manager.manifest.Models[1].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	return manager, driver
}

func TestExplicitGPUsAssignInOrderAndSurviveRefresh(t *testing.T) {
	manager, driver := newExplicitTestManager(t)
	op, noOp, err := manager.Load("alpha", LoadRequest{Instances: ptr(2), GPUs: []int{3, 1, 2, 0}})
	if err != nil || noOp {
		t.Fatalf("Load = (%+v, %v, %v)", op, noOp, err)
	}
	if !reflect.DeepEqual(op.GPUs, []int{3, 1, 2, 0}) {
		t.Fatalf("operation gpus = %v", op.GPUs)
	}
	// Both pairs span the two groups.
	if len(op.Warnings) != 2 || !strings.Contains(op.Warnings[0], "span 2 PCIe groups") {
		t.Fatalf("warnings = %q", op.Warnings)
	}
	waitOperation(t, manager, op.ID, "succeeded")

	driver.mu.Lock()
	labels := map[string][]int{}
	for id, state := range driver.states {
		labels[id] = state.AssignedGPUs
	}
	driver.mu.Unlock()
	want := map[string][]int{"alpha": {1, 3}, "alpha--2": {0, 2}}
	if !reflect.DeepEqual(labels, want) {
		t.Fatalf("container GPUs = %v, want %v", labels, want)
	}

	// Reconciliation reads the placement back from the containers and a
	// repeat auto load is a no-op that keeps it.
	manager.refresh(context.Background())
	if _, noOp, err := manager.Load("alpha", LoadRequest{Instances: ptr(2)}); err != nil || !noOp {
		t.Fatalf("repeat load = (%v, %v)", noOp, err)
	}
	status, _ := manager.Status("alpha")
	got := map[string][]int{}
	for _, instance := range status.Instances {
		got[instance.InstanceID] = instance.AssignedGPUs
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("status GPUs after refresh = %v, want %v", got, want)
	}
}

func TestExplicitGPUsWithinOneGroupHaveNoWarnings(t *testing.T) {
	manager, _ := newExplicitTestManager(t)
	op, _, err := manager.Load("alpha", LoadRequest{GPUs: []int{2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	if len(op.Warnings) != 0 || op.TargetInstances != 1 {
		t.Fatalf("op = %+v", op)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ := manager.Status("alpha")
	if !reflect.DeepEqual(status.AssignedGPUs, []int{2, 3}) {
		t.Fatalf("assigned = %v", status.AssignedGPUs)
	}
}

func TestExplicitGPUsValidation(t *testing.T) {
	manager, _ := newExplicitTestManager(t)
	// beta holds GPU 1.
	op, _, err := manager.Load("beta", LoadRequest{GPUs: []int{1}})
	if err != nil {
		t.Fatal(err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0, 1}, {2, 3}, {4, 5}}
	manager.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0}, {Index: 1}, {Index: 2}, {Index: 3, MemoryUsedMiB: 4096}, {Index: 6},
	}}

	for _, tc := range []struct {
		name    string
		modelID string
		request LoadRequest
		code    string
	}{
		{"wrong count", "alpha", LoadRequest{GPUs: []int{0}}, GPUErrInvalid},
		{"wrong count for replicas", "alpha", LoadRequest{Instances: ptr(2), GPUs: []int{0, 2}}, GPUErrInvalid},
		{"duplicate", "alpha", LoadRequest{GPUs: []int{0, 0}}, GPUErrInvalid},
		{"negative", "alpha", LoadRequest{GPUs: []int{-1, 0}}, GPUErrInvalid},
		{"unknown", "alpha", LoadRequest{GPUs: []int{0, 9}}, GPUErrInvalid},
		{"held by another deployment", "alpha", LoadRequest{GPUs: []int{0, 1}}, GPUErrUnavailable},
		{"above memory threshold", "alpha", LoadRequest{GPUs: []int{2, 3}}, GPUErrUnavailable},
		{"no new instances", "beta", LoadRequest{GPUs: []int{2}}, GPUErrNotApplicable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := manager.Load(tc.modelID, tc.request)
			wantGPUSelectionError(t, err, tc.code)
		})
	}

	// In the topology but missing from nvidia-smi (4), or in nvidia-smi but
	// outside the topology (6): both exist, so both are accepted.
	op, _, err = manager.Load("alpha", LoadRequest{GPUs: []int{4, 6}})
	if err != nil {
		t.Fatalf("topology-or-snapshot GPUs rejected: %v", err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	if len(op.Warnings) != 1 {
		t.Fatalf("ungrouped GPU should warn: %q", op.Warnings)
	}
	status, _ := manager.Status("alpha")
	if !reflect.DeepEqual(status.AssignedGPUs, []int{4, 6}) {
		t.Fatalf("assigned = %v", status.AssignedGPUs)
	}
	if _, _, err := manager.Load("alpha", LoadRequest{Instances: ptr(1), GPUs: []int{0, 2}}); err == nil {
		t.Fatal("scale-to-same-count with gpus succeeded")
	} else {
		wantGPUSelectionError(t, err, GPUErrNotApplicable)
	}
}

func TestExplicitGPUsRequirePlacement(t *testing.T) {
	driver := newFakeDriver()
	manager := newTestManager(t, driver)
	_, _, err := manager.Load("alpha", LoadRequest{GPUs: []int{0}})
	wantGPUSelectionError(t, err, GPUErrNotSupported)
	if _, _, err := manager.Load("missing", LoadRequest{GPUs: []int{0}}); err == nil || errors.As(err, new(*GPUSelectionError)) {
		t.Fatalf("unknown model error = %v", err)
	}
}

func TestExplicitGPUsRestartFailedReplicaOnItsOwnGPUs(t *testing.T) {
	manager, driver := newExplicitTestManager(t)
	op, _, err := manager.Load("alpha", LoadRequest{GPUs: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	driver.mu.Lock()
	state := driver.states["alpha"]
	state.Running, state.ExitCode = false, 1
	driver.states["alpha"] = state
	driver.mu.Unlock()
	manager.refresh(context.Background())

	op, _, err = manager.Load("alpha", LoadRequest{GPUs: []int{1, 0}})
	if err != nil {
		t.Fatalf("restart on own GPUs: %v", err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
}

func TestConcurrentExplicitLoadsRaceForTheSameGPU(t *testing.T) {
	manager, _ := newExplicitTestManager(t)
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	provider := &blockingGPUProvider{
		devices: []gpu.Device{{Index: 0}, {Index: 1}, {Index: 2}, {Index: 3}},
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
			op, _, err := manager.Load(id, LoadRequest{GPUs: []int{3}})
			results <- result{op, err}
		}()
	}
	// Both requests passed preflight and are inside the unlocked snapshot.
	<-provider.entered
	<-provider.entered
	close(provider.release)

	var won, lost int
	for range 2 {
		r := <-results
		switch {
		case r.err == nil:
			won++
			waitOperation(t, manager, r.op.ID, "succeeded")
		default:
			wantGPUSelectionError(t, r.err, GPUErrUnavailable)
			lost++
		}
	}
	if won != 1 || lost != 1 {
		t.Fatalf("won=%d lost=%d, want exactly one winner", won, lost)
	}
	alpha, _ := manager.Status("alpha")
	beta, _ := manager.Status("beta")
	if len(alpha.AssignedGPUs)+len(beta.AssignedGPUs) != 1 {
		t.Fatalf("GPU 3 assigned twice: alpha=%v beta=%v", alpha.AssignedGPUs, beta.AssignedGPUs)
	}
}

func TestExplicitAndAutomaticLoadsShareReservations(t *testing.T) {
	manager, _ := newExplicitTestManager(t)
	op, _, err := manager.Load("beta", LoadRequest{GPUs: []int{0}})
	if err != nil {
		t.Fatal(err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	// alpha (TP2) cannot use group {0,1} any more.
	op, _, err = manager.Load("alpha", LoadRequest{})
	if err != nil {
		t.Fatal(err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ := manager.Status("alpha")
	if !reflect.DeepEqual(status.AssignedGPUs, []int{2, 3}) {
		t.Fatalf("auto placement = %v, want [2 3]", status.AssignedGPUs)
	}
}

type countingGPUProvider struct {
	mu      sync.Mutex
	devices []gpu.Device
	err     error
	calls   atomic.Int64
	gate    chan struct{}
}

func (p *countingGPUProvider) Snapshot(ctx context.Context) ([]gpu.Device, error) {
	p.calls.Add(1)
	if p.gate != nil {
		select {
		case <-p.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.devices, p.err
}

func (p *countingGPUProvider) fail(err error) {
	p.mu.Lock()
	p.err = err
	p.mu.Unlock()
}

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

func TestGPUInventoryCachesSingleFlightsAndServesStale(t *testing.T) {
	manager, _ := newExplicitTestManager(t)
	clock := &fakeClock{now: time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC)}
	manager.now = clock.Now
	provider := &countingGPUProvider{
		devices: []gpu.Device{{Index: 0, Name: "GPU zero", MemoryTotalMiB: 100, MemoryFreeMiB: 100}, {Index: 7, MemoryFreeMiB: 1}},
		gate:    make(chan struct{}),
	}
	manager.gpus = provider

	var wg sync.WaitGroup
	inventories := make([]GPUInventory, 8)
	for i := range inventories {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			inventories[i], err = manager.GPUInventory(context.Background())
			if err != nil {
				t.Errorf("GPUInventory: %v", err)
			}
		}()
	}
	time.Sleep(20 * time.Millisecond)
	close(provider.gate)
	wg.Wait()
	if calls := provider.calls.Load(); calls != 1 {
		t.Fatalf("concurrent inventory ran nvidia-smi %d times, want 1", calls)
	}
	inventory := inventories[0]
	if inventory.Stale || !inventory.SampledAt.Equal(clock.Now()) || !reflect.DeepEqual(inventory.Groups, [][]int{{0, 1}, {2, 3}}) {
		t.Fatalf("inventory = %+v", inventory)
	}
	if g := inventory.GPUs[0]; g.Group == nil || *g.Group != 0 || g.Name != "GPU zero" || g.Assigned == nil || len(g.Assigned) != 0 {
		t.Fatalf("gpu 0 = %+v", g)
	}
	if inventory.GPUs[1].Group != nil {
		t.Fatalf("ungrouped GPU has group %v", *inventory.GPUs[1].Group)
	}

	clock.Advance(4 * time.Second)
	if _, err := manager.GPUInventory(context.Background()); err != nil || provider.calls.Load() != 1 {
		t.Fatalf("fresh snapshot not reused: calls=%d err=%v", provider.calls.Load(), err)
	}

	clock.Advance(2 * time.Second)
	provider.fail(errors.New("nvidia-smi exploded"))
	inventory, err := manager.GPUInventory(context.Background())
	if err != nil || !inventory.Stale || provider.calls.Load() != 2 {
		t.Fatalf("stale fallback = (%+v, %v), calls=%d", inventory, err, provider.calls.Load())
	}
	// A recent failure is cached too, so a broken driver is not re-queried
	// per request.
	if _, err := manager.GPUInventory(context.Background()); err != nil || provider.calls.Load() != 2 {
		t.Fatalf("failure not cached: calls=%d err=%v", provider.calls.Load(), err)
	}

	clock.Advance(time.Minute)
	if _, err := manager.GPUInventory(context.Background()); !errors.Is(err, ErrGPUQueryFailed) {
		t.Fatalf("expired snapshot error = %v, want ErrGPUQueryFailed", err)
	}
}

func TestGPUInventoryListsAssignmentsIncludingLoading(t *testing.T) {
	manager, driver := newExplicitTestManager(t)
	driver.blockStart = make(chan struct{})
	defer close(driver.blockStart)
	op, _, err := manager.Load("alpha", LoadRequest{GPUs: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := manager.GPUInventory(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []GPUAssignment{{ModelID: "alpha", InstanceID: "alpha"}}
	for _, g := range inventory.GPUs {
		switch g.Index {
		case 0, 1:
			if !reflect.DeepEqual(g.Assigned, want) {
				t.Fatalf("GPU %d assigned = %+v", g.Index, g.Assigned)
			}
		default:
			if len(g.Assigned) != 0 {
				t.Fatalf("GPU %d assigned = %+v", g.Index, g.Assigned)
			}
		}
	}
	if status, _ := manager.Status("alpha"); status.Phase != PhaseLoading {
		t.Fatalf("phase = %s, want loading (op %s)", status.Phase, op.ID)
	}
}

func TestStaticGPUCount(t *testing.T) {
	groups := [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}}
	for _, tc := range []struct {
		value  string
		groups [][]int
		want   int
	}{
		{"", groups, 0},
		{"all", groups, 8},
		{"all", nil, 0},
		{"2", nil, 2},
		{"count=3", nil, 3},
		{"count=all,capabilities=compute", groups, 8},
		{"device=0,1,2", nil, 3},
		{`"device=0,1"`, nil, 2},
		{"device=GPU-abc", nil, 1},
		{"device=0,1,capabilities=utility", nil, 2},
	} {
		if got := staticGPUCount(tc.value, tc.groups); got != tc.want {
			t.Fatalf("staticGPUCount(%q) = %d, want %d", tc.value, got, tc.want)
		}
	}
}

func TestModelsExposeStaticGPUCount(t *testing.T) {
	manager := newTestManager(t, newFakeDriver())
	manager.manifest.Models[0].GPUs = "device=0,1"
	if got := manager.Models()[0].GPUCount; got != 2 {
		t.Fatalf("GPUCount = %d, want 2", got)
	}
	if got := manager.Models()[1].GPUCount; got != 0 {
		t.Fatalf("GPU-less GPUCount = %d, want 0", got)
	}
}
