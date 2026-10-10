package fleet

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

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
	op, _, err := manager.Activate("alpha")
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
