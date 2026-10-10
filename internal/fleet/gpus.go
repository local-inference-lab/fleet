package fleet

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

// GPU selection error codes, returned verbatim as API error codes.
const (
	GPUErrNotSupported  = "gpus_not_supported"
	GPUErrNotApplicable = "gpus_not_applicable"
	GPUErrInvalid       = "invalid_gpus"
	GPUErrUnavailable   = "gpus_unavailable"
)

// GPUSelectionError rejects an explicit GPU list on a load request. Code is
// one of the GPUErr* constants.
type GPUSelectionError struct {
	Code string
	Err  error
}

func (e *GPUSelectionError) Error() string { return e.Err.Error() }
func (e *GPUSelectionError) Unwrap() error { return e.Err }

func gpuSelectionErrorf(code, format string, args ...any) *GPUSelectionError {
	return &GPUSelectionError{Code: code, Err: fmt.Errorf(format, args...)}
}

// validateGPUListShape checks what needs no state: indices are non-negative
// and unique.
func validateGPUListShape(gpus []int) error {
	seen := make(map[int]bool, len(gpus))
	for _, index := range gpus {
		if index < 0 {
			return gpuSelectionErrorf(GPUErrInvalid, "gpus contains negative index %d", index)
		}
		if seen[index] {
			return gpuSelectionErrorf(GPUErrInvalid, "gpus contains duplicate index %d", index)
		}
		seen[index] = true
	}
	return nil
}

// startIndexesLocked lists the instance indexes an activation to target would
// start: every index up to target without a keepable instance. It mirrors the
// keep rule in planInstancesLocked.
func (m *Manager) startIndexesLocked(modelID string, target int) []int {
	kept := map[int]bool{}
	for _, instance := range m.statuses[modelID].Instances {
		if instance.Index <= target && keepableInstance(instance) {
			kept[instance.Index] = true
		}
	}
	var starts []int
	for index := 1; index <= target; index++ {
		if !kept[index] {
			starts = append(starts, index)
		}
	}
	return starts
}

// checkGPUCountLocked requires an explicit GPU list to cover exactly the
// instances this request starts.
func (m *Manager) checkGPUCountLocked(model manifest.Model, target int, gpus []int) error {
	starts := m.startIndexesLocked(model.ID, target)
	if len(starts) == 0 {
		return gpuSelectionErrorf(GPUErrNotApplicable, "gpus apply only to newly started instances, and this request starts none of %s's %d instances", model.ID, target)
	}
	if want := model.Placement.GPUCount * len(starts); len(gpus) != want {
		return gpuSelectionErrorf(GPUErrInvalid, "gpus lists %d GPUs, want %d (gpu_count %d x %d new instances)", len(gpus), want, model.Placement.GPUCount, len(starts))
	}
	return nil
}

// validateExplicitGPUsLocked checks an operator's GPU pick against the
// snapshot and current reservations, and splits it into one ascending GPU
// list per started instance. Topology group rules are deliberately not
// enforced; a pick spanning groups only produces a warning.
func (m *Manager) validateExplicitGPUsLocked(cfg *manifest.Manifest, model manifest.Model, startIndexes []int, devices []gpu.Device, snapshotErr error, gpus []int) ([][]int, []string, error) {
	topology := gpu.Topology{Groups: cfg.Runtime.GPUTopology.Groups, MaxUsedMemoryMiB: cfg.Runtime.GPUTopology.MaxUsedMemoryMiB}
	byIndex := make(map[int]gpu.Device, len(devices))
	if snapshotErr == nil {
		for _, device := range devices {
			byIndex[device.Index] = device
		}
	}
	var unknown []int
	for _, index := range gpus {
		_, inSnapshot := byIndex[index]
		_, inTopology := topology.GroupOf(index)
		if !inSnapshot && !inTopology {
			unknown = append(unknown, index)
		}
	}
	if len(unknown) != 0 {
		return nil, nil, gpuSelectionErrorf(GPUErrInvalid, "gpus %s are not in the GPU snapshot or the manifest topology", formatIndexes(unknown))
	}
	if snapshotErr != nil {
		return nil, nil, snapshotErr
	}

	// The instances being (re)started give their old GPUs back, so a failed
	// replica may be restarted on its own cards.
	restarting := make(map[string]bool, len(startIndexes))
	for _, index := range startIndexes {
		restarting[instanceID(model.ID, index)] = true
	}
	holders := m.gpuHoldersLocked()
	var busy []string
	for _, index := range gpus {
		for _, holder := range holders[index] {
			if holder.ModelID == model.ID && restarting[holder.InstanceID] {
				continue
			}
			busy = append(busy, fmt.Sprintf("GPU %d is assigned to %s", index, holder.InstanceID))
			break
		}
		if device, ok := byIndex[index]; ok && topology.OverThreshold(device) {
			busy = append(busy, fmt.Sprintf("GPU %d uses %d MiB, above max_used_memory_mib %d", index, device.MemoryUsedMiB, topology.MaxUsedMemoryMiB))
		}
	}
	if len(busy) != 0 {
		return nil, nil, &GPUSelectionError{Code: GPUErrUnavailable, Err: errors.New(strings.Join(busy, "; "))}
	}

	count := model.Placement.GPUCount
	perInstance := make([][]int, len(startIndexes))
	var warnings []string
	for i, index := range startIndexes {
		chunk := append([]int(nil), gpus[i*count:(i+1)*count]...)
		sort.Ints(chunk)
		perInstance[i] = chunk
		if count > 4 || len(topology.Groups) == 0 {
			continue
		}
		spanned := map[int]bool{}
		for _, gpuIndex := range chunk {
			group, ok := topology.GroupOf(gpuIndex)
			if !ok {
				group = -1 - gpuIndex // outside every group: its own island
			}
			spanned[group] = true
		}
		if len(spanned) > 1 {
			warnings = append(warnings, fmt.Sprintf("instance %s GPUs %s span %d PCIe groups; tensor-parallel traffic will cross group boundaries", instanceID(model.ID, index), formatIndexes(chunk), len(spanned)))
		}
	}
	return perInstance, warnings, nil
}

func formatIndexes(indexes []int) string {
	parts := make([]string, len(indexes))
	for i, index := range indexes {
		parts[i] = strconv.Itoa(index)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// GPUAssignment names a Fleet deployment instance holding a GPU.
type GPUAssignment struct {
	ModelID    string `json:"model_id"`
	InstanceID string `json:"instance_id"`
}

// gpuHoldersLocked maps each GPU to the Fleet instances holding it: every
// instance that is not unloaded (loading, stopping and failed ones included,
// matching what blocks allocation), plus a legacy instance-less active status.
func (m *Manager) gpuHoldersLocked() map[int][]GPUAssignment {
	holders := map[int][]GPUAssignment{}
	for modelID, status := range m.statuses {
		for _, instance := range status.Instances {
			if instance.Phase == PhaseUnloaded {
				continue
			}
			for _, index := range instance.AssignedGPUs {
				holders[index] = append(holders[index], GPUAssignment{ModelID: modelID, InstanceID: instance.InstanceID})
			}
		}
		if len(status.Instances) == 0 && statusActive(status) {
			for _, index := range status.AssignedGPUs {
				holders[index] = append(holders[index], GPUAssignment{ModelID: modelID, InstanceID: modelID})
			}
		}
	}
	for index := range holders {
		list := holders[index]
		sort.Slice(list, func(i, j int) bool {
			if list[i].ModelID != list[j].ModelID {
				return list[i].ModelID < list[j].ModelID
			}
			return list[i].InstanceID < list[j].InstanceID
		})
	}
	return holders
}

const (
	// gpuInventoryMaxAge is how long the inventory endpoint reuses a
	// snapshot (or a failure) before running nvidia-smi again.
	gpuInventoryMaxAge = 5 * time.Second
	// gpuInventoryStaleLimit is the oldest snapshot served, marked stale,
	// when nvidia-smi fails.
	gpuInventoryStaleLimit = 60 * time.Second
)

// ErrGPUQueryFailed reports that no usable GPU snapshot is available.
var ErrGPUQueryFailed = errors.New("GPU query failed")

// gpuSnapshotCache remembers the last nvidia-smi result and single-flights
// inventory refreshes. Its mutex is never held while nvidia-smi runs.
type gpuSnapshotCache struct {
	mu        sync.Mutex
	devices   []gpu.Device
	sampledAt time.Time // last success
	failErr   error
	failedAt  time.Time
	inflight  chan struct{}
}

// store records a snapshot taken by anyone (the inventory refresh or a load).
func (c *gpuSnapshotCache) store(devices []gpu.Device, err error, at time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		if at.After(c.failedAt) {
			c.failErr, c.failedAt = err, at
		}
		return
	}
	if at.After(c.sampledAt) {
		c.devices, c.sampledAt = append([]gpu.Device(nil), devices...), at
	}
}

// cachedGPUSnapshot returns a snapshot at most gpuInventoryMaxAge old,
// running at most one nvidia-smi at a time. On failure it falls back to a
// snapshot under gpuInventoryStaleLimit old, reported as stale.
func (m *Manager) cachedGPUSnapshot(ctx context.Context) ([]gpu.Device, time.Time, bool, error) {
	c := &m.gpuCache
	for attempt := 0; ; attempt++ {
		c.mu.Lock()
		now := m.now()
		failedRecently := c.failErr != nil && c.failedAt.After(c.sampledAt) && now.Sub(c.failedAt) < gpuInventoryMaxAge
		switch {
		case failedRecently || (attempt > 0 && c.failErr != nil && c.failedAt.After(c.sampledAt)):
			devices, sampledAt, err := append([]gpu.Device(nil), c.devices...), c.sampledAt, c.failErr
			c.mu.Unlock()
			if !sampledAt.IsZero() && now.Sub(sampledAt) < gpuInventoryStaleLimit {
				return devices, sampledAt, true, nil
			}
			return nil, time.Time{}, false, fmt.Errorf("%w: %v", ErrGPUQueryFailed, err)
		case !c.sampledAt.IsZero() && (now.Sub(c.sampledAt) < gpuInventoryMaxAge || attempt > 0):
			devices, sampledAt := append([]gpu.Device(nil), c.devices...), c.sampledAt
			c.mu.Unlock()
			return devices, sampledAt, false, nil
		}
		if c.inflight == nil {
			done := make(chan struct{})
			c.inflight = done
			go m.refreshGPUSnapshot(done)
		}
		wait := c.inflight
		c.mu.Unlock()
		select {
		case <-wait:
		case <-ctx.Done():
			return nil, time.Time{}, false, ctx.Err()
		}
	}
}

// refreshGPUSnapshot runs nvidia-smi detached from any one request, so a
// client hanging up does not cancel the refresh other callers wait on.
func (m *Manager) refreshGPUSnapshot(done chan struct{}) {
	ctx, cancel := context.WithTimeout(context.Background(), gpuSnapshotTimeout)
	devices, err := m.gpus.Snapshot(ctx)
	cancel()
	m.gpuCache.store(devices, err, m.now())
	m.gpuCache.mu.Lock()
	if m.gpuCache.inflight == done {
		m.gpuCache.inflight = nil
	}
	m.gpuCache.mu.Unlock()
	close(done)
}

// GPUInfo is one GPU in the inventory API.
type GPUInfo struct {
	Index              int             `json:"index"`
	Name               string          `json:"name"`
	MemoryTotalMiB     int             `json:"memory_total_mib"`
	MemoryUsedMiB      int             `json:"memory_used_mib"`
	MemoryFreeMiB      int             `json:"memory_free_mib"`
	UtilizationPercent *int            `json:"utilization_percent"`
	Group              *int            `json:"group"`
	Assigned           []GPUAssignment `json:"assigned"`
}

// GPUInventory is the GET /v1/gpus response.
type GPUInventory struct {
	SampledAt time.Time `json:"sampled_at"`
	Stale     bool      `json:"stale,omitempty"`
	Groups    [][]int   `json:"groups"`
	GPUs      []GPUInfo `json:"gpus"`
}

// GPUInventory joins the cached nvidia-smi snapshot with the manifest
// topology and Fleet's current GPU assignments.
func (m *Manager) GPUInventory(ctx context.Context) (GPUInventory, error) {
	devices, sampledAt, stale, err := m.cachedGPUSnapshot(ctx)
	if err != nil {
		return GPUInventory{}, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	groups := make([][]int, 0, len(m.manifest.Runtime.GPUTopology.Groups))
	for _, group := range m.manifest.Runtime.GPUTopology.Groups {
		groups = append(groups, append([]int{}, group...))
	}
	topology := gpu.Topology{Groups: groups}
	holders := m.gpuHoldersLocked()
	inventory := GPUInventory{SampledAt: sampledAt.UTC(), Stale: stale, Groups: groups, GPUs: make([]GPUInfo, 0, len(devices))}
	for _, device := range devices {
		info := GPUInfo{
			Index: device.Index, Name: device.Name,
			MemoryTotalMiB: device.MemoryTotalMiB, MemoryUsedMiB: device.MemoryUsedMiB, MemoryFreeMiB: device.MemoryFreeMiB,
			Assigned: append([]GPUAssignment{}, holders[device.Index]...),
		}
		if device.UtilizationPercent != nil {
			utilization := *device.UtilizationPercent
			info.UtilizationPercent = &utilization
		}
		if group, ok := topology.GroupOf(device.Index); ok {
			info.Group = &group
		}
		inventory.GPUs = append(inventory.GPUs, info)
	}
	return inventory, nil
}

// staticGPUCount derives a GPU count from a model's Docker --gpus value
// ("all", "2", "count=2", "device=0,1"; optionally quoted). It returns 0 when
// the count is unknown, e.g. "all" without a manifest topology.
func staticGPUCount(value string, groups [][]int) int {
	value = strings.Trim(strings.TrimSpace(value), "\"'")
	if value == "" {
		return 0
	}
	all := 0
	for _, group := range groups {
		all += len(group)
	}
	if value == "all" {
		return all
	}
	if count, err := strconv.Atoi(value); err == nil && count > 0 {
		return count
	}
	devices, counting := 0, false
	for _, part := range strings.Split(value, ",") {
		part = strings.Trim(strings.TrimSpace(part), "\"'")
		key, rest, hasKey := strings.Cut(part, "=")
		switch {
		case hasKey && key == "count":
			counting = false
			if rest == "all" {
				return all
			}
			if count, err := strconv.Atoi(rest); err == nil && count > 0 {
				return count
			}
		case hasKey && key == "device":
			counting = rest != ""
			if counting {
				devices++
			}
		case hasKey:
			counting = false
		case counting && part != "":
			devices++
		}
	}
	return devices
}
