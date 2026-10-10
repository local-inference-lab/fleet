package gpu

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Device struct {
	Index         int
	MemoryUsedMiB int
	MemoryFreeMiB int
}

type Provider interface {
	Snapshot(context.Context) ([]Device, error)
}

type NVIDIAProvider struct {
	Binary string
}

func NewNVIDIAProvider(binary string) NVIDIAProvider {
	if binary == "" {
		binary = "nvidia-smi"
	}
	return NVIDIAProvider{Binary: binary}
}

// DefaultSnapshotTimeout bounds nvidia-smi when the caller set no deadline; a
// wedged driver otherwise hangs the query indefinitely.
const DefaultSnapshotTimeout = 10 * time.Second

func (p NVIDIAProvider) Snapshot(ctx context.Context) ([]Device, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, DefaultSnapshotTimeout)
		defer cancel()
	}
	out, err := exec.CommandContext(ctx, p.Binary,
		"--query-gpu=index,memory.used,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("query GPUs with nvidia-smi: %w", err)
	}
	return ParseNVIDIASMI(string(out))
}

func ParseNVIDIASMI(raw string) ([]Device, error) {
	var devices []Device
	for lineNo, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 3 {
			return nil, fmt.Errorf("parse nvidia-smi line %d: expected 3 columns", lineNo+1)
		}
		index, err := atoiColumn(parts[0])
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi line %d index: %w", lineNo+1, err)
		}
		used, err := atoiColumn(parts[1])
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi line %d memory.used: %w", lineNo+1, err)
		}
		free, err := atoiColumn(parts[2])
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi line %d memory.free: %w", lineNo+1, err)
		}
		devices = append(devices, Device{Index: index, MemoryUsedMiB: used, MemoryFreeMiB: free})
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i].Index < devices[j].Index })
	return devices, nil
}

func atoiColumn(value string) (int, error) {
	return strconv.Atoi(strings.TrimSpace(value))
}

type Topology struct {
	Groups           [][]int
	MaxUsedMemoryMiB int
}

type Allocator struct {
	Topology Topology
}

var ErrInsufficient = errors.New("insufficient available GPUs")

// candidate is one topology-valid GPU set under consideration.
type candidate struct {
	picked  []int // ascending
	total   int   // summed MemoryFreeMiB
	minimum int   // smallest per-GPU MemoryFreeMiB
	groups  []int // group order used to build it, for deterministic ties
}

// Allocate picks count GPUs, excluding reserved ones and any above the
// topology's MaxUsedMemoryMiB. Among every placement the topology allows
// (count <= 4: one group; 6: a full four-GPU group plus two from another; 8:
// every group; no topology: any GPUs) it returns the one with the most total
// free memory, breaking ties by the larger per-GPU minimum and then by lowest
// group order and index. The result is in ascending index order, which is how
// it lands in the container's GPU label and in Docker's device list.
func (a Allocator) Allocate(devices []Device, count int, reserved []int) ([]int, error) {
	if count <= 0 {
		return nil, nil
	}
	free := a.availableFree(devices, reserved)
	groups := normalizeGroups(a.Topology.Groups)

	var candidates []candidate
	if len(groups) == 0 {
		indexes := make([]int, 0, len(free))
		for index := range free {
			indexes = append(indexes, index)
		}
		if picked, ok := pickMostFree(free, indexes, count); ok {
			candidates = append(candidates, newCandidate(free, nil, picked))
		}
	} else {
		candidates = topologyCandidates(free, groups, count)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("%w: need %d GPUs", ErrInsufficient, count)
	}
	best := candidates[0]
	for _, next := range candidates[1:] {
		if better(next, best) {
			best = next
		}
	}
	return best.picked, nil
}

func topologyCandidates(free map[int]int, groups [][]int, count int) []candidate {
	var candidates []candidate
	switch {
	case count <= 4:
		for i, group := range groups {
			if picked, ok := pickMostFree(free, group, count); ok {
				candidates = append(candidates, newCandidate(free, []int{i}, picked))
			}
		}
	case count == 6:
		for i, primary := range groups {
			if len(primary) != 4 {
				continue
			}
			full, ok := pickMostFree(free, primary, len(primary))
			if !ok {
				continue
			}
			for j, secondary := range groups {
				if i == j {
					continue
				}
				extra, ok := pickMostFree(free, secondary, 2)
				if ok {
					candidates = append(candidates, newCandidate(free, []int{i, j}, append(append([]int{}, full...), extra...)))
				}
			}
		}
	case count == 8:
		var picked []int
		order := make([]int, 0, len(groups))
		for i, group := range groups {
			part, ok := pickMostFree(free, group, len(group))
			if !ok {
				return nil
			}
			picked = append(picked, part...)
			order = append(order, i)
		}
		if len(picked) == count {
			candidates = append(candidates, newCandidate(free, order, picked))
		}
	}
	return candidates
}

// pickMostFree returns the count available indexes from pool with the most
// free memory (lower index first on ties).
func pickMostFree(free map[int]int, pool []int, count int) ([]int, bool) {
	available := make([]int, 0, len(pool))
	for _, index := range pool {
		if _, ok := free[index]; ok {
			available = append(available, index)
		}
	}
	if len(available) < count {
		return nil, false
	}
	sort.Slice(available, func(i, j int) bool {
		if free[available[i]] != free[available[j]] {
			return free[available[i]] > free[available[j]]
		}
		return available[i] < available[j]
	})
	return append([]int(nil), available[:count]...), true
}

func newCandidate(free map[int]int, groups []int, picked []int) candidate {
	sort.Ints(picked)
	c := candidate{picked: picked, groups: groups}
	for i, index := range picked {
		c.total += free[index]
		if i == 0 || free[index] < c.minimum {
			c.minimum = free[index]
		}
	}
	return c
}

func better(left, right candidate) bool {
	if left.total != right.total {
		return left.total > right.total
	}
	if left.minimum != right.minimum {
		return left.minimum > right.minimum
	}
	if c := compareInts(left.groups, right.groups); c != 0 {
		return c < 0
	}
	return compareInts(left.picked, right.picked) < 0
}

func compareInts(left, right []int) int {
	for i := 0; i < len(left) && i < len(right); i++ {
		if left[i] != right[i] {
			if left[i] < right[i] {
				return -1
			}
			return 1
		}
	}
	return len(left) - len(right)
}

// availableFree maps each allocatable GPU to its free memory.
func (a Allocator) availableFree(devices []Device, reserved []int) map[int]int {
	reservedSet := make(map[int]bool, len(reserved))
	for _, index := range reserved {
		reservedSet[index] = true
	}
	available := make(map[int]int, len(devices))
	for _, device := range devices {
		if reservedSet[device.Index] || a.Topology.OverThreshold(device) {
			continue
		}
		available[device.Index] = device.MemoryFreeMiB
	}
	return available
}

// OverThreshold reports whether a GPU already uses more memory than the
// topology allows a new placement to share.
func (t Topology) OverThreshold(device Device) bool {
	return t.MaxUsedMemoryMiB > 0 && device.MemoryUsedMiB > t.MaxUsedMemoryMiB
}

// GroupOf returns the index of the topology group containing gpu.
func (t Topology) GroupOf(gpu int) (int, bool) {
	for i, group := range t.Groups {
		for _, index := range group {
			if index == gpu {
				return i, true
			}
		}
	}
	return 0, false
}

func normalizeGroups(groups [][]int) [][]int {
	result := make([][]int, 0, len(groups))
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		copyGroup := append([]int(nil), group...)
		result = append(result, copyGroup)
	}
	return result
}
