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

// Device is one GPU from an nvidia-smi snapshot. Index, MemoryUsedMiB and
// MemoryFreeMiB are all the allocator needs; the rest feeds the inventory API.
type Device struct {
	Index          int
	Name           string
	MemoryTotalMiB int
	MemoryUsedMiB  int
	MemoryFreeMiB  int
	// UtilizationPercent is nil when nvidia-smi reports it as unavailable.
	UtilizationPercent *int
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

// QueryFields is the nvidia-smi --query-gpu column list ParseNVIDIASMI expects.
const QueryFields = "index,name,memory.total,memory.used,memory.free,utilization.gpu"

func (p NVIDIAProvider) Snapshot(ctx context.Context) ([]Device, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, DefaultSnapshotTimeout)
		defer cancel()
	}
	out, err := exec.CommandContext(ctx, p.Binary,
		"--query-gpu="+QueryFields,
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("query GPUs with nvidia-smi: %w", err)
	}
	return ParseNVIDIASMI(string(out))
}

// ParseNVIDIASMI parses `--query-gpu=` QueryFields CSV output. The legacy
// three-column `index,memory.used,memory.free` form is still accepted. GPU
// names may contain spaces (and, defensively, commas): the name is everything
// between the first column and the last four.
func ParseNVIDIASMI(raw string) ([]Device, error) {
	var devices []Device
	for lineNo, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		device, err := parseDeviceLine(parts)
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi line %d: %w", lineNo+1, err)
		}
		devices = append(devices, device)
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i].Index < devices[j].Index })
	return devices, nil
}

func parseDeviceLine(parts []string) (Device, error) {
	var device Device
	var err error
	if len(parts) == 3 {
		if device.Index, err = atoiColumn("index", parts[0]); err != nil {
			return Device{}, err
		}
		if device.MemoryUsedMiB, err = atoiColumn("memory.used", parts[1]); err != nil {
			return Device{}, err
		}
		if device.MemoryFreeMiB, err = atoiColumn("memory.free", parts[2]); err != nil {
			return Device{}, err
		}
		return device, nil
	}
	if len(parts) < 6 {
		return Device{}, fmt.Errorf("expected 6 columns, got %d", len(parts))
	}
	tail := parts[len(parts)-4:]
	if device.Index, err = atoiColumn("index", parts[0]); err != nil {
		return Device{}, err
	}
	device.Name = strings.TrimSpace(strings.Join(parts[1:len(parts)-4], ","))
	if device.MemoryTotalMiB, err = atoiColumn("memory.total", tail[0]); err != nil {
		return Device{}, err
	}
	if device.MemoryUsedMiB, err = atoiColumn("memory.used", tail[1]); err != nil {
		return Device{}, err
	}
	if device.MemoryFreeMiB, err = atoiColumn("memory.free", tail[2]); err != nil {
		return Device{}, err
	}
	if value := strings.TrimSpace(tail[3]); !unavailableValue(value) {
		utilization, err := atoiColumn("utilization.gpu", value)
		if err != nil {
			return Device{}, err
		}
		device.UtilizationPercent = &utilization
	}
	return device, nil
}

// unavailableValue recognizes nvidia-smi placeholders such as "[N/A]" and
// "[Not Supported]" (some drivers omit the brackets).
func unavailableValue(value string) bool {
	if value == "" {
		return true
	}
	if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		return true
	}
	switch strings.ToLower(value) {
	case "n/a", "not supported", "unknown error":
		return true
	}
	return false
}

func atoiColumn(name, value string) (int, error) {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return 0, fmt.Errorf("%s: invalid value %q", name, strings.TrimSpace(value))
	}
	if parsed < 0 {
		return 0, fmt.Errorf("%s: negative value %d", name, parsed)
	}
	return parsed, nil
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
	picked  []int // device order handed to Docker
	sorted  []int // ascending copy, for deterministic tie-breaks
	total   int   // summed MemoryFreeMiB
	minimum int   // smallest per-GPU MemoryFreeMiB
	groups  []int // group order used to build it, for deterministic ties
}

// Allocate picks count GPUs, excluding reserved ones and any above the
// topology's MaxUsedMemoryMiB. Among every placement the topology allows
// (count <= 4: one group; 6: a full four-GPU group plus two from another; 8:
// every group; no topology: any GPUs) it returns the one with the most total
// free memory, breaking ties by the larger per-GPU minimum and then by lowest
// group order and index.
//
// The returned order is the device order handed to Docker and stored in the
// container label, so it determines the rank -> GPU mapping of PCIe allreduce
// collectives. It follows topology traversal: GPUs within a group in manifest
// order; TP6 lists the full primary group, then the two extra GPUs; TP8
// concatenates the groups in manifest order. Without a topology it is
// ascending.
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
		sort.Ints(indexes)
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

// pickMostFree chooses the count available indexes from pool with the most
// free memory and returns them in pool order. Ties go to the earlier pool
// position, i.e. the group's manifest order, as first-fit placement did.
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
	ranked := append([]int(nil), available...)
	sort.SliceStable(ranked, func(i, j int) bool { return free[ranked[i]] > free[ranked[j]] })
	chosen := make(map[int]bool, count)
	for _, index := range ranked[:count] {
		chosen[index] = true
	}
	picked := make([]int, 0, count)
	for _, index := range available {
		if chosen[index] {
			picked = append(picked, index)
		}
	}
	return picked, true
}

func newCandidate(free map[int]int, groups []int, picked []int) candidate {
	sorted := append([]int(nil), picked...)
	sort.Ints(sorted)
	c := candidate{picked: picked, sorted: sorted, groups: groups}
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
	return compareInts(left.sorted, right.sorted) < 0
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
