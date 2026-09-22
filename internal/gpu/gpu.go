package gpu

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"
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

func (p NVIDIAProvider) Snapshot(ctx context.Context) ([]Device, error) {
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

func (a Allocator) Allocate(devices []Device, count int, reserved []int) ([]int, error) {
	if count <= 0 {
		return nil, nil
	}
	available := a.availableSet(devices, reserved)
	groups := normalizeGroups(a.Topology.Groups)
	if len(groups) == 0 {
		return selectFromSorted(available, count)
	}

	switch {
	case count <= 4:
		for _, group := range groups {
			if picked, ok := selectFromGroup(available, group, count); ok {
				return picked, nil
			}
		}
	case count == 6:
		for i, primary := range groups {
			full, ok := selectFromGroup(available, primary, len(primary))
			if !ok || len(full) != 4 {
				continue
			}
			for j, secondary := range groups {
				if i == j {
					continue
				}
				extra, ok := selectFromGroup(available, secondary, 2)
				if ok {
					return append(append([]int{}, full...), extra...), nil
				}
			}
		}
	case count == 8:
		var picked []int
		for _, group := range groups {
			part, ok := selectFromGroup(available, group, len(group))
			if !ok {
				return nil, fmt.Errorf("%w: need %d GPUs", ErrInsufficient, count)
			}
			picked = append(picked, part...)
		}
		if len(picked) == count {
			return picked, nil
		}
	}
	return nil, fmt.Errorf("%w: need %d GPUs", ErrInsufficient, count)
}

func (a Allocator) availableSet(devices []Device, reserved []int) map[int]bool {
	reservedSet := make(map[int]bool, len(reserved))
	for _, index := range reserved {
		reservedSet[index] = true
	}
	maxUsed := a.Topology.MaxUsedMemoryMiB
	available := make(map[int]bool, len(devices))
	for _, device := range devices {
		if reservedSet[device.Index] {
			continue
		}
		if maxUsed > 0 && device.MemoryUsedMiB > maxUsed {
			continue
		}
		available[device.Index] = true
	}
	return available
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

func selectFromGroup(available map[int]bool, group []int, count int) ([]int, bool) {
	picked := make([]int, 0, count)
	for _, index := range group {
		if available[index] {
			picked = append(picked, index)
			if len(picked) == count {
				return picked, true
			}
		}
	}
	return nil, false
}

func selectFromSorted(available map[int]bool, count int) ([]int, error) {
	indexes := make([]int, 0, len(available))
	for index := range available {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	if len(indexes) < count {
		return nil, fmt.Errorf("%w: need %d GPUs", ErrInsufficient, count)
	}
	return indexes[:count], nil
}
