package gpu

import (
	"errors"
	"reflect"
	"testing"
)

func TestParseNVIDIASMI(t *testing.T) {
	devices, err := ParseNVIDIASMI("0, 10, 97800\n1, 0, 97810\n")
	if err != nil {
		t.Fatalf("ParseNVIDIASMI() error = %v", err)
	}
	want := []Device{{Index: 0, MemoryUsedMiB: 10, MemoryFreeMiB: 97800}, {Index: 1, MemoryUsedMiB: 0, MemoryFreeMiB: 97810}}
	if !reflect.DeepEqual(devices, want) {
		t.Fatalf("devices = %+v, want %+v", devices, want)
	}
}

func TestAllocatorRespectsTopologyGroups(t *testing.T) {
	allocator := Allocator{Topology: Topology{
		Groups:           [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}},
		MaxUsedMemoryMiB: 1024,
	}}
	devices := allFreeDevices()

	picked, err := allocator.Allocate(devices, 4, nil)
	if err != nil {
		t.Fatalf("Allocate TP4 error = %v", err)
	}
	if !reflect.DeepEqual(picked, []int{0, 1, 6, 7}) {
		t.Fatalf("TP4 picked %v", picked)
	}

	picked, err = allocator.Allocate(devices, 2, []int{0, 1, 6, 7})
	if err != nil {
		t.Fatalf("Allocate TP2 error = %v", err)
	}
	if !reflect.DeepEqual(picked, []int{2, 3}) {
		t.Fatalf("TP2 picked %v", picked)
	}
}

func TestAllocatorAllowsSparseAvailableWithinGroup(t *testing.T) {
	allocator := Allocator{Topology: Topology{
		Groups:           [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}},
		MaxUsedMemoryMiB: 1024,
	}}
	picked, err := allocator.Allocate(allFreeDevices(), 2, []int{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Allocate TP2 sparse error = %v", err)
	}
	if !reflect.DeepEqual(picked, []int{0, 7}) {
		t.Fatalf("sparse TP2 picked %v", picked)
	}
}

func TestAllocatorTP4FallsBackToAnotherCompleteGroup(t *testing.T) {
	allocator := Allocator{Topology: Topology{
		Groups:           [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}},
		MaxUsedMemoryMiB: 1024,
	}}
	devices := allFreeDevices()
	devices[0].MemoryUsedMiB = 2048

	picked, err := allocator.Allocate(devices, 4, nil)
	if err != nil {
		t.Fatalf("Allocate TP4 fallback error = %v", err)
	}
	if !reflect.DeepEqual(picked, []int{2, 3, 4, 5}) {
		t.Fatalf("TP4 fallback picked %v", picked)
	}
}

func TestAllocatorTP6UsesFullGroupPlusTwo(t *testing.T) {
	allocator := Allocator{Topology: Topology{
		Groups:           [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}},
		MaxUsedMemoryMiB: 1024,
	}}
	picked, err := allocator.Allocate(allFreeDevices(), 6, nil)
	if err != nil {
		t.Fatalf("Allocate TP6 error = %v", err)
	}
	if !reflect.DeepEqual(picked, []int{0, 1, 6, 7, 2, 3}) {
		t.Fatalf("TP6 picked %v", picked)
	}
}

func TestAllocatorFiltersBusyGPUs(t *testing.T) {
	allocator := Allocator{Topology: Topology{
		Groups:           [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}},
		MaxUsedMemoryMiB: 1024,
	}}
	devices := allFreeDevices()
	devices[0].MemoryUsedMiB = 2048
	_, err := allocator.Allocate(devices, 8, nil)
	if !errors.Is(err, ErrInsufficient) {
		t.Fatalf("Allocate busy TP8 error = %v, want ErrInsufficient", err)
	}
}

func allFreeDevices() []Device {
	return []Device{
		{Index: 0}, {Index: 1}, {Index: 2}, {Index: 3},
		{Index: 4}, {Index: 5}, {Index: 6}, {Index: 7},
	}
}
