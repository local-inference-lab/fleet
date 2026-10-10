package gpu

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func intPtr(value int) *int { return &value }

func TestParseNVIDIASMI(t *testing.T) {
	raw := "1, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 1234, 96000, 37\n" +
		"0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 10, 97800, [N/A]\n" +
		"\n"
	devices, err := ParseNVIDIASMI(raw)
	if err != nil {
		t.Fatalf("ParseNVIDIASMI() error = %v", err)
	}
	want := []Device{
		{Index: 0, Name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", MemoryTotalMiB: 97887, MemoryUsedMiB: 10, MemoryFreeMiB: 97800},
		{Index: 1, Name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", MemoryTotalMiB: 97887, MemoryUsedMiB: 1234, MemoryFreeMiB: 96000, UtilizationPercent: intPtr(37)},
	}
	if !reflect.DeepEqual(devices, want) {
		t.Fatalf("devices = %+v, want %+v", devices, want)
	}
}

func TestParseNVIDIASMIRobustness(t *testing.T) {
	devices, err := ParseNVIDIASMI("0, Weird, Name, With Commas, 100, 40, 60, [Not Supported]\n")
	if err != nil {
		t.Fatalf("comma name error = %v", err)
	}
	if devices[0].Name != "Weird, Name, With Commas" || devices[0].MemoryFreeMiB != 60 || devices[0].UtilizationPercent != nil {
		t.Fatalf("comma name parsed as %+v", devices[0])
	}

	legacy, err := ParseNVIDIASMI("0, 10, 97800\n1, 0, 97810\n")
	if err != nil {
		t.Fatalf("legacy error = %v", err)
	}
	if want := []Device{{Index: 0, MemoryUsedMiB: 10, MemoryFreeMiB: 97800}, {Index: 1, MemoryUsedMiB: 0, MemoryFreeMiB: 97810}}; !reflect.DeepEqual(legacy, want) {
		t.Fatalf("legacy = %+v, want %+v", legacy, want)
	}

	for _, bad := range []string{
		"0, GPU, 100, [N/A], 60, 1", // memory is never optional
		"0, GPU, 100, 40, 60",       // too few columns
		"x, GPU, 100, 40, 60, 1",    // bad index
		"0, GPU, 100, 40, 60, lots", // bad utilization
		"-1, GPU, 100, 40, 60, 1",   // negative index
	} {
		if _, err := ParseNVIDIASMI(bad); err == nil {
			t.Fatalf("ParseNVIDIASMI(%q) succeeded", bad)
		}
	}
}

var b12xGroups = [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}}

// devicesWithFree builds the eight b12x GPUs with the given free memory
// (indexed by GPU) and no used memory.
func devicesWithFree(free ...int) []Device {
	devices := make([]Device, len(free))
	for i, value := range free {
		devices[i] = Device{Index: i, MemoryFreeMiB: value}
	}
	return devices
}

func TestAllocatorPicksMostFreeVRAM(t *testing.T) {
	even := devicesWithFree(100, 100, 100, 100, 100, 100, 100, 100)
	// Group B {2,3,4,5} has more free memory overall; GPU 6 is the single
	// emptiest card.
	uneven := devicesWithFree(50, 60, 90, 80, 85, 70, 100, 40)

	tests := []struct {
		name     string
		groups   [][]int
		maxUsed  int
		devices  []Device
		count    int
		reserved []int
		want     []int
		wantErr  bool
	}{
		{name: "tp1 tie takes lowest group and index", groups: b12xGroups, devices: even, count: 1, want: []int{0}},
		{name: "tp1 takes emptiest card anywhere", groups: b12xGroups, devices: uneven, count: 1, want: []int{6}},
		{name: "tp2 picks group with most free pair", groups: b12xGroups, devices: uneven, count: 2, want: []int{2, 4}},
		{name: "tp2 tie takes first group", groups: b12xGroups, devices: even, count: 2, want: []int{0, 1}},
		{name: "tp3 picks most free trio", groups: b12xGroups, devices: uneven, count: 3, want: []int{2, 3, 4}},
		{name: "tp4 picks group with most total free", groups: b12xGroups, devices: uneven, count: 4, want: []int{2, 3, 4, 5}},
		{name: "tp4 tie takes first group", groups: b12xGroups, devices: even, count: 4, want: []int{0, 1, 6, 7}},
		{name: "tp6 full group plus most free pair from the other", groups: b12xGroups, devices: uneven, count: 6, want: []int{1, 2, 3, 4, 5, 6}},
		{name: "tp6 tie is ascending first group plus first pair", groups: b12xGroups, devices: even, count: 6, want: []int{0, 1, 2, 3, 6, 7}},
		{name: "tp8 takes every GPU in ascending order", groups: b12xGroups, devices: uneven, count: 8, want: []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{
			name: "tie on total breaks on higher minimum", groups: b12xGroups, count: 2,
			// Group A best pair 100+40=140 (min 40); group B best pair 70+70=140 (min 70).
			devices: devicesWithFree(100, 40, 70, 70, 10, 10, 0, 0),
			want:    []int{2, 3},
		},
		{
			name: "reserved GPUs are skipped", groups: b12xGroups, devices: uneven, count: 2,
			reserved: []int{2, 4}, want: []int{1, 6},
		},
		{
			name: "threshold excludes busy GPUs", groups: b12xGroups, maxUsed: 1024, count: 4,
			devices: func() []Device {
				d := devicesWithFree(50, 60, 90, 80, 85, 70, 100, 40)
				d[3].MemoryUsedMiB = 2048
				return d
			}(),
			want: []int{0, 1, 6, 7},
		},
		{
			name: "threshold makes tp8 impossible", groups: b12xGroups, maxUsed: 1024, count: 8,
			devices: func() []Device {
				d := devicesWithFree(100, 100, 100, 100, 100, 100, 100, 100)
				d[0].MemoryUsedMiB = 2048
				return d
			}(),
			wantErr: true,
		},
		{name: "tp6 needs a complete group", groups: b12xGroups, devices: uneven, count: 6, reserved: []int{0, 2}, wantErr: true},
		{name: "tp4 does not straddle groups", groups: b12xGroups, devices: uneven, count: 4, reserved: []int{0, 2}, wantErr: true},
		{name: "unsupported count with topology", groups: b12xGroups, devices: uneven, count: 5, wantErr: true},
		{name: "no topology takes most free anywhere", devices: uneven, count: 3, want: []int{2, 4, 6}},
		{name: "no topology still honors reservations", devices: uneven, count: 2, reserved: []int{6}, want: []int{2, 4}},
		{name: "no topology insufficient", devices: uneven, count: 9, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allocator := Allocator{Topology: Topology{Groups: tt.groups, MaxUsedMemoryMiB: tt.maxUsed}}
			picked, err := allocator.Allocate(tt.devices, tt.count, tt.reserved)
			if tt.wantErr {
				if !errors.Is(err, ErrInsufficient) {
					t.Fatalf("Allocate() = (%v, %v), want ErrInsufficient", picked, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Allocate() error = %v", err)
			}
			if !reflect.DeepEqual(picked, tt.want) {
				t.Fatalf("Allocate() = %v, want %v", picked, tt.want)
			}
		})
	}
}

func TestAllocatorIsDeterministicAcrossDeviceOrder(t *testing.T) {
	allocator := Allocator{Topology: Topology{Groups: b12xGroups}}
	devices := devicesWithFree(100, 100, 100, 100, 100, 100, 100, 100)
	reversed := make([]Device, len(devices))
	for i := range devices {
		reversed[len(devices)-1-i] = devices[i]
	}
	for range 20 {
		picked, err := allocator.Allocate(reversed, 2, nil)
		if err != nil || !reflect.DeepEqual(picked, []int{0, 1}) {
			t.Fatalf("Allocate() = (%v, %v), want [0 1]", picked, err)
		}
	}
}

func TestAllocatorTP2AroundExistingTP4(t *testing.T) {
	allocator := Allocator{Topology: Topology{Groups: b12xGroups, MaxUsedMemoryMiB: 1024}}
	picked, err := allocator.Allocate(devicesWithFree(0, 0, 0, 0, 0, 0, 0, 0), 2, []int{0, 1, 6, 7})
	if err != nil || !reflect.DeepEqual(picked, []int{2, 3}) {
		t.Fatalf("Allocate() = (%v, %v), want [2 3]", picked, err)
	}
	picked, err = allocator.Allocate(devicesWithFree(0, 0, 0, 0, 0, 0, 0, 0), 2, []int{1, 2, 3, 4, 5, 6})
	if err != nil || !reflect.DeepEqual(picked, []int{0, 7}) {
		t.Fatalf("sparse Allocate() = (%v, %v), want [0 7]", picked, err)
	}
}

func TestTopologyHelpers(t *testing.T) {
	topology := Topology{Groups: b12xGroups, MaxUsedMemoryMiB: 1024}
	if group, ok := topology.GroupOf(6); !ok || group != 0 {
		t.Fatalf("GroupOf(6) = (%d, %v)", group, ok)
	}
	if _, ok := topology.GroupOf(9); ok {
		t.Fatal("GroupOf(9) found a group")
	}
	if topology.OverThreshold(Device{MemoryUsedMiB: 1024}) || !topology.OverThreshold(Device{MemoryUsedMiB: 1025}) {
		t.Fatal("OverThreshold boundary is wrong")
	}
	if (Topology{}).OverThreshold(Device{MemoryUsedMiB: 1 << 20}) {
		t.Fatal("zero threshold must not exclude GPUs")
	}
	if !strings.Contains(QueryFields, "utilization.gpu") {
		t.Fatal("QueryFields misses utilization")
	}
}
