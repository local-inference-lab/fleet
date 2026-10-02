package fleet

import (
	"bytes"
	"context"
	"encoding/json"
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
	// One container listing per reconcile, only models that have containers
	// are stopped, and readiness is observed with a single listing.
	want := []string{"list", "stop:alpha:true", "start:beta", "list"}
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

func TestLifecycleLogsCompletedOperations(t *testing.T) {
	for _, tc := range []struct {
		name      string
		model     string
		operation string
		initial   bool
		startErr  error
		stopErr   error
		wantState string
		wantLogs  []string
	}{
		{"load", "alpha", "activate", false, nil, nil, "succeeded", []string{"model loaded:alpha:INFO"}},
		{"unload", "alpha", "unload", true, nil, nil, "succeeded", []string{"model unloaded:alpha:INFO"}},
		{"load failure", "alpha", "activate", false, errors.New("start failed"), nil, "failed", []string{"model load failed:alpha:ERROR"}},
		{"unload failure", "alpha", "unload", true, nil, errors.New("stop failed"), "failed", []string{"model unload failed:alpha:ERROR"}},
		{"switch", "beta", "activate", true, nil, nil, "succeeded", []string{"model unloaded:alpha:INFO", "model loaded:beta:INFO"}},
		{"switch unload failure", "beta", "activate", true, nil, errors.New("stop failed"), "failed", []string{"model unload failed:alpha:ERROR", "model load failed:beta:ERROR"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			driver := newFakeDriver()
			if tc.initial {
				driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
			}
			driver.startErr = tc.startErr
			driver.stopErr = tc.stopErr
			manager := newTestManager(t, driver)
			var output bytes.Buffer
			manager.logger = slog.New(slog.NewJSONHandler(&output, nil))

			var op Operation
			var err error
			if tc.operation == "unload" {
				op, _, err = manager.Unload(tc.model)
			} else {
				op, _, err = manager.Activate(tc.model)
			}
			if err != nil {
				t.Fatalf("%s(%s): %v", tc.operation, tc.model, err)
			}
			waitOperation(t, manager, op.ID, tc.wantState)

			decoder := json.NewDecoder(bytes.NewReader(output.Bytes()))
			var logs []string
			for decoder.More() {
				var record struct {
					Level       string `json:"level"`
					Message     string `json:"msg"`
					ModelID     string `json:"model_id"`
					OperationID string `json:"operation_id"`
					Error       string `json:"error"`
				}
				if err := decoder.Decode(&record); err != nil {
					t.Fatalf("decode log: %v", err)
				}
				if record.OperationID != op.ID {
					t.Fatalf("log operation ID = %q, want %q", record.OperationID, op.ID)
				}
				if record.Level == "ERROR" && record.Error == "" {
					t.Fatalf("failure log has no error: %+v", record)
				}
				logs = append(logs, record.Message+":"+record.ModelID+":"+record.Level)
			}
			if !reflect.DeepEqual(logs, tc.wantLogs) {
				t.Fatalf("logs = %v, want %v", logs, tc.wantLogs)
			}
		})
	}
}

func TestLifecycleLogsLoadResourceFailure(t *testing.T) {
	manager := newTestManager(t, newFakeDriver())
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	manager.gpus = fakeGPUProvider{}
	var output bytes.Buffer
	manager.logger = slog.New(slog.NewJSONHandler(&output, nil))

	_, _, err := manager.Activate("alpha")
	var insufficient *InsufficientResourcesError
	if !errors.As(err, &insufficient) {
		t.Fatalf("Activate() error = %v, want insufficient resources", err)
	}
	var record struct {
		Level   string `json:"level"`
		Message string `json:"msg"`
		ModelID string `json:"model_id"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("decode log: %v", err)
	}
	if record.Level != "ERROR" || record.Message != "model load failed" || record.ModelID != "alpha" || record.Error == "" {
		t.Fatalf("resource failure log = %+v", record)
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

func TestRefreshAdoptsRunningContainersAfterRestart(t *testing.T) {
	for _, health := range []string{"healthy", "unhealthy"} {
		t.Run(health, func(t *testing.T) {
			driver := newFakeDriver()
			driver.states["alpha"] = deployment.ContainerState{
				Exists: true, Running: true, Status: "running", Health: health, AssignedGPUs: []int{0, 1},
			}
			manager := newTestManager(t, driver)
			status, _ := manager.Status("alpha")
			if status.Desired != "ready" || !reflect.DeepEqual(status.AssignedGPUs, []int{0, 1}) {
				t.Fatalf("recovered container status = %+v", status)
			}
			if missing, _ := manager.Status("beta"); missing.Desired != "unloaded" {
				t.Fatalf("missing container status = %+v", missing)
			}
			for _, call := range driver.calls {
				if call != "list" && !strings.HasPrefix(call, "inspect:") {
					t.Fatalf("discovery mutated a container: %s", call)
				}
			}
		})
	}
}

func TestRefreshAdoptsRunningContainerAfterInitialInspectError(t *testing.T) {
	driver := newFakeDriver()
	driver.inspectErr = errors.New("temporary inspection failure")
	manager := newTestManager(t, driver)
	driver.inspectErr = nil
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	manager.refresh(context.Background())
	status, _ := manager.Status("alpha")
	if status.Phase != PhaseReady || status.Desired != "ready" {
		t.Fatalf("recovered container status = %+v", status)
	}
}

func TestRefreshPreservesExplicitUnloadIntent(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running"}
	manager := newTestManager(t, driver)
	manager.updateTransition("alpha", PhaseStopping, "unloaded")
	manager.refresh(context.Background())
	status, _ := manager.Status("alpha")
	if status.Phase != PhaseStopping || status.Desired != "unloaded" {
		t.Fatalf("unload intent was overwritten: %+v", status)
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

func TestActivateInstancesScalesReplicasAndKeepsGPUAndPortsDisjoint(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	manager.manifest.Models[0].MemoryLimit = "40g"
	manager.manifest.Models[0].MemorySwapLimit = "40g"

	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")

	status, _ := manager.Status("alpha")
	if status.ReadyInstances != 2 || status.DesiredInstances != 2 || status.Phase != PhaseReady {
		t.Fatalf("status after scale up = %+v", status)
	}
	if got := []string{status.Instances[0].InstanceID, status.Instances[1].InstanceID}; !reflect.DeepEqual(got, []string{"alpha", "alpha--2"}) {
		t.Fatalf("instance IDs = %v", got)
	}
	if got := []int{status.Instances[0].Port, status.Instances[1].Port}; !reflect.DeepEqual(got, []int{9001, 9002}) {
		t.Fatalf("ports = %v", got)
	}
	if reflect.DeepEqual(status.Instances[0].AssignedGPUs, status.Instances[1].AssignedGPUs) {
		t.Fatalf("replicas used overlapping GPU assignment: %+v", status.Instances)
	}
	driver.mu.Lock()
	startedModels := append([]manifest.Model(nil), driver.startedModels...)
	driver.mu.Unlock()
	var startedAlpha []manifest.Model
	for _, model := range startedModels {
		if strings.HasPrefix(model.ID, "alpha") {
			startedAlpha = append(startedAlpha, model)
		}
	}
	if len(startedAlpha) != 2 {
		t.Fatalf("started alpha models = %+v, want two replicas", startedAlpha)
	}
	for _, model := range startedAlpha {
		if model.MemoryLimit != "40g" || model.MemorySwapLimit != "40g" {
			t.Fatalf("replica %q memory limits = %q/%q, want 40g/40g", model.ID, model.MemoryLimit, model.MemorySwapLimit)
		}
	}

	target = 1
	op, noOp, err = manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 1) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ = manager.Status("alpha")
	if status.ReadyInstances != 1 || len(status.Instances) != 1 || status.Instances[0].InstanceID != "alpha" {
		t.Fatalf("status after scale down = %+v", status)
	}
	driver.mu.Lock()
	_, extraExists := driver.states["alpha--2"]
	driver.mu.Unlock()
	if extraExists {
		t.Fatal("scaled down instance alpha--2 still exists")
	}
}

func TestActivateInstancesScaleDownStopFailureRetainsReplicaGPUReservation(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0}, {1}}
	manager.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0, MemoryUsedMiB: 0},
		{Index: 1, MemoryUsedMiB: 0},
	}}
	manager.manifest.Models[1].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	manager.manifest.Models[1].Command = []string{"serve", "--host", "0.0.0.0", "--port", "9003"}
	manager.manifest.Models[1].Readiness = &manifest.Readiness{URL: "http://127.0.0.1:9003/health", SuccessStatus: 200}

	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	before, _ := manager.Status("alpha")
	if before.ReadyInstances != 2 || len(before.Instances) != 2 {
		t.Fatalf("status after scale up = %+v", before)
	}
	stoppedReplicaGPUs := append([]int(nil), before.Instances[1].AssignedGPUs...)

	driver.stopErr = errors.New("stop failed")
	target = 1
	op, noOp, err = manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 1) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "failed")

	status, _ := manager.Status("alpha")
	if status.Instances[0].InstanceID != "alpha" || status.Instances[0].Phase != PhaseReady || !reflect.DeepEqual(status.Instances[0].AssignedGPUs, before.Instances[0].AssignedGPUs) {
		t.Fatalf("healthy sibling not preserved after stop failure: before=%+v after=%+v", before, status)
	}
	if len(status.Instances) != 2 || status.Instances[1].InstanceID != "alpha--2" || !reflect.DeepEqual(status.Instances[1].AssignedGPUs, stoppedReplicaGPUs) {
		t.Fatalf("stopping replica GPU metadata lost after stop failure: before=%+v after=%+v", before, status)
	}

	_, _, err = manager.Activate("beta")
	var insufficient *InsufficientResourcesError
	if !errors.As(err, &insufficient) {
		t.Fatalf("Activate(beta) error = %v, want insufficient resources from retained alpha reservations", err)
	}
}

func TestActivateInstancesScaleUpReservesExistingReplicaGPU(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	target := 1
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 1) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	first, _ := manager.Status("alpha")
	if len(first.Instances) != 1 {
		t.Fatalf("initial status = %+v", first)
	}

	target = 2
	op, noOp, err = manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ := manager.Status("alpha")
	if reflect.DeepEqual(status.Instances[0].AssignedGPUs, status.Instances[1].AssignedGPUs) {
		t.Fatalf("scale-up reused existing GPU: before=%+v after=%+v", first, status)
	}
}

func TestActivateInstancesDedupesIdenticalExplicitInFlightRequest(t *testing.T) {
	driver := newFakeDriver()
	driver.blockStart = make(chan struct{})
	manager := newReplicaTestManager(t, driver)
	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	same, sameNoOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil {
		t.Fatalf("second ActivateInstances(alpha, 2) error = %v", err)
	}
	if !sameNoOp || same.ID != op.ID {
		t.Fatalf("second ActivateInstances(alpha, 2) = (%+v, %v), want same op", same, sameNoOp)
	}
	close(driver.blockStart)
	waitOperation(t, manager, op.ID, "succeeded")
}

func TestActivateInstancesInsufficientResourcesDoesNotMutateExistingInstances(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	before, _ := manager.Status("alpha")

	target = 4
	_, _, err = manager.ActivateInstances("alpha", &target)
	var insufficient *InsufficientResourcesError
	if !errors.As(err, &insufficient) {
		t.Fatalf("ActivateInstances(alpha, 4) error = %v, want insufficient resources", err)
	}
	after, _ := manager.Status("alpha")
	if !reflect.DeepEqual(before.Instances, after.Instances) || after.ReadyInstances != before.ReadyInstances {
		t.Fatalf("status mutated after failed preflight:\nbefore=%+v\nafter=%+v", before, after)
	}
}

func TestActivateInstancesTP4ReplicasUseDisjointGPUSets(t *testing.T) {
	driver := newFakeDriver()
	manager := newReplicaTestManager(t, driver)
	manager.manifest.Runtime.ModelPortRange = &manifest.PortRange{Start: 9001, End: 9002}
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0, 1, 6, 7}, {2, 3, 4, 5}}
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 4, Strategy: "pcie_affinity"}
	manager.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0, MemoryUsedMiB: 0}, {Index: 1, MemoryUsedMiB: 0},
		{Index: 2, MemoryUsedMiB: 0}, {Index: 3, MemoryUsedMiB: 0},
		{Index: 4, MemoryUsedMiB: 0}, {Index: 5, MemoryUsedMiB: 0},
		{Index: 6, MemoryUsedMiB: 0}, {Index: 7, MemoryUsedMiB: 0},
	}}
	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ := manager.Status("alpha")
	if intersects(status.Instances[0].AssignedGPUs, status.Instances[1].AssignedGPUs) {
		t.Fatalf("TP4 replicas overlap GPUs: %+v", status.Instances)
	}
}

func TestActivateSwitchIgnoresOtherModelGPUsOnlyWhenConcurrentDeploymentsDisabled(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", AssignedGPUs: []int{0, 1, 2, 3}}
	manager := newTestManager(t, driver)
	manager.manifest.Runtime.ConcurrentDeployments = false
	manager.manifest.Runtime.GPUTopology.Groups = [][]int{{0, 1, 2, 3}}
	manager.manifest.Models[1].Placement = &manifest.Placement{GPUCount: 4, Strategy: "pcie_affinity"}
	manager.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0, MemoryUsedMiB: 0}, {Index: 1, MemoryUsedMiB: 0},
		{Index: 2, MemoryUsedMiB: 0}, {Index: 3, MemoryUsedMiB: 0},
	}}

	op, noOp, err := manager.Activate("beta")
	if err != nil || noOp {
		t.Fatalf("Activate(beta) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
}

func TestActivateRestartsStoppedFailedBaseInstance(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{
		Exists: true, Running: false, Status: "exited", ExitCode: 1, Port: 9001, AssignedGPUs: []int{0},
	}
	manager := newReplicaTestManager(t, driver)
	status, _ := manager.Status("alpha")
	if status.Phase != PhaseFailed {
		t.Fatalf("initial status = %+v, want failed", status)
	}

	op, noOp, err := manager.Activate("alpha")
	if err != nil || noOp {
		t.Fatalf("Activate(alpha) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")
	status, _ = manager.Status("alpha")
	if status.Phase != PhaseReady || status.ReadyInstances != 1 || status.Instances[0].Port != 9001 {
		t.Fatalf("restarted status = %+v", status)
	}
}

func TestActivateRepairsFailedReplicaWithoutRestartingHealthySibling(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9001, AssignedGPUs: []int{0}}
	driver.states["alpha--2"] = deployment.ContainerState{
		Exists: true, Running: false, Status: "exited", ExitCode: 1, Port: 9002, AssignedGPUs: []int{1},
	}
	manager := newReplicaTestManager(t, driver)
	target := 2
	op, noOp, err := manager.ActivateInstances("alpha", &target)
	if err != nil || noOp {
		t.Fatalf("ActivateInstances(alpha, 2) = (%+v, %v, %v)", op, noOp, err)
	}
	waitOperation(t, manager, op.ID, "succeeded")

	driver.mu.Lock()
	calls := append([]string(nil), driver.calls...)
	driver.mu.Unlock()
	startAlpha := 0
	startReplica := 0
	for _, call := range calls {
		if call == "start:alpha" {
			startAlpha++
		}
		if call == "start:alpha--2" {
			startReplica++
		}
	}
	if startAlpha != 0 || startReplica != 1 {
		t.Fatalf("start calls alpha=%d alpha--2=%d, calls=%v", startAlpha, startReplica, calls)
	}
	status, _ := manager.Status("alpha")
	if status.Phase != PhaseReady || status.ReadyInstances != 2 || status.Instances[1].Port != 9002 {
		t.Fatalf("repaired status = %+v", status)
	}
}

func TestRefreshRecoversReplicaContainersAfterRestart(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9001, AssignedGPUs: []int{0}}
	driver.states["alpha--2"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9002, AssignedGPUs: []int{1}}
	manager := newReplicaTestManager(t, driver)

	status, _ := manager.Status("alpha")
	if status.Desired != "ready" || status.ReadyInstances != 2 || len(status.Instances) != 2 {
		t.Fatalf("recovered replica status = %+v", status)
	}
	if got := []int{status.Instances[0].Port, status.Instances[1].Port}; !reflect.DeepEqual(got, []int{9001, 9002}) {
		t.Fatalf("recovered ports = %v", got)
	}
}

func TestReloadManifestRejectsActiveReplicaPortCollision(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9001, AssignedGPUs: []int{0}}
	driver.states["alpha--2"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9002, AssignedGPUs: []int{1}}
	manager := newReplicaTestManager(t, driver)
	next := loadFleetManifestForTest(t)
	next.Runtime.ConcurrentDeployments = true
	next.Runtime.ModelPortRange = &manifest.PortRange{Start: 9001, End: 9003}
	next.Runtime.GPUTopology.Groups = [][]int{{0, 1}, {2, 3}}
	next.Models[0] = manager.manifest.Models[0]
	next.Models[1].Command = []string{"serve", "--port", "9002"}
	next.Models[1].Readiness = &manifest.Readiness{URL: "http://127.0.0.1:9002/health", SuccessStatus: 200}

	err := manager.ReloadManifest(next)
	if err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("ReloadManifest() error = %v, want port collision", err)
	}
}

func TestReloadManifestRejectsMixedReadyFailedActiveModelChange(t *testing.T) {
	driver := newFakeDriver()
	driver.states["alpha"] = deployment.ContainerState{Exists: true, Running: true, Status: "running", Port: 9001, AssignedGPUs: []int{0}}
	driver.states["alpha--2"] = deployment.ContainerState{Exists: true, Running: false, Status: "exited", ExitCode: 1, Port: 9002, AssignedGPUs: []int{1}}
	manager := newReplicaTestManager(t, driver)
	next := loadFleetManifestForTest(t)
	next.Runtime = manager.manifest.Runtime
	next.Models = append([]manifest.Model(nil), manager.manifest.Models...)
	next.Models[0].Image = "replacement"

	err := manager.ReloadManifest(next)
	if err == nil || !strings.Contains(err.Error(), "cannot be changed") {
		t.Fatalf("ReloadManifest() error = %v, want active-model rejection", err)
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

func TestModelsExposeConfiguredGPUCount(t *testing.T) {
	manager := newTestManager(t, newFakeDriver())
	manager.manifest.Models[0].Placement = &manifest.Placement{GPUCount: 4, Strategy: "pcie_affinity"}
	if got := manager.Models()[0].GPUCount; got != 4 {
		t.Fatalf("GPUCount = %d, want 4", got)
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

func newReplicaTestManager(t *testing.T, driver *fakeDriver) *Manager {
	t.Helper()
	cfg := loadFleetManifestForTest(t)
	cfg.Runtime.ConcurrentDeployments = true
	cfg.Runtime.ModelPortRange = &manifest.PortRange{Start: 9001, End: 9003}
	cfg.Runtime.GPUTopology.Groups = [][]int{{0, 1}, {2, 3}}
	cfg.Runtime.GPUTopology.MaxUsedMemoryMiB = 1024
	cfg.Models[0].Command = []string{"serve", "--host", "0.0.0.0", "--port", "9001"}
	cfg.Models[0].Ports = []manifest.Port{{HostIP: "127.0.0.1", HostPort: 9001, ContainerPort: 9001, Protocol: "tcp"}}
	cfg.Models[0].Readiness = &manifest.Readiness{URL: "http://127.0.0.1:9001/health", SuccessStatus: 200}
	cfg.Models[0].Placement = &manifest.Placement{GPUCount: 1, Strategy: "pcie_affinity"}
	m := NewManager(cfg, driver, slog.Default())
	m.gpus = fakeGPUProvider{devices: []gpu.Device{
		{Index: 0, MemoryUsedMiB: 0}, {Index: 1, MemoryUsedMiB: 0},
		{Index: 2, MemoryUsedMiB: 0}, {Index: 3, MemoryUsedMiB: 0},
	}}
	m.client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("ok")),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})}
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

func intersects(left, right []int) bool {
	seen := map[int]bool{}
	for _, value := range left {
		seen[value] = true
	}
	for _, value := range right {
		if seen[value] {
			return true
		}
	}
	return false
}

type fakeDriver struct {
	mu            sync.Mutex
	states        map[string]deployment.ContainerState
	calls         []string
	startedModels []manifest.Model
	blockStart    chan struct{}
	inspectErr    error
	startErr      error
	stopErr       error
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
	d.startedModels = append(d.startedModels, model)
	if d.startErr != nil {
		return d.startErr
	}
	d.states[model.ID] = deployment.ContainerState{Exists: true, Running: true, Status: "running", AssignedGPUs: assigned, Port: model.InstancePort}
	return nil
}

func (d *fakeDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "stop:"+model.ID+":"+strconv.FormatBool(remove))
	if d.stopErr != nil {
		return d.stopErr
	}
	delete(d.states, model.ID)
	return nil
}

func (d *fakeDriver) List(ctx context.Context) ([]deployment.ContainerState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "list")
	if d.inspectErr != nil {
		return nil, d.inspectErr
	}
	return fakeList(d.states), nil
}

// fakeList derives Fleet's identity labels from the fake's instance-ID keys.
func fakeList(states map[string]deployment.ContainerState) []deployment.ContainerState {
	var result []deployment.ContainerState
	for id, state := range states {
		state.Exists = true
		state.InstanceID = id
		state.ModelID, state.InstanceIndex = id, 1
		if base, suffix, ok := strings.Cut(id, "--"); ok {
			if index, err := strconv.Atoi(suffix); err == nil {
				state.ModelID, state.InstanceIndex = base, index
			}
		}
		result = append(result, state)
	}
	return result
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
	if d.inspectErr != nil {
		return deployment.ContainerState{}, d.inspectErr
	}
	state, ok := d.states[model.ID]
	if !ok {
		return deployment.ContainerState{}, deployment.ErrNotFound
	}
	state.Exists = true
	return state, nil
}

func TestLifecycleErrorsRedactManifestSecrets(t *testing.T) {
	driver := newFakeDriver()
	driver.startErr = errors.New("create failed: API_KEY=secret-value-123\x1b[31m\nforged")
	manager := newTestManager(t, driver)
	manager.manifest.Models[0].Environment = map[string]string{"API_KEY": "secret-value-123"}

	op, _, err := manager.Activate("alpha")
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		op, _ = manager.Operation(op.ID)
		if op.State == "failed" || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	status, _ := manager.Status("alpha")
	for _, message := range []string{op.Error, status.LastError, status.Instances[0].LastError} {
		if message == "" || strings.Contains(message, "secret-value-123") || strings.ContainsAny(message, "\x1b\n") {
			t.Fatalf("unsanitized lifecycle error %q (op=%+v status=%+v)", message, op, status)
		}
	}
}

func newDiscardLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func writeFile(path, body string) error { return os.WriteFile(path, []byte(body), 0o600) }
