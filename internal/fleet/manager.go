package fleet

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/gpu"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

type Phase string

const (
	PhaseUnloaded  Phase = "unloaded"
	PhaseLoading   Phase = "loading"
	PhaseReady     Phase = "ready"
	PhaseUnhealthy Phase = "unhealthy"
	PhaseStopping  Phase = "stopping"
	PhaseFailed    Phase = "failed"
	PhaseUnknown   Phase = "unknown"
)

type ModelInfo struct {
	ID           string `json:"id"`
	Description  string `json:"description,omitempty"`
	Image        string `json:"image"`
	GPUCount     int    `json:"gpu_count,omitempty"`
	MaxInstances int    `json:"max_instances,omitempty"`
}

type Status struct {
	ModelID          string           `json:"model_id"`
	Phase            Phase            `json:"phase"`
	Desired          string           `json:"desired_state"`
	DesiredInstances int              `json:"desired_instances,omitempty"`
	ReadyInstances   int              `json:"ready_instances,omitempty"`
	Instances        []InstanceStatus `json:"instances,omitempty"`
	Container        string           `json:"container_status,omitempty"`
	Health           string           `json:"health,omitempty"`
	ExitCode         int              `json:"exit_code,omitempty"`
	OOMKilled        bool             `json:"oom_killed,omitempty"`
	AssignedGPUs     []int            `json:"assigned_gpus,omitempty"`
	LastError        string           `json:"last_error,omitempty"`
	LastChecked      time.Time        `json:"last_checked"`
}

type InstanceStatus struct {
	InstanceID   string    `json:"instance_id"`
	Index        int       `json:"index"`
	Port         int       `json:"port,omitempty"`
	Phase        Phase     `json:"phase"`
	Container    string    `json:"container_status,omitempty"`
	Health       string    `json:"health,omitempty"`
	ExitCode     int       `json:"exit_code,omitempty"`
	OOMKilled    bool      `json:"oom_killed,omitempty"`
	AssignedGPUs []int     `json:"assigned_gpus,omitempty"`
	LastError    string    `json:"last_error,omitempty"`
	LastChecked  time.Time `json:"last_checked"`
}

type Operation struct {
	ID              string     `json:"id"`
	Kind            string     `json:"kind"`
	ModelID         string     `json:"model_id"`
	TargetInstances int        `json:"instances,omitempty"`
	State           string     `json:"state"`
	Error           string     `json:"error,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	StartedAt       *time.Time `json:"started_at,omitempty"`
	FinishedAt      *time.Time `json:"finished_at,omitempty"`
}

type ConflictError struct {
	Operation Operation
}

func (e *ConflictError) Error() string { return "another lifecycle operation is in progress" }

type InsufficientResourcesError struct {
	ModelID string
	Err     error
}

func (e *InsufficientResourcesError) Error() string {
	return fmt.Sprintf("insufficient GPU resources for %s: %v", e.ModelID, e.Err)
}

func (e *InsufficientResourcesError) Unwrap() error { return e.Err }

type Manager struct {
	manifest *manifest.Manifest
	driver   deployment.Driver
	gpus     gpu.Provider
	logger   *slog.Logger
	client   *http.Client

	mu         sync.RWMutex
	statuses   map[string]Status
	operations map[string]Operation
	inFlight   string
	cancel     context.CancelFunc
	done       chan struct{}
}

func NewManager(cfg *manifest.Manifest, driver deployment.Driver, logger *slog.Logger) *Manager {
	return NewManagerWithGPUProvider(cfg, driver, gpu.NewNVIDIAProvider("nvidia-smi"), logger)
}

func NewManagerWithGPUProvider(cfg *manifest.Manifest, driver deployment.Driver, provider gpu.Provider, logger *slog.Logger) *Manager {
	statuses := make(map[string]Status, len(cfg.Models))
	for _, model := range cfg.Models {
		statuses[model.ID] = Status{ModelID: model.ID, Phase: PhaseUnknown, Desired: "unloaded"}
	}
	return &Manager{
		manifest:   cfg,
		driver:     driver,
		gpus:       provider,
		logger:     logger,
		client:     &http.Client{Timeout: 3 * time.Second},
		statuses:   statuses,
		operations: make(map[string]Operation),
		done:       make(chan struct{}),
	}
}

func (m *Manager) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	go func() {
		defer close(m.done)
		for {
			m.refresh(ctx)
			timer := time.NewTimer(m.pollDuration())
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
	}()
}

func (m *Manager) Close() {
	if m.cancel == nil {
		return
	}
	m.cancel()
	<-m.done
}

func (m *Manager) Models() []ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	models := make([]ModelInfo, 0, len(m.manifest.Models))
	for _, model := range m.manifest.Models {
		gpuCount := 0
		if model.Placement != nil {
			gpuCount = model.Placement.GPUCount
		}
		models = append(models, ModelInfo{
			ID: model.ID, Description: model.Description, Image: model.Image,
			GPUCount: gpuCount, MaxInstances: m.maxInstancesLocked(model),
		})
	}
	return models
}

func (m *Manager) maxInstancesLocked(model manifest.Model) int {
	return maxInstancesForManifest(m.manifest, model)
}

func maxInstancesForManifest(cfg *manifest.Manifest, model manifest.Model) int {
	limit := 1
	if model.Placement != nil && model.Placement.GPUCount > 0 {
		total := 0
		for _, group := range cfg.Runtime.GPUTopology.Groups {
			total += len(group)
		}
		limit = total / model.Placement.GPUCount
	}
	if limit < 1 {
		limit = 1
	}
	portLimit := portCapacityForManifest(cfg, model)
	if portLimit < limit {
		limit = portLimit
	}
	if limit > maxRequestedInstances {
		return maxRequestedInstances
	}
	return limit
}

func (m *Manager) portCapacityLocked(model manifest.Model) int {
	return portCapacityForManifest(m.manifest, model)
}

func portCapacityForManifest(cfg *manifest.Manifest, model manifest.Model) int {
	if cfg.Runtime.ModelPortRange == nil {
		return 1
	}
	base := basePort(model)
	if base == 0 {
		return 1
	}
	configured := configuredBasePorts(cfg)
	count := 1
	for port := cfg.Runtime.ModelPortRange.Start; port <= cfg.Runtime.ModelPortRange.End; port++ {
		if port == base {
			continue
		}
		if configured[port] {
			continue
		}
		count++
	}
	return count
}

// ReloadManifest atomically replaces the active manifest after checking that
// the update cannot orphan or silently mutate a running deployment.
func (m *Manager) ReloadManifest(next *manifest.Manifest) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.inFlight != "" {
		return errors.New("a lifecycle operation is in progress")
	}
	if next.API.Listen != m.manifest.API.Listen {
		return errors.New("api.listen cannot be changed without restarting fleet")
	}
	if next.Runtime.DockerBinary != m.manifest.Runtime.DockerBinary {
		return errors.New("runtime.docker_binary cannot be changed without restarting fleet")
	}
	if !reflect.DeepEqual(next.Runtime.ModelPortRange, m.manifest.Runtime.ModelPortRange) && m.hasActiveDeploymentLocked() {
		return errors.New("runtime.model_port_range cannot be changed while deployments are active")
	}
	if err := m.validateActiveInstancesAgainstManifestLocked(next); err != nil {
		return err
	}

	for _, current := range m.manifest.Models {
		replacement, present := next.Model(current.ID)
		status := m.statuses[current.ID]
		if manifestModelCanChange(status) {
			continue
		}
		if !present {
			return fmt.Errorf("model %q cannot be removed while its deployment is active", current.ID)
		}
		if !reflect.DeepEqual(current, replacement) {
			return fmt.Errorf("model %q cannot be changed while its deployment is active", current.ID)
		}
	}

	statuses := make(map[string]Status, len(next.Models))
	for _, model := range next.Models {
		status, present := m.statuses[model.ID]
		if !present {
			status = Status{ModelID: model.ID, Phase: PhaseUnknown, Desired: "unloaded"}
		}
		statuses[model.ID] = status
	}
	m.manifest = next
	m.statuses = statuses
	return nil
}

func manifestModelCanChange(status Status) bool {
	return !statusHasLiveInstances(status) && (status.Phase == PhaseUnloaded || status.Phase == PhaseFailed)
}

func (m *Manager) hasActiveDeploymentLocked() bool {
	for _, status := range m.statuses {
		if statusHasLiveInstances(status) || statusActive(status) {
			return true
		}
	}
	return false
}

func (m *Manager) validateActiveInstancesAgainstManifestLocked(next *manifest.Manifest) error {
	nextBasePorts := configuredBasePorts(next)
	for _, status := range m.statuses {
		if !statusHasLiveInstances(status) {
			continue
		}
		for _, instance := range status.Instances {
			if instance.Index > 1 && instance.Port != 0 && nextBasePorts[instance.Port] {
				return fmt.Errorf("active replica %q port %d collides with a configured model port", instance.InstanceID, instance.Port)
			}
		}
		model, present := next.Model(status.ModelID)
		if !present {
			continue
		}
		if max := maxInstancesForManifest(next, model); max < maxInstanceIndex(status) {
			return fmt.Errorf("model %q active instance index exceeds new max_instances %d", status.ModelID, max)
		}
	}
	return nil
}

func (m *Manager) pollDuration() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.manifest.Runtime.PollDuration()
}

func (m *Manager) Statuses() []Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]Status, 0, len(m.statuses))
	for _, status := range m.statuses {
		result = append(result, cloneStatus(status))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ModelID < result[j].ModelID })
	return result
}

func (m *Manager) Status(modelID string) (Status, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	status, ok := m.statuses[modelID]
	return cloneStatus(status), ok
}

func (m *Manager) Operation(id string) (Operation, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	op, ok := m.operations[id]
	return op, ok
}

func (m *Manager) Activate(modelID string) (Operation, bool, error) {
	return m.ActivateInstances(modelID, nil)
}

func (m *Manager) ActivateInstances(modelID string, requested *int) (Operation, bool, error) {
	m.mu.Lock()
	cfg := m.manifest
	model, ok := cfg.Model(modelID)
	if !ok {
		m.mu.Unlock()
		return Operation{}, false, fmt.Errorf("unknown model %q", modelID)
	}
	if m.inFlight != "" {
		op := m.operations[m.inFlight]
		if op.Kind == "activate" && op.ModelID == modelID && (requested == nil || op.TargetInstances == *requested) {
			m.mu.Unlock()
			return op, true, nil
		}
		m.mu.Unlock()
		return Operation{}, false, &ConflictError{Operation: op}
	}
	target := m.targetInstancesLocked(modelID, requested)
	if target < 1 || target > maxRequestedInstances {
		m.mu.Unlock()
		return Operation{}, false, fmt.Errorf("instances must be between 1 and %d", maxRequestedInstances)
	}
	if target > m.maxInstancesLocked(model) {
		m.mu.Unlock()
		err := fmt.Errorf("requested %d instances, maximum is %d", target, m.maxInstancesLocked(model))
		m.logger.Error("model load failed", "model_id", modelID, "error", err)
		return Operation{}, false, &InsufficientResourcesError{ModelID: modelID, Err: err}
	}
	plan, err := m.planInstancesLocked(cfg, model, target)
	if err != nil {
		m.mu.Unlock()
		m.logger.Error("model load failed", "model_id", modelID, "error", err)
		return Operation{}, false, &InsufficientResourcesError{ModelID: modelID, Err: err}
	}
	if m.isReadyNoopLocked(modelID, cfg, target) {
		status := m.statuses[modelID]
		status.Desired = "ready"
		status.DesiredInstances = target
		m.statuses[modelID] = status
		m.mu.Unlock()
		return Operation{}, true, nil
	}
	m.markPlanLocked(modelID, plan)
	op := newOperation("activate", modelID, target)
	m.operations[op.ID] = op
	m.inFlight = op.ID
	m.mu.Unlock()
	go m.runActivate(op.ID, cfg, model, plan)
	return op, false, nil
}

func (m *Manager) isReadyNoopLocked(modelID string, cfg *manifest.Manifest, target int) bool {
	status := m.statuses[modelID]
	if status.Phase != PhaseReady || status.ReadyInstances != target || len(status.Instances) != target {
		return false
	}
	if cfg.Runtime.ConcurrentDeployments {
		return true
	}
	for id, status := range m.statuses {
		if id != modelID && status.Phase != PhaseUnloaded {
			return false
		}
	}
	return true
}

func (m *Manager) targetInstancesLocked(modelID string, requested *int) int {
	if requested != nil {
		return *requested
	}
	status := m.statuses[modelID]
	if len(status.Instances) > 1 {
		return len(status.Instances)
	}
	if status.ReadyInstances > 1 {
		return status.ReadyInstances
	}
	return 1
}

func (m *Manager) planInstancesLocked(cfg *manifest.Manifest, model manifest.Model, target int) (instancePlan, error) {
	current := sortedInstances(m.statuses[model.ID].Instances)
	currentByIndex := map[int]InstanceStatus{}
	kept := map[int]bool{}
	plan := instancePlan{target: target}
	for _, instance := range current {
		currentByIndex[instance.Index] = instance
		if instance.Index <= target && keepableInstance(instance) {
			plan.keep = append(plan.keep, instance)
			kept[instance.Index] = true
		}
	}
	for i := len(current) - 1; i >= 0; i-- {
		if current[i].Index > target {
			plan.stop = append(plan.stop, plannedInstance{
				model: cloneInstanceModel(model, current[i].Index, current[i].Port),
				gpus:  append([]int(nil), current[i].AssignedGPUs...),
				port:  current[i].Port,
				index: current[i].Index,
			})
		}
	}
	if len(plan.keep) == target {
		return plan, nil
	}

	var startIndexes []int
	extraPortsNeeded := 0
	for index := 1; index <= target; index++ {
		if kept[index] {
			continue
		}
		startIndexes = append(startIndexes, index)
		if index > 1 && currentByIndex[index].Port == 0 {
			extraPortsNeeded++
		}
	}
	ports, err := m.allocatePortsLocked(model, extraPortsNeeded)
	if err != nil {
		return instancePlan{}, err
	}
	var devices []gpu.Device
	var allocator gpu.Allocator
	var reserved []int
	if model.Placement != nil {
		devices, err = m.gpus.Snapshot(context.Background())
		if err != nil {
			return instancePlan{}, err
		}
		allocator = gpu.Allocator{Topology: gpu.Topology{
			Groups:           cfg.Runtime.GPUTopology.Groups,
			MaxUsedMemoryMiB: cfg.Runtime.GPUTopology.MaxUsedMemoryMiB,
		}}
		reserved = m.reservedGPUsForPlanLocked(cfg, model.ID)
	}
	for _, index := range startIndexes {
		currentInstance := currentByIndex[index]
		port := basePort(model)
		if index > 1 {
			port = currentInstance.Port
			if port == 0 {
				port = ports[0]
				ports = ports[1:]
			}
		}
		if owner, taken := m.activePortOwnerLocked(model.ID, port); taken {
			return instancePlan{}, fmt.Errorf("port %d is already used by %s", port, owner)
		}
		assigned := append([]int(nil), currentInstance.AssignedGPUs...)
		if model.Placement != nil {
			if len(assigned) == 0 {
				assigned, err = allocator.Allocate(devices, model.Placement.GPUCount, reserved)
				if err != nil {
					return instancePlan{}, err
				}
			}
			reserved = append(reserved, assigned...)
		}
		plan.start = append(plan.start, plannedInstance{
			model: cloneInstanceModel(model, index, port),
			gpus:  append([]int(nil), assigned...),
			port:  port,
			index: index,
		})
	}
	return plan, nil
}

func keepableInstance(instance InstanceStatus) bool {
	switch instance.Phase {
	case PhaseReady, PhaseLoading, PhaseUnhealthy:
		return true
	default:
		return false
	}
}

func (m *Manager) allocatePortsLocked(model manifest.Model, needed int) ([]int, error) {
	if needed <= 0 {
		return nil, nil
	}
	if m.manifest.Runtime.ModelPortRange == nil {
		return nil, errors.New("runtime.model_port_range is required for multiple instances")
	}
	configured := m.configuredBasePortsLocked()
	reserved := m.reservedPortsLocked()
	var ports []int
	for port := m.manifest.Runtime.ModelPortRange.Start; port <= m.manifest.Runtime.ModelPortRange.End; port++ {
		if port == basePort(model) || configured[port] || reserved[port] {
			continue
		}
		ports = append(ports, port)
		if len(ports) == needed {
			return ports, nil
		}
	}
	return nil, fmt.Errorf("need %d extra ports, found %d", needed, len(ports))
}

func (m *Manager) configuredBasePortsLocked() map[int]bool {
	return configuredBasePorts(m.manifest)
}

func configuredBasePorts(cfg *manifest.Manifest) map[int]bool {
	ports := map[int]bool{}
	for _, model := range cfg.Models {
		if port := basePort(model); port != 0 {
			ports[port] = true
		}
	}
	return ports
}

func (m *Manager) reservedPortsLocked() map[int]bool {
	ports := map[int]bool{}
	for _, status := range m.statuses {
		for _, instance := range status.Instances {
			if instance.Index > 1 && instance.Port != 0 && instance.Phase != PhaseUnloaded {
				ports[instance.Port] = true
			}
		}
	}
	return ports
}

func (m *Manager) activePortOwnerLocked(exceptModelID string, port int) (string, bool) {
	if port == 0 {
		return "", false
	}
	for modelID, status := range m.statuses {
		if modelID == exceptModelID {
			continue
		}
		for _, instance := range status.Instances {
			if instance.Port == port && instance.Phase != PhaseUnloaded {
				return instance.InstanceID, true
			}
		}
	}
	return "", false
}

func (m *Manager) reservedGPUsLocked(exceptModelID string) []int {
	return m.reservedGPUsForPlanLocked(m.manifest, exceptModelID)
}

func (m *Manager) reservedGPUsForPlanLocked(cfg *manifest.Manifest, exceptModelID string) []int {
	var reserved []int
	for id, status := range m.statuses {
		if id == exceptModelID {
			for _, instance := range status.Instances {
				if instance.Phase != PhaseUnloaded {
					reserved = append(reserved, instance.AssignedGPUs...)
				}
			}
			continue
		}
		if !cfg.Runtime.ConcurrentDeployments {
			continue
		}
		for _, instance := range status.Instances {
			if instance.Phase != PhaseUnloaded {
				reserved = append(reserved, instance.AssignedGPUs...)
			}
		}
		if len(status.Instances) == 0 && statusActive(status) {
			reserved = append(reserved, status.AssignedGPUs...)
		}
	}
	return reserved
}

func (m *Manager) markPlanLocked(modelID string, plan instancePlan) {
	now := time.Now().UTC()
	instances := append([]InstanceStatus(nil), plan.keep...)
	for _, start := range plan.start {
		instances = append(instances, InstanceStatus{
			InstanceID: instanceID(modelID, start.index), Index: start.index, Port: start.port,
			Phase: PhaseLoading, AssignedGPUs: append([]int(nil), start.gpus...), LastChecked: now,
		})
	}
	for _, stop := range plan.stop {
		instances = append(instances, InstanceStatus{
			InstanceID: instanceID(modelID, stop.index), Index: stop.index, Port: stop.port,
			Phase: PhaseStopping, AssignedGPUs: append([]int(nil), stop.gpus...), LastChecked: now,
		})
	}
	status := aggregateStatus(modelID, "ready", instances, false)
	status.DesiredInstances = plan.target
	m.statuses[modelID] = status
}

func (m *Manager) Unload(modelID string) (Operation, bool, error) {
	m.mu.Lock()
	cfg := m.manifest
	model, ok := cfg.Model(modelID)
	if !ok {
		m.mu.Unlock()
		return Operation{}, false, fmt.Errorf("unknown model %q", modelID)
	}
	if m.inFlight != "" {
		op := m.operations[m.inFlight]
		if op.Kind == "unload" && op.ModelID == modelID {
			m.mu.Unlock()
			return op, true, nil
		}
		m.mu.Unlock()
		return Operation{}, false, &ConflictError{Operation: op}
	}
	if status := m.statuses[modelID]; status.Phase == PhaseUnloaded && status.Container == "" {
		status.Desired = "unloaded"
		m.statuses[modelID] = status
		m.mu.Unlock()
		return Operation{}, true, nil
	}
	op := newOperation("unload", modelID, 0)
	m.operations[op.ID] = op
	m.inFlight = op.ID
	m.mu.Unlock()
	go m.runUnload(op.ID, cfg, model)
	return op, false, nil
}

func newOperation(kind, modelID string, targetInstances int) Operation {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		panic(fmt.Sprintf("generate operation id: %v", err))
	}
	return Operation{ID: hex.EncodeToString(buf), Kind: kind, ModelID: modelID, TargetInstances: targetInstances, State: "pending", CreatedAt: time.Now().UTC()}
}

func (m *Manager) beginOperation(id string) {
	now := time.Now().UTC()
	m.mu.Lock()
	op := m.operations[id]
	op.State = "running"
	op.StartedAt = &now
	m.operations[id] = op
	m.mu.Unlock()
}

func (m *Manager) finishOperation(id string, err error) {
	m.mu.RLock()
	op := m.operations[id]
	m.mu.RUnlock()
	if err != nil {
		if op.Kind == "unload" {
			m.logger.Error("model unload failed", "model_id", op.ModelID, "operation_id", id, "error", err)
		} else {
			m.logger.Error("model load failed", "model_id", op.ModelID, "operation_id", id, "error", err)
		}
	} else if op.Kind == "unload" {
		m.logger.Info("model unloaded", "model_id", op.ModelID, "operation_id", id)
	} else {
		m.logger.Info("model loaded", "model_id", op.ModelID, "operation_id", id)
	}

	now := time.Now().UTC()
	m.mu.Lock()
	op.State = "succeeded"
	if err != nil {
		op.State = "failed"
		op.Error = err.Error()
	}
	op.FinishedAt = &now
	m.operations[id] = op
	if m.inFlight == id {
		m.inFlight = ""
	}
	m.mu.Unlock()
}

func (m *Manager) runActivate(operationID string, cfg *manifest.Manifest, model manifest.Model, plan instancePlan) {
	modelID := model.ID
	m.beginOperation(operationID)
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Runtime.ReadinessDuration())
	defer cancel()

	if !cfg.Runtime.ConcurrentDeployments {
		for _, other := range cfg.Models {
			if other.ID == modelID {
				continue
			}
			wasLoaded := m.currentStatus(other.ID).Phase != PhaseUnloaded
			m.updateTransition(other.ID, PhaseStopping, "unloaded")
			opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.OperationDuration())
			err := m.stopAllInstances(opCtx, other, cfg.Runtime.RemoveOnUnload)
			opCancel()
			if err != nil {
				m.setFailure(other.ID, err)
				m.logger.Error("model unload failed", "model_id", other.ID, "operation_id", operationID, "error", err)
				m.finishOperation(operationID, fmt.Errorf("unload %s before switch: %w", other.ID, err))
				return
			}
			m.setUnloaded(other.ID)
			if wasLoaded {
				m.logger.Info("model unloaded", "model_id", other.ID, "operation_id", operationID)
			}
		}
	}

	for _, stop := range plan.stop {
		opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.OperationDuration())
		err := m.driver.Stop(opCtx, stop.model, cfg.Runtime.RemoveOnUnload)
		opCancel()
		if err != nil {
			m.setFailure(modelID, err)
			m.finishOperation(operationID, err)
			return
		}
		m.removeInstance(modelID, stop.index)
	}

	for _, start := range plan.start {
		m.setInstancePhase(modelID, start.index, PhaseLoading, "")
		opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.OperationDuration())
		err := m.driver.Start(opCtx, start.model, start.gpus)
		opCancel()
		if err != nil {
			m.setInstanceFailure(modelID, start.index, err)
			m.finishOperation(operationID, err)
			return
		}
	}

	ticker := time.NewTicker(cfg.Runtime.PollDuration())
	defer ticker.Stop()
	for {
		status := m.probe(ctx, model, cfg.Runtime)
		if status.ReadyInstances == plan.target && status.Phase == PhaseReady {
			status.Desired = "ready"
			status.DesiredInstances = plan.target
			m.storeStatus(status)
			m.finishOperation(operationID, nil)
			return
		}
		if status.Phase == PhaseFailed || (status.Phase == PhaseUnloaded && status.ExitCode != 0) {
			m.storeStatus(status)
			err := fmt.Errorf("model %s exited before becoming ready", modelID)
			m.finishOperation(operationID, err)
			return
		}
		status.Phase = PhaseLoading
		status.Desired = "ready"
		status.DesiredInstances = plan.target
		m.storeStatus(status)
		select {
		case <-ctx.Done():
			err := fmt.Errorf("model %s readiness timed out: %w", modelID, ctx.Err())
			m.setFailure(modelID, err)
			m.finishOperation(operationID, err)
			return
		case <-ticker.C:
		}
	}
}

func (m *Manager) runUnload(operationID string, cfg *manifest.Manifest, model manifest.Model) {
	modelID := model.ID
	m.beginOperation(operationID)
	m.updateTransition(modelID, PhaseStopping, "unloaded")
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Runtime.OperationDuration())
	defer cancel()
	if err := m.stopAllInstances(ctx, model, cfg.Runtime.RemoveOnUnload); err != nil {
		m.setFailure(modelID, err)
		m.finishOperation(operationID, err)
		return
	}
	m.setUnloaded(modelID)
	m.finishOperation(operationID, nil)
}

func (m *Manager) refresh(ctx context.Context) {
	m.mu.RLock()
	cfg := m.manifest
	m.mu.RUnlock()
	for _, model := range cfg.Models {
		status := m.probe(ctx, model, cfg.Runtime)
		m.mu.RLock()
		if m.manifest != cfg {
			m.mu.RUnlock()
			return
		}
		current := m.statuses[model.ID]
		if m.inFlight != "" && m.operations[m.inFlight].ModelID == model.ID {
			m.mu.RUnlock()
			continue
		}
		m.mu.RUnlock()
		// On restart, recover intent from an existing running container. Unknown
		// also covers a failed first inspection; explicit stopping intent stays intact.
		if current.Phase == PhaseUnknown && current.Desired == "unloaded" &&
			(status.Phase == PhaseReady || status.Phase == PhaseUnhealthy) {
			status.Desired = "ready"
		}
		if current.Phase == PhaseLoading && status.Phase != PhaseFailed {
			status.Phase = PhaseLoading
			status.Desired = "ready"
			status.DesiredInstances = current.DesiredInstances
			status.Instances = mergePendingInstances(status.Instances, current.Instances)
			if len(status.AssignedGPUs) == 0 {
				status.AssignedGPUs = append([]int(nil), current.AssignedGPUs...)
			}
		}
		if current.Phase == PhaseStopping && status.Phase != PhaseUnloaded {
			status.Phase = PhaseStopping
			status.Desired = "unloaded"
		}
		m.mu.Lock()
		if m.manifest != cfg {
			m.mu.Unlock()
			return
		}
		latest := m.statuses[model.ID]
		if m.inFlight != "" && m.operations[m.inFlight].ModelID == model.ID {
			m.mu.Unlock()
			continue
		}
		if latest.Phase != current.Phase || latest.LastChecked != current.LastChecked || latest.DesiredInstances != current.DesiredInstances {
			m.mu.Unlock()
			continue
		}
		m.statuses[status.ModelID] = status
		m.mu.Unlock()
	}
}

func (m *Manager) probe(parent context.Context, model manifest.Model, runtime manifest.RuntimeConfig) Status {
	ctx, cancel := context.WithTimeout(parent, runtime.OperationDuration())
	defer cancel()
	m.mu.RLock()
	current := m.statuses[model.ID]
	current = cloneStatus(current)
	maxInstances := m.maxInstancesLocked(model)
	if knownMax := maxInstanceIndex(current); knownMax > maxInstances {
		maxInstances = knownMax
	}
	m.mu.RUnlock()
	desired := current.Desired
	if desired == "" {
		desired = "unloaded"
	}
	knownPorts := map[int]int{1: basePort(model)}
	for _, instance := range current.Instances {
		if instance.Port != 0 {
			knownPorts[instance.Index] = instance.Port
		}
	}
	var instances []InstanceStatus
	for index := 1; index <= maxInstances; index++ {
		port := knownPorts[index]
		status, found := m.probeInstance(ctx, model, runtime, index, port)
		if found {
			instances = append(instances, status)
		}
	}
	status := aggregateStatus(model.ID, desired, instances, true)
	if len(instances) == 0 {
		status.LastChecked = time.Now().UTC()
	}
	return status
}

func (m *Manager) probeInstance(parent context.Context, model manifest.Model, runtime manifest.RuntimeConfig, index, port int) (InstanceStatus, bool) {
	now := time.Now().UTC()
	instanceModel := cloneInstanceModel(model, index, port)
	state, err := m.driver.Inspect(parent, instanceModel)
	if errors.Is(err, deployment.ErrNotFound) {
		return InstanceStatus{}, false
	}
	status := InstanceStatus{
		InstanceID: instanceID(model.ID, index), Index: index, Port: port,
		LastChecked: now,
	}
	if state.Port != 0 {
		status.Port = state.Port
		instanceModel = cloneInstanceModel(model, index, state.Port)
	}
	if err != nil {
		status.Phase = PhaseUnknown
		status.LastError = err.Error()
		return status, true
	}
	status.Container = state.Status
	status.Health = state.Health
	status.ExitCode = state.ExitCode
	status.OOMKilled = state.OOMKilled
	status.AssignedGPUs = append([]int(nil), state.AssignedGPUs...)
	status.LastError = state.Error
	if !state.Running {
		status.Phase = PhaseUnloaded
		if state.ExitCode != 0 || state.OOMKilled || state.Error != "" {
			status.Phase = PhaseFailed
		}
		return status, true
	}
	if state.Health == "unhealthy" {
		status.Phase = PhaseUnhealthy
		return status, true
	}
	if instanceModel.Readiness != nil {
		request, requestErr := http.NewRequestWithContext(parent, http.MethodGet, instanceModel.Readiness.URL, nil)
		if requestErr != nil {
			status.Phase = PhaseUnhealthy
			status.LastError = requestErr.Error()
			return status, true
		}
		response, requestErr := m.client.Do(request)
		if requestErr != nil {
			status.Phase = PhaseUnhealthy
			status.LastError = requestErr.Error()
			return status, true
		}
		response.Body.Close()
		if response.StatusCode != instanceModel.Readiness.SuccessStatus {
			status.Phase = PhaseUnhealthy
			status.LastError = fmt.Sprintf("readiness returned HTTP %d", response.StatusCode)
			return status, true
		}
	}
	status.Phase = PhaseReady
	return status, true
}

func (m *Manager) currentStatus(modelID string) Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneStatus(m.statuses[modelID])
}

func (m *Manager) storeStatus(status Status) {
	m.mu.Lock()
	m.statuses[status.ModelID] = status
	m.mu.Unlock()
}

func (m *Manager) updateTransition(modelID string, phase Phase, desired string) {
	m.mu.Lock()
	status := m.statuses[modelID]
	status.Phase = phase
	status.Desired = desired
	status.LastError = ""
	status.LastChecked = time.Now().UTC()
	m.statuses[modelID] = status
	m.mu.Unlock()
}

func (m *Manager) setFailure(modelID string, err error) {
	m.mu.Lock()
	status := m.statuses[modelID]
	status.Phase = PhaseFailed
	status.LastError = err.Error()
	status.LastChecked = time.Now().UTC()
	m.statuses[modelID] = status
	m.mu.Unlock()
}

func (m *Manager) setInstancePhase(modelID string, index int, phase Phase, err string) {
	m.mu.Lock()
	status := cloneStatus(m.statuses[modelID])
	for i := range status.Instances {
		if status.Instances[i].Index == index {
			status.Instances[i].Phase = phase
			status.Instances[i].LastError = err
			status.Instances[i].LastChecked = time.Now().UTC()
			break
		}
	}
	status = aggregateStatus(modelID, status.Desired, status.Instances, false)
	m.statuses[modelID] = status
	m.mu.Unlock()
}

func (m *Manager) setInstanceFailure(modelID string, index int, err error) {
	m.setInstancePhase(modelID, index, PhaseFailed, err.Error())
}

func (m *Manager) removeInstance(modelID string, index int) {
	m.mu.Lock()
	status := m.statuses[modelID]
	var instances []InstanceStatus
	for _, instance := range status.Instances {
		if instance.Index != index {
			instances = append(instances, instance)
		}
	}
	status = aggregateStatus(modelID, status.Desired, instances, true)
	m.statuses[modelID] = status
	m.mu.Unlock()
}

func (m *Manager) setUnloaded(modelID string) {
	m.storeStatus(Status{ModelID: modelID, Phase: PhaseUnloaded, Desired: "unloaded", LastChecked: time.Now().UTC()})
}

func (m *Manager) stopAllInstances(ctx context.Context, model manifest.Model, remove bool) error {
	status := m.currentStatus(model.ID)
	instances := sortedInstances(status.Instances)
	if len(instances) == 0 {
		instances = []InstanceStatus{{InstanceID: model.ID, Index: 1, Port: basePort(model)}}
	}
	for i := len(instances) - 1; i >= 0; i-- {
		instance := instances[i]
		if err := m.driver.Stop(ctx, cloneInstanceModel(model, instance.Index, instance.Port), remove); err != nil {
			return err
		}
	}
	return nil
}
