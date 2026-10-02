package fleet

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
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
	InstanceID string `json:"instance_id"`
	Index      int    `json:"index"`
	// Port is always present: API clients (llmconduit) treat it as required.
	Port         int       `json:"port"`
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

// InvalidRequestError reports a lifecycle request that can never succeed as
// written (for example an out-of-range replica count).
type InvalidRequestError struct {
	Err error
}

func (e *InvalidRequestError) Error() string { return e.Err.Error() }
func (e *InvalidRequestError) Unwrap() error { return e.Err }

const (
	// gpuSnapshotTimeout bounds nvidia-smi on the request path; it runs
	// without the manager lock so a slow driver cannot freeze status reads.
	gpuSnapshotTimeout = 5 * time.Second
	// loadingPollInterval is how often an activation checks readiness, so a
	// model becomes routable within about a second of its health turning green.
	loadingPollInterval = time.Second
	// probeTimeout bounds one readiness HTTP probe; probes run in parallel.
	probeTimeout = 2 * time.Second
	// maxParallelProbes bounds concurrent readiness probes per reconcile.
	maxParallelProbes = 16
	// operationTTL and maxOperations bound the finished-operation history.
	operationTTL  = time.Hour
	maxOperations = 256
)

type Manager struct {
	manifest *manifest.Manifest
	driver   deployment.Driver
	gpus     gpu.Provider
	logger   *slog.Logger
	client   *http.Client
	now      func() time.Time

	// mu guards the maps below and is only held for in-memory work, never
	// across Docker, nvidia-smi, or readiness I/O.
	mu         sync.RWMutex
	statuses   map[string]Status
	operations map[string]Operation
	// inFlight maps a model ID to its running lifecycle operation. With
	// concurrent deployments each model has at most one operation; without
	// them the map holds at most one entry in total (exclusive switching).
	inFlight map[string]string
	// generations is bumped by every lifecycle write to a model's status so
	// reconciliation can discard results observed before that write.
	generations map[string]uint64

	// lifecycleMu serializes container stop/create/start across operations so
	// two GPU-heavy profiles never start at the same instant. It is released
	// before the (potentially very long) readiness wait.
	lifecycleMu sync.Mutex

	cancel context.CancelFunc
	done   chan struct{}
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
		manifest:    cfg,
		driver:      driver,
		gpus:        provider,
		logger:      logger,
		client:      &http.Client{Timeout: 3 * time.Second},
		now:         func() time.Time { return time.Now().UTC() },
		statuses:    statuses,
		operations:  make(map[string]Operation),
		inFlight:    make(map[string]string),
		generations: make(map[string]uint64),
		done:        make(chan struct{}),
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

	if len(m.inFlight) != 0 {
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

// inFlightLocked applies the lifecycle admission rules. A request identical to
// the model's running operation joins it (dup); any other request for that
// model conflicts. Without concurrent deployments, every operation is
// exclusive: a request for any model conflicts with a running one.
func (m *Manager) inFlightLocked(cfg *manifest.Manifest, modelID, kind string, requested *int) (Operation, bool, error) {
	if id, ok := m.inFlight[modelID]; ok {
		op := m.operations[id]
		if op.Kind == kind && (kind != "activate" || requested == nil || op.TargetInstances == *requested) {
			return op, true, nil
		}
		return Operation{}, false, &ConflictError{Operation: op}
	}
	if !cfg.Runtime.ConcurrentDeployments {
		for _, id := range m.inFlight {
			return Operation{}, false, &ConflictError{Operation: m.operations[id]}
		}
	}
	return Operation{}, false, nil
}

// ownedLocked reports whether a running operation is responsible for the
// model's status; reconciliation must not overwrite it meanwhile.
func (m *Manager) ownedLocked(modelID string) bool {
	if _, ok := m.inFlight[modelID]; ok {
		return true
	}
	return !m.manifest.Runtime.ConcurrentDeployments && len(m.inFlight) != 0
}

func (m *Manager) Activate(modelID string) (Operation, bool, error) {
	return m.ActivateInstances(modelID, nil)
}

func (m *Manager) ActivateInstances(modelID string, requested *int) (Operation, bool, error) {
	for attempt := 0; attempt < 3; attempt++ {
		op, noOp, retry, err := m.tryActivate(modelID, requested)
		if !retry {
			return op, noOp, err
		}
	}
	return Operation{}, false, &ConflictError{}
}

// tryActivate admits an activation in two short critical sections around the
// GPU snapshot: nvidia-smi can be slow, so it runs unlocked, and everything it
// informs (admission, reservations, allocation) is re-evaluated afterwards.
func (m *Manager) tryActivate(modelID string, requested *int) (Operation, bool, bool, error) {
	m.mu.Lock()
	cfg := m.manifest
	model, ok := cfg.Model(modelID)
	if !ok {
		m.mu.Unlock()
		return Operation{}, false, false, fmt.Errorf("unknown model %q", modelID)
	}
	op, dup, err := m.activationPreflightLocked(cfg, model, requested)
	m.mu.Unlock()
	if op.Kind == "noop" {
		return Operation{}, true, false, nil
	}
	if err != nil || dup {
		return op, dup, false, err
	}

	var devices []gpu.Device
	var snapshotErr error
	if model.Placement != nil {
		ctx, cancel := context.WithTimeout(context.Background(), gpuSnapshotTimeout)
		devices, snapshotErr = m.gpus.Snapshot(ctx)
		cancel()
	}

	m.mu.Lock()
	if m.manifest != cfg {
		m.mu.Unlock()
		return Operation{}, false, true, nil
	}
	op, dup, err = m.activationPreflightLocked(cfg, model, requested)
	if op.Kind == "noop" {
		m.mu.Unlock()
		return Operation{}, true, false, nil
	}
	if err != nil || dup {
		m.mu.Unlock()
		return op, dup, false, err
	}
	target := op.TargetInstances
	plan, err := m.planInstancesLocked(cfg, model, target, devices, snapshotErr)
	if err != nil {
		m.mu.Unlock()
		m.logger.Error("model load failed", "model_id", modelID, "error", sanitizeFor(model, err.Error()))
		return Operation{}, false, false, &InsufficientResourcesError{ModelID: modelID, Err: err}
	}
	m.markPlanLocked(modelID, plan)
	op = newOperation("activate", modelID, target, m.now())
	m.addOperationLocked(op)
	m.mu.Unlock()
	go m.runActivate(op.ID, cfg, model, plan)
	return op, false, false, nil
}

// activationPreflightLocked returns a joined duplicate operation, a "noop"
// marker for an already-satisfied request, or a pending operation shell whose
// TargetInstances is the validated replica target.
func (m *Manager) activationPreflightLocked(cfg *manifest.Manifest, model manifest.Model, requested *int) (Operation, bool, error) {
	op, dup, err := m.inFlightLocked(cfg, model.ID, "activate", requested)
	if err != nil || dup {
		return op, dup, err
	}
	target := m.targetInstancesLocked(model.ID, requested)
	if target < 1 || target > maxRequestedInstances {
		return Operation{}, false, &InvalidRequestError{Err: fmt.Errorf("instances must be between 1 and %d", maxRequestedInstances)}
	}
	if max := m.maxInstancesLocked(model); target > max {
		err := fmt.Errorf("requested %d instances, maximum is %d", target, max)
		m.logger.Error("model load failed", "model_id", model.ID, "error", err)
		return Operation{}, false, &InsufficientResourcesError{ModelID: model.ID, Err: err}
	}
	if m.isReadyNoopLocked(model.ID, cfg, target) {
		status := m.statuses[model.ID]
		status.Desired = "ready"
		status.DesiredInstances = target
		m.statuses[model.ID] = status
		return Operation{Kind: "noop"}, false, nil
	}
	return Operation{Kind: "activate", TargetInstances: target}, false, nil
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

func (m *Manager) planInstancesLocked(cfg *manifest.Manifest, model manifest.Model, target int, devices []gpu.Device, snapshotErr error) (instancePlan, error) {
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
	var allocator gpu.Allocator
	var reserved []int
	if model.Placement != nil {
		if snapshotErr != nil {
			return instancePlan{}, snapshotErr
		}
		allocator = gpu.Allocator{Topology: gpu.Topology{
			Groups:           cfg.Runtime.GPUTopology.Groups,
			MaxUsedMemoryMiB: cfg.Runtime.GPUTopology.MaxUsedMemoryMiB,
		}}
		// Reservations come from current state, not the snapshot, so GPUs
		// promised to another in-flight activation are never handed out twice.
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
	configured := configuredBasePorts(m.manifest)
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
	now := m.now()
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
	m.writeStatusLocked(status)
}

func (m *Manager) Unload(modelID string) (Operation, bool, error) {
	m.mu.Lock()
	cfg := m.manifest
	model, ok := cfg.Model(modelID)
	if !ok {
		m.mu.Unlock()
		return Operation{}, false, fmt.Errorf("unknown model %q", modelID)
	}
	if op, dup, err := m.inFlightLocked(cfg, modelID, "unload", nil); err != nil || dup {
		m.mu.Unlock()
		return op, dup, err
	}
	if status := m.statuses[modelID]; status.Phase == PhaseUnloaded && status.Container == "" && len(status.Instances) == 0 {
		status.Desired = "unloaded"
		m.statuses[modelID] = status
		m.mu.Unlock()
		return Operation{}, true, nil
	}
	op := newOperation("unload", modelID, 0, m.now())
	m.addOperationLocked(op)
	m.transitionLocked(modelID, PhaseStopping, "unloaded")
	m.mu.Unlock()
	go m.runUnload(op.ID, cfg, model)
	return op, false, nil
}

func newOperation(kind, modelID string, targetInstances int, now time.Time) Operation {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		panic(fmt.Sprintf("generate operation id: %v", err))
	}
	return Operation{ID: hex.EncodeToString(buf), Kind: kind, ModelID: modelID, TargetInstances: targetInstances, State: "pending", CreatedAt: now}
}

func (m *Manager) addOperationLocked(op Operation) {
	m.evictOperationsLocked()
	m.operations[op.ID] = op
	m.inFlight[op.ModelID] = op.ID
}

// evictOperationsLocked drops finished operations older than operationTTL and
// then the oldest finished ones beyond maxOperations. Running operations are
// never evicted, so a client polling one always finds it.
func (m *Manager) evictOperationsLocked() {
	now := m.now()
	var finished []Operation
	for id, op := range m.operations {
		if op.FinishedAt == nil {
			continue
		}
		if now.Sub(*op.FinishedAt) > operationTTL {
			delete(m.operations, id)
			continue
		}
		finished = append(finished, op)
	}
	excess := len(m.operations) - (maxOperations - 1)
	if excess <= 0 {
		return
	}
	sort.Slice(finished, func(i, j int) bool { return finished[i].FinishedAt.Before(*finished[j].FinishedAt) })
	for i := 0; i < excess && i < len(finished); i++ {
		delete(m.operations, finished[i].ID)
	}
}

func (m *Manager) beginOperation(id string) {
	now := m.now()
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
	message := ""
	if err != nil {
		message = m.sanitizeLocked(op.ModelID, err.Error())
	}
	m.mu.RUnlock()
	// Log before publishing the terminal state so an observer that sees the
	// finished operation also sees its log record.
	switch {
	case err != nil && op.Kind == "unload":
		m.logger.Error("model unload failed", "model_id", op.ModelID, "operation_id", id, "error", message)
	case err != nil:
		m.logger.Error("model load failed", "model_id", op.ModelID, "operation_id", id, "error", message)
	case op.Kind == "unload":
		m.logger.Info("model unloaded", "model_id", op.ModelID, "operation_id", id)
	default:
		m.logger.Info("model loaded", "model_id", op.ModelID, "operation_id", id)
	}

	now := m.now()
	m.mu.Lock()
	op = m.operations[id]
	op.State = "succeeded"
	if err != nil {
		op.State = "failed"
		op.Error = message
	}
	op.FinishedAt = &now
	m.operations[id] = op
	if m.inFlight[op.ModelID] == id {
		delete(m.inFlight, op.ModelID)
	}
	m.evictOperationsLocked()
	m.mu.Unlock()
}

func (m *Manager) runActivate(operationID string, cfg *manifest.Manifest, model manifest.Model, plan instancePlan) {
	modelID := model.ID
	m.beginOperation(operationID)
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Runtime.ReadinessDuration())
	defer cancel()

	if len(plan.start) != 0 {
		if puller, ok := m.driver.(deployment.ImagePuller); ok {
			pullCtx, pullCancel := context.WithTimeout(ctx, cfg.Runtime.PullDuration())
			err := puller.EnsureImage(pullCtx, model.Image)
			pullCancel()
			if err != nil {
				for _, start := range plan.start {
					m.setInstanceFailure(modelID, start.index, err)
				}
				m.finishOperation(operationID, err)
				return
			}
		}
	}

	m.lifecycleMu.Lock()
	err := m.applyPlan(ctx, operationID, cfg, model, plan)
	m.lifecycleMu.Unlock()
	if err != nil {
		m.finishOperation(operationID, err)
		return
	}
	m.awaitReady(ctx, operationID, cfg, model, plan.target)
}

// applyPlan performs the container mutations of an activation while holding
// lifecycleMu: an exclusive switch first stops every other model that has
// containers (in parallel), then surplus replicas stop and new ones start.
func (m *Manager) applyPlan(ctx context.Context, operationID string, cfg *manifest.Manifest, model manifest.Model, plan instancePlan) error {
	modelID := model.ID
	if !cfg.Runtime.ConcurrentDeployments {
		if err := m.stopOtherModels(ctx, operationID, cfg, modelID); err != nil {
			return err
		}
	}

	for _, stop := range plan.stop {
		opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.StopDuration(model))
		err := m.driver.Stop(opCtx, stop.model, cfg.Runtime.RemoveOnUnload)
		opCancel()
		if err != nil {
			m.setFailure(modelID, err)
			return err
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
			return err
		}
	}
	return nil
}

func (m *Manager) stopOtherModels(ctx context.Context, operationID string, cfg *manifest.Manifest, modelID string) error {
	var others []manifest.Model
	for _, other := range cfg.Models {
		if other.ID == modelID {
			continue
		}
		status := m.currentStatus(other.ID)
		// A model observed with no containers has nothing to stop; Unknown
		// (never inspected) is stopped defensively.
		if status.Phase == PhaseUnloaded && len(status.Instances) == 0 {
			continue
		}
		others = append(others, other)
	}
	errs := make([]error, len(others))
	var wg sync.WaitGroup
	for i, other := range others {
		m.updateTransition(other.ID, PhaseStopping, "unloaded")
		wg.Add(1)
		go func() {
			defer wg.Done()
			opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.StopDuration(other))
			defer opCancel()
			errs[i] = m.stopAllInstances(opCtx, other, cfg.Runtime.RemoveOnUnload)
		}()
	}
	wg.Wait()
	var failed error
	for i, other := range others {
		if errs[i] != nil {
			m.setFailure(other.ID, errs[i])
			m.logger.Error("model unload failed", "model_id", other.ID, "operation_id", operationID, "error", sanitizeFor(other, errs[i].Error()))
			if failed == nil {
				failed = fmt.Errorf("unload %s before switch: %w", other.ID, errs[i])
			}
			continue
		}
		m.setUnloaded(other.ID)
		m.logger.Info("model unloaded", "model_id", other.ID, "operation_id", operationID)
	}
	return failed
}

func (m *Manager) awaitReady(ctx context.Context, operationID string, cfg *manifest.Manifest, model manifest.Model, target int) {
	modelID := model.ID
	interval := cfg.Runtime.PollDuration()
	if interval > loadingPollInterval {
		interval = loadingPollInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		status := m.observeModel(ctx, cfg, model)
		if status.ReadyInstances == target && status.Phase == PhaseReady {
			status.Desired = "ready"
			status.DesiredInstances = target
			m.storeStatus(status)
			m.finishOperation(operationID, nil)
			return
		}
		if status.Phase == PhaseFailed || (status.Phase == PhaseUnloaded && status.ExitCode != 0) {
			status.Desired = "ready"
			m.storeStatus(status)
			m.finishOperation(operationID, fmt.Errorf("model %s exited before becoming ready", modelID))
			return
		}
		status.Phase = PhaseLoading
		status.Desired = "ready"
		status.DesiredInstances = target
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
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Runtime.StopDuration(model))
	defer cancel()
	if err := m.stopAllInstances(ctx, model, cfg.Runtime.RemoveOnUnload); err != nil {
		m.setFailure(modelID, err)
		m.finishOperation(operationID, err)
		return
	}
	m.setUnloaded(modelID)
	m.finishOperation(operationID, nil)
}

// refresh reconciles every model's status from one container listing plus
// parallel readiness probes. A result is discarded when a lifecycle write
// happened while it was being observed (generation changed) or when a running
// operation owns the model's status.
func (m *Manager) refresh(ctx context.Context) {
	m.mu.RLock()
	cfg := m.manifest
	generations := make(map[string]uint64, len(cfg.Models))
	desired := make(map[string]string, len(cfg.Models))
	for _, model := range cfg.Models {
		generations[model.ID] = m.generations[model.ID]
		desired[model.ID] = m.statuses[model.ID].Desired
	}
	m.mu.RUnlock()

	observed := m.observe(ctx, cfg, cfg.Models, desired)

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.manifest != cfg || ctx.Err() != nil {
		return
	}
	for _, model := range cfg.Models {
		if m.generations[model.ID] != generations[model.ID] || m.ownedLocked(model.ID) {
			continue
		}
		current := m.statuses[model.ID]
		status := observed[model.ID]
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
		m.statuses[model.ID] = status
	}
}

// observeModel is refresh for one model, used while an activation waits.
func (m *Manager) observeModel(ctx context.Context, cfg *manifest.Manifest, model manifest.Model) Status {
	return m.observe(ctx, cfg, []manifest.Model{model}, map[string]string{model.ID: "ready"})[model.ID]
}

type probeTarget struct {
	modelID  string
	position int
	url      string
	success  int
	model    manifest.Model
}

func (m *Manager) observe(ctx context.Context, cfg *manifest.Manifest, models []manifest.Model, desired map[string]string) map[string]Status {
	now := m.now()
	states, listErr := m.driver.List(ctx)
	byModel := map[string][]deployment.ContainerState{}
	for _, state := range states {
		byModel[state.ModelID] = append(byModel[state.ModelID], state)
	}

	instances := make(map[string][]InstanceStatus, len(models))
	var probes []probeTarget
	for _, model := range models {
		if listErr != nil {
			instances[model.ID] = []InstanceStatus{{
				InstanceID: model.ID, Index: 1, Port: basePort(model), Phase: PhaseUnknown,
				LastError: sanitizeFor(model, listErr.Error()), LastChecked: now,
			}}
			continue
		}
		modelDesired := desired[model.ID]
		for _, state := range byModel[model.ID] {
			instance, needsProbe := instanceFromState(model, state, modelDesired, now)
			instances[model.ID] = append(instances[model.ID], instance)
			if needsProbe {
				probeModel := cloneInstanceModel(model, instance.Index, instance.Port)
				probes = append(probes, probeTarget{
					modelID: model.ID, position: len(instances[model.ID]) - 1,
					url: probeModel.Readiness.URL, success: probeModel.Readiness.SuccessStatus, model: model,
				})
			}
		}
	}

	results := make([]string, len(probes))
	semaphore := make(chan struct{}, maxParallelProbes)
	var wg sync.WaitGroup
	for i, probe := range probes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			results[i] = m.probeReadiness(ctx, probe.url, probe.success)
		}()
	}
	wg.Wait()
	for i, probe := range probes {
		instance := &instances[probe.modelID][probe.position]
		if results[i] != "" {
			instance.Phase = PhaseUnhealthy
			instance.LastError = sanitizeFor(probe.model, results[i])
			continue
		}
		instance.Phase = PhaseReady
	}

	statuses := make(map[string]Status, len(models))
	for _, model := range models {
		modelDesired := desired[model.ID]
		if modelDesired == "" {
			modelDesired = "unloaded"
		}
		status := aggregateStatus(model.ID, modelDesired, instances[model.ID], true)
		if len(instances[model.ID]) == 0 {
			status.LastChecked = now
		}
		statuses[model.ID] = status
	}
	return statuses
}

// instanceFromState converts container state to an instance status. It
// reports whether the instance still needs an HTTP readiness probe.
func instanceFromState(model manifest.Model, state deployment.ContainerState, desired string, now time.Time) (InstanceStatus, bool) {
	index := state.InstanceIndex
	if index < 1 {
		index = 1
	}
	port := state.Port
	if port == 0 && index == 1 {
		port = basePort(model)
	}
	status := InstanceStatus{
		InstanceID: instanceID(model.ID, index), Index: index, Port: port,
		Container: state.Status, Health: state.Health, ExitCode: state.ExitCode, OOMKilled: state.OOMKilled,
		AssignedGPUs: append([]int(nil), state.AssignedGPUs...),
		LastError:    sanitizeFor(model, state.Error), LastChecked: now,
	}
	if !state.Running {
		status.Phase = PhaseUnloaded
		if (state.ExitCode != 0 || state.OOMKilled || state.Error != "") && !requestedStopExit(state, desired) {
			status.Phase = PhaseFailed
		}
		return status, false
	}
	if state.Health == "unhealthy" {
		status.Phase = PhaseUnhealthy
		return status, false
	}
	if model.Readiness == nil {
		status.Phase = PhaseReady
		return status, false
	}
	status.Phase = PhaseUnhealthy
	return status, true
}

// requestedStopExit recognizes the exit codes of a container Fleet asked to
// stop: 143 (SIGTERM honored) or 137 (SIGKILL after the stop timeout). They are
// a clean unload, not a failure, unless the kernel OOM killer was involved.
func requestedStopExit(state deployment.ContainerState, desired string) bool {
	return desired == "unloaded" && !state.OOMKilled && state.Error == "" &&
		(state.ExitCode == 137 || state.ExitCode == 143)
}

func (m *Manager) probeReadiness(parent context.Context, url string, success int) string {
	ctx, cancel := context.WithTimeout(parent, probeTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err.Error()
	}
	response, err := m.client.Do(request)
	if err != nil {
		return err.Error()
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 64<<10))
	response.Body.Close()
	if response.StatusCode != success {
		return fmt.Sprintf("readiness returned HTTP %d", response.StatusCode)
	}
	return ""
}

func (m *Manager) currentStatus(modelID string) Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneStatus(m.statuses[modelID])
}

// writeStatusLocked is the single lifecycle write path: it stores the status
// and bumps the model's generation so in-progress reconciliation drops its
// now-stale observation.
func (m *Manager) writeStatusLocked(status Status) {
	m.statuses[status.ModelID] = status
	m.generations[status.ModelID]++
}

func (m *Manager) storeStatus(status Status) {
	m.mu.Lock()
	m.writeStatusLocked(status)
	m.mu.Unlock()
}

func (m *Manager) updateTransition(modelID string, phase Phase, desired string) {
	m.mu.Lock()
	m.transitionLocked(modelID, phase, desired)
	m.mu.Unlock()
}

func (m *Manager) transitionLocked(modelID string, phase Phase, desired string) {
	status := m.statuses[modelID]
	status.Phase = phase
	status.Desired = desired
	status.LastError = ""
	status.LastChecked = m.now()
	m.writeStatusLocked(status)
}

func (m *Manager) setFailure(modelID string, err error) {
	m.mu.Lock()
	status := m.statuses[modelID]
	status.Phase = PhaseFailed
	status.LastError = m.sanitizeLocked(modelID, err.Error())
	status.LastChecked = m.now()
	m.writeStatusLocked(status)
	m.mu.Unlock()
}

func (m *Manager) setInstancePhase(modelID string, index int, phase Phase, err string) {
	m.mu.Lock()
	status := cloneStatus(m.statuses[modelID])
	for i := range status.Instances {
		if status.Instances[i].Index == index {
			status.Instances[i].Phase = phase
			status.Instances[i].LastError = m.sanitizeLocked(modelID, err)
			status.Instances[i].LastChecked = m.now()
			break
		}
	}
	m.writeStatusLocked(aggregateStatus(modelID, status.Desired, status.Instances, false))
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
	m.writeStatusLocked(aggregateStatus(modelID, status.Desired, instances, true))
	m.mu.Unlock()
}

func (m *Manager) setUnloaded(modelID string) {
	m.storeStatus(Status{ModelID: modelID, Phase: PhaseUnloaded, Desired: "unloaded", LastChecked: m.now()})
}

// stopAllInstances stops a model's replicas in parallel; each stop is bounded
// by ctx, which callers derive from runtime.stop_timeout.
func (m *Manager) stopAllInstances(ctx context.Context, model manifest.Model, remove bool) error {
	status := m.currentStatus(model.ID)
	instances := sortedInstances(status.Instances)
	if len(instances) == 0 {
		instances = []InstanceStatus{{InstanceID: model.ID, Index: 1, Port: basePort(model)}}
	}
	errs := make([]error, len(instances))
	var wg sync.WaitGroup
	for i, instance := range instances {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = m.driver.Stop(ctx, cloneInstanceModel(model, instance.Index, instance.Port), remove)
		}()
	}
	wg.Wait()
	return errors.Join(errs...)
}

// sanitizeLocked prepares Docker, runtime, or probe error text for the API by
// redacting the model's manifest environment values and bounding its size.
func (m *Manager) sanitizeLocked(modelID, message string) string {
	if message == "" {
		return ""
	}
	model, _ := m.manifest.Model(modelID)
	return sanitizeFor(model, message)
}

func sanitizeFor(model manifest.Model, message string) string {
	if message == "" {
		return ""
	}
	return deployment.SanitizeMessage(message, deployment.EnvironmentValues(model))
}
