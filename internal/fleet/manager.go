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
	ID          string `json:"id"`
	Description string `json:"description,omitempty"`
	Image       string `json:"image"`
}

type Status struct {
	ModelID      string    `json:"model_id"`
	Phase        Phase     `json:"phase"`
	Desired      string    `json:"desired_state"`
	Container    string    `json:"container_status,omitempty"`
	Health       string    `json:"health,omitempty"`
	ExitCode     int       `json:"exit_code,omitempty"`
	OOMKilled    bool      `json:"oom_killed,omitempty"`
	AssignedGPUs []int     `json:"assigned_gpus,omitempty"`
	LastError    string    `json:"last_error,omitempty"`
	LastChecked  time.Time `json:"last_checked"`
}

type Operation struct {
	ID         string     `json:"id"`
	Kind       string     `json:"kind"`
	ModelID    string     `json:"model_id"`
	State      string     `json:"state"`
	Error      string     `json:"error,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
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
		models = append(models, ModelInfo{ID: model.ID, Description: model.Description, Image: model.Image})
	}
	return models
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
	return status.Phase == PhaseUnloaded || status.Phase == PhaseFailed
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
		result = append(result, status)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ModelID < result[j].ModelID })
	return result
}

func (m *Manager) Status(modelID string) (Status, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	status, ok := m.statuses[modelID]
	return status, ok
}

func (m *Manager) Operation(id string) (Operation, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	op, ok := m.operations[id]
	return op, ok
}

func (m *Manager) Activate(modelID string) (Operation, bool, error) {
	m.mu.Lock()
	cfg := m.manifest
	model, ok := cfg.Model(modelID)
	if !ok {
		m.mu.Unlock()
		return Operation{}, false, fmt.Errorf("unknown model %q", modelID)
	}
	if m.inFlight != "" {
		op := m.operations[m.inFlight]
		if op.Kind == "activate" && op.ModelID == modelID {
			m.mu.Unlock()
			return op, true, nil
		}
		m.mu.Unlock()
		return Operation{}, false, &ConflictError{Operation: op}
	}
	if m.isReadyNoopLocked(modelID, cfg) {
		status := m.statuses[modelID]
		status.Desired = "ready"
		m.statuses[modelID] = status
		m.mu.Unlock()
		return Operation{}, true, nil
	}
	var assignedGPUs []int
	if model.Placement != nil {
		devices, err := m.gpus.Snapshot(context.Background())
		if err != nil {
			m.mu.Unlock()
			return Operation{}, false, &InsufficientResourcesError{ModelID: modelID, Err: err}
		}
		allocator := gpu.Allocator{Topology: gpu.Topology{
			Groups:           cfg.Runtime.GPUTopology.Groups,
			MaxUsedMemoryMiB: cfg.Runtime.GPUTopology.MaxUsedMemoryMiB,
		}}
		var reserved []int
		for id, status := range m.statuses {
			if id == modelID {
				continue
			}
			switch status.Phase {
			case PhaseReady, PhaseLoading, PhaseUnhealthy:
				reserved = append(reserved, status.AssignedGPUs...)
			}
		}
		assignedGPUs, err = allocator.Allocate(devices, model.Placement.GPUCount, reserved)
		if err != nil {
			m.mu.Unlock()
			return Operation{}, false, &InsufficientResourcesError{ModelID: modelID, Err: err}
		}
		status := m.statuses[modelID]
		status.Phase = PhaseLoading
		status.Desired = "ready"
		status.AssignedGPUs = append([]int(nil), assignedGPUs...)
		status.LastChecked = time.Now().UTC()
		m.statuses[modelID] = status
	}
	op := newOperation("activate", modelID)
	m.operations[op.ID] = op
	m.inFlight = op.ID
	m.mu.Unlock()
	go m.runActivate(op.ID, cfg, model, assignedGPUs)
	return op, false, nil
}

func (m *Manager) isReadyNoopLocked(modelID string, cfg *manifest.Manifest) bool {
	if m.statuses[modelID].Phase != PhaseReady {
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
	op := newOperation("unload", modelID)
	m.operations[op.ID] = op
	m.inFlight = op.ID
	m.mu.Unlock()
	go m.runUnload(op.ID, cfg, model)
	return op, false, nil
}

func newOperation(kind, modelID string) Operation {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		panic(fmt.Sprintf("generate operation id: %v", err))
	}
	return Operation{ID: hex.EncodeToString(buf), Kind: kind, ModelID: modelID, State: "pending", CreatedAt: time.Now().UTC()}
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
	now := time.Now().UTC()
	m.mu.Lock()
	op := m.operations[id]
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

func (m *Manager) runActivate(operationID string, cfg *manifest.Manifest, model manifest.Model, assignedGPUs []int) {
	modelID := model.ID
	m.beginOperation(operationID)
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Runtime.ReadinessDuration())
	defer cancel()

	if !cfg.Runtime.ConcurrentDeployments {
		for _, other := range cfg.Models {
			if other.ID == modelID {
				continue
			}
			m.updateTransition(other.ID, PhaseStopping, "unloaded")
			opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.OperationDuration())
			err := m.driver.Stop(opCtx, other, cfg.Runtime.RemoveOnUnload)
			opCancel()
			if err != nil {
				m.setFailure(other.ID, err)
				m.finishOperation(operationID, fmt.Errorf("unload %s before switch: %w", other.ID, err))
				return
			}
			m.setUnloaded(other.ID)
		}
	}

	m.updateTransition(modelID, PhaseLoading, "ready")
	m.setAssignedGPUs(modelID, assignedGPUs)
	opCtx, opCancel := context.WithTimeout(ctx, cfg.Runtime.OperationDuration())
	err := m.driver.Start(opCtx, model, assignedGPUs)
	opCancel()
	if err != nil {
		m.setFailure(modelID, err)
		m.finishOperation(operationID, err)
		return
	}

	ticker := time.NewTicker(cfg.Runtime.PollDuration())
	defer ticker.Stop()
	for {
		status := m.probe(ctx, model, cfg.Runtime)
		if status.Phase == PhaseReady {
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
	if err := m.driver.Stop(ctx, model, cfg.Runtime.RemoveOnUnload); err != nil {
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
		m.mu.RUnlock()
		if current.Phase == PhaseLoading && status.Phase != PhaseReady && status.Phase != PhaseFailed {
			status.Phase = PhaseLoading
			status.Desired = "ready"
			status.AssignedGPUs = append([]int(nil), current.AssignedGPUs...)
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
		m.statuses[status.ModelID] = status
		m.mu.Unlock()
	}
}

func (m *Manager) probe(parent context.Context, model manifest.Model, runtime manifest.RuntimeConfig) Status {
	ctx, cancel := context.WithTimeout(parent, runtime.OperationDuration())
	defer cancel()
	now := time.Now().UTC()
	state, err := m.driver.Inspect(ctx, model)
	if errors.Is(err, deployment.ErrNotFound) {
		return Status{ModelID: model.ID, Phase: PhaseUnloaded, Desired: "unloaded", LastChecked: now}
	}
	if err != nil {
		return Status{ModelID: model.ID, Phase: PhaseUnknown, Desired: m.desired(model.ID), LastError: err.Error(), LastChecked: now}
	}
	status := Status{
		ModelID: model.ID, Desired: m.desired(model.ID), Container: state.Status,
		Health: state.Health, ExitCode: state.ExitCode, OOMKilled: state.OOMKilled,
		AssignedGPUs: state.AssignedGPUs, LastError: state.Error, LastChecked: now,
	}
	if !state.Running {
		status.Phase = PhaseUnloaded
		if state.ExitCode != 0 || state.OOMKilled || state.Error != "" {
			status.Phase = PhaseFailed
		}
		return status
	}
	if state.Health == "unhealthy" {
		status.Phase = PhaseUnhealthy
		return status
	}
	if model.Readiness != nil {
		request, requestErr := http.NewRequestWithContext(parent, http.MethodGet, model.Readiness.URL, nil)
		if requestErr != nil {
			status.Phase = PhaseUnhealthy
			status.LastError = requestErr.Error()
			return status
		}
		response, requestErr := m.client.Do(request)
		if requestErr != nil {
			status.Phase = PhaseUnhealthy
			status.LastError = requestErr.Error()
			return status
		}
		response.Body.Close()
		if response.StatusCode != model.Readiness.SuccessStatus {
			status.Phase = PhaseUnhealthy
			status.LastError = fmt.Sprintf("readiness returned HTTP %d", response.StatusCode)
			return status
		}
	}
	status.Phase = PhaseReady
	return status
}

func (m *Manager) desired(modelID string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.statuses[modelID].Desired
}

func (m *Manager) currentStatus(modelID string) Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.statuses[modelID]
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

func (m *Manager) setAssignedGPUs(modelID string, assigned []int) {
	m.mu.Lock()
	status := m.statuses[modelID]
	status.AssignedGPUs = append([]int(nil), assigned...)
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

func (m *Manager) setUnloaded(modelID string) {
	m.storeStatus(Status{ModelID: modelID, Phase: PhaseUnloaded, Desired: "unloaded", LastChecked: time.Now().UTC()})
}
