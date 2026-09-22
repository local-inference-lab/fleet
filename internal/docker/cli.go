package docker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

const managedLabel = "ai.local-inference-lab.lil-fleet.managed"
const gpuLabel = "ai.local-inference-lab.lil-fleet.gpus"

type CLIDriver struct {
	binary string
}

func NewCLIDriver(binary string) *CLIDriver {
	return &CLIDriver{binary: binary}
}

func ContainerName(modelID string) string { return "lil-fleet-" + modelID }

func (d *CLIDriver) Start(ctx context.Context, model manifest.Model, assignedGPUs []int) error {
	state, err := d.Inspect(ctx, model)
	if err != nil && !errors.Is(err, deployment.ErrNotFound) {
		return err
	}
	fingerprint, err := modelFingerprint(model)
	if err != nil {
		return err
	}
	if state.Exists {
		actual, labelErr := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.fingerprint\" }}", ContainerName(model.ID))
		if labelErr == nil && strings.TrimSpace(actual) == fingerprint && equalGPUIndices(state.AssignedGPUs, assignedGPUs) {
			if state.Running {
				return nil
			}
			_, err = d.output(ctx, "start", ContainerName(model.ID))
			return err
		}
		if _, err := d.output(ctx, "rm", "-f", ContainerName(model.ID)); err != nil {
			return fmt.Errorf("remove stale container: %w", err)
		}
	}

	args := []string{"create", "--name", ContainerName(model.ID),
		"--label", managedLabel + "=true",
		"--label", "ai.local-inference-lab.lil-fleet.model=" + model.ID,
		"--label", "ai.local-inference-lab.lil-fleet.fingerprint=" + fingerprint,
		"--label", gpuLabel + "=" + joinGPUIndices(assignedGPUs),
	}
	keys := make([]string, 0, len(model.Environment))
	for key := range model.Environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		args = append(args, "--env", key+"="+model.Environment[key])
	}
	for _, mount := range model.Mounts {
		value := "type=bind,source=" + mount.Source + ",target=" + mount.Target
		if mount.ReadOnly {
			value += ",readonly"
		}
		args = append(args, "--mount", value)
	}
	for _, port := range model.Ports {
		mapping := fmt.Sprintf("%s:%d:%d/%s", port.HostIP, port.HostPort, port.ContainerPort, port.Protocol)
		args = append(args, "--publish", mapping)
	}
	if len(assignedGPUs) != 0 {
		// Docker's --gpus parser requires the inner device request to retain
		// quotes when multiple comma-separated IDs are passed as one argv item.
		args = append(args, "--gpus", "\"device="+joinGPUIndices(assignedGPUs)+"\"")
	} else if model.GPUs != "" {
		args = append(args, "--gpus", model.GPUs)
	}
	if model.ShmSize != "" {
		args = append(args, "--shm-size", model.ShmSize)
	}
	if model.IPC != "" {
		args = append(args, "--ipc", model.IPC)
	}
	if model.NetworkMode != "" {
		args = append(args, "--network", model.NetworkMode)
	}
	if model.Entrypoint != "" {
		args = append(args, "--entrypoint", model.Entrypoint)
	}
	for _, value := range model.SecurityOpt {
		args = append(args, "--security-opt", value)
	}
	for _, value := range model.Ulimits {
		args = append(args, "--ulimit", value)
	}
	if model.Init {
		args = append(args, "--init")
	}
	if model.Restart != "" {
		args = append(args, "--restart", model.Restart)
	}
	args = append(args, model.Image)
	args = append(args, model.Command...)
	if _, err := d.output(ctx, args...); err != nil {
		return fmt.Errorf("create container: %w", err)
	}
	if _, err := d.output(ctx, "start", ContainerName(model.ID)); err != nil {
		return fmt.Errorf("start container: %w", err)
	}
	return nil
}

func (d *CLIDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	state, err := d.Inspect(ctx, model)
	if errors.Is(err, deployment.ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if !state.Exists {
		return nil
	}
	var args []string
	if remove {
		args = []string{"rm", "-f", ContainerName(model.ID)}
	} else {
		args = []string{"stop", ContainerName(model.ID)}
	}
	if _, err := d.output(ctx, args...); err != nil {
		return fmt.Errorf("stop container: %w", err)
	}
	return nil
}

func (d *CLIDriver) Inspect(ctx context.Context, model manifest.Model) (deployment.ContainerState, error) {
	raw, err := d.output(ctx, "inspect", "--format", "{{json .State}}", ContainerName(model.ID))
	if err != nil {
		message := strings.ToLower(err.Error())
		if strings.Contains(message, "no such object") || strings.Contains(message, "no such container") {
			return deployment.ContainerState{}, deployment.ErrNotFound
		}
		return deployment.ContainerState{}, fmt.Errorf("inspect container: %w", err)
	}
	var state struct {
		Status    string `json:"Status"`
		Running   bool   `json:"Running"`
		ExitCode  int    `json:"ExitCode"`
		OOMKilled bool   `json:"OOMKilled"`
		Error     string `json:"Error"`
		Health    *struct {
			Status string `json:"Status"`
		} `json:"Health"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &state); err != nil {
		return deployment.ContainerState{}, fmt.Errorf("decode docker state: %w", err)
	}
	managed, err := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.managed\" }}", ContainerName(model.ID))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect managed label: %w", err)
	}
	if strings.TrimSpace(managed) != "true" {
		return deployment.ContainerState{}, fmt.Errorf("container %q is unmanaged", ContainerName(model.ID))
	}
	health := ""
	if state.Health != nil {
		health = state.Health.Status
	}
	gpus, err := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.gpus\" }}", ContainerName(model.ID))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect gpu label: %w", err)
	}
	assigned, err := parseGPUIndices(strings.TrimSpace(gpus))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect gpu label: %w", err)
	}
	return deployment.ContainerState{
		Exists: true, Running: state.Running, Status: state.Status, Health: health,
		ExitCode: state.ExitCode, OOMKilled: state.OOMKilled, Error: state.Error, AssignedGPUs: assigned,
	}, nil
}

func joinGPUIndices(indices []int) string {
	values := make([]string, len(indices))
	for i, index := range indices {
		values[i] = fmt.Sprintf("%d", index)
	}
	return strings.Join(values, ",")
}

func parseGPUIndices(value string) ([]int, error) {
	if value == "" || value == "<no value>" {
		return nil, nil
	}
	parts := strings.Split(value, ",")
	result := make([]int, len(parts))
	for i, part := range parts {
		parsed, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("invalid GPU index %q", part)
		}
		result[i] = parsed
	}
	return result, nil
}

func equalGPUIndices(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func (d *CLIDriver) output(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, d.binary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
		}
		return "", fmt.Errorf("docker %s: %s", args[0], message)
	}
	return string(output), nil
}

func modelFingerprint(model manifest.Model) (string, error) {
	encoded, err := json.Marshal(model)
	if err != nil {
		return "", fmt.Errorf("fingerprint model: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
