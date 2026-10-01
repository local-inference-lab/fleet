package docker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

const managedLabel = "ai.local-inference-lab.lil-fleet.managed"
const gpuLabel = "ai.local-inference-lab.lil-fleet.gpus"
const modelLabel = "ai.local-inference-lab.lil-fleet.model"
const instanceLabel = "ai.local-inference-lab.lil-fleet.instance"
const instanceIndexLabel = "ai.local-inference-lab.lil-fleet.instance_index"
const portLabel = "ai.local-inference-lab.lil-fleet.port"

type CLIDriver struct {
	binary string
}

func NewCLIDriver(binary string) *CLIDriver {
	return &CLIDriver{binary: binary}
}

func ContainerName(modelID string) string { return "lil-fleet-" + modelID }

func containerName(model manifest.Model) string {
	if model.InstanceID != "" {
		return ContainerName(model.InstanceID)
	}
	return ContainerName(model.ID)
}

func profileID(model manifest.Model) string {
	if model.InstanceProfileID != "" {
		return model.InstanceProfileID
	}
	return model.ID
}

func instanceID(model manifest.Model) string {
	if model.InstanceID != "" {
		return model.InstanceID
	}
	return model.ID
}

func instanceIndex(model manifest.Model) int {
	if model.InstanceIndex != 0 {
		return model.InstanceIndex
	}
	return 1
}

func (d *CLIDriver) Start(ctx context.Context, model manifest.Model, assignedGPUs []int) error {
	name := containerName(model)
	state, err := d.Inspect(ctx, model)
	if err != nil && !errors.Is(err, deployment.ErrNotFound) {
		return err
	}
	fingerprint, err := d.fingerprint(ctx, model)
	if err != nil {
		return err
	}
	if state.Exists {
		actual, labelErr := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.fingerprint\" }}", name)
		if labelErr == nil && strings.TrimSpace(actual) == fingerprint && equalGPUIndices(state.AssignedGPUs, assignedGPUs) {
			if state.Running {
				return nil
			}
			_, err = d.output(ctx, "start", name)
			return err
		}
		if _, err := d.output(ctx, "rm", "-f", name); err != nil {
			return fmt.Errorf("remove stale container: %w", err)
		}
	}

	args := []string{"create", "--name", name,
		"--label", managedLabel + "=true",
		"--label", modelLabel + "=" + profileID(model),
		"--label", instanceLabel + "=" + instanceID(model),
		"--label", instanceIndexLabel + "=" + strconv.Itoa(instanceIndex(model)),
		"--label", portLabel + "=" + strconv.Itoa(model.InstancePort),
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
		source := mountSource(model, mount)
		if mount.PerModel {
			ensurePerModelDirectory(source)
		}
		value := "type=bind,source=" + source + ",target=" + mount.Target
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
	if model.MemoryLimit != "" {
		args = append(args, "--memory", model.MemoryLimit)
	}
	if model.MemorySwapLimit != "" {
		args = append(args, "--memory-swap", model.MemorySwapLimit)
	}
	if model.Entrypoint != "" {
		args = append(args, "--entrypoint", model.Entrypoint)
	}
	args = append(args, hardeningArgs(model)...)
	for _, value := range model.SecurityOpt {
		args = append(args, "--security-opt", value)
	}
	for _, value := range model.Ulimits {
		args = append(args, "--ulimit", value)
	}
	if model.Init {
		args = append(args, "--init")
	}
	if model.Privileged {
		args = append(args, "--privileged")
	}
	if seconds := model.StopTimeoutSeconds(); seconds != 0 {
		args = append(args, "--stop-timeout", strconv.Itoa(seconds))
	}
	if model.Restart != "" {
		args = append(args, "--restart", model.Restart)
	}
	args = append(args, model.Image)
	args = append(args, model.Command...)
	if _, err := d.output(ctx, args...); err != nil {
		return fmt.Errorf("create container: %w", err)
	}
	if _, err := d.output(ctx, "start", name); err != nil {
		return fmt.Errorf("start container: %w", err)
	}
	return nil
}

func (d *CLIDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	name := containerName(model)
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
		args = []string{"rm", "-f", name}
	} else {
		args = []string{"stop", name}
	}
	if _, err := d.output(ctx, args...); err != nil {
		return fmt.Errorf("stop container: %w", err)
	}
	return nil
}

func (d *CLIDriver) Inspect(ctx context.Context, model manifest.Model) (deployment.ContainerState, error) {
	name := containerName(model)
	raw, err := d.output(ctx, "inspect", "--format", "{{json .State}}", name)
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
	managed, err := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.managed\" }}", name)
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect managed label: %w", err)
	}
	if strings.TrimSpace(managed) != "true" {
		return deployment.ContainerState{}, fmt.Errorf("container %q is unmanaged", name)
	}
	health := ""
	if state.Health != nil {
		health = state.Health.Status
	}
	gpus, err := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.gpus\" }}", name)
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect gpu label: %w", err)
	}
	assigned, err := parseGPUIndices(strings.TrimSpace(gpus))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect gpu label: %w", err)
	}
	portValue, err := d.output(ctx, "inspect", "--format", "{{ index .Config.Labels \"ai.local-inference-lab.lil-fleet.port\" }}", name)
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect port label: %w", err)
	}
	port, err := parsePort(strings.TrimSpace(portValue))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect port label: %w", err)
	}
	return deployment.ContainerState{
		Exists: true, Running: state.Running, Status: state.Status, Health: health,
		ExitCode: state.ExitCode, OOMKilled: state.OOMKilled, Error: state.Error, AssignedGPUs: assigned, Port: port,
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

func parsePort(value string) (int, error) {
	if value == "" || value == "<no value>" {
		return 0, nil
	}
	port, err := strconv.Atoi(value)
	if err != nil || port < 0 || port > 65535 {
		return 0, fmt.Errorf("invalid port %q", value)
	}
	return port, nil
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

// hardeningVersion is part of the fingerprint so containers created before a
// change to the default hardening are recreated on their next start.
const hardeningVersion = "2"

// hardeningArgs applies least-privilege defaults. Privileged profiles opt out
// explicitly; everything else drops every capability (add back individual ones
// with cap_add) and cannot gain privileges through setuid binaries. Neither
// default affects GPU access, which the NVIDIA runtime hook sets up outside the
// container's capability set.
func hardeningArgs(model manifest.Model) []string {
	var args []string
	if !model.Privileged {
		args = append(args, "--cap-drop", "ALL")
		for _, capability := range model.CapAdd {
			args = append(args, "--cap-add", capability)
		}
		if !hasSecurityOpt(model.SecurityOpt, "no-new-privileges") {
			args = append(args, "--security-opt", "no-new-privileges=true")
		}
	}
	if model.User != "" {
		args = append(args, "--user", model.User)
	}
	if model.ReadOnly {
		args = append(args, "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev")
	}
	return args
}

func hasSecurityOpt(options []string, name string) bool {
	for _, option := range options {
		if option == name || strings.HasPrefix(option, name+"=") || strings.HasPrefix(option, name+":") {
			return true
		}
	}
	return false
}

// mountSource returns the host path for a bind mount. Per-model mounts get a
// subdirectory named after the profile (replicas share their profile's
// directory) so writable caches cannot be read or poisoned across profiles.
// The container target is unchanged, so paths such as VLLM_CACHE_ROOT=/cache/x
// keep working.
func mountSource(model manifest.Model, mount manifest.Mount) string {
	if !mount.PerModel {
		return mount.Source
	}
	return path.Join(mount.Source, profileID(model))
}

// ensurePerModelDirectory creates a per-model cache directory when the parent
// is visible to Fleet. When Fleet runs in a container without that host path,
// creation is skipped and Docker reports the missing bind source instead.
func ensurePerModelDirectory(dir string) {
	if info, err := os.Stat(filepath.Dir(dir)); err != nil || !info.IsDir() {
		return
	}
	_ = os.Mkdir(dir, 0o700)
}

type fingerprintInput struct {
	Model            manifest.Model    `json:"model"`
	HardeningVersion string            `json:"hardening_version"`
	ImageID          string            `json:"image_id,omitempty"`
	SeccompProfiles  map[string]string `json:"seccomp_profiles,omitempty"`
}

// fingerprint identifies everything that shapes a container: the manifest
// profile, the hardening defaults, the contents of referenced seccomp profiles,
// and (best effort) the local image ID so a retagged image is picked up.
func (d *CLIDriver) fingerprint(ctx context.Context, model manifest.Model) (string, error) {
	input := fingerprintInput{Model: model, HardeningVersion: hardeningVersion}
	for _, option := range model.SecurityOpt {
		profile, ok := manifest.SeccompProfilePath(option)
		if !ok {
			continue
		}
		contents, err := os.ReadFile(profile)
		if err != nil {
			return "", fmt.Errorf("read seccomp profile: %w", err)
		}
		sum := sha256.Sum256(contents)
		if input.SeccompProfiles == nil {
			input.SeccompProfiles = map[string]string{}
		}
		input.SeccompProfiles[profile] = hex.EncodeToString(sum[:])
	}
	if id, err := d.output(ctx, "image", "inspect", "--format", "{{.Id}}", model.Image); err == nil {
		input.ImageID = strings.TrimSpace(id)
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("fingerprint model: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
