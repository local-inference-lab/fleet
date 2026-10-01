package docker

import (
	"bytes"
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
	"time"

	"github.com/local-inference-lab/fleet/internal/deployment"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

const managedLabel = "ai.local-inference-lab.lil-fleet.managed"
const gpuLabel = "ai.local-inference-lab.lil-fleet.gpus"
const modelLabel = "ai.local-inference-lab.lil-fleet.model"
const instanceLabel = "ai.local-inference-lab.lil-fleet.instance"
const instanceIndexLabel = "ai.local-inference-lab.lil-fleet.instance_index"
const portLabel = "ai.local-inference-lab.lil-fleet.port"
const fingerprintLabel = "ai.local-inference-lab.lil-fleet.fingerprint"

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
	ctx, cancel := withDefaultTimeout(ctx, mutateTimeout)
	defer cancel()
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
		if state.Fingerprint == fingerprint && equalGPUIndices(state.AssignedGPUs, assignedGPUs) {
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
		"--label", fingerprintLabel + "=" + fingerprint,
		"--label", gpuLabel + "=" + joinGPUIndices(assignedGPUs),
	}
	envArgs, cliEnv := environmentArgs(model.Environment)
	args = append(args, envArgs...)
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
	secrets := deployment.EnvironmentValues(model)
	if _, err := d.run(ctx, cliEnv, secrets, args...); err != nil {
		return fmt.Errorf("create container: %w", err)
	}
	if _, err := d.run(ctx, nil, secrets, "start", name); err != nil {
		return fmt.Errorf("start container: %w", err)
	}
	return nil
}

// environmentArgs passes environment values to docker create through the CLI
// process environment ("--env KEY") instead of argv ("--env KEY=VALUE"), so
// secrets are not visible in the host process table. Keys that would also
// reconfigure the Docker CLI itself (DOCKER_HOST, proxies, HOME, ...) cannot be
// routed that way without redirecting the daemon connection, so they stay on
// argv; they are configuration, not credentials.
func environmentArgs(environment map[string]string) ([]string, []string) {
	keys := make([]string, 0, len(environment))
	for key := range environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var args, cliEnv []string
	for _, key := range keys {
		if affectsDockerCLI(key) {
			args = append(args, "--env", key+"="+environment[key])
			continue
		}
		args = append(args, "--env", key)
		cliEnv = append(cliEnv, key+"="+environment[key])
	}
	return args, cliEnv
}

func affectsDockerCLI(key string) bool {
	upper := strings.ToUpper(key)
	for _, prefix := range []string{"DOCKER_", "BUILDX_", "COMPOSE_", "XDG_", "LD_", "GO", "SSL_CERT_"} {
		if strings.HasPrefix(upper, prefix) {
			return true
		}
	}
	switch upper {
	case "HOME", "PATH", "TMPDIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY":
		return true
	}
	return false
}

func (d *CLIDriver) Stop(ctx context.Context, model manifest.Model, remove bool) error {
	ctx, cancel := withDefaultTimeout(ctx, mutateTimeout)
	defer cancel()
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
		if isNotFound(err) {
			return nil
		}
		return fmt.Errorf("stop container: %w", err)
	}
	return nil
}

// Inspect reads one container's state and Fleet labels with a single
// docker container inspect call.
func (d *CLIDriver) Inspect(ctx context.Context, model manifest.Model) (deployment.ContainerState, error) {
	ctx, cancel := withDefaultTimeout(ctx, readTimeout)
	defer cancel()
	name := containerName(model)
	raw, err := d.output(ctx, "container", "inspect", name)
	if err != nil {
		if isNotFound(err) {
			return deployment.ContainerState{}, deployment.ErrNotFound
		}
		return deployment.ContainerState{}, fmt.Errorf("inspect container: %w", err)
	}
	var docs []inspectDocument
	if err := json.Unmarshal([]byte(raw), &docs); err != nil || len(docs) != 1 {
		return deployment.ContainerState{}, fmt.Errorf("decode docker inspect output")
	}
	if docs[0].Config.Labels[managedLabel] != "true" {
		return deployment.ContainerState{}, fmt.Errorf("container %q is unmanaged", name)
	}
	return docs[0].state()
}

// List returns all Fleet-managed containers using one docker ps and one
// docker container inspect, regardless of how many models or replicas exist.
func (d *CLIDriver) List(ctx context.Context) ([]deployment.ContainerState, error) {
	ctx, cancel := withDefaultTimeout(ctx, readTimeout)
	defer cancel()
	ids, err := d.output(ctx, "ps", "--all", "--quiet", "--no-trunc", "--filter", "label="+managedLabel+"=true")
	if err != nil {
		return nil, fmt.Errorf("list containers: %w", err)
	}
	fields := strings.Fields(ids)
	if len(fields) == 0 {
		return nil, nil
	}
	raw, err := d.output(ctx, append([]string{"container", "inspect"}, fields...)...)
	// A container removed between ps and inspect makes docker exit non-zero
	// while still printing the remaining containers.
	if err != nil && strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("inspect containers: %w", err)
	}
	var docs []inspectDocument
	if decodeErr := json.Unmarshal([]byte(raw), &docs); decodeErr != nil {
		if err != nil {
			return nil, fmt.Errorf("inspect containers: %w", err)
		}
		return nil, fmt.Errorf("decode docker inspect output: %w", decodeErr)
	}
	states := make([]deployment.ContainerState, 0, len(docs))
	for _, doc := range docs {
		if doc.Config.Labels[managedLabel] != "true" {
			continue
		}
		state, err := doc.state()
		if err != nil {
			return nil, err
		}
		states = append(states, state)
	}
	return states, nil
}

// EnsureImage pulls the image only when it is not present locally.
func (d *CLIDriver) EnsureImage(ctx context.Context, image string) error {
	inspectCtx, cancel := withDefaultTimeout(ctx, readTimeout)
	_, err := d.output(inspectCtx, "image", "inspect", "--format", "{{.Id}}", image)
	cancel()
	if err == nil {
		return nil
	}
	if !isNotFound(err) {
		return fmt.Errorf("inspect image: %w", err)
	}
	pullCtx, cancel := withDefaultTimeout(ctx, defaultPullTimeout)
	defer cancel()
	if _, err := d.output(pullCtx, "pull", "--quiet", image); err != nil {
		return fmt.Errorf("pull image: %w", err)
	}
	return nil
}

type inspectDocument struct {
	Name  string `json:"Name"`
	State struct {
		Status    string `json:"Status"`
		Running   bool   `json:"Running"`
		ExitCode  int    `json:"ExitCode"`
		OOMKilled bool   `json:"OOMKilled"`
		Error     string `json:"Error"`
		Health    *struct {
			Status string `json:"Status"`
		} `json:"Health"`
	} `json:"State"`
	Config struct {
		Labels map[string]string `json:"Labels"`
	} `json:"Config"`
}

func (doc inspectDocument) state() (deployment.ContainerState, error) {
	labels := doc.Config.Labels
	assigned, err := parseGPUIndices(strings.TrimSpace(labels[gpuLabel]))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect gpu label: %w", err)
	}
	port, err := parsePort(strings.TrimSpace(labels[portLabel]))
	if err != nil {
		return deployment.ContainerState{}, fmt.Errorf("inspect port label: %w", err)
	}
	index := 1
	if value := strings.TrimSpace(labels[instanceIndexLabel]); value != "" {
		index, err = strconv.Atoi(value)
		if err != nil || index < 1 {
			return deployment.ContainerState{}, fmt.Errorf("invalid instance index label %q", value)
		}
	}
	health := ""
	if doc.State.Health != nil {
		health = doc.State.Health.Status
	}
	return deployment.ContainerState{
		Exists: true, Running: doc.State.Running, Status: doc.State.Status, Health: health,
		ExitCode: doc.State.ExitCode, OOMKilled: doc.State.OOMKilled, Error: doc.State.Error,
		AssignedGPUs: assigned, Port: port,
		Name: strings.TrimPrefix(doc.Name, "/"), ModelID: labels[modelLabel], InstanceID: labels[instanceLabel],
		InstanceIndex: index, Fingerprint: labels[fingerprintLabel],
	}, nil
}

func isNotFound(err error) bool {
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "no such object") || strings.Contains(message, "no such container") ||
		strings.Contains(message, "no such image")
}

const (
	// readTimeout bounds inspect/ps calls that should answer immediately.
	readTimeout = 30 * time.Second
	// mutateTimeout bounds create/start/stop when the caller set no deadline.
	mutateTimeout = 10 * time.Minute
	// defaultPullTimeout bounds a pull when the caller set no deadline.
	defaultPullTimeout = time.Hour
)

// withDefaultTimeout applies limit unless the caller already set an earlier
// deadline, so every Docker exec is bounded even when called with Background.
func withDefaultTimeout(ctx context.Context, limit time.Duration) (context.Context, context.CancelFunc) {
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= limit {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, limit)
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
	return d.run(ctx, nil, nil, args...)
}

// run executes the Docker CLI without a shell. Extra environment entries are
// appended to Fleet's own environment (later duplicates win). Failures report
// only sanitized stderr, never argv, with manifest secrets redacted.
func (d *CLIDriver) run(ctx context.Context, env []string, secrets []string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, d.binary, args...)
	if len(env) != 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &limitedWriter{buffer: &stderr, limit: 64 << 10}
	if err := cmd.Run(); err != nil {
		message := strings.TrimSpace(stderr.String())
		if message == "" {
			message = err.Error()
			if ctx.Err() != nil {
				message = ctx.Err().Error()
			}
		}
		return stdout.String(), &commandError{subcommand: args[0], message: deployment.SanitizeMessage(message, secrets)}
	}
	return stdout.String(), nil
}

type commandError struct {
	subcommand string
	message    string
}

func (e *commandError) Error() string { return "docker " + e.subcommand + ": " + e.message }

// limitedWriter keeps the first limit bytes so a runaway stderr stream cannot
// grow Fleet's memory without bound.
type limitedWriter struct {
	buffer *bytes.Buffer
	limit  int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if remaining := w.limit - w.buffer.Len(); remaining > 0 {
		if len(p) > remaining {
			w.buffer.Write(p[:remaining])
		} else {
			w.buffer.Write(p)
		}
	}
	return len(p), nil
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
	inspectCtx, cancel := withDefaultTimeout(ctx, readTimeout)
	defer cancel()
	if id, err := d.output(inspectCtx, "image", "inspect", "--format", "{{.Id}}", model.Image); err == nil {
		input.ImageID = strings.TrimSpace(id)
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("fingerprint model: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
