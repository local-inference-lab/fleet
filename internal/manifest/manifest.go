package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	validID            = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)
	reservedReplicaID  = regexp.MustCompile(`--([2-9]|[1-5][0-9]|6[0-4])$`)
	validRestartPolicy = regexp.MustCompile(`^(no|always|unless-stopped|on-failure(:[1-9][0-9]*)?)$`)
	validMemoryLimit   = regexp.MustCompile(`^[1-9][0-9]*[bBkKmMgGtTpP]?$`)
)

type Manifest struct {
	Version int           `json:"version"`
	API     APIConfig     `json:"api"`
	Runtime RuntimeConfig `json:"runtime"`
	Models  []Model       `json:"models"`
}

type APIConfig struct {
	Listen string `json:"listen"`
}

type RuntimeConfig struct {
	DockerBinary          string      `json:"docker_binary"`
	PollInterval          string      `json:"poll_interval"`
	OperationTimeout      string      `json:"operation_timeout"`
	ReadinessTimeout      string      `json:"readiness_timeout"`
	RemoveOnUnload        bool        `json:"remove_on_unload"`
	ConcurrentDeployments bool        `json:"concurrent_deployments,omitempty"`
	GPUTopology           GPUTopology `json:"gpu_topology,omitempty"`
	ModelPortRange        *PortRange  `json:"model_port_range,omitempty"`
	pollInterval          time.Duration
	operationTimeout      time.Duration
	readinessTimeout      time.Duration
}

type PortRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

type GPUTopology struct {
	Groups           [][]int `json:"groups,omitempty"`
	MaxUsedMemoryMiB int     `json:"max_used_memory_mib,omitempty"`
}

type Model struct {
	ID              string            `json:"id"`
	Description     string            `json:"description,omitempty"`
	Image           string            `json:"image"`
	Command         []string          `json:"command,omitempty"`
	Environment     map[string]string `json:"environment,omitempty"`
	Mounts          []Mount           `json:"mounts,omitempty"`
	Ports           []Port            `json:"ports,omitempty"`
	GPUs            string            `json:"gpus,omitempty"`
	Placement       *Placement        `json:"placement,omitempty"`
	ShmSize         string            `json:"shm_size,omitempty"`
	IPC             string            `json:"ipc,omitempty"`
	Entrypoint      string            `json:"entrypoint,omitempty"`
	NetworkMode     string            `json:"network_mode,omitempty"`
	MemoryLimit     string            `json:"memory_limit,omitempty"`
	MemorySwapLimit string            `json:"memory_swap_limit,omitempty"`
	SecurityOpt     []string          `json:"security_opt,omitempty"`
	Ulimits         []string          `json:"ulimits,omitempty"`
	Restart         string            `json:"restart,omitempty"`
	Init            bool              `json:"init,omitempty"`
	Privileged      bool              `json:"privileged,omitempty"`
	StopTimeout     string            `json:"stop_timeout,omitempty"`
	Readiness       *Readiness        `json:"readiness,omitempty"`
	stopTimeout     time.Duration

	InstanceProfileID string `json:"-"`
	InstanceID        string `json:"-"`
	InstanceIndex     int    `json:"-"`
	InstancePort      int    `json:"-"`
}

type Placement struct {
	GPUCount int    `json:"gpu_count"`
	Strategy string `json:"strategy,omitempty"`
}

type Mount struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"read_only,omitempty"`
}

type Port struct {
	HostIP        string `json:"host_ip,omitempty"`
	HostPort      int    `json:"host_port"`
	ContainerPort int    `json:"container_port"`
	Protocol      string `json:"protocol,omitempty"`
}

type Readiness struct {
	URL           string `json:"url"`
	SuccessStatus int    `json:"success_status,omitempty"`
}

func Load(path string) (*Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat manifest: %w", err)
	}
	if info.Size() > 4<<20 {
		return nil, errors.New("manifest exceeds 4 MiB limit")
	}

	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	var m Manifest
	if err := decoder.Decode(&m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	if err := ensureEOF(decoder); err != nil {
		return nil, err
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	if err := m.resolveSeccompProfiles(filepath.Dir(path)); err != nil {
		return nil, err
	}
	return &m, nil
}

// resolveSeccompProfiles anchors relative seccomp profile paths to the manifest
// directory. The Docker CLI reads the profile client-side relative to its own
// working directory, which differs between a source checkout, a systemd unit,
// and the controller container, so a relative path would silently depend on
// how Fleet was launched.
func (m *Manifest) resolveSeccompProfiles(manifestDir string) error {
	base, err := filepath.Abs(manifestDir)
	if err != nil {
		return fmt.Errorf("resolve manifest directory: %w", err)
	}
	for i := range m.Models {
		model := &m.Models[i]
		for j, option := range model.SecurityOpt {
			profile, ok := SeccompProfilePath(option)
			if !ok {
				continue
			}
			if !filepath.IsAbs(profile) {
				profile = filepath.Join(base, profile)
			}
			info, err := os.Stat(profile)
			if err != nil {
				return fmt.Errorf("model %q: seccomp profile: %w", model.ID, err)
			}
			if !info.Mode().IsRegular() {
				return fmt.Errorf("model %q: seccomp profile %q is not a regular file", model.ID, profile)
			}
			model.SecurityOpt[j] = "seccomp=" + profile
		}
	}
	return nil
}

// SeccompProfilePath returns the file referenced by a seccomp security option,
// or false for non-seccomp options and the built-in "unconfined" profile.
func SeccompProfilePath(option string) (string, bool) {
	name, value, ok := strings.Cut(option, "=")
	if !ok {
		name, value, ok = strings.Cut(option, ":")
	}
	if !ok || name != "seccomp" || value == "" || value == "unconfined" || value == "builtin" {
		return "", false
	}
	return value, true
}

func ensureEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("manifest must contain exactly one JSON document")
		}
		return fmt.Errorf("decode trailing manifest content: %w", err)
	}
	return nil
}

func (m *Manifest) validate() error {
	if m.Version != 1 {
		return fmt.Errorf("unsupported manifest version %d (want 1)", m.Version)
	}
	if m.API.Listen == "" {
		m.API.Listen = "127.0.0.1:8090"
	}
	if m.Runtime.DockerBinary == "" {
		m.Runtime.DockerBinary = "docker"
	}
	var err error
	if m.Runtime.pollInterval, err = durationOrDefault(m.Runtime.PollInterval, 2*time.Second); err != nil {
		return fmt.Errorf("runtime.poll_interval: %w", err)
	}
	if m.Runtime.operationTimeout, err = durationOrDefault(m.Runtime.OperationTimeout, 30*time.Second); err != nil {
		return fmt.Errorf("runtime.operation_timeout: %w", err)
	}
	if m.Runtime.readinessTimeout, err = durationOrDefault(m.Runtime.ReadinessTimeout, 10*time.Minute); err != nil {
		return fmt.Errorf("runtime.readiness_timeout: %w", err)
	}
	if len(m.Models) == 0 {
		return errors.New("manifest must define at least one model")
	}
	if m.Runtime.GPUTopology.MaxUsedMemoryMiB == 0 {
		m.Runtime.GPUTopology.MaxUsedMemoryMiB = 1024
	}
	if m.Runtime.GPUTopology.MaxUsedMemoryMiB < 0 {
		return errors.New("runtime.gpu_topology.max_used_memory_mib must not be negative")
	}
	if portRange := m.Runtime.ModelPortRange; portRange != nil {
		if portRange.Start < 1 || portRange.End > 65535 || portRange.Start > portRange.End {
			return errors.New("runtime.model_port_range must be a valid inclusive TCP port range")
		}
	}
	seenGPU := map[int]bool{}
	for i, group := range m.Runtime.GPUTopology.Groups {
		if len(group) == 0 {
			return fmt.Errorf("runtime.gpu_topology.groups[%d] must not be empty", i)
		}
		for _, index := range group {
			if index < 0 || seenGPU[index] {
				return fmt.Errorf("runtime.gpu_topology.groups contains invalid or duplicate GPU %d", index)
			}
			seenGPU[index] = true
		}
	}
	seen := make(map[string]struct{}, len(m.Models))
	for i := range m.Models {
		model := &m.Models[i]
		if !validID.MatchString(model.ID) {
			return fmt.Errorf("models[%d].id %q must match %s", i, model.ID, validID)
		}
		if reservedReplicaID.MatchString(model.ID) {
			return fmt.Errorf("models[%d].id %q uses the reserved replica suffix namespace", i, model.ID)
		}
		if _, ok := seen[model.ID]; ok {
			return fmt.Errorf("duplicate model id %q", model.ID)
		}
		seen[model.ID] = struct{}{}
		if strings.TrimSpace(model.Image) == "" {
			return fmt.Errorf("model %q: image is required", model.ID)
		}
		// The image is the first positional docker-create argument; a leading
		// dash would be parsed as another Docker option.
		if strings.HasPrefix(model.Image, "-") || strings.ContainsFunc(model.Image, unsafeArgumentRune) {
			return fmt.Errorf("model %q: image must not start with '-' or contain whitespace or control characters", model.ID)
		}
		if strings.HasPrefix(model.Entrypoint, "-") {
			return fmt.Errorf("model %q: entrypoint must not start with '-'", model.ID)
		}
		if model.Restart != "" && !validRestartPolicy.MatchString(model.Restart) {
			return fmt.Errorf("model %q: invalid restart policy %q", model.ID, model.Restart)
		}
		if err := validateMemoryLimits(model); err != nil {
			return err
		}
		if model.StopTimeout != "" {
			model.stopTimeout, err = parseStopTimeout(model.StopTimeout)
			if err != nil {
				return fmt.Errorf("model %q: stop_timeout: %w", model.ID, err)
			}
		}
		if model.Placement != nil {
			if !m.Runtime.ConcurrentDeployments {
				return fmt.Errorf("model %q: placement requires runtime.concurrent_deployments", model.ID)
			}
			if len(m.Runtime.GPUTopology.Groups) == 0 {
				return fmt.Errorf("model %q: placement requires runtime.gpu_topology.groups", model.ID)
			}
			switch model.Placement.GPUCount {
			case 1, 2, 3, 4, 6, 8:
			default:
				return fmt.Errorf("model %q: placement.gpu_count must be one of 1, 2, 3, 4, 6, or 8", model.ID)
			}
			if model.Placement.GPUCount > len(seenGPU) {
				return fmt.Errorf("model %q: placement.gpu_count is outside configured topology", model.ID)
			}
			if model.Placement.Strategy == "" {
				model.Placement.Strategy = "pcie_affinity"
			}
			if model.Placement.Strategy != "pcie_affinity" {
				return fmt.Errorf("model %q: unsupported placement strategy %q", model.ID, model.Placement.Strategy)
			}
			if model.GPUs != "" {
				return fmt.Errorf("model %q: gpus and placement are mutually exclusive", model.ID)
			}
		}
		for j, mount := range model.Mounts {
			if !strings.HasPrefix(mount.Source, "/") || !strings.HasPrefix(mount.Target, "/") {
				return fmt.Errorf("model %q mount %d: source and target must be absolute", model.ID, j)
			}
			// --mount is a comma-separated key=value list; a comma in a path
			// would inject extra mount options such as a writable bind.
			if strings.ContainsAny(mount.Source+mount.Target, ",\"") || strings.ContainsFunc(mount.Source+mount.Target, unsafeArgumentRune) {
				return fmt.Errorf("model %q mount %d: source and target must not contain commas, quotes, or control characters", model.ID, j)
			}
		}
		for j := range model.Ports {
			port := &model.Ports[j]
			if port.HostPort < 1 || port.HostPort > 65535 || port.ContainerPort < 1 || port.ContainerPort > 65535 {
				return fmt.Errorf("model %q port %d: ports must be between 1 and 65535", model.ID, j)
			}
			if port.HostIP == "" {
				port.HostIP = "127.0.0.1"
			}
			if net.ParseIP(port.HostIP) == nil {
				return fmt.Errorf("model %q port %d: host_ip must be an IP address", model.ID, j)
			}
			if port.Protocol == "" {
				port.Protocol = "tcp"
			}
			if port.Protocol != "tcp" && port.Protocol != "udp" {
				return fmt.Errorf("model %q port %d: protocol must be tcp or udp", model.ID, j)
			}
			if m.Runtime.ModelPortRange != nil && !m.Runtime.ModelPortRange.Contains(port.HostPort) {
				return fmt.Errorf("model %q port %d: host port %d is outside runtime.model_port_range", model.ID, j, port.HostPort)
			}
		}
		if model.NetworkMode == "host" {
			if err := validateHostNetworkBind(*model); err != nil {
				return fmt.Errorf("model %q: %w", model.ID, err)
			}
		}
		if model.Readiness != nil {
			if model.Readiness.URL == "" {
				return fmt.Errorf("model %q: readiness.url is required", model.ID)
			}
			if model.Readiness.SuccessStatus == 0 {
				model.Readiness.SuccessStatus = 200
			}
		}
		if m.Runtime.ModelPortRange != nil {
			servePort, err := commandPort(model.Command)
			if err != nil {
				return fmt.Errorf("model %q: %w", model.ID, err)
			}
			if !m.Runtime.ModelPortRange.Contains(servePort) {
				return fmt.Errorf("model %q: command port %d is outside runtime.model_port_range", model.ID, servePort)
			}
			if model.Readiness == nil {
				return fmt.Errorf("model %q: readiness is required when runtime.model_port_range is configured", model.ID)
			}
			if err := validateReadinessURL(model.ID, model.Readiness.URL, servePort, *m.Runtime.ModelPortRange); err != nil {
				return err
			}
		}
	}
	sort.Slice(m.Models, func(i, j int) bool { return m.Models[i].ID < m.Models[j].ID })
	return nil
}

func unsafeArgumentRune(r rune) bool {
	return r < 0x20 || r == 0x7f || r == ' ' || r == '\t'
}

func validateMemoryLimits(model *Model) error {
	memoryBytes := int64(0)
	if model.MemoryLimit != "" {
		var err error
		memoryBytes, err = parseDockerMemoryQuantity(model.MemoryLimit)
		if err != nil {
			return fmt.Errorf("model %q: memory_limit: %w", model.ID, err)
		}
	}
	if model.MemorySwapLimit == "" {
		return nil
	}
	if model.MemoryLimit == "" {
		return fmt.Errorf("model %q: memory_swap_limit requires memory_limit", model.ID)
	}
	if model.MemorySwapLimit == "-1" {
		return nil
	}
	swapBytes, err := parseDockerMemoryQuantity(model.MemorySwapLimit)
	if err != nil {
		return fmt.Errorf("model %q: memory_swap_limit: %w", model.ID, err)
	}
	if swapBytes < memoryBytes {
		return fmt.Errorf("model %q: memory_swap_limit must be -1 or greater than or equal to memory_limit", model.ID)
	}
	return nil
}

func parseDockerMemoryQuantity(value string) (int64, error) {
	if !validMemoryLimit.MatchString(value) {
		return 0, errors.New("must be a positive Docker memory quantity")
	}
	suffix := value[len(value)-1]
	number := value
	multiplier := int64(1)
	switch suffix {
	case 'b', 'B':
		number = value[:len(value)-1]
	case 'k', 'K':
		number = value[:len(value)-1]
		multiplier = 1024
	case 'm', 'M':
		number = value[:len(value)-1]
		multiplier = 1024 * 1024
	case 'g', 'G':
		number = value[:len(value)-1]
		multiplier = 1024 * 1024 * 1024
	case 't', 'T':
		number = value[:len(value)-1]
		multiplier = 1024 * 1024 * 1024 * 1024
	case 'p', 'P':
		number = value[:len(value)-1]
		multiplier = 1024 * 1024 * 1024 * 1024 * 1024
	}
	parsed, err := strconv.ParseInt(number, 10, 64)
	if err != nil || parsed <= 0 || parsed > (1<<63-1)/multiplier {
		return 0, errors.New("must fit in signed 64-bit bytes")
	}
	return parsed * multiplier, nil
}

func (r PortRange) Contains(port int) bool {
	return port >= r.Start && port <= r.End
}

func commandPort(command []string) (int, error) {
	port := 0
	for index := 0; index < len(command); index++ {
		value := ""
		switch {
		case command[index] == "--port":
			if index+1 >= len(command) {
				return 0, errors.New("--port requires a value")
			}
			index++
			value = command[index]
		case strings.HasPrefix(command[index], "--port="):
			value = strings.TrimPrefix(command[index], "--port=")
		default:
			continue
		}
		if port != 0 {
			return 0, errors.New("command must contain exactly one --port")
		}
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 1 || parsed > 65535 {
			return 0, fmt.Errorf("invalid command port %q", value)
		}
		port = parsed
	}
	if port == 0 {
		return 0, errors.New("command must contain exactly one --port when runtime.model_port_range is configured")
	}
	return port, nil
}

// validateHostNetworkBind keeps host-networked inference servers off public
// interfaces. Host networking has no publish step, so a server bound to 0.0.0.0
// or :: is reachable, unauthenticated, from every network the host joins. vLLM,
// lil-serve, and the GLM serve scripts all default to 0.0.0.0, so an absent bind
// is treated as public too.
func validateHostNetworkBind(model Model) error {
	explicit := false
	for index := 0; index < len(model.Command); index++ {
		value := ""
		switch argument := model.Command[index]; {
		case argument == "--host":
			if index+1 >= len(model.Command) {
				return errors.New("--host requires a value")
			}
			index++
			value = model.Command[index]
		case strings.HasPrefix(argument, "--host="):
			value = strings.TrimPrefix(argument, "--host=")
		default:
			continue
		}
		explicit = true
		if !isLoopbackHost(value) {
			return fmt.Errorf("network_mode \"host\" would expose --host %q on every host interface; bind to 127.0.0.1 or ::1", value)
		}
	}
	if value, ok := model.Environment["HOST"]; ok {
		explicit = true
		if !isLoopbackHost(value) {
			return fmt.Errorf("network_mode \"host\" would expose HOST=%q on every host interface; bind to 127.0.0.1 or ::1", value)
		}
	}
	if !explicit {
		return errors.New("network_mode \"host\" requires an explicit loopback bind (--host 127.0.0.1 or environment HOST=127.0.0.1) because model servers default to all interfaces")
	}
	return nil
}

func isLoopbackHost(value string) bool {
	host := strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(value), "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func validateReadinessURL(modelID, raw string, servePort int, portRange PortRange) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "http" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("model %q: readiness.url must be a plain loopback HTTP URL", modelID)
	}
	host := parsed.Hostname()
	ip := net.ParseIP(host)
	if !strings.EqualFold(host, "localhost") && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("model %q: readiness.url must use a loopback host", modelID)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil || !portRange.Contains(port) {
		return fmt.Errorf("model %q: readiness.url port must be inside runtime.model_port_range", modelID)
	}
	if port != servePort {
		return fmt.Errorf("model %q: readiness.url port %d does not match command port %d", modelID, port, servePort)
	}
	return nil
}

func durationOrDefault(value string, fallback time.Duration) (time.Duration, error) {
	if value == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(value)
	if err != nil {
		return 0, err
	}
	if d <= 0 {
		return 0, errors.New("must be greater than zero")
	}
	return d, nil
}

func parseStopTimeout(value string) (time.Duration, error) {
	d, err := time.ParseDuration(value)
	if err != nil {
		return 0, err
	}
	if d < time.Second {
		return 0, errors.New("must be at least 1s")
	}
	if d%time.Second != 0 {
		return 0, errors.New("must be a whole number of seconds")
	}
	return d, nil
}

func (r RuntimeConfig) PollDuration() time.Duration      { return r.pollInterval }
func (r RuntimeConfig) OperationDuration() time.Duration { return r.operationTimeout }
func (r RuntimeConfig) ReadinessDuration() time.Duration { return r.readinessTimeout }

func (m Model) StopTimeoutSeconds() int {
	d := m.stopTimeout
	if d == 0 && m.StopTimeout != "" {
		var err error
		d, err = parseStopTimeout(m.StopTimeout)
		if err != nil {
			return 0
		}
	}
	return int(d / time.Second)
}

func (m *Manifest) Model(id string) (Model, bool) {
	for _, model := range m.Models {
		if model.ID == id {
			return model, true
		}
	}
	return Model{}, false
}
