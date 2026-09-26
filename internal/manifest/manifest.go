package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	validID            = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)
	validRestartPolicy = regexp.MustCompile(`^(no|always|unless-stopped|on-failure(:[1-9][0-9]*)?)$`)
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
	ID          string            `json:"id"`
	Description string            `json:"description,omitempty"`
	Image       string            `json:"image"`
	Command     []string          `json:"command,omitempty"`
	Environment map[string]string `json:"environment,omitempty"`
	Mounts      []Mount           `json:"mounts,omitempty"`
	Ports       []Port            `json:"ports,omitempty"`
	GPUs        string            `json:"gpus,omitempty"`
	Placement   *Placement        `json:"placement,omitempty"`
	ShmSize     string            `json:"shm_size,omitempty"`
	IPC         string            `json:"ipc,omitempty"`
	Entrypoint  string            `json:"entrypoint,omitempty"`
	NetworkMode string            `json:"network_mode,omitempty"`
	SecurityOpt []string          `json:"security_opt,omitempty"`
	Ulimits     []string          `json:"ulimits,omitempty"`
	Restart     string            `json:"restart,omitempty"`
	Init        bool              `json:"init,omitempty"`
	Privileged  bool              `json:"privileged,omitempty"`
	StopTimeout string            `json:"stop_timeout,omitempty"`
	Readiness   *Readiness        `json:"readiness,omitempty"`
	stopTimeout time.Duration
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
	return &m, nil
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
		if _, ok := seen[model.ID]; ok {
			return fmt.Errorf("duplicate model id %q", model.ID)
		}
		seen[model.ID] = struct{}{}
		if strings.TrimSpace(model.Image) == "" {
			return fmt.Errorf("model %q: image is required", model.ID)
		}
		if model.Restart != "" && !validRestartPolicy.MatchString(model.Restart) {
			return fmt.Errorf("model %q: invalid restart policy %q", model.ID, model.Restart)
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
		}
		for j := range model.Ports {
			port := &model.Ports[j]
			if port.HostPort < 1 || port.HostPort > 65535 || port.ContainerPort < 1 || port.ContainerPort > 65535 {
				return fmt.Errorf("model %q port %d: ports must be between 1 and 65535", model.ID, j)
			}
			if port.HostIP == "" {
				port.HostIP = "127.0.0.1"
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
