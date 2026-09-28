package fleet

import (
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"

	"github.com/local-inference-lab/fleet/internal/manifest"
)

const maxRequestedInstances = 64

type instancePlan struct {
	target int
	start  []plannedInstance
	stop   []plannedInstance
	keep   []InstanceStatus
}

type plannedInstance struct {
	model manifest.Model
	gpus  []int
	port  int
	index int
}

func instanceID(modelID string, index int) string {
	if index <= 1 {
		return modelID
	}
	return fmt.Sprintf("%s--%d", modelID, index)
}

func basePort(model manifest.Model) int {
	if model.InstancePort != 0 {
		return model.InstancePort
	}
	if port, ok := commandPort(model.Command); ok {
		return port
	}
	for _, published := range model.Ports {
		if published.HostPort != 0 {
			return published.HostPort
		}
	}
	return 0
}

func cloneInstanceModel(model manifest.Model, index, port int) manifest.Model {
	clone := model
	clone.ID = instanceID(model.ID, index)
	clone.InstanceProfileID = model.ID
	clone.InstanceID = clone.ID
	clone.InstanceIndex = index
	clone.InstancePort = port
	clone.Command = rewriteCommandPort(model.Command, port)
	clone.Ports = rewritePorts(model.Ports, basePort(model), port)
	clone.Readiness = rewriteReadiness(model.Readiness, port)
	return clone
}

func commandPort(command []string) (int, bool) {
	for i := 0; i < len(command); i++ {
		switch {
		case command[i] == "--port" && i+1 < len(command):
			port, err := strconv.Atoi(command[i+1])
			return port, err == nil
		case strings.HasPrefix(command[i], "--port="):
			port, err := strconv.Atoi(strings.TrimPrefix(command[i], "--port="))
			return port, err == nil
		}
	}
	return 0, false
}

func rewriteCommandPort(command []string, port int) []string {
	if port == 0 {
		return append([]string(nil), command...)
	}
	out := append([]string(nil), command...)
	for i := 0; i < len(out); i++ {
		switch {
		case out[i] == "--port" && i+1 < len(out):
			out[i+1] = strconv.Itoa(port)
			return out
		case strings.HasPrefix(out[i], "--port="):
			out[i] = "--port=" + strconv.Itoa(port)
			return out
		}
	}
	return out
}

func rewritePorts(ports []manifest.Port, oldPort, newPort int) []manifest.Port {
	out := append([]manifest.Port(nil), ports...)
	if newPort == 0 {
		return out
	}
	for i := range out {
		if out[i].HostPort == oldPort {
			out[i].HostPort = newPort
		}
		if out[i].ContainerPort == oldPort {
			out[i].ContainerPort = newPort
		}
	}
	return out
}

func rewriteReadiness(readiness *manifest.Readiness, port int) *manifest.Readiness {
	if readiness == nil || port == 0 {
		if readiness == nil {
			return nil
		}
		copy := *readiness
		return &copy
	}
	copy := *readiness
	parsed, err := url.Parse(copy.URL)
	if err == nil {
		parsed.Host = parsed.Hostname() + ":" + strconv.Itoa(port)
		copy.URL = parsed.String()
	}
	return &copy
}

func aggregateStatus(modelID string, desired string, instances []InstanceStatus, nowZero bool) Status {
	sort.Slice(instances, func(i, j int) bool { return instances[i].Index < instances[j].Index })
	status := Status{ModelID: modelID, Phase: PhaseUnloaded, Desired: desired, DesiredInstances: len(instances), Instances: instances}
	for _, instance := range instances {
		status.LastChecked = instance.LastChecked
		if instance.Phase == PhaseReady {
			status.ReadyInstances++
		}
		if instance.Index == 1 || status.Container == "" {
			status.Container = instance.Container
			status.Health = instance.Health
			status.ExitCode = instance.ExitCode
			status.OOMKilled = instance.OOMKilled
			status.LastError = instance.LastError
		}
		status.AssignedGPUs = append(status.AssignedGPUs, instance.AssignedGPUs...)
		switch instance.Phase {
		case PhaseFailed:
			status.Phase = PhaseFailed
		case PhaseLoading:
			if status.Phase != PhaseFailed {
				status.Phase = PhaseLoading
			}
		case PhaseStopping:
			if status.Phase != PhaseFailed && status.Phase != PhaseLoading {
				status.Phase = PhaseStopping
			}
		case PhaseUnhealthy:
			if status.Phase != PhaseFailed && status.Phase != PhaseLoading && status.Phase != PhaseStopping {
				status.Phase = PhaseUnhealthy
			}
		case PhaseReady:
			if status.Phase == PhaseUnloaded {
				status.Phase = PhaseReady
			}
		case PhaseUnknown:
			if status.Phase == PhaseUnloaded {
				status.Phase = PhaseUnknown
			}
		}
	}
	if len(instances) == 0 {
		status.DesiredInstances = 0
		if nowZero {
			status.ReadyInstances = 0
		}
	}
	return status
}

func statusActive(status Status) bool {
	switch status.Phase {
	case PhaseReady, PhaseLoading, PhaseUnhealthy, PhaseStopping:
		return true
	default:
		return false
	}
}

func statusHasLiveInstances(status Status) bool {
	for _, instance := range status.Instances {
		if instance.Phase != PhaseUnloaded {
			return true
		}
	}
	return false
}

func maxInstanceIndex(status Status) int {
	max := 0
	for _, instance := range status.Instances {
		if instance.Index > max {
			max = instance.Index
		}
	}
	return max
}

func mergePendingInstances(probed, current []InstanceStatus) []InstanceStatus {
	seen := map[int]bool{}
	out := append([]InstanceStatus(nil), probed...)
	for _, instance := range probed {
		seen[instance.Index] = true
	}
	for _, instance := range current {
		if seen[instance.Index] || instance.Phase != PhaseLoading {
			continue
		}
		out = append(out, instance)
	}
	return sortedInstances(out)
}

func sortedInstances(instances []InstanceStatus) []InstanceStatus {
	out := append([]InstanceStatus(nil), instances...)
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

func cloneStatus(status Status) Status {
	status.AssignedGPUs = append([]int(nil), status.AssignedGPUs...)
	status.Instances = cloneInstances(status.Instances)
	return status
}

func cloneInstances(instances []InstanceStatus) []InstanceStatus {
	out := append([]InstanceStatus(nil), instances...)
	for i := range out {
		out[i].AssignedGPUs = append([]int(nil), out[i].AssignedGPUs...)
	}
	return out
}
