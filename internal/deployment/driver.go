package deployment

import (
	"context"
	"errors"

	"github.com/local-inference-lab/fleet/internal/manifest"
)

var ErrNotFound = errors.New("deployment not found")

type ContainerState struct {
	Exists       bool
	Running      bool
	Status       string
	Health       string
	ExitCode     int
	OOMKilled    bool
	Error        string
	AssignedGPUs []int
}

type Driver interface {
	Start(context.Context, manifest.Model, []int) error
	Stop(context.Context, manifest.Model, bool) error
	Inspect(context.Context, manifest.Model) (ContainerState, error)
}
