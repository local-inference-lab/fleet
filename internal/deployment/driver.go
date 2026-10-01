package deployment

import (
	"context"
	"errors"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

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
	Port         int
}

type Driver interface {
	Start(context.Context, manifest.Model, []int) error
	Stop(context.Context, manifest.Model, bool) error
	Inspect(context.Context, manifest.Model) (ContainerState, error)
}

// MaxMessageBytes caps operator-visible error text taken from Docker, the
// container runtime, or readiness probes.
const MaxMessageBytes = 1024

// SanitizeMessage makes untrusted diagnostic text safe to store in operation
// and status errors: manifest secrets are redacted, control characters are
// replaced (so text cannot forge log lines or terminal escapes), and the result
// is capped at MaxMessageBytes on a UTF-8 boundary.
func SanitizeMessage(message string, secrets []string) string {
	ordered := make([]string, 0, len(secrets))
	for _, secret := range secrets {
		// Very short values ("0", "1", "on") are configuration flags, not
		// secrets; redacting them would only shred the message.
		if len(secret) >= 4 {
			ordered = append(ordered, secret)
		}
	}
	sort.Slice(ordered, func(i, j int) bool { return len(ordered[i]) > len(ordered[j]) })
	for _, secret := range ordered {
		message = strings.ReplaceAll(message, secret, "[redacted]")
	}
	message = strings.Map(func(r rune) rune {
		if r == utf8.RuneError || unicode.IsControl(r) {
			return ' '
		}
		return r
	}, message)
	message = strings.Join(strings.Fields(message), " ")
	if len(message) <= MaxMessageBytes {
		return message
	}
	cut := MaxMessageBytes - len("...")
	for cut > 0 && !utf8.RuneStart(message[cut]) {
		cut--
	}
	return message[:cut] + "..."
}

// EnvironmentValues returns a model's environment values for redaction.
func EnvironmentValues(model manifest.Model) []string {
	values := make([]string, 0, len(model.Environment))
	for _, value := range model.Environment {
		values = append(values, value)
	}
	return values
}
