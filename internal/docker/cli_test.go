package docker

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local-inference-lab/fleet/internal/manifest"
)

func TestStartPassesManifestValuesAsLiteralArguments(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "args.log")
	statePath := filepath.Join(dir, "exists")
	scriptPath := filepath.Join(dir, "docker")
	script := `#!/bin/sh
if [ "$1" = "inspect" ] && [ ! -f "$FAKE_DOCKER_STATE" ]; then
  echo "error: no such object" >&2
  exit 1
fi
for arg in "$@"; do
  printf '%s\n' "$arg" >> "$FAKE_DOCKER_LOG"
done
if [ "$1" = "create" ]; then
  : > "$FAKE_DOCKER_STATE"
fi
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake docker: %v", err)
	}
	t.Setenv("FAKE_DOCKER_LOG", logPath)
	t.Setenv("FAKE_DOCKER_STATE", statePath)

	driver := NewCLIDriver(scriptPath)
	model := manifest.Model{
		ID:      "safe-model",
		Image:   "example/image@sha256:abc",
		Command: []string{"serve", "model with spaces", "; touch /tmp/not-executed"},
		Environment: map[string]string{
			"TOKEN": "value with spaces; still literal",
		},
		Mounts:  []manifest.Mount{{Source: "/host/models", Target: "/models", ReadOnly: true}},
		Ports:   []manifest.Port{{HostIP: "127.0.0.1", HostPort: 8000, ContainerPort: 8000, Protocol: "tcp"}},
		GPUs:    "all",
		Restart: "unless-stopped",
	}
	if err := driver.Start(context.Background(), model, []int{0, 1}); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read fake Docker args: %v", err)
	}
	args := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for _, want := range []string{
		"create", "--name", "lil-fleet-safe-model", "--env", "TOKEN=value with spaces; still literal",
		"--mount", "type=bind,source=/host/models,target=/models,readonly",
		"--publish", "127.0.0.1:8000:8000/tcp", "example/image@sha256:abc",
		"--gpus", "\"device=0,1\"",
		"--restart", "unless-stopped",
		"model with spaces", "; touch /tmp/not-executed", "start",
	} {
		if !contains(args, want) {
			t.Fatalf("Docker args missing %q:\n%s", want, raw)
		}
	}
}

func TestStopRefusesUnmanagedContainer(t *testing.T) {
	dir := t.TempDir()
	scriptPath := filepath.Join(dir, "docker")
	script := `#!/bin/sh
case "$3" in
  *json*) printf '%s\n' '{"Status":"running","Running":true,"ExitCode":0}' ;;
  *managed*) printf '%s\n' 'false' ;;
esac
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake docker: %v", err)
	}
	err := NewCLIDriver(scriptPath).Stop(context.Background(), manifest.Model{ID: "safe-model"}, true)
	if err == nil || !strings.Contains(err.Error(), "unmanaged") {
		t.Fatalf("Stop() error = %v, want unmanaged-container refusal", err)
	}
}

func TestDockerErrorsDoNotEchoCommandArguments(t *testing.T) {
	dir := t.TempDir()
	scriptPath := filepath.Join(dir, "docker")
	if err := os.WriteFile(scriptPath, []byte("#!/bin/sh\necho failed >&2\nexit 1\n"), 0o700); err != nil {
		t.Fatalf("write fake docker: %v", err)
	}
	_, err := NewCLIDriver(scriptPath).output(context.Background(), "create", "--env", "TOKEN=top-secret")
	if err == nil {
		t.Fatal("output() unexpectedly succeeded")
	}
	if strings.Contains(err.Error(), "top-secret") {
		t.Fatalf("error leaked argument: %v", err)
	}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
