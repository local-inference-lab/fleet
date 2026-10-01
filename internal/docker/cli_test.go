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
		Mounts:          []manifest.Mount{{Source: "/host/models", Target: "/models", ReadOnly: true}},
		Ports:           []manifest.Port{{HostIP: "127.0.0.1", HostPort: 8000, ContainerPort: 8000, Protocol: "tcp"}},
		GPUs:            "all",
		MemoryLimit:     "40g",
		MemorySwapLimit: "40g",
		Restart:         "unless-stopped",
		Init:            true,
		Privileged:      true,
		StopTimeout:     "2m0s",
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
		"--label", "ai.local-inference-lab.lil-fleet.model=safe-model",
		"--label", "ai.local-inference-lab.lil-fleet.instance=safe-model",
		"--label", "ai.local-inference-lab.lil-fleet.instance_index=1",
		"--label", "ai.local-inference-lab.lil-fleet.port=0",
		"--mount", "type=bind,source=/host/models,target=/models,readonly",
		"--publish", "127.0.0.1:8000:8000/tcp", "example/image@sha256:abc",
		"--gpus", "\"device=0,1\"",
		"--memory", "40g", "--memory-swap", "40g",
		"--init", "--privileged", "--stop-timeout", "120",
		"--restart", "unless-stopped",
		"model with spaces", "; touch /tmp/not-executed", "start",
	} {
		if !contains(args, want) {
			t.Fatalf("Docker args missing %q:\n%s", want, raw)
		}
	}
}

func TestStartLabelsReplicaInstanceAndUsesReplicaName(t *testing.T) {
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

	model := manifest.Model{
		ID: "alpha--2", Image: "example/image",
		InstanceProfileID: "alpha", InstanceID: "alpha--2", InstanceIndex: 2, InstancePort: 9002,
		Command: []string{"serve", "--port", "9002"},
		Ports:   []manifest.Port{{HostIP: "127.0.0.1", HostPort: 9002, ContainerPort: 9002, Protocol: "tcp"}},
	}
	if err := NewCLIDriver(scriptPath).Start(context.Background(), model, []int{1}); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read fake Docker args: %v", err)
	}
	args := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for _, want := range []string{
		"--name", "lil-fleet-alpha--2",
		"--label", "ai.local-inference-lab.lil-fleet.model=alpha",
		"--label", "ai.local-inference-lab.lil-fleet.instance=alpha--2",
		"--label", "ai.local-inference-lab.lil-fleet.instance_index=2",
		"--label", "ai.local-inference-lab.lil-fleet.port=9002",
		"--publish", "127.0.0.1:9002:9002/tcp",
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

func TestHardeningArgsDropPrivilegesUnlessProfileIsPrivileged(t *testing.T) {
	got := hardeningArgs(manifest.Model{CapAdd: []string{"DAC_OVERRIDE"}, User: "1000:1000", ReadOnly: true})
	want := []string{
		"--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
		"--security-opt", "no-new-privileges=true",
		"--user", "1000:1000", "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev",
	}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("hardening args = %q, want %q", got, want)
	}
	if got := hardeningArgs(manifest.Model{SecurityOpt: []string{"no-new-privileges=false"}}); contains(got, "no-new-privileges=true") {
		t.Fatalf("explicit no-new-privileges override was replaced: %q", got)
	}
	if got := hardeningArgs(manifest.Model{Privileged: true}); len(got) != 0 {
		t.Fatalf("privileged hardening args = %q, want none", got)
	}
}

func TestStartBindsPerModelCacheSubdirectory(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "args.log")
	scriptPath := writeFakeDocker(t, dir)
	t.Setenv("FAKE_DOCKER_LOG", logPath)
	t.Setenv("FAKE_DOCKER_STATE", filepath.Join(dir, "exists"))
	cache := filepath.Join(dir, "cache")
	if err := os.Mkdir(cache, 0o700); err != nil {
		t.Fatal(err)
	}

	model := manifest.Model{
		ID: "alpha--2", Image: "example/image", InstanceProfileID: "alpha", InstanceID: "alpha--2", InstanceIndex: 2,
		Mounts: []manifest.Mount{{Source: cache, Target: "/cache", PerModel: true}},
	}
	if err := NewCLIDriver(scriptPath).Start(context.Background(), model, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(strings.TrimSpace(string(raw)), "\n")
	want := "type=bind,source=" + filepath.Join(cache, "alpha") + ",target=/cache"
	if !contains(args, want) || !contains(args, "--cap-drop") || !contains(args, "no-new-privileges=true") {
		t.Fatalf("Docker args missing per-model mount or hardening:\n%s", raw)
	}
	if info, err := os.Stat(filepath.Join(cache, "alpha")); err != nil || !info.IsDir() || info.Mode().Perm() != 0o700 {
		t.Fatalf("per-model cache directory = %v, %v", info, err)
	}
}

func TestFingerprintTracksSeccompContentsAndImageID(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "seccomp.json")
	if err := os.WriteFile(profile, []byte(`{"defaultAction":"SCMP_ACT_ERRNO"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	imageID := filepath.Join(dir, "image-id")
	if err := os.WriteFile(imageID, []byte("sha256:one"), 0o600); err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(dir, "docker")
	script := "#!/bin/sh\nif [ \"$1\" = image ]; then cat \"$FAKE_IMAGE_ID\"; fi\n"
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAKE_IMAGE_ID", imageID)
	driver := NewCLIDriver(scriptPath)
	model := manifest.Model{ID: "a", Image: "example/image", SecurityOpt: []string{"seccomp=" + profile}}

	first, err := driver.fingerprint(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(profile, []byte(`{"defaultAction":"SCMP_ACT_ALLOW"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	second, err := driver.fingerprint(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(imageID, []byte("sha256:two"), 0o600); err != nil {
		t.Fatal(err)
	}
	third, err := driver.fingerprint(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	if first == second || second == third {
		t.Fatalf("fingerprints did not change: %s %s %s", first, second, third)
	}
}

func writeFakeDocker(t *testing.T, dir string) string {
	t.Helper()
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
	return scriptPath
}
