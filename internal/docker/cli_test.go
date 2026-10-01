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
  printf '%s' "$TOKEN" > "$FAKE_DOCKER_ENV"
fi
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake docker: %v", err)
	}
	envPath := filepath.Join(dir, "env.log")
	t.Setenv("FAKE_DOCKER_LOG", logPath)
	t.Setenv("FAKE_DOCKER_STATE", statePath)
	t.Setenv("FAKE_DOCKER_ENV", envPath)

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
		"create", "--name", "lil-fleet-safe-model", "--env", "TOKEN",
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
	if strings.Contains(string(raw), "value with spaces") {
		t.Fatalf("environment value leaked onto Docker argv:\n%s", raw)
	}
	if env, err := os.ReadFile(envPath); err != nil || string(env) != "value with spaces; still literal" {
		t.Fatalf("docker create environment TOKEN = %q, %v", env, err)
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

func TestEnvironmentArgsKeepDockerCLISettingsOnArgv(t *testing.T) {
	args, env := environmentArgs(map[string]string{"API_KEY": "secret", "DOCKER_HOST": "unix:///x", "HOME": "/root", "http_proxy": "http://p"})
	wantArgs := []string{"--env", "API_KEY", "--env", "DOCKER_HOST=unix:///x", "--env", "HOME=/root", "--env", "http_proxy=http://p"}
	if strings.Join(args, " ") != strings.Join(wantArgs, " ") || strings.Join(env, " ") != "API_KEY=secret" {
		t.Fatalf("environmentArgs = %q / %q", args, env)
	}
}

func TestDockerErrorsAreSanitizedAndRedacted(t *testing.T) {
	dir := t.TempDir()
	scriptPath := filepath.Join(dir, "docker")
	script := "#!/bin/sh\nprintf 'bad \\033[31mvalue hunter2-secret\\nforged line %0900d' 0 >&2\nprintf 'x%.0s' $(seq 1 2000) >&2\nexit 1\n"
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	_, err := NewCLIDriver(scriptPath).run(context.Background(), nil, []string{"hunter2-secret"}, "create")
	if err == nil {
		t.Fatal("run() unexpectedly succeeded")
	}
	message := err.Error()
	if strings.Contains(message, "hunter2-secret") || strings.ContainsAny(message, "\x1b\n") || !strings.Contains(message, "[redacted]") {
		t.Fatalf("unsanitized docker error: %q", message)
	}
	if len(message) > len("docker create: ")+1024 {
		t.Fatalf("docker error length = %d, want capped", len(message))
	}
}
