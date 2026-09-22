package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestLoadAppliesDefaultsAndSortsImmutableModelProfiles(t *testing.T) {
	cfg := loadManifest(t, `{
		"version": 1,
		"runtime": {
			"operation_timeout": "250ms",
			"readiness_timeout": "1s"
		},
		"models": [
			{
				"id": "qwen3",
				"description": "Qwen profile",
				"image": "ghcr.io/local-inference-lab/qwen:latest",
				"command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
				"environment": {"MAX_MODEL_LEN": "32768"},
				"mounts": [{"source": "/models/qwen", "target": "/models/qwen", "read_only": true}],
				"ports": [{"host_port": 8001, "container_port": 8000}],
				"gpus": "device=0",
				"shm_size": "16g",
				"ipc": "host",
				"readiness": {"url": "http://127.0.0.1:8001/health"}
			},
			{
				"id": "glm",
				"image": "ghcr.io/local-inference-lab/glm:latest"
			}
		]
	}`)

	if cfg.API.Listen != "127.0.0.1:8090" {
		t.Fatalf("default listen = %q", cfg.API.Listen)
	}
	if cfg.Runtime.DockerBinary != "docker" {
		t.Fatalf("default docker binary = %q", cfg.Runtime.DockerBinary)
	}
	if cfg.Runtime.PollDuration() != 2*time.Second {
		t.Fatalf("default poll interval = %s", cfg.Runtime.PollDuration())
	}
	if cfg.Runtime.OperationDuration() != 250*time.Millisecond {
		t.Fatalf("operation timeout = %s", cfg.Runtime.OperationDuration())
	}
	if cfg.Runtime.ReadinessDuration() != time.Second {
		t.Fatalf("readiness timeout = %s", cfg.Runtime.ReadinessDuration())
	}
	if got := []string{cfg.Models[0].ID, cfg.Models[1].ID}; got[0] != "glm" || got[1] != "qwen3" {
		t.Fatalf("models not sorted by id: %v", got)
	}
	qwen, ok := cfg.Model("qwen3")
	if !ok {
		t.Fatal("qwen3 model missing")
	}
	if qwen.Ports[0].HostIP != "127.0.0.1" || qwen.Ports[0].Protocol != "tcp" {
		t.Fatalf("port defaults not applied: %+v", qwen.Ports[0])
	}
	if qwen.Readiness.SuccessStatus != 200 {
		t.Fatalf("readiness success default = %d", qwen.Readiness.SuccessStatus)
	}
}

func TestLoadRejectsUnknownFieldsAndInvalidProfiles(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name: "unknown top-level override",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "image"}],
				"overrides": {"MAX_MODEL_LEN": "999"}
			}`,
			wantErr: "unknown field",
		},
		{
			name: "unknown model override",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "image", "native_args": ["--unsafe"]}]
			}`,
			wantErr: "unknown field",
		},
		{
			name: "duplicate ids",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one"}, {"id": "a", "image": "two"}]
			}`,
			wantErr: "duplicate model id",
		},
		{
			name: "relative mount",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "mounts": [{"source": "models/a", "target": "/models/a"}]}]
			}`,
			wantErr: "source and target must be absolute",
		},
		{
			name: "invalid port protocol",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "ports": [{"host_port": 8000, "container_port": 8000, "protocol": "sctp"}]}]
			}`,
			wantErr: "protocol must be tcp or udp",
		},
		{
			name: "invalid model port range",
			body: `{
				"version": 1,
				"runtime": {"model_port_range": {"start": 8108, "end": 8101}},
				"models": [{"id": "a", "image": "one"}]
			}`,
			wantErr: "model_port_range must be a valid",
		},
		{
			name: "command port outside model range",
			body: `{
				"version": 1,
				"runtime": {"model_port_range": {"start": 8101, "end": 8108}},
				"models": [{
					"id": "a", "image": "one", "command": ["serve", "--port", "9000"],
					"readiness": {"url": "http://127.0.0.1:9000/health"}
				}]
			}`,
			wantErr: "command port 9000 is outside",
		},
		{
			name: "readiness URL must remain on loopback",
			body: `{
				"version": 1,
				"runtime": {"model_port_range": {"start": 8101, "end": 8108}},
				"models": [{
					"id": "a", "image": "one", "command": ["serve", "--port=8101"],
					"readiness": {"url": "http://169.254.169.254:8101/health"}
				}]
			}`,
			wantErr: "readiness.url must use a loopback host",
		},
		{
			name: "readiness port must match command",
			body: `{
				"version": 1,
				"runtime": {"model_port_range": {"start": 8101, "end": 8108}},
				"models": [{
					"id": "a", "image": "one", "command": ["serve", "--port", "8101"],
					"readiness": {"url": "http://127.0.0.1:8102/health"}
				}]
			}`,
			wantErr: "does not match command port",
		},
		{
			name: "unsupported placement count",
			body: `{
				"version": 1,
				"runtime": {
					"concurrent_deployments": true,
					"gpu_topology": {"groups": [[0,1,6,7],[2,3,4,5]]}
				},
				"models": [{"id": "a", "image": "one", "placement": {"gpu_count": 5}}]
			}`,
			wantErr: "placement.gpu_count must be one of",
		},
		{
			name: "duplicate topology gpu",
			body: `{
				"version": 1,
				"runtime": {
					"concurrent_deployments": true,
					"gpu_topology": {"groups": [[0,1],[1,2]]}
				},
				"models": [{"id": "a", "image": "one"}]
			}`,
			wantErr: "duplicate GPU",
		},
		{
			name: "invalid restart policy",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "restart": "unless-stopped; reboot"}]
			}`,
			wantErr: "invalid restart policy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Load(writeManifest(t, tt.body))
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Load error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestB12XFleetManifestIsValid(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "fleet.b12x.json"))
	if err != nil {
		t.Fatalf("Load(fleet.b12x.json) error = %v", err)
	}
	if len(cfg.Models) != 9 {
		t.Fatalf("model count = %d, want 9", len(cfg.Models))
	}
	if !cfg.Runtime.ConcurrentDeployments {
		t.Fatal("B12X manifest must enable concurrent deployments")
	}
	if cfg.Runtime.ModelPortRange == nil || cfg.Runtime.ModelPortRange.Start != 8101 || cfg.Runtime.ModelPortRange.End != 8109 {
		t.Fatalf("B12X model port range = %+v, want 8101-8109", cfg.Runtime.ModelPortRange)
	}
	for _, id := range []string{"ds41-flash-tp4-ssd", "mimo-v26-flash-tp4"} {
		t.Run(id+"/isolation", func(t *testing.T) {
			model, ok := cfg.Model(id)
			if !ok {
				t.Fatalf("model %q missing", id)
			}
			// TP workers share a container, not the host IPC namespace. Keep
			// enough private shared memory without disabling Docker's filter.
			if model.IPC != "private" || model.ShmSize != "32g" {
				t.Fatalf("IPC/shared memory = %q/%q, want private/32g", model.IPC, model.ShmSize)
			}
			if id == "ds41-flash-tp4-ssd" {
				if len(model.SecurityOpt) != 1 || model.SecurityOpt[0] != "seccomp=overrides/seccomp-deepseek-io-uring.json" {
					t.Fatalf("DeepSeek security options = %v, want the scoped io_uring policy", model.SecurityOpt)
				}
			} else if len(model.SecurityOpt) != 0 {
				t.Fatalf("MiMo security options = %v, want Docker defaults", model.SecurityOpt)
			}
		})
	}
	mimo, ok := cfg.Model("mimo-v26-flash-tp4")
	if !ok {
		t.Fatal("MiMo 2.6 Flash profile missing")
	}
	if mimo.Placement == nil || mimo.Placement.GPUCount != 4 || mimo.Placement.Strategy != "pcie_affinity" {
		t.Fatalf("MiMo placement = %+v, want topology-aware TP4", mimo.Placement)
	}
	if mimo.Restart != "unless-stopped" {
		t.Fatalf("MiMo restart policy = %q", mimo.Restart)
	}
	if len(mimo.Mounts) == 0 || mimo.Mounts[0].Source != "/srv/fleet/models/MiMo-V2.6-Flash-RL" || !mimo.Mounts[0].ReadOnly {
		t.Fatalf("MiMo model mount = %+v", mimo.Mounts)
	}
	if mimo.Environment["VLLM_PCIE_ALLREDUCE_BACKEND"] != "b12x" {
		t.Fatalf("MiMo PCIe all-reduce backend = %q", mimo.Environment["VLLM_PCIE_ALLREDUCE_BACKEND"])
	}
	foundMMBackend := false
	for index, argument := range mimo.Command {
		if argument == "--mm-encoder-attn-backend" && index+1 < len(mimo.Command) && mimo.Command[index+1] == "TRITON_ATTN" {
			foundMMBackend = true
			break
		}
	}
	if !foundMMBackend {
		t.Fatal("MiMo must use the Triton multimodal encoder backend")
	}
}

func TestDeepSeekSeccompOnlyAddsRequiredIOUringCalls(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "overrides", "seccomp-deepseek-io-uring.json"))
	if err != nil {
		t.Fatal(err)
	}
	var profile map[string]any
	if err := json.Unmarshal(raw, &profile); err != nil {
		t.Fatal(err)
	}
	rules, ok := profile["syscalls"].([]any)
	if !ok || len(rules) == 0 {
		t.Fatal("seccomp syscall rules missing")
	}
	exception, ok := rules[0].(map[string]any)
	if !ok {
		t.Fatal("io_uring exception missing")
	}
	delete(exception, "comment")
	want := map[string]any{
		"names":  []any{"io_uring_setup", "io_uring_enter", "io_uring_register"},
		"action": "SCMP_ACT_ALLOW",
	}
	if !reflect.DeepEqual(exception, want) {
		t.Fatalf("io_uring exception = %v, want %v", exception, want)
	}
	profile["syscalls"] = rules[1:]
	base, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	// Canonical JSON SHA-256 of Docker 29.4.1's vendored default profile.
	// Rebase deliberately on Docker upgrades; do not quietly widen the policy.
	const defaultProfileSHA256 = "7ce699efbba58df5691185a87189ecc0a47ff01c48ec8fc5708465954b672979"
	if got := fmt.Sprintf("%x", sha256.Sum256(base)); got != defaultProfileSHA256 {
		t.Fatalf("base seccomp policy differs from the pinned Docker default: %s", got)
	}
}

func loadManifest(t *testing.T, body string) *Manifest {
	t.Helper()
	cfg, err := Load(writeManifest(t, body))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	return cfg
}

func writeManifest(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fleet.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}
