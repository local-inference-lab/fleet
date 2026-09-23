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
	if len(cfg.Models) != 10 {
		t.Fatalf("model count = %d, want 10", len(cfg.Models))
	}
	if !cfg.Runtime.ConcurrentDeployments {
		t.Fatal("B12X manifest must enable concurrent deployments")
	}
	if cfg.Runtime.ModelPortRange == nil || cfg.Runtime.ModelPortRange.Start != 8101 || cfg.Runtime.ModelPortRange.End != 8109 {
		t.Fatalf("B12X model port range = %+v, want 8101-8109", cfg.Runtime.ModelPortRange)
	}
	for _, id := range []string{"ds41-flash-tp4-ssd", "mimo-v26-flash-tp4", "mimo-v26-pro-tp8"} {
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

	pro, ok := cfg.Model("mimo-v26-pro-tp8")
	if !ok {
		t.Fatal("MiMo 2.6 Pro TP8 profile missing")
	}
	if pro.Image != mimo.Image {
		t.Fatalf("MiMo Pro image = %q, want same Flash image %q", pro.Image, mimo.Image)
	}
	if pro.Placement == nil || pro.Placement.GPUCount != 8 || pro.Placement.Strategy != "pcie_affinity" {
		t.Fatalf("MiMo Pro placement = %+v, want topology-aware TP8", pro.Placement)
	}
	if len(pro.Command) < 2 || pro.Command[0] != "serve" || pro.Command[1] != "/model" {
		t.Fatalf("MiMo Pro command model path = %v, want serve /model", pro.Command)
	}
	if got := commandArgValue(t, pro.Command, "--served-model-name"); got != "MiMo-V2.6-Pro-RL" {
		t.Fatalf("MiMo Pro served model = %q, want MiMo-V2.6-Pro-RL", got)
	}
	if got := commandArgValue(t, pro.Command, "--tensor-parallel-size"); got != "8" {
		t.Fatalf("MiMo Pro tensor parallel size = %q, want 8", got)
	}
	if got := commandArgValue(t, pro.Command, "--safetensors-load-strategy"); got != "lazy" {
		t.Fatalf("MiMo Pro safetensors load strategy = %q, want lazy", got)
	}
	if commandHasArg(pro.Command, "--load-format") {
		t.Fatalf("MiMo Pro command must use the standard safetensors loader, not --load-format b12x: %v", pro.Command)
	}
	if got := commandArgValue(t, pro.Command, "--port"); got != "8109" || got != commandArgValue(t, mimo.Command, "--port") {
		t.Fatalf("MiMo Pro port = %q and Flash port = %q, want shared 8109 guarded by TP8 full-GPU placement", got, commandArgValue(t, mimo.Command, "--port"))
	}
	if pro.Environment["CUDA_VISIBLE_DEVICES"] != "0,1,2,3,4,5,6,7" {
		t.Fatalf("MiMo Pro CUDA_VISIBLE_DEVICES = %q, want all eight logical GPUs", pro.Environment["CUDA_VISIBLE_DEVICES"])
	}
	if len(pro.Mounts) == 0 || pro.Mounts[0].Source != "/srv/fleet/models/MiMo-V2.6-Pro-RL" || pro.Mounts[0].Target != "/model" || !pro.Mounts[0].ReadOnly {
		t.Fatalf("MiMo Pro model mount = %+v, want read-only Pro checkpoint at /model", pro.Mounts)
	}
	speculative := speculativeConfig(t, pro)
	if speculative["method"] != "dflash" {
		t.Fatalf("MiMo Pro speculative method = %v, want dflash", speculative["method"])
	}
	if speculative["model"] != "/model/dflash" {
		t.Fatalf("MiMo Pro draft path = %v, want /model/dflash under the Pro checkpoint mount", speculative["model"])
	}
	if speculative["model"] == "/opt/mimo-v26/dflash-fixed" {
		t.Fatal("MiMo Pro must not use the image's baked Flash draft")
	}
	if speculative["num_speculative_tokens"] != float64(7) {
		t.Fatalf("MiMo Pro speculative token count = %v, want 7", speculative["num_speculative_tokens"])
	}
	if speculative["draft_tensor_parallel_size"] != float64(8) {
		t.Fatalf("MiMo Pro draft TP = %v, want TP8", speculative["draft_tensor_parallel_size"])
	}
}

func commandArgValue(t *testing.T, command []string, name string) string {
	t.Helper()
	for index, argument := range command {
		if argument == name && index+1 < len(command) {
			return command[index+1]
		}
		if strings.HasPrefix(argument, name+"=") {
			return strings.TrimPrefix(argument, name+"=")
		}
	}
	t.Fatalf("command missing %s: %v", name, command)
	return ""
}

func commandHasArg(command []string, name string) bool {
	for _, argument := range command {
		if argument == name || strings.HasPrefix(argument, name+"=") {
			return true
		}
	}
	return false
}

func speculativeConfig(t *testing.T, model Model) map[string]any {
	t.Helper()
	raw := commandArgValue(t, model.Command, "--speculative-config")
	var config map[string]any
	if err := json.Unmarshal([]byte(raw), &config); err != nil {
		t.Fatalf("%s speculative config is not JSON: %v", model.ID, err)
	}
	return config
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
