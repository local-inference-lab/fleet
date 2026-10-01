package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
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
				"memory_limit": "40g",
				"memory_swap_limit": "40g",
				"privileged": true,
				"stop_timeout": "2m0s",
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
	if !qwen.Privileged || qwen.StopTimeoutSeconds() != 120 {
		t.Fatalf("privileged/stop timeout = %v/%d, want true/120", qwen.Privileged, qwen.StopTimeoutSeconds())
	}
	if qwen.MemoryLimit != "40g" || qwen.MemorySwapLimit != "40g" {
		t.Fatalf("memory limits = %q/%q, want 40g/40g", qwen.MemoryLimit, qwen.MemorySwapLimit)
	}
	glm, ok := cfg.Model("glm")
	if !ok {
		t.Fatal("glm model missing")
	}
	if glm.MemoryLimit != "" || glm.MemorySwapLimit != "" {
		t.Fatalf("default memory limits = %q/%q, want empty", glm.MemoryLimit, glm.MemorySwapLimit)
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
			name: "reserved replica suffix",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "alpha--2", "image": "one"}]
			}`,
			wantErr: "reserved replica suffix",
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
		{
			name: "invalid stop timeout",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "stop_timeout": "500ms"}]
			}`,
			wantErr: "stop_timeout",
		},
		{
			name: "invalid memory limit",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "memory_limit": "40gb"}]
			}`,
			wantErr: "memory_limit",
		},
		{
			name: "memory swap requires memory",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "memory_swap_limit": "40g"}]
			}`,
			wantErr: "memory_swap_limit requires memory_limit",
		},
		{
			name: "memory swap must not be below memory",
			body: `{
				"version": 1,
				"runtime": {},
				"models": [{"id": "a", "image": "one", "memory_limit": "40g", "memory_swap_limit": "39g"}]
			}`,
			wantErr: "memory_swap_limit must be -1",
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

func TestLoadAllowsUnlimitedMemorySwapWhenMemoryLimitIsSet(t *testing.T) {
	cfg := loadManifest(t, `{
		"version": 1,
		"runtime": {},
		"models": [{"id": "a", "image": "one", "memory_limit": "1m", "memory_swap_limit": "-1"}]
	}`)
	model, ok := cfg.Model("a")
	if !ok {
		t.Fatal("model a missing")
	}
	if model.MemoryLimit != "1m" || model.MemorySwapLimit != "-1" {
		t.Fatalf("memory limits = %q/%q, want 1m/-1", model.MemoryLimit, model.MemorySwapLimit)
	}
}

func TestB12XFleetManifestIsValid(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "fleet.b12x.json"))
	if err != nil {
		t.Fatalf("Load(fleet.b12x.json) error = %v", err)
	}
	if len(cfg.Models) != 15 {
		t.Fatalf("model count = %d, want 15", len(cfg.Models))
	}
	if !cfg.Runtime.ConcurrentDeployments {
		t.Fatal("B12X manifest must enable concurrent deployments")
	}
	if cfg.Runtime.ModelPortRange == nil || cfg.Runtime.ModelPortRange.Start != 8101 || cfg.Runtime.ModelPortRange.End != 8121 {
		t.Fatalf("B12X model port range = %+v, want 8101-8121", cfg.Runtime.ModelPortRange)
	}
	for _, id := range []string{"ds41-flash-tp4-ssd", "mimo-v26-flash-opus55", "mimo-v26-pro-tp8", "mimo-v26-flash-mopd-tp4"} {
		t.Run(id+"/isolation", func(t *testing.T) {
			model, ok := cfg.Model(id)
			if !ok {
				t.Fatalf("model %q missing", id)
			}
			if model.ShmSize != "32g" {
				t.Fatalf("shared memory = %q, want 32g", model.ShmSize)
			}
			if id == "ds41-flash-tp4-ssd" || id == "mimo-v26-flash-mopd-tp4" {
				wantIPC := "private"
				if id == "mimo-v26-flash-mopd-tp4" {
					wantIPC = "host"
				}
				if model.IPC != wantIPC || len(model.SecurityOpt) != 1 || model.SecurityOpt[0] != "seccomp=overrides/seccomp-deepseek-io-uring.json" {
					t.Fatalf("io_uring profile IPC/security = %q/%v", model.IPC, model.SecurityOpt)
				}
			} else if model.IPC != "host" || len(model.SecurityOpt) != 1 || model.SecurityOpt[0] != "seccomp=unconfined" {
				t.Fatalf("MiMo IPC/security = %q/%v", model.IPC, model.SecurityOpt)
			}
		})
	}
	deepseek, ok := cfg.Model("ds41-flash-tp4-ssd")
	if !ok {
		t.Fatal("DeepSeek V4.1 profile missing")
	}
	if len(deepseek.Mounts) < 2 || deepseek.Mounts[0].Source != "/mnt/llm_stuff/models/deepseek-v4.1-flash" || deepseek.Mounts[1].Source != "/mnt/llm_stuff/cache" {
		t.Fatalf("DeepSeek mounts = %+v, want host model and cache under /mnt/llm_stuff", deepseek.Mounts)
	}
	glmFlash, ok := cfg.Model("glm53-flash-tp4")
	if !ok {
		t.Fatal("GLM 5.3 Flash profile missing")
	}
	const glmFlashImage = "lil-fleet/glm53-flash:mixed-device-20260924"
	if glmFlash.Image != glmFlashImage {
		t.Fatalf("GLM 5.3 Flash image = %q, want %q", glmFlash.Image, glmFlashImage)
	}
	if len(glmFlash.Mounts) < 2 || glmFlash.Mounts[0].Source != "/mnt/llm_stuff/models/GLM-5.3-Flash-NVFP4" || glmFlash.Mounts[1].Source != "/mnt/llm_stuff/cache" {
		t.Fatalf("GLM 5.3 Flash mounts = %+v, want host model and cache under /mnt/llm_stuff", glmFlash.Mounts)
	}
	glm53, ok := cfg.Model("glm53-tp8")
	if !ok {
		t.Fatal("GLM 5.3 TP8 profile missing")
	}
	if glm53.Image != "joninco/vllm:glm53-b12x686450b7-vllm5b28d30b59-r27" || glm53.Entrypoint != "" {
		t.Fatalf("GLM 5.3 TP8 image/entrypoint = %q/%q", glm53.Image, glm53.Entrypoint)
	}
	if glm53.Environment["PORT"] != "8104" || glm53.Environment["SERVED_MODEL_NAME"] != "GLM-5.3-NVFP4" || commandArgValue(t, glm53.Command, "--port") != "8104" {
		t.Fatalf("GLM 5.3 TP8 serving settings = port %q, model %q, command %v", glm53.Environment["PORT"], glm53.Environment["SERVED_MODEL_NAME"], glm53.Command)
	}
	if glm53.Environment["MAX_MODEL_LEN"] != "515000" || glm53.Environment["GPU_MEMORY_UTILIZATION"] != "0.95" {
		t.Fatalf("GLM 5.3 TP8 context/memory = %q/%q", glm53.Environment["MAX_MODEL_LEN"], glm53.Environment["GPU_MEMORY_UTILIZATION"])
	}
	var glmSpec map[string]any
	if err := json.Unmarshal([]byte(glm53.Environment["SPEC_CONFIG"]), &glmSpec); err != nil || glmSpec["method"] != "mtp" || glmSpec["num_speculative_tokens"] != float64(3) || glmSpec["moe_backend"] != "triton" {
		t.Fatalf("GLM 5.3 TP8 MTP config = %v, error %v", glmSpec, err)
	}
	var glmOverlay struct {
		QuantizationConfig struct {
			Ignore []string `json:"ignore"`
		} `json:"quantization_config"`
	}
	overlay, err := os.ReadFile(filepath.Join("..", "..", "overrides", "glm53-r27-config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(overlay, &glmOverlay); err != nil {
		t.Fatalf("GLM 5.3 TP8 config overlay: %v", err)
	}
	if !slices.Contains(glmOverlay.QuantizationConfig.Ignore, "model.layers.78*") {
		t.Fatal("GLM 5.3 TP8 BF16 MTP layer must be excluded from NVFP4 quantization")
	}
	if !glm53.Privileged || glm53.StopTimeoutSeconds() != 120 || glm53.NetworkMode != "host" || glm53.IPC != "host" {
		t.Fatalf("GLM 5.3 TP8 container settings = privileged %v, stop %d, network %q, IPC %q", glm53.Privileged, glm53.StopTimeoutSeconds(), glm53.NetworkMode, glm53.IPC)
	}
	if len(glm53.Mounts) != 4 || glm53.Mounts[0].Source != "/mnt/llm_stuff/models/GLM-5.3-NVFP4" || glm53.Mounts[0].Target != glm53.Environment["MODEL"] || !glm53.Mounts[0].ReadOnly || glm53.Mounts[3].Target != glm53.Environment["MODEL"]+"/config.json" || !glm53.Mounts[3].ReadOnly {
		t.Fatalf("GLM 5.3 TP8 model mounts = %+v", glm53.Mounts)
	}
	if glm53.Mounts[1].Source != "/home/clay/research/lil-fleet/.cache/glm53-r27/jit" || glm53.Mounts[2].Source != "/home/clay/research/lil-fleet/.cache/glm53-r27/container-tmp" {
		t.Fatalf("GLM 5.3 TP8 writable cache mounts = %+v", glm53.Mounts)
	}
	mimo, ok := cfg.Model("mimo-v26-flash-opus55")
	if !ok {
		t.Fatal("MiMo 2.6 Flash profile missing")
	}
	if mimo.Image != "madeby561/vllm:mimo-v26-flash-b12x-20260923-rc4" {
		t.Fatalf("MiMo image = %q, want the working rc4 image", mimo.Image)
	}
	if mimo.Placement == nil || mimo.Placement.GPUCount != 4 || mimo.Placement.Strategy != "pcie_affinity" {
		t.Fatalf("MiMo placement = %+v, want topology-aware TP4", mimo.Placement)
	}
	if mimo.Restart != "unless-stopped" {
		t.Fatalf("MiMo restart policy = %q", mimo.Restart)
	}
	if len(mimo.Mounts) == 0 || mimo.Mounts[0].Source != "/mnt/llm_stuff/models/MiMo-V2.6-Flash-RL" || !mimo.Mounts[0].ReadOnly {
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

	if got := commandArgValue(t, mimo.Command, "--linear-backend"); got != "b12x" {
		t.Fatalf("MiMo linear backend = %q, want b12x", got)
	}
	if got := commandArgValue(t, mimo.Command, "--max-num-scheduled-tokens"); got != "2048" {
		t.Fatalf("MiMo max scheduled tokens = %q, want 2048", got)
	}
	if got := commandArgValue(t, mimo.Command, "--prefill-compute-share"); got != "0.8" {
		t.Fatalf("MiMo prefill compute share = %q, want 0.8", got)
	}
	if mimo.Environment["VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE"] != "2MB" || mimo.Environment["VLLM_PCIE_DMA_MIN_BYTES"] != "off" {
		t.Fatalf("MiMo PCIe tuning = twoshot %q DMA minimum %q", mimo.Environment["VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE"], mimo.Environment["VLLM_PCIE_DMA_MIN_BYTES"])
	}
	speculative := speculativeConfig(t, mimo)
	if speculative["method"] != "dflash" {
		t.Fatalf("MiMo Flash speculative method = %v, want dflash", speculative["method"])
	}
	if speculative["model"] != "/opt/mimo-v26/dflash-fixed" {
		t.Fatalf("MiMo Flash draft path = %v, want the image's /opt/mimo-v26/dflash-fixed", speculative["model"])
	}
	if speculative["num_speculative_tokens"] != float64(7) {
		t.Fatalf("MiMo Flash speculative token count = %v, want 7", speculative["num_speculative_tokens"])
	}
	if speculative["draft_tensor_parallel_size"] != float64(4) {
		t.Fatalf("MiMo Flash draft TP = %v, want TP4", speculative["draft_tensor_parallel_size"])
	}
	if speculative["draft_sample_method"] != "probabilistic" || speculative["rejection_sample_method"] != "standard" {
		t.Fatalf("MiMo Opus55 sampling = draft %v rejection %v", speculative["draft_sample_method"], speculative["rejection_sample_method"])
	}

	pro, ok := cfg.Model("mimo-v26-pro-tp8")
	if !ok {
		t.Fatal("MiMo 2.6 Pro TP8 profile missing")
	}
	if pro.Image != mimo.Image {
		t.Fatalf("MiMo Pro image = %q, want Opus55 image %q", pro.Image, mimo.Image)
	}
	if pro.Placement == nil || pro.Placement.GPUCount != 8 || pro.Placement.Strategy != "pcie_affinity" {
		t.Fatalf("MiMo Pro placement = %+v, want topology-aware TP8", pro.Placement)
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
	if got := commandArgValue(t, pro.Command, "--linear-backend"); got != "b12x" {
		t.Fatalf("MiMo Pro linear backend = %q, want b12x", got)
	}
	if got := commandArgValue(t, pro.Command, "--max-num-scheduled-tokens"); got != "2048" {
		t.Fatalf("MiMo Pro max scheduled tokens = %q, want 2048", got)
	}
	if len(pro.Command) < 2 || pro.Command[0] != "serve" || pro.Command[1] != "/model" {
		t.Fatalf("MiMo Pro command model path = %v, want serve /model", pro.Command)
	}
	if commandHasArg(pro.Command, "--load-format") {
		t.Fatalf("MiMo Pro must use the standard lazy safetensors loader, not a --load-format override: %v", pro.Command)
	}
	if got, flash := commandArgValue(t, pro.Command, "--port"), commandArgValue(t, mimo.Command, "--port"); got != "8109" || got != flash {
		t.Fatalf("MiMo Pro port = %q and Flash port = %q, want shared 8109 guarded by TP8 full-GPU placement", got, flash)
	}
	if pro.Environment["CUDA_VISIBLE_DEVICES"] != "0,1,2,3,4,5,6,7" {
		t.Fatalf("MiMo Pro CUDA_VISIBLE_DEVICES = %q, want all eight logical GPUs", pro.Environment["CUDA_VISIBLE_DEVICES"])
	}
	if len(pro.Mounts) == 0 || pro.Mounts[0].Source != "/mnt/llm_stuff/models/MiMo-V2.6-Pro-RL" || pro.Mounts[0].Target != "/model" || !pro.Mounts[0].ReadOnly {
		t.Fatalf("MiMo Pro model mount = %+v, want read-only Pro checkpoint at /model", pro.Mounts)
	}
	proSpeculative := speculativeConfig(t, pro)
	if proSpeculative["model"] != "/model/dflash" {
		t.Fatalf("MiMo Pro draft path = %v, want /model/dflash", proSpeculative["model"])
	}
	if proSpeculative["method"] != "dflash" || proSpeculative["num_speculative_tokens"] != float64(7) {
		t.Fatalf("MiMo Pro speculative method/tokens = %v/%v, want dflash/7", proSpeculative["method"], proSpeculative["num_speculative_tokens"])
	}
	if proSpeculative["draft_tensor_parallel_size"] != float64(8) {
		t.Fatalf("MiMo Pro draft TP = %v, want TP8", proSpeculative["draft_tensor_parallel_size"])
	}
	if proSpeculative["draft_sample_method"] != "probabilistic" || proSpeculative["rejection_sample_method"] != "standard" {
		t.Fatalf("MiMo Pro Opus55 sampling = draft %v rejection %v", proSpeculative["draft_sample_method"], proSpeculative["rejection_sample_method"])
	}
}

func TestDerivedTP4Profiles(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "fleet.b12x.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		id, image, modelDir, servedName, port string
	}{
		{"swift15-flash-next-nvfp4-tp4", "lil-fleet/b12x:f20ab3bad7de", "/mnt/llm_stuff/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4", "Swift-1.5-Flash-Next-NVFP4", "8112"},
		{"mimo-v26-flash-mopd-tp4", "local/mimo-v26-pro:karmic-kraken-av-20260926", "/mnt/llm_stuff/models/MiMo-V2.6-Flash-MOPD", "MiMo-V2.6-Flash-MOPD", "8113"},
		{"qwen38-flash-next-qad4000-tp4", "lil-fleet/b12x:f20ab3bad7de", "/mnt/llm_stuff/models/Qwen3.8-Flash-Next-NVFP4-QAD-4000", "Qwen3.8-Flash-Next-QAD-4000", "8110"},
		{"glm53-flash-qad-tvn3500-tp4", "lil-fleet/glm53-flash:mixed-device-20260924", "/mnt/llm_stuff/models/GLM-5.3-Flash-NVFP4-QAD-TVN-3500", "GLM-5.3-Flash-NVFP4-QAD-TVN-3500", "8111"},
	} {
		t.Run(test.id, func(t *testing.T) {
			model, ok := cfg.Model(test.id)
			if !ok {
				t.Fatal("profile missing")
			}
			if model.Image != test.image || model.Placement == nil || model.Placement.GPUCount != 4 {
				t.Fatalf("image/placement = %q/%+v", model.Image, model.Placement)
			}
			if commandArgValue(t, model.Command, "--served-model-name") != test.servedName ||
				commandArgValue(t, model.Command, "--port") != test.port ||
				commandArgValue(t, model.Command, "--tensor-parallel-size") != "4" {
				t.Fatalf("profile command = %v", model.Command)
			}
			if len(model.Mounts) < 1 || model.Mounts[0].Source != test.modelDir || !model.Mounts[0].ReadOnly {
				t.Fatalf("profile checkpoint mount = %+v", model.Mounts)
			}
			if model.Readiness == nil || model.Readiness.URL != "http://127.0.0.1:"+test.port+"/health" {
				t.Fatalf("profile readiness = %+v", model.Readiness)
			}
		})
	}
	mopd, _ := cfg.Model("mimo-v26-flash-mopd-tp4")
	if len(mopd.SecurityOpt) != 1 || mopd.SecurityOpt[0] != "seccomp=overrides/seccomp-deepseek-io-uring.json" {
		t.Fatalf("MiMo MOPD security options = %v, want narrow io_uring profile", mopd.SecurityOpt)
	}
	if got := speculativeConfig(t, mopd); got["model"] != "/model/dflash" || got["draft_tensor_parallel_size"] != float64(4) {
		t.Fatalf("MiMo MOPD DFlash configuration = %v", got)
	}
	glm, _ := cfg.Model("glm53-flash-qad-tvn3500-tp4")
	if glm.Environment["B12X_TUNING_DEVICE_CLASS"] != "sm120-188sm" {
		t.Fatalf("GLM QAD mixed-GPU tuning class = %q", glm.Environment["B12X_TUNING_DEVICE_CLASS"])
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

func TestMiMoProEncoderDriverCompatibility(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "fleet.b12x.json"))
	if err != nil {
		t.Fatal(err)
	}
	pro, ok := cfg.Model("mimo-v26-pro-tp8")
	if !ok {
		t.Fatal("MiMo Pro profile missing")
	}
	if got := commandArgValue(t, pro.Command, "--mm-encoder-attn-backend"); got != "TRITON_ATTN" {
		t.Fatalf("MiMo Pro encoder attention backend = %q, want TRITON_ATTN for driver compatibility", got)
	}
}
