# Fleet

`lil-fleet` is a small Docker control plane for switching between a fixed set of local inference deployments. It follows the Local Inference Lab pattern: model and hardware tuning belongs to declarative deployment configuration; the API only chooses a configured profile, loads it, unloads it, and reports its state.

The service intentionally has no endpoint for changing images, commands, environment variables, mounts, ports, GPU allocation, shared memory, IPC, or readiness checks. Those values come from a strict JSON manifest that Fleet watches and reloads. Unknown manifest and API fields are rejected.

## Run it

Requirements are Go 1.26.8+ and a working Docker CLI/daemon. The minimum patch
version includes standard-library security fixes; rebuild existing binaries when
updating the toolchain. Copy and edit the example first:

```sh
cp fleet.example.json fleet.json
go run ./cmd/lil-fleet -manifest fleet.json
```

Protect the control-plane routes with a bearer token file:

```sh
go run ./cmd/lil-fleet -manifest fleet.json -token-file .secrets/api-token
curl -H "Authorization: Bearer $(cat .secrets/api-token)" http://127.0.0.1:8090/v1/models
```

`/healthz` and `/readyz` remain unauthenticated for local probes. All `/v1/*` routes require the token when `-token-file` is supplied. Fleet refuses to start without `-token-file` when `api.listen` is not a loopback address; pass `-insecure-no-auth` only if something else authenticates every request. Keep the token file mode `0600`; Fleet warns at startup when it is readable by group or others. The HTTP server bounds header, body, response, and idle time (5s/30s/60s/120s); every route answers from memory, so no request is long-lived.

The example model images are illustrative floating tags. For a reproducible deployment, replace them with image digests, pin model revisions in each command, and use host paths that exist on the Docker daemon host.

### Running on the host (recommended)

[`contrib/lil-fleet.service`](contrib/lil-fleet.service) is a hardened systemd
unit: an unprivileged `fleet` user in the `docker` group, the token delivered as
a systemd credential, no capabilities, a read-only view of the filesystem except
the per-model cache roots, and a `@system-service` syscall filter. Adjust paths
and `ReadWritePaths` to your manifest. Docker group membership is still
root-equivalent; the unit limits the Fleet process, not what Docker will do on
its behalf.

### Running the controller in Docker

[`compose.yaml`](compose.yaml) runs Fleet without the Docker socket. A
[socket-proxy](https://github.com/wollomatic/socket-proxy) sidecar with no
network opens the socket and exposes a filtered unix socket that Fleet uses via
`DOCKER_HOST`. It allows only `_ping`, `version`, container list/inspect/
create/start/stop/delete, and image inspect/pull; exec, build, volume, network,
system, and swarm APIs are refused. `-allowbindmountfrom` additionally limits
bind-mount sources to `FLEET_ALLOWED_BIND_ROOTS` (comma-separated; include every
manifest mount root and per-model cache root). This narrows a compromised Fleet's
reach but is not a sandbox: Fleet must create GPU containers, and the manifest
can still ask for privileged ones.

```sh
export DOCKER_GID=$(getent group docker | cut -d: -f3)
export FLEET_UID=$(id -u) FLEET_GID=$(id -g)
export FLEET_ALLOWED_BIND_ROOTS=/srv/fleet
docker compose up -d --build
```

Both containers run read-only, with all capabilities dropped and
`no-new-privileges`; the Fleet image runs as a non-root user. Notes:

- Fleet uses host networking because readiness URLs must be loopback
  (`127.0.0.1:<port>`). With bridge networking, `127.0.0.1` would be the Fleet
  container itself and every probe would fail. Keep `api.listen` on
  `127.0.0.1:8090`; Fleet refuses a non-loopback listen without a token.
- Relative `seccomp=` paths resolve against the manifest directory
  (`/etc/lil-fleet`), and the Docker CLI reads the profile inside the Fleet
  container, so the compose file mounts `./overrides` at
  `/etc/lil-fleet/overrides`.
- To let Fleet create `per_model` cache directories, mount each cache root at its
  host path; otherwise create the directories on the host.
- The manifest is a single-file bind mount. Editors that replace the file (new
  inode) are not seen inside the container; edit in place or restart.
- If a new Docker CLI version needs another endpoint, run the proxy once with
  `-loglevel=DEBUG` to see the denied request.

## API

List the configured profiles and their current state:

```sh
curl http://127.0.0.1:8090/v1/models
```

Inspect the GPUs, their free memory, PCIe group, and which Fleet deployments hold them:

```sh
curl http://127.0.0.1:8090/v1/gpus
```

```json
{
  "sampled_at": "2026-10-10T12:00:00Z",
  "groups": [[0,1,6,7],[2,3,4,5]],
  "gpus": [
    {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "memory_total_mib": 97887, "memory_used_mib": 1234, "memory_free_mib": 96000, "utilization_percent": 0, "group": 0, "assigned": [{"model_id": "ds41-flash-tp4-ssd", "instance_id": "ds41-flash-tp4-ssd"}]}
  ]
}
```

`groups` is the manifest topology (`[]` without one) and `group` indexes into it (`null` for a GPU outside every group). `utilization_percent` is `null` when `nvidia-smi` reports it unavailable. `assigned` lists every Fleet instance holding the GPU, including loading and stopping ones, and is `[]` for a free GPU. The snapshot is reused for about 5 seconds, and concurrent requests share one `nvidia-smi` run, so polling this endpoint does not spawn a process per request. When `nvidia-smi` fails, the last snapshot is returned with `"stale": true` while it is under a minute old; otherwise the response is `503` with code `gpu_query_failed`. Each `/v1/models` entry also reports `gpu_count`: the placement GPU count, or the count implied by a static `gpus` value.

Load a model. By default this is an exclusive switch. When `runtime.concurrent_deployments` is enabled, Fleet allocates free GPUs from the manifest topology and keeps compatible deployments resident. An empty request body keeps the legacy behavior: ensure at least one instance is running, or preserve the current replica count when the model is already scaled above one.

```sh
curl -i -X POST http://127.0.0.1:8090/v1/models/qwen3-8b/load
```

To request an exact replica count, send a strict JSON body with `instances` (and optionally `gpus`, below); other fields are rejected. The value must be between 1 and 64 and must fit the manifest's GPU topology and dynamic port pool. For example, a TP4 profile on the B12X manifest can run as two disjoint four-GPU instances:

```sh
curl -i -X POST http://127.0.0.1:8090/v1/models/qwen38-flash-next-qad4000-tp4/load \
  -H 'content-type: application/json' \
  -d '{"instances":2}'
```

For a model with `placement`, a load may also choose the GPUs for the instances it starts with `gpus`. The list must hold `placement.gpu_count` indices per newly started instance; the first `gpu_count` go to the first new instance, and so on (each instance's GPUs are stored in ascending order and persist in the container labels like automatic picks). Omitting `gpus`, or sending `[]`, keeps automatic placement. Fleet does not enforce PCIe group rules for an explicit pick; a pick that spans groups for a TP1 to TP4 model succeeds with a `warnings` array in the response:

```sh
curl -i -X POST http://127.0.0.1:8090/v1/models/qwen38-flash-next-tp2-ssd/load \
  -H 'content-type: application/json' \
  -d '{"instances":1,"gpus":[2,3]}'
```

```json
{"changed": true, "operation": {"id": "…", "kind": "activate", "model_id": "qwen38-flash-next-tp2-ssd", "instances": 1, "gpus": [2,3], "state": "pending", "created_at": "…"}}
```

| Status | Code | Cause |
|-|-|-|
| 400 | `gpus_not_supported` | The model has no `placement` |
| 400 | `gpus_not_applicable` | `gpus` is non-empty but the request starts no new instance (count unchanged or lower) |
| 400 | `invalid_gpus` | Not an integer array, a negative, duplicate, or unknown index (in neither `nvidia-smi` nor the topology), or the length is not `gpu_count` times the new instances |
| 409 | `gpus_unavailable` | A GPU is assigned to another deployment or instance (including one still loading) or is above `max_used_memory_mib` |

The response is `202 Accepted`, includes an operation whose `instances` field is the target count, and sets a `Location` header. Poll that operation and the deployment:

```sh
curl http://127.0.0.1:8090/v1/operations/OPERATION_ID
curl http://127.0.0.1:8090/v1/deployments/qwen3-8b
```

An alternate switch endpoint is useful for clients that do not put identifiers in paths. Its strict body accepts only `model_id`:

```sh
curl -i -X POST http://127.0.0.1:8090/v1/deployments/switch \
  -H 'content-type: application/json' \
  -d '{"model_id":"mistral-7b"}'
```

Unload a model:

```sh
curl -i -X POST http://127.0.0.1:8090/v1/models/mistral-7b/unload
```

Unload stops all instances of that profile. Lifecycle calls are idempotent per model: a duplicate in-flight call (same kind and, for loads, the same `instances` and either no `gpus` or the same `gpus`) returns the same operation, and a conflicting call for that model returns `409 Conflict`. Without `runtime.concurrent_deployments`, switching is exclusive, so any lifecycle call during a running operation returns `409`. With it, each model has its own operation: a model waiting up to `readiness_timeout` for readiness does not block loads or unloads of other models. Container stop/create/start steps are still serialized fleet-wide by a short lock, so two GPU-heavy profiles never start at the same instant, and GPUs are reserved under the manager lock before any container is created.

Note that an empty load body and `{"instances":1}` differ when a model already runs replicas: the empty body keeps the current replica count, while `{"instances":1}` scales down to one.

Loading is observed every second (or `runtime.poll_interval` if shorter), so a model is reported ready within about a second of its probe passing. An exclusive switch stops only models that actually have containers, and stops them in parallel. Each stop is bounded by `runtime.stop_timeout` (default `operation_timeout`, never less than the model's own `stop_timeout` plus 15s). A missing image is pulled before create under `runtime.pull_timeout` (default `30m`), outside the lifecycle lock. Every Docker and `nvidia-smi` call has a deadline; the GPU snapshot runs outside the manager lock with a 5s limit.

Finished operations are kept for one hour, and at most 256 are retained; older ones return `404` from `/v1/operations/{id}`. Running operations are never evicted.

## Patched B12X fleet

[`fleet.b12x.json`](fleet.b12x.json) contains example RTX PRO 6000 profiles for an eight-GPU system with two PCIe groups. Copy it to the ignored `fleet.json`, then adjust the example topology groups `[0,1,6,7]` and `[2,3,4,5]` and `/mnt/llm_stuff` mount paths for the Docker daemon host. Build its pinned runtime overrides first:

```sh
docker build -f Dockerfile.b12x-override \
  --build-arg B12X_REF=f20ab3bad7def65f6211403fb73f9b1e8a33dfb0 \
  -t lil-fleet/b12x:f20ab3bad7de .
cp fleet.b12x.json fleet.json
# Adjust fleet.json for the host before starting the controller.
go run ./cmd/lil-fleet -manifest fleet.json -token-file .secrets/api-token
```

The MiMo Opus55 profiles use the published `madeby561/vllm:mimo-v26-flash-b12x-20260923-rc4` image. For hosts that need the mixed-device tuning override, optionally build `Dockerfile.mimo-b12x-override` with `docker build -f Dockerfile.mimo-b12x-override -t lil-fleet/mimo-v26-flash-opus55:20260923c-b12x-mixed-device .`, then set the relevant profiles' `image` fields in `fleet.json` to that local tag.

TP1 through TP4 stay within one PCIe group; TP6 takes one complete group plus two GPUs from the other; TP8 requires all GPUs. Among the placements those rules allow, Fleet picks the GPU set with the most total free memory in the latest `nvidia-smi` snapshot (ties go to the larger per-GPU minimum, then to the lowest group and index), so a TP2 load lands on the emptiest cards rather than the first free ones. Assigned GPUs are always listed in ascending index order. Replica loads reserve disjoint GPU sets and distinct ports from the shared `8101` through `8121` pool before creating containers, so `{"instances":2}` works for TP4 when both four-GPU groups are free. Live `nvidia-smi` memory use prevents Fleet from allocating GPUs occupied by workloads it did not create. A placement that cannot fit returns `409 Conflict` without creating a container.

The DeepSeek profile bounds host startup memory with one B12X compiler per TP rank, a 64-entry compiled-object memory cache, and 40 GiB RAM with no swap per replica. Its [startup wrapper](overrides/lil-serve-b12x-preparation-bound.sh) applies a checked, idempotent [B12X preparation patch](overrides/apply_b12x_preparation_bound.py) before the requested Karmic image's launcher: `LIL_B12X_PREPARATION_FACTORY_CACHE=0` releases discarded factory results and `LIL_B12X_PREPARATION_RACE_BATCH=2` limits concurrent tuning candidates. Autotuning stays enabled; kernel artifacts remain under the mounted SSD cache. The patch checks the image's source before writing; revalidate it when upgrading the image. Run its GPU-free tests inside the image with `python overrides/test_b12x_preparation_bound.py --source-root /opt/venv/lib/python3.12/site-packages/b12x -v`.

The same wrapper applies an opt-in [vLLM sequence-info patch](overrides/apply_vllm_sequence_info.py) because this image otherwise exposes `/server_info` only through development mode. With `LIL_VLLM_SEQUENCE_INFO=1`, it returns just `vllm_config.scheduler_config.max_num_seqs`; llmconduit can discover 32 slots per running replica without enabling vLLM development controls. The source match is checked and idempotent. Revalidate it on image upgrades with `python overrides/test_vllm_sequence_info.py -v` inside the exact image.

DeepSeek uses disk-backed Engram tables. Qwen Flash Next TP2/TP3 use disk-backed PLE tables; TP4 disables PLE CPU offload so its tables remain on GPU. GLM 5.3 TP6 and Qwen Flash Next TP3 are labeled experimental because upstream has not published qualified recipes for those shapes.

The `swift15-flash-next-nvfp4-tp4` profile serves [UkisAI Swift 1.5 Flash Next NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) as `Swift-1.5-Flash-Next-NVFP4` on port `8112`. It reuses the Qwen Flash Next TP4 runtime with PLE tables in VRAM and allocates one four-GPU PCIe group. Place the checkpoint in `/mnt/llm_stuff/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` before loading it. Like every host-networked B12X profile, it listens only on `127.0.0.1`; reach it through a local proxy or the llmconduit worker on this host.

The `mimo-v26-flash-mopd-tp4` profile serves [MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) on port `8113` from `/mnt/llm_stuff/models/MiMo-V2.6-Flash-MOPD`. It uses four GPUs, the Karmic Kraken MiMo runtime, FP8 KV cache, and the bundled TP4 DFlash draft. Its fast bounce loader uses the existing Docker-default-based seccomp profile that adds only the three `io_uring` syscalls.

The `mimo-v26-flash-opus55` profile serves `XiaomiMiMo/MiMo-V2.6-Flash-RL` from this host's `/mnt/llm_stuff/models/MiMo-V2.6-Flash-RL` checkpoint with the Opus55 attention/L2 prefetch path and probabilistic DFlash verification. It forces A16 activations for B12X FP4 MoE instead of the default W4A8 path. Its TP4 placement takes one complete PCIe group, depending on availability; the fixed `CUDA_VISIBLE_DEVICES=0,1,2,3` refers to the four logical devices exposed inside the allocated container. The optional derived image adds the host-specific `B12X_TUNING_DEVICE_CLASS` patch so mixed Max-Q and standard RTX PRO 6000 cards agree on one SM120A tuning identity. The multimodal encoder remains on Triton attention for compatibility with the installed NVIDIA driver.

The `mimo-v26-pro-tp8` profile applies the same Opus55 attention, L2 prefetch, forced-A16 B12X FP4 MoE, PCIe collective, and probabilistic verification settings to `XiaomiMiMo/MiMo-V2.6-Pro-RL`. Pro retains its model-specific constraints: all eight GPUs, the lazy safetensors loader, a smaller sequence concurrency, and its TP8 DFlash draft mounted at `/model/dflash` from `/mnt/llm_stuff/models/MiMo-V2.6-Pro-RL`. Flash and Pro share port `8109` because TP8 Pro cannot run concurrently with the TP4 Flash profile. Download the official Pro checkpoint at revision `73875d00b30a89ef8cc353a0b60b0e9f9561952d`, and keep its DFlash draft under the same checkpoint directory (`dflash/`) so the container sees it as `/model/dflash`. Pro uses vLLM's standard lazy safetensors loader instead of `--load-format b12x` because the B12X direct loader rejects Pro's audio-encoder weights; B12X remains the execution backend for kernels and collectives.

The `glm53-tp8` profile uses `joninco/vllm:glm53-b12x686450b7-vllm5b28d30b59-r27` with B12X PCIe collectives, NVFP4, and three-token MTP. This host's checkpoint stores MTP layer 78 in BF16 but omits it from the ModelOpt `ignore` list; a read-only [config overlay](overrides/glm53-r27-config.json) adds `model.layers.78*` so the draft backend can load it. GPU memory utilization is `0.95` as requested; the maximum context is 515,000 tokens. It serves `GLM-5.3-NVFP4` on port `8104`, mounts the local checkpoint at `/data/models/local-inference-lab/GLM-5.3-NVFP4`, and uses isolated writable JIT and temporary directories under `/home/clay/research/lil-fleet/.cache/glm53-r27`. The profile uses host networking, host IPC, all eight GPUs, Docker privileged mode, a two-minute stop timeout, and Fleet readiness checks at `/health`. The image pins [joninco/b12x commit `686450b7`](https://github.com/joninco/b12x/commit/686450b72a5665ebeeb571ed5c1729b190a829da).

The GLM 5.3 Flash profile pins the Karmic Kraken beta image by immutable GHCR digest. The true-TP6 GLM profile is an explicit compatibility exception: it pins the documented Eldritch head-padding runtime (including its matching bundled B12X), because current B12X cannot be overlaid on that older vLLM API and the current Karmic runtime has dropped the virtual-TP padding layer.

### DeepSeek and MiMo isolation

The DeepSeek and MiMo profiles use private IPC namespaces with 32 GiB of `/dev/shm`;
their workers share memory within the container without joining host IPC. MiMo
Flash RL and Pro use Docker's default seccomp filter. DeepSeek's disk-backed
Engram reader and MiMo Flash MOPD's bounce loader need `io_uring_setup`,
`io_uring_enter`, and `io_uring_register`, so they use
[`overrides/seccomp-deepseek-io-uring.json`](overrides/seccomp-deepseek-io-uring.json):
the [Docker 29.4.1 default profile](https://github.com/moby/moby/blob/docker-v29.4.1/vendor/github.com/moby/profiles/seccomp/default.json)
with only those three syscalls added. No extra capabilities are granted.
The upstream policy is Copyright The Moby Authors, under the
[Apache 2.0 license](overrides/LICENSE.seccomp); the added rule carries the
modification notice. A regression test checks the rest of the policy against
the pinned upstream JSON checksum.

Fleet resolves a relative `seccomp=` path against the manifest's directory when
it loads the manifest, and rejects the manifest if the file is missing, so the
profile no longer depends on the controller's working directory. The Docker CLI
reads the profile client-side, so when Fleet runs in a container the profile must
be mounted at the same path as the manifest's `overrides/` directory (see
[Running the controller in Docker](#running-the-controller-in-docker)). Fleet
fingerprints the profile's contents, so editing it recreates the container on
the next load. Running containers are not updated until they are unloaded and
loaded again.
Rebase the custom profile when upgrading Docker so it receives future default
policy improvements.

This is defense in depth, not VM isolation: GPU access, host networking, existing
mounts, and memory limits remain unchanged. In particular, Docker blocks
[`io_uring` by default for security reasons](https://docs.docker.com/engine/security/seccomp/).
The io_uring exception retains that kernel attack surface and cannot restrict
individual io_uring operations to reads through ordinary seccomp argument rules.
Keep the host kernel and NVIDIA driver patched; do not reuse this exception for
models that work with the default filter.

### Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness |
| `GET` | `/readyz` | Docker backend availability |
| `GET` | `/v1/models` | Safe manifest metadata plus status |
| `GET` | `/v1/models/{id}` | One configured model plus status |
| `POST` | `/v1/models/{id}/load` | Load or switch to a configured model |
| `POST` | `/v1/models/{id}/unload` | Stop/remove a configured model |
| `POST` | `/v1/deployments/switch` | Switch using `{"model_id": "..."}` |
| `GET` | `/v1/deployments` | All deployment states |
| `GET` | `/v1/deployments/{id}` | One deployment state |
| `GET` | `/v1/operations/{id}` | Async lifecycle operation state |

Full schemas are in [`openapi.yaml`](openapi.yaml).

## Manifest boundary

`fleet.json` is the administrative interface. The API never writes it. Fleet checks the file once per second and atomically applies each valid change while retaining the last valid configuration if a write is incomplete or invalid. Model additions and changes to unloaded models are live; removing or changing an active model is retried after that model is unloaded. Changes to `api.listen` or `runtime.docker_binary` still require a controller restart. The file supports:

- controller listen address and Docker binary;
- reconciliation, command, readiness, stop (`stop_timeout`), and image pull (`pull_timeout`) timeouts;
- an optional inclusive `runtime.model_port_range` that requires every model command,
  published host port, and loopback HTTP readiness probe to use the same port inside
  that range;
- model ID, description, container image, command, and validated Docker restart policy;
- environment, bind mounts (optionally `per_model`), ports, static GPU request or topology-aware placement, entrypoint, network/IPC mode, security options, `cap_add`, `user`, `read_only`, ulimits, shared memory, and optional Docker memory limits;
- an optional HTTP readiness probe.

Set a model's `memory_limit` to pass Docker `--memory`; set `memory_swap_limit`
with it to pass Docker `--memory-swap`. Use equal values such as
`"memory_limit": "40g"` and `"memory_swap_limit": "40g"` when a workload should
be killed inside its own container instead of growing into host swap. Docker's
`-1` swap value is accepted for unlimited swap, but `memory_swap_limit` cannot be
set without `memory_limit`, and finite swap must be at least the memory limit.

Profiles with `network_mode: "host"` skip Docker port publishing, so whatever address the server binds is reachable from every network the host joins, without authentication. Fleet therefore rejects a host-networked profile unless it binds loopback explicitly: every `--host`/`--host=` argument and any `HOST` environment value must be `127.0.0.0/8`, `::1`, or `localhost`, and at least one of them must be present, because vLLM, lil-serve, and the GLM serve scripts all default to `0.0.0.0`. The B12X profiles pass `--host 127.0.0.1` (or `HOST=127.0.0.1` for `glm53-tp8`, whose entrypoint reads `HOST`). Local consumers such as the llmconduit worker, including a rootless container that reaches host loopback through `slirp4netns:allow_host_loopback`, are unaffected. Bridge-networked profiles such as those in `fleet.example.json` keep `--host 0.0.0.0` inside the container and publish only to `127.0.0.1` by default.

### Container hardening

Fleet starts every non-privileged model container with `--cap-drop ALL` and
`--security-opt no-new-privileges=true`. GPU access is unaffected because the
NVIDIA runtime hook configures devices outside the container's capability set.
Profiles opt back in narrowly:

- `cap_add`: individual Linux capabilities (for example `["DAC_OVERRIDE"]`);
  `ALL` is rejected, so use `privileged` when a profile really needs everything;
- `user`: run as a uid/name with an optional `:group`;
- `read_only`: mount the root filesystem read-only with a private `/tmp` tmpfs;
- a `security_opt` entry such as `no-new-privileges=false` replaces the default.

`privileged: true` profiles (such as `glm53-tp8`) receive none of these defaults.
Without capabilities, container root obeys ordinary file permissions, so a
writable mount must be owned by the container user or the profile must add
`DAC_OVERRIDE`. The B12X cache root `/mnt/llm_stuff/cache` is owner-only and
owned by the host user, so its non-privileged profiles add only `DAC_OVERRIDE`.

A writable mount with `"per_model": true` binds `<source>/<model-id>` instead of
the shared source. The container target is unchanged, so environment paths such
as `VLLM_CACHE_ROOT=/cache/mimo-v26-pro` and `B12X_COMPILE_CACHE_DIR=/cache/b12x`
keep working while each profile gets its own cache tree (replicas share their
profile's directory). Fleet creates the subdirectory with mode `0700` when it can
see the parent path; otherwise create it on the Docker host before loading. The
B12X manifest marks every `/mnt/llm_stuff/cache` mount `per_model`, so the first
load of each profile after upgrading starts with a cold compile/tuning cache
unless you seed it, for example
`sudo cp -a /mnt/llm_stuff/cache/b12x /mnt/llm_stuff/cache/<model-id>/`.

Container fingerprints cover the profile, these hardening defaults, the contents
of referenced seccomp profiles, and the local image ID. A container whose
fingerprint changed is recreated the next time it is started; running containers
are left alone until they are unloaded.

Profiles that Fleet can co-schedule must use distinct base ports. Mutually exclusive profiles may share a port; TP8 Pro shares `8109` with TP4 Flash because it requires all GPUs. Replicas use the same inclusive `runtime.model_port_range` as a dynamic port pool, and the B12X manifest restricts model listeners to `8101` through `8121`; keep the surrounding host and container firewall rules synchronized with that range.

Environment and commands are deliberately omitted from API responses so secrets and privileged launch arguments do not leak. Docker commands use direct argument execution rather than a shell, and model IDs are validated before they can contribute to deterministic container names.

## Deployment states

- `unloaded`: no container exists, or it is cleanly stopped (including exit 137/143 after a requested stop);
- `loading`: the container exists but its readiness probe has not succeeded;
- `ready`: Docker is running and the manifest probe succeeded;
- `unhealthy`: Docker or the application probe reports unhealthy;
- `stopping`: an unload is running;
- `failed`: start, stop, unexpected exit, timeout, or OOM failure;
- `unknown`: Docker could not be inspected.

The controller reconciles Docker state on startup and at `runtime.poll_interval` with one `docker ps` plus one `docker container inspect` for all managed containers, then probes readiness in parallel (2s per probe). Status is not based only on API intent. An observation made while a lifecycle operation changed the model, or while an operation owns it, is discarded instead of overwriting newer state. Deployment status includes aggregate compatibility fields plus `desired_instances`, `ready_instances`, and per-instance `instances[]` with the concrete instance ID, index, port, phase, GPU assignment, and health. Managed containers are named `lil-fleet-{model-id}` for the first instance and `lil-fleet-{model-id}--N` for replicas, and each is labeled with the profile model and a manifest fingerprint. A changed manifest causes a stale stopped or running container to be replaced on its next load.

## Security notes

Docker daemon access is effectively host administration. Keep this API on loopback or a protected administrative network, always with `-token-file` when it is reachable by anyone else. The current API is a control plane only; inference traffic goes directly to each manifest-defined port, which must stay on loopback (host networking) or a loopback-published port. Model environment values reach `docker create` through the CLI's environment rather than argv (except keys the Docker CLI itself reads, such as `DOCKER_*`, `HOME`, `PATH`, and proxy variables). Docker and runtime error text is stripped of control characters, capped at 1 KiB, and redacted of manifest environment values before it appears in operations, status, or logs.

Run all checks with:

```sh
make check
```
