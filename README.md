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

`/healthz` and `/readyz` remain unauthenticated for local probes. All `/v1/*` routes require the token when `-token-file` is supplied.

The example model images are illustrative floating tags. For a reproducible deployment, replace them with image digests, pin model revisions in each command, and use host paths that exist on the Docker daemon host.

The included Compose service can run the controller in Docker, but mounting the Docker socket gives it host-level control. Its manifest is mounted read-only:

```sh
docker compose up -d --build
```

When using Compose, set `api.listen` to `0.0.0.0:8090` inside `fleet.json`. Bind the published API port to loopback or put authentication and TLS in front of it.

## API

List the configured profiles and their current state:

```sh
curl http://127.0.0.1:8090/v1/models
```

Load a model. By default this is an exclusive switch. When `runtime.concurrent_deployments` is enabled, Fleet allocates free GPUs from the manifest topology and keeps compatible deployments resident.

```sh
curl -i -X POST http://127.0.0.1:8090/v1/models/qwen3-8b/load
```

The response is `202 Accepted`, includes an operation, and sets a `Location` header. Poll that operation and the deployment:

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

Lifecycle calls are idempotent. A duplicate in-flight call returns the same operation; a conflicting call returns `409 Conflict`. The controller permits only one lifecycle operation at a time so two GPU-heavy profiles cannot race into memory.

## Patched B12X fleet

[`fleet.b12x.json`](fleet.b12x.json) contains example RTX PRO 6000 profiles for an eight-GPU system with two PCIe groups. Copy it to the ignored `fleet.json`, then adjust the example topology groups `[0,1,6,7]` and `[2,3,4,5]` and `/srv/fleet` mount paths for the Docker daemon host. Build its pinned runtime overrides first:

```sh
docker build -f Dockerfile.b12x-override \
  --build-arg B12X_REF=f20ab3bad7def65f6211403fb73f9b1e8a33dfb0 \
  -t lil-fleet/b12x:f20ab3bad7de .
docker build -f Dockerfile.mimo-b12x-override \
  -t lil-fleet/mimo-v26-flash:b12x-mixed-device .
cp fleet.b12x.json fleet.json
# Adjust fleet.json for the host before starting the controller.
go run ./cmd/lil-fleet -manifest fleet.json -token-file .secrets/api-token
```

TP1 through TP4 stay within one PCIe group; TP6 takes one complete group plus two GPUs from the other; TP8 requires all GPUs. Live `nvidia-smi` memory use prevents Fleet from allocating GPUs occupied by workloads it did not create. A placement that cannot fit returns `409 Conflict` without creating a container.

DeepSeek uses disk-backed Engram tables. Qwen Flash Next TP2/TP3 use disk-backed PLE tables; TP4 disables PLE CPU offload so its tables remain on GPU. GLM 5.3 TP6 and Qwen Flash Next TP3 are labeled experimental because upstream has not published qualified recipes for those shapes.

The `mimo-v26-flash-tp4` profile serves `XiaomiMiMo/MiMo-V2.6-Flash-RL` from the example path `/srv/fleet/models/MiMo-V2.6-Flash-RL` with DFlash speculative decoding. Its TP4 placement takes one complete PCIe group, depending on availability; the fixed `CUDA_VISIBLE_DEVICES=0,1,2,3` refers to the four logical devices exposed inside the allocated container. For groups mixing Max-Q and standard RTX PRO 6000 cards, the model-specific image adds an opt-in `B12X_TUNING_DEVICE_CLASS` identity. This lets B12X agree on one SM120A tuning selection across ranks while leaving its executable caches bound to each physical GPU. The multimodal encoder uses Triton attention explicitly for compatibility with drivers that cannot execute the image's newer FlashAttention PTX.

The `mimo-v26-pro-tp8` profile serves `XiaomiMiMo/MiMo-V2.6-Pro-RL` across all eight GPUs while reusing the MiMo Flash B12X image. Download the official Pro checkpoint at revision `73875d00b30a89ef8cc353a0b60b0e9f9561952d` into `/srv/fleet/models/MiMo-V2.6-Pro-RL`, and keep the DFlash draft under that same checkpoint at `/srv/fleet/models/MiMo-V2.6-Pro-RL/dflash` so the container sees it as `/model/dflash`. The Pro profile shares port `8109` with Flash only because TP8 reserves every GPU and cannot run concurrently with the TP4 Flash profile; do not reuse a host port for profiles that Fleet can co-schedule.

Pro intentionally uses vLLM's standard lazy safetensors loader because this image's B12X direct loader rejects Pro audio-encoder weights. B12X remains the execution backend for collectives and kernels.

The main image is the current Karmic release with B12X pinned from source. The true-TP6 GLM profile is an explicit compatibility exception: it pins the documented Eldritch head-padding runtime (including its matching bundled B12X), because current B12X cannot be overlaid on that older vLLM API and the current Karmic runtime has dropped the virtual-TP padding layer.

### DeepSeek and MiMo isolation

The DeepSeek and MiMo profiles use private IPC namespaces with 32 GiB of `/dev/shm`;
their workers share memory within the container without joining host IPC. MiMo
uses Docker's default seccomp filter. DeepSeek's disk-backed Engram reader needs
`io_uring_setup`, `io_uring_enter`, and `io_uring_register`, so it uses
[`overrides/seccomp-deepseek-io-uring.json`](overrides/seccomp-deepseek-io-uring.json):
the [Docker 29.4.1 default profile](https://github.com/moby/moby/blob/docker-v29.4.1/vendor/github.com/moby/profiles/seccomp/default.json)
with only those three syscalls added. No extra capabilities are granted.
The upstream policy is Copyright The Moby Authors, under the
[Apache 2.0 license](overrides/LICENSE.seccomp); the added rule carries the
modification notice. A regression test checks the rest of the policy against
the pinned upstream JSON checksum.

Run Fleet from the repository root (including the service's `WorkingDirectory`)
so Docker's CLI can read that relative profile path. For another working
directory, use an absolute `seccomp=` path available to the Docker CLI. Manifest
changes require a controller restart and a model unload/load to recreate the
container; restarting the controller alone does not update running containers.
Rebase the custom profile when upgrading Docker so it receives future default
policy improvements.

This is defense in depth, not VM isolation: GPU access, host networking, existing
mounts, and memory limits remain unchanged. In particular, Docker blocks
[`io_uring` by default for security reasons](https://docs.docker.com/engine/security/seccomp/).
The DeepSeek exception retains that kernel attack surface and cannot restrict
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
- reconciliation, command, and readiness timeouts;
- an optional inclusive `runtime.model_port_range` that requires every model command,
  published host port, and loopback HTTP readiness probe to use the same port inside
  that range;
- model ID, description, container image, command, and validated Docker restart policy;
- environment, bind mounts, ports, static GPU request or topology-aware placement, entrypoint, network/IPC mode, security options, ulimits, and shared memory;
- an optional HTTP readiness probe.

Exclusive profiles may share a host inference port. Profiles that Fleet can co-schedule must use distinct ports; the B12X Pro profile may share Flash's port only because it reserves all eight GPUs. The B12X manifest restricts model listeners to `8101` through `8109`; keep the surrounding host and container firewall rules synchronized with that range.

Environment and commands are deliberately omitted from API responses so secrets and privileged launch arguments do not leak. Docker commands use direct argument execution rather than a shell, and model IDs are validated before they can contribute to deterministic container names.

## Deployment states

- `unloaded`: no container exists, or it is cleanly stopped;
- `loading`: the container exists but its readiness probe has not succeeded;
- `ready`: Docker is running and the manifest probe succeeded;
- `unhealthy`: Docker or the application probe reports unhealthy;
- `stopping`: an unload is running;
- `failed`: start, stop, exit, timeout, or OOM failure;
- `unknown`: Docker could not be inspected.

The controller reconciles Docker state on startup and at `runtime.poll_interval`; status is not based only on API intent. Managed containers are named `lil-fleet-{model-id}` and labeled with the model and a manifest fingerprint. A changed manifest causes a stale stopped or running container to be replaced on its next load.

## Security notes

Docker daemon access is effectively host administration. Keep this API on loopback or a protected administrative network. Add authentication at a reverse proxy before exposing it to other users. The current API is a control plane only; inference traffic goes directly to each manifest-defined published port.

Run all checks with:

```sh
make check
```
