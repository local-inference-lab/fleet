"""Run Step 5 Preview NVFP4 on SM120 with B12X MoE and padded TP6 attention.

Applies to the vllm/vllm-openai:step-5-preview image. Every replacement must
match exactly once, so an image upgrade fails the build instead of silently
skipping a hunk.

* Step 5 DSA accepts SM120 and scores the sparse indexer with the portable
  Triton kernel already used on Hopper; FlashInfer's fp8_paged_mqa_logits
  rejects compute capability 12.0.
* B12X MoE runs Step 5's clamped SwiGLU (swiglustep, limit 7) as SiLU with
  swiglu_limit=7. B12X clamps the gate before SiLU and Step 5 clamps after; the
  two differ only for gate > 7 (6.9936 vs 7.0).
* Unquantized MoE layers (Step 5 keeps layers 88-90 in BF16) fall back to the
  automatic backend when --moe-backend names a quantization-only B12X backend.
* `text_config.head_padding` declares the checkpoint's original attention
  geometry while the served config declares a padded one. Q heads are padded
  with zero heads to fill virtual KV groups; each KV head and its sparse
  indexer provider group are replicated across the virtual groups split from
  it. Zero Q/gate rows and zero o_proj columns contribute nothing, and
  replicated indexer heads select the same regions, so outputs are unchanged.
  The MTP draft config receives the same text_config override.
* VLLM_PCIE_IPC_MAX_TOKENS caps FlashInfer's PCIe IPC all-reduce at the batch
  sizes where it beats NCCL.
* FLASHINFER_AUTOTUNE_GPU_NAME lets mixed Max-Q and Workstation ranks share
  FlashInfer tuning caches.
* VLLM_STEP5_FUSE_SWA_GATE=1 enables the fused sliding-window decode and
  head-wise gate path, which upstream limits to SM103.
* Step 5's FP32 router GEMM kernel runs on SM120 instead of the 4x slower
  F.linear fallback.
* VLLM_STEP5_MOE_REDUCE_BF16=1 all-reduces the combined MoE output in BF16
  instead of FP32, as attention o_proj already does.
* Vocabulary embeddings widen their padding to a multiple of 64 * TP when the
  64-padded vocabulary would not shard evenly (128896 tokens at TP6).
"""

from pathlib import Path


root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
step5 = root / "models/step5/nvidia"


def patch(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    source = path.read_text()
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise SystemExit(f"unexpected Step 5 source in {path}: {original[:80]!r}")
        source = source.replace(original, replacement)
    path.write_text(source)


patch(
    step5 / "sparse_attention.py",
    (
        (
            "        return capability in (DeviceCapability(9, 0), DeviceCapability(10, 3))",
            "        return capability in (\n"
            "            DeviceCapability(9, 0),\n"
            "            DeviceCapability(10, 3),\n"
            "            DeviceCapability(12, 0),\n"
            "        )",
        ),
    ),
)

patch(
    step5 / "flashinfer_sparse.py",
    (
        (
            "        if torch.cuda.get_device_capability(device) == (9, 0):",
            "        if torch.cuda.get_device_capability(device) in ((9, 0), (12, 0)):",
        ),
    ),
)

patch(
    root / "model_executor/layers/fused_moe/b12x.py",
    (
        (
            "def _b12x_activation_name(activation: MoEActivation) -> str:\n"
            "    if activation == MoEActivation.SILU:\n",
            "def _b12x_activation_name(activation: MoEActivation) -> str:\n"
            "    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP):\n",
        ),
        (
            "        limit = self.quant_config.gemm1_clamp_limit\n"
            "        if limit is None:\n"
            "            limit = self.moe_config.swiglu_limit\n",
            "        limit = self.quant_config.gemm1_clamp_limit\n"
            "        if limit is None:\n"
            "            limit = self.moe_config.swiglu_limit\n"
            "        if limit is None and activation == MoEActivation.SWIGLUSTEP:\n"
            "            limit = 7.0\n",
        ),
        (
            "        return activation in (\n"
            "            MoEActivation.SILU,\n"
            "            MoEActivation.SITU,\n"
            "            MoEActivation.SWIGLUOAI_UNINTERLEAVE,\n",
            "        return activation in (\n"
            "            MoEActivation.SILU,\n"
            "            MoEActivation.SWIGLUSTEP,\n"
            "            MoEActivation.SITU,\n"
            "            MoEActivation.SWIGLUOAI_UNINTERLEAVE,\n",
        ),
    ),
)

patch(
    root / "model_executor/layers/fused_moe/oracle/unquantized.py",
    (
        (
            '    if runner_backend not in ["auto", "humming"]:',
            '    if runner_backend not in ["auto", "humming", "b12x", "flashinfer_b12x"]:',
        ),
    ),
)

PAD_HELPERS = '''

_STEP5_PADDED_ATTENTION_WEIGHT = re.compile(
    r"(?:^|\\.)layers\\.\\d+\\.self_attn\\."
    r"(q_proj\\.weight|g_proj\\.weight|ssmax_s|o_proj\\.weight|k_proj\\.weight|"
    r"v_proj\\.weight|sparse_indexer_q\\.weight|sparse_indexer_w\\.weight)$"
)


def _step5_head_padding(config: Any) -> dict[str, int] | None:
    """Return the checkpoint-to-served attention geometry, if padded."""
    padding = getattr(config, "head_padding", None)
    if not padding:
        return None
    source_heads = int(padding["num_attention_heads"])
    source_groups = int(padding["num_attention_groups"])
    source_index_heads = int(padding["sparse_indexer_num_heads"])
    heads = int(config.num_attention_heads)
    groups = int(config.num_attention_groups)
    other = getattr(config, "attention_other_setting", None) or {}
    if other and (
        int(other["num_attention_heads"]) != heads
        or int(other["num_attention_groups"]) != groups
    ):
        raise ValueError("Step5 head padding requires uniform attention geometry.")
    if (
        source_groups <= 0
        or source_heads % source_groups
        or groups % source_groups
        or heads % groups
    ):
        raise ValueError(
            "Step5 head padding cannot map "
            f"{source_heads} heads in {source_groups} groups to "
            f"{heads} heads in {groups} groups."
        )
    replicas = groups // source_groups
    heads_per_group = heads // groups
    source_heads_per_group = source_heads // source_groups
    if heads_per_group * replicas < source_heads_per_group:
        raise ValueError(
            f"Step5 head padding leaves no room for {source_heads_per_group} "
            f"heads in {replicas} groups of {heads_per_group}."
        )
    sparse_config = get_step5_sparse_config(config)
    if sparse_config is not None and (
        int(sparse_config.sparse_indexer_num_heads) != source_index_heads * replicas
        or int(sparse_config.index_tp_size) != groups
    ):
        raise ValueError(
            "Step5 head padding must replicate each sparse indexer provider "
            f"group {replicas} times across {groups} provider groups."
        )
    return {
        "source_groups": source_groups,
        "source_heads_per_group": source_heads_per_group,
        "source_index_heads_per_group": source_index_heads // source_groups,
        "replicas": replicas,
        "padded_heads_per_source_group": heads_per_group * replicas,
    }


def _step5_pad_heads(
    tensor: torch.Tensor, dim: int, groups: int, heads: int, padded: int
) -> torch.Tensor:
    moved = tensor.movedim(dim, 0)
    grouped = moved.reshape(groups, heads, -1, *moved.shape[1:])
    result = grouped.new_zeros(groups, padded, *grouped.shape[2:])
    result[:, :heads] = grouped
    return result.reshape(-1, *moved.shape[1:]).movedim(0, dim)


def _step5_replicate_groups(
    tensor: torch.Tensor, groups: int, replicas: int
) -> torch.Tensor:
    grouped = tensor.reshape(groups, -1, *tensor.shape[1:])
    return grouped.repeat_interleave(replicas, dim=0).reshape(-1, *tensor.shape[1:])


def pad_step5_attention_weight(
    config: Any, name: str, tensor: torch.Tensor
) -> torch.Tensor:
    """Expand a checkpoint attention tensor to the served padded geometry."""
    match = _STEP5_PADDED_ATTENTION_WEIGHT.search(name)
    padding = _step5_head_padding(config) if match is not None else None
    if padding is None:
        return tensor
    kind = match.group(1)
    groups = padding["source_groups"]
    if kind in (
        "k_proj.weight",
        "v_proj.weight",
        "sparse_indexer_q.weight",
        "sparse_indexer_w.weight",
    ):
        return _step5_replicate_groups(tensor, groups, padding["replicas"])
    return _step5_pad_heads(
        tensor,
        1 if kind == "o_proj.weight" else 0,
        groups,
        padding["source_heads_per_group"],
        padding["padded_heads_per_source_group"],
    )
'''

patch(
    step5 / "model.py",
    (
        (
            "def _step5_groups_per_rank(total_groups: int, tp_size: int) -> int:",
            PAD_HELPERS.lstrip("\n")
            + "\n\ndef _step5_groups_per_rank(total_groups: int, tp_size: int) -> int:",
        ),
        ("import copy\n", "import copy\nimport os\n"),
        (
            "            combined = (\n"
            "                moe_output\n"
            "                if shared_output is None\n"
            "                else self._add_for_residual(moe_output, shared_output)\n"
            "            )\n",
            "            if shared_output is None:\n"
            "                combined = moe_output\n"
            "            elif (\n"
            "                self.moe.fuse_all_reduce\n"
            "                and os.environ.get(\"VLLM_STEP5_MOE_REDUCE_BF16\") == \"1\"\n"
            "            ):\n"
            "                # All-reduce the MoE contribution in BF16 like attention\n"
            "                # o_proj; the FP32 residual add follows unchanged.\n"
            "                combined = moe_output + shared_output\n"
            "            else:\n"
            "                combined = self._add_for_residual(moe_output, shared_output)\n",
        ),
        (
            "            and current_platform.is_device_capability(103)\n"
            "            and typing.cast(Any, self.attn.impl).supports_headwise_gate()\n",
            "            and (\n"
            "                current_platform.is_device_capability(103)\n"
            "                or os.environ.get(\"VLLM_STEP5_FUSE_SWA_GATE\") == \"1\"\n"
            "            )\n"
            "            and typing.cast(Any, self.attn.impl).supports_headwise_gate()\n",
        ),
        (
            "        def _load_weight(name: str, loaded_weight: torch.Tensor) -> set[str]:\n"
            "            loaded_params: set[str] = set()\n",
            "        def _load_weight(name: str, loaded_weight: torch.Tensor) -> set[str]:\n"
            "            loaded_params: set[str] = set()\n"
            "            loaded_weight = pad_step5_attention_weight(\n"
            "                config, name, loaded_weight\n"
            "            )\n",
        ),
    ),
)

# Step 5's 128896-token vocabulary is a multiple of 64 but not of 64 * 6.
# Only shapes that would otherwise fail to shard get the wider padding.
patch(
    root / "model_executor/layers/vocab_parallel_embedding.py",
    (
        (
            "        self.padding_size = padding_size\n",
            "        if self.tp_size > 1 and pad_vocab_size(num_embeddings, padding_size) % (\n"
            "            self.tp_size\n"
            "        ):\n"
            "            padding_size *= self.tp_size\n"
            "        self.padding_size = padding_size\n",
        ),
    ),
)

# FlashInfer's PCIe IPC all-reduce beats NCCL only for a few decode tokens on
# PCIe-only TP8 (8 us vs 17 us at one token; 300 us vs 44 us at 64 tokens), and
# its tuner never compares against NCCL. VLLM_PCIE_IPC_MAX_TOKENS sends larger
# all-reduces to the next backend.
patch(
    root / "distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py",
    (
        (
            "            and inp.shape[1] == self.hidden_dim\n",
            "            and inp.shape[1] == self.hidden_dim\n"
            "            and (\n"
            "                not _MAX_TOKENS or inp.shape[0] <= _MAX_TOKENS\n"
            "            )\n",
        ),
        (
            "from collections.abc import Sequence\n",
            "import os\nfrom collections.abc import Sequence\n",
        ),
        (
            "_SUPPORTED_WORLD_SIZES = (2, 4, 8)\n",
            "_SUPPORTED_WORLD_SIZES = (2, 4, 8)\n"
            "_MAX_TOKENS = int(os.environ.get(\"VLLM_PCIE_IPC_MAX_TOKENS\", \"0\"))\n",
        ),
    ),
)

# FlashInfer drops cached tuning recorded on a differently named GPU. Max-Q and
# Workstation RTX PRO 6000 cards share GB202 silicon, but in one TP group their
# ranks then disagree about the PCIe IPC tune cache and the workspace refuses to
# build. FLASHINFER_AUTOTUNE_GPU_NAME gives every rank one recorded identity.
patch(
    root.parent / "flashinfer/autotuner/autotuner.py",
    (
        (
            '        meta["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())\n',
            '        meta["gpu"] = os.environ.get(\n'
            '            "FLASHINFER_AUTOTUNE_GPU_NAME"\n'
            '        ) or torch.cuda.get_device_name(torch.cuda.current_device())\n',
        ),
    ),
)

# The FP32 router kernel built for Step 5's 4096 x 352 gate also runs on SM120:
# 4.6 us vs 20 us for the F.linear fallback at four tokens, with lower error
# against an FP64 reference.
patch(
    root / "model_executor/layers/fused_moe/router/gate_linear.py",
    (
        (
            "                    current_platform.is_cuda()\n"
            "                    and (is_hopper or is_blackwell)\n"
            "                    and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES\n",
            "                    current_platform.is_cuda()\n"
            "                    and (\n"
            "                        is_hopper\n"
            "                        or is_blackwell\n"
            "                        or current_platform.is_device_capability((12, 0))\n"
            "                    )\n"
            "                    and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES\n",
        ),
    ),
)

# Dict hf_overrides normally stay on the target. The MTP draft is built from
# the flattened text config and must see the same padded geometry, both for
# vLLM's TP head checks and for padding its own attention weights.
patch(
    root / "config/speculative.py",
    (
        (
            "    @staticmethod\n"
            "    def compose_draft_hf_overrides(\n",
            "    @staticmethod\n"
            "    def _apply_step5_head_padding_override(\n"
            "        text_overrides: dict[str, Any],\n"
            "        hf_config: PreTrainedConfig,\n"
            "    ) -> PreTrainedConfig:\n"
            "        hf_config = SpeculativeConfig.hf_config_override(hf_config)\n"
            "\n"
            "        def merge(target: Any, updates: dict[str, Any]) -> None:\n"
            "            for key, value in updates.items():\n"
            "                current = (\n"
            "                    target.get(key)\n"
            "                    if isinstance(target, dict)\n"
            "                    else getattr(target, key, None)\n"
            "                )\n"
            "                if isinstance(value, dict) and isinstance(current, dict):\n"
            "                    merge(current, value)\n"
            "                elif isinstance(target, dict):\n"
            "                    target[key] = copy.deepcopy(value)\n"
            "                else:\n"
            "                    setattr(target, key, copy.deepcopy(value))\n"
            "\n"
            "        merge(hf_config, text_overrides)\n"
            "        return hf_config\n"
            "\n"
            "    @staticmethod\n"
            "    def compose_draft_hf_overrides(\n",
        ),
        (
            "        if not callable(target_hf_overrides):\n"
            "            return SpeculativeConfig.hf_config_override\n",
            "        text_overrides = (\n"
            "            target_hf_overrides.get(\"text_config\")\n"
            "            if isinstance(target_hf_overrides, dict)\n"
            "            else None\n"
            "        )\n"
            "        if isinstance(text_overrides, dict) and \"head_padding\" in text_overrides:\n"
            "            return functools.partial(\n"
            "                SpeculativeConfig._apply_step5_head_padding_override,\n"
            "                text_overrides,\n"
            "            )\n"
            "        if not callable(target_hf_overrides):\n"
            "            return SpeculativeConfig.hf_config_override\n",
        ),
    ),
)

patch(
    step5 / "mtp.py",
    (
        (
            "    _validate_fused_qkv_indexer_weights,\n)\n",
            "    _validate_fused_qkv_indexer_weights,\n"
            "    pad_step5_attention_weight,\n)\n",
        ),
        (
            "            if \"rotary_emb.inv_freq\" in name:\n"
            "                continue\n"
            "            if name.startswith(\"layers.\"):\n",
            "            if \"rotary_emb.inv_freq\" in name:\n"
            "                continue\n"
            "            loaded_weight = pad_step5_attention_weight(\n"
            "                self.config, name, loaded_weight\n"
            "            )\n"
            "            if name.startswith(\"layers.\"):\n",
        ),
    ),
)
