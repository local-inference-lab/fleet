from pathlib import Path


warmup = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/warmup/qwen_triton_warmup.py"
)
source = warmup.read_text()
replacements = (
    (
        "    feature_size = int(config.hv * config.v)",
        "    # Match the loaded norm width; per-head weights are shared across heads.\n"
        "    feature_size = int(config.norm_weight.numel())\n"
        "    rows_per_token = int(config.hv * config.v) // feature_size",
    ),
    (
        "        x = torch.empty((length, feature_size), dtype=config.conv_dtype, device=device)",
        "        x = torch.empty(\n"
        "            (length * rows_per_token, feature_size),\n"
        "            dtype=config.conv_dtype,\n"
        "            device=device,\n"
        "        )",
    ),
)
for original, replacement in replacements:
    if source.count(original) != 1:
        raise SystemExit("unexpected Qwen GDN layer norm warmup implementation")
    source = source.replace(original, replacement)

warmup.write_text(source)
