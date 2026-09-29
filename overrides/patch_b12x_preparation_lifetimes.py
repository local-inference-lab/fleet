from pathlib import Path


backend = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/b12x.py"
)
source = backend.read_text()
original = "            owners=(scratch, q, output, page_table, cache_seqlens, cu_seqlens_q),"
replacement = """            # Priming is synchronous; serving binds fresh workspace tensors.
            # Only benchmark calls need to retain their synthetic inputs.
            owners=(scratch, q, output, page_table, cache_seqlens, cu_seqlens_q)
            if benchmark
            else (),"""

if source.count(original) != 1:
    raise SystemExit("unexpected B12X attention preparation lifetime implementation")

backend.write_text(source.replace(original, replacement))
