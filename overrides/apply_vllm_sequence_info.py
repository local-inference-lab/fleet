#!/usr/bin/env python3
"""Opt-in lightweight vLLM sequence-capacity endpoint patch.

This intentionally adds only a read-only `/server_info` route in normal serve
mode when LIL_VLLM_SEQUENCE_INFO=1. It does not enable vLLM dev routes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


RELATIVE_TARGET = Path("entrypoints/serve/__init__.py")
MARKER = "# LIL_VLLM_SEQUENCE_INFO lightweight endpoint"

ORIGINAL_BLOCK = '''def register_vllm_serve_api_routers(app: FastAPI):
    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)
'''

PATCHED_BLOCK = '''def register_vllm_serve_api_routers(app: FastAPI):
    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)

    # LIL_VLLM_SEQUENCE_INFO lightweight endpoint
    import os
    from vllm import envs
    if (not envs.VLLM_SERVER_DEV_MODE and
            os.environ.get("LIL_VLLM_SEQUENCE_INFO", "").lower() in {"1", "true", "yes"}):
        @app.get("/server_info")
        async def lil_vllm_sequence_info(config_format: str = "json"):
            scheduler_config = app.state.vllm_config.scheduler_config
            return {
                "vllm_config": {
                    "scheduler_config": {
                        "max_num_seqs": scheduler_config.max_num_seqs,
                    },
                },
            }
'''


def target_path(root: Path) -> Path:
    return root / RELATIVE_TARGET


def patch_source(text: str) -> str:
    if MARKER in text:
        if PATCHED_BLOCK not in text:
            raise ValueError("source contains marker but not the expected patched block")
        return text
    if text.count(ORIGINAL_BLOCK) != 1:
        raise ValueError("expected exact vLLM serve router block once")
    return text.replace(ORIGINAL_BLOCK, PATCHED_BLOCK)


def apply(root: Path) -> bool:
    path = target_path(root)
    text = path.read_text()
    patched = patch_source(text)
    if patched == text:
        return False
    path.write_text(patched)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    changed = apply(args.vllm_root)
    print(f"{target_path(args.vllm_root)}: {'patched' if changed else 'already patched'}")


if __name__ == "__main__":
    main()
