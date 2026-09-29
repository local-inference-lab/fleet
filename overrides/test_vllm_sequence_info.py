#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent
PATCHER_PATH = ARTIFACT_DIR / "apply_vllm_sequence_info.py"

spec = importlib.util.spec_from_file_location("apply_vllm_sequence_info", PATCHER_PATH)
patcher = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(patcher)


class FakeRoute:
    def __init__(self, path: str, endpoint):
        self.path = path
        self.endpoint = endpoint


class FakeApp:
    def __init__(self, max_num_seqs: int = 32):
        scheduler = types.SimpleNamespace(max_num_seqs=max_num_seqs)
        self.state = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(scheduler_config=scheduler)
        )
        self.routes: list[FakeRoute] = []

    def get(self, path: str):
        def decorator(func):
            self.routes.append(FakeRoute(path, func))
            return func

        return decorator


def install_fake_modules(dev_mode: bool, enabled: bool) -> None:
    vllm_mod = types.ModuleType("vllm")
    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_SERVER_DEV_MODE = dev_mode

    os.environ["LIL_VLLM_SEQUENCE_INFO"] = "1" if enabled else "0"
    vllm_mod.envs = envs_mod

    instrumentator = types.ModuleType("vllm.entrypoints.serve.instrumentator")
    instrumentator.register_instrumentator_api_routers = lambda app: None

    lora_router = types.ModuleType("vllm.entrypoints.serve.lora.api_router")
    profile_router = types.ModuleType("vllm.entrypoints.serve.profile.api_router")
    tokenize_router = types.ModuleType("vllm.entrypoints.serve.tokenize.api_router")
    for module in (lora_router, profile_router, tokenize_router):
        module.attach_router = lambda app: None

    modules = {
        "vllm": vllm_mod,
        "vllm.envs": envs_mod,
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.serve": types.ModuleType("vllm.entrypoints.serve"),
        "vllm.entrypoints.serve.instrumentator": instrumentator,
        "vllm.entrypoints.serve.lora": types.ModuleType(
            "vllm.entrypoints.serve.lora"
        ),
        "vllm.entrypoints.serve.lora.api_router": lora_router,
        "vllm.entrypoints.serve.profile": types.ModuleType(
            "vllm.entrypoints.serve.profile"
        ),
        "vllm.entrypoints.serve.profile.api_router": profile_router,
        "vllm.entrypoints.serve.tokenize": types.ModuleType(
            "vllm.entrypoints.serve.tokenize"
        ),
        "vllm.entrypoints.serve.tokenize.api_router": tokenize_router,
    }
    sys.modules.update(modules)


def exec_patched_register(dev_mode: bool, enabled: bool, max_num_seqs: int = 32):
    install_fake_modules(dev_mode=dev_mode, enabled=enabled)
    namespace = {
        "__name__": "vllm.entrypoints.serve",
        "__package__": "vllm.entrypoints.serve",
    }
    exec(patcher.PATCHED_BLOCK, namespace)
    app = FakeApp(max_num_seqs=max_num_seqs)
    namespace["register_vllm_serve_api_routers"](app)
    return app


class VllmSequenceInfoPatchTest(unittest.IsolatedAsyncioTestCase):
    def test_patch_is_idempotent(self):
        once = patcher.patch_source(patcher.ORIGINAL_BLOCK)
        twice = patcher.patch_source(once)
        self.assertEqual(once, twice)

    def test_malformed_source_rejects(self):
        with self.assertRaisesRegex(ValueError, "expected exact"):
            patcher.patch_source("def register_vllm_serve_api_routers(app):\n    pass\n")

    def test_default_env_false_adds_no_route(self):
        app = exec_patched_register(dev_mode=False, enabled=False)
        self.assertEqual([], [route.path for route in app.routes])

    def test_dev_mode_skips_lightweight_route(self):
        app = exec_patched_register(dev_mode=True, enabled=True)
        self.assertEqual([], [route.path for route in app.routes])

    async def test_enabled_route_reports_only_max_num_seqs(self):
        app = exec_patched_register(dev_mode=False, enabled=True, max_num_seqs=77)
        self.assertEqual(["/server_info"], [route.path for route in app.routes])

        body = await app.routes[0].endpoint(config_format="json")

        self.assertEqual(
            {"vllm_config": {"scheduler_config": {"max_num_seqs": 77}}},
            body,
        )
        rendered = repr(body).lower()
        for forbidden in ("api", "key", "secret", "token", "env", "system", "sleep"):
            self.assertNotIn(forbidden, rendered)

    def test_no_mutating_dev_routes_are_registered(self):
        app = exec_patched_register(dev_mode=False, enabled=True)
        self.assertEqual(["/server_info"], [route.path for route in app.routes])
        self.assertNotIn("/sleep", [route.path for route in app.routes])
        self.assertNotIn("/reset_prefix_cache", [route.path for route in app.routes])

    def test_apply_writes_expected_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / patcher.RELATIVE_TARGET
            target.parent.mkdir(parents=True)
            target.write_text(patcher.ORIGINAL_BLOCK)

            self.assertTrue(patcher.apply(root))
            self.assertIn(patcher.MARKER, target.read_text())
            self.assertFalse(patcher.apply(root))


if __name__ == "__main__":
    unittest.main()
