from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext


ROOT = pathlib.Path(__file__).resolve().parent
arguments = argparse.ArgumentParser(add_help=False)
arguments.add_argument("--source-root", type=pathlib.Path,
                       default=pathlib.Path("/opt/venv/lib/python3.12/site-packages/b12x"))
options, unittest_arguments = arguments.parse_known_args()
sys.argv[1:] = unittest_arguments
SOURCE_ROOT = options.source_root
if (SOURCE_ROOT / "_lib" / "program_cache.py").exists():
    SOURCE_PROGRAM = SOURCE_ROOT / "_lib" / "program_cache.py"
    SOURCE_SESSION = SOURCE_ROOT / "preparation" / "session.py"
else:
    SOURCE_PROGRAM = SOURCE_ROOT / "program_cache.py"
    SOURCE_SESSION = SOURCE_ROOT / "session.py"
FIXTURE_DIRECTORY = tempfile.TemporaryDirectory(prefix="ds41-prep-tested-source.")
FIXTURE_ROOT = pathlib.Path(FIXTURE_DIRECTORY.name)
shutil.copyfile(SOURCE_PROGRAM, FIXTURE_ROOT / "program_cache.py")
shutil.copyfile(SOURCE_SESSION, FIXTURE_ROOT / "session.py")
subprocess.run([sys.executable, str(ROOT / "apply_b12x_preparation_bound.py"),
                str(FIXTURE_ROOT)], check=True, stdout=subprocess.DEVNULL)


class EnvPatch:
    def __init__(self, **values):
        self.values = values
        self.previous = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, *_exc):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_program_cache():
    for name in (
        "b12x._lib.program_cache",
        "b12x._lib.compile_plan",
        "b12x._lib",
        "b12x",
    ):
        sys.modules.pop(name, None)

    package = types.ModuleType("b12x")
    package.__path__ = []
    lib = types.ModuleType("b12x._lib")
    lib.__path__ = []
    compile_plan = types.ModuleType("b12x._lib.compile_plan")
    compile_plan.CompiledCuTeProgram = type("CompiledCuTeProgram", (), {})
    compile_plan._RESIDENT_PROGRAMS = set()
    compile_plan.evict_unretained_triton = lambda retained: None
    compile_plan.program_keys = lambda value: set()
    compile_plan.retained_program_keys = lambda: set()

    sys.modules["b12x"] = package
    sys.modules["b12x._lib"] = lib
    sys.modules["b12x._lib.compile_plan"] = compile_plan

    spec = importlib.util.spec_from_file_location(
        "b12x._lib.program_cache", FIXTURE_ROOT / "program_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_session_subset():
    source = (FIXTURE_ROOT / "session.py").read_text()
    parsed = ast.parse(source)
    wanted = {
        "_default_compile_workers",
        "_RACE_BATCH_ENV",
        "_env_positive_int",
        "_default_race_batch",
        "PreparationSession",
    }
    body = []
    for node in parsed.body:
        name = getattr(node, "name", None)
        if name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                body.append(node)

    module_ast = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module_ast)

    class FrozenMapping(dict):
        pass

    class DetectedDevice:
        def __init__(self):
            self.identity = None

    globals_ = {
        "os": os,
        "threading": __import__("threading"),
        "Path": pathlib.Path,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "SURVIVOR_ROUNDS": 7,
        "DEFAULT_SAMPLES": 3,
        "DetectedDevice": DetectedDevice,
        "detect_device": lambda device: DetectedDevice(),
        "FrozenMapping": FrozenMapping,
    }
    exec(compile(module_ast, str(FIXTURE_ROOT / "session.py"), "exec"), globals_)
    return globals_


class PreparationProgramCacheTests(unittest.TestCase):
    def test_default_factory_cache_memoizes_results(self):
        module = load_program_cache()
        calls = []

        class Payload:
            pass

        @module.program_cache(scope="preparation")
        def factory(value):
            calls.append(value)
            return Payload()

        with EnvPatch(LIL_B12X_PREPARATION_FACTORY_CACHE=None):
            cache = module.PreparationProgramCache()
            with cache.activate():
                first = factory(1)
                second = factory(1)

        self.assertIs(first, second)
        self.assertEqual(calls, [1])
        self.assertEqual(len(cache._caches), 1)
        retained = weakref.ref(first)
        del first, second
        gc.collect()
        self.assertIsNotNone(retained())
        cache.clear()
        gc.collect()
        self.assertIsNone(retained())

    def test_opt_out_calls_factory_without_retaining_factory_cache(self):
        module = load_program_cache()
        calls = []

        class Payload:
            pass

        @module.program_cache(scope="preparation")
        def factory(value):
            calls.append(value)
            return Payload()

        with EnvPatch(LIL_B12X_PREPARATION_FACTORY_CACHE="0"):
            cache = module.PreparationProgramCache()
            with cache.activate():
                first = factory(1)
                second = factory(1)

        selected = weakref.ref(first)
        loser = weakref.ref(second)
        del second
        gc.collect()
        self.assertIs(selected(), first)
        self.assertIsNone(loser())
        self.assertEqual(calls, [1, 1])
        self.assertEqual(cache._caches, {})

    def test_invalid_factory_cache_env_is_rejected(self):
        module = load_program_cache()

        @module.program_cache(scope="preparation")
        def factory():
            return object()

        with EnvPatch(LIL_B12X_PREPARATION_FACTORY_CACHE="maybe"):
            cache = module.PreparationProgramCache()
            with cache.activate():
                with self.assertRaisesRegex(ValueError, "LIL_B12X_PREPARATION_FACTORY_CACHE"):
                    factory()


class PreparationSessionRaceBatchTests(unittest.TestCase):
    def test_default_race_batch_remains_constructor_default(self):
        module = load_session_subset()
        with EnvPatch(LIL_B12X_PREPARATION_RACE_BATCH=None):
            session = module["PreparationSession"]()
        self.assertEqual(session.race_batch, 32)

    def test_env_race_batch_override_is_applied(self):
        module = load_session_subset()
        with EnvPatch(LIL_B12X_PREPARATION_RACE_BATCH="4"):
            session = module["PreparationSession"]()
        self.assertEqual(session.race_batch, 4)

    def test_invalid_race_batch_env_is_rejected(self):
        module = load_session_subset()
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value):
                with EnvPatch(LIL_B12X_PREPARATION_RACE_BATCH=value):
                    with self.assertRaisesRegex(ValueError, "LIL_B12X_PREPARATION_RACE_BATCH"):
                        module["PreparationSession"]()


class RuntimePatcherTests(unittest.TestCase):
    def make_package_root(self):
        work = tempfile.TemporaryDirectory(prefix="ds41-prep-patcher-test.")
        root = pathlib.Path(work.name)
        (root / "_lib").mkdir()
        (root / "preparation").mkdir()
        shutil.copyfile(SOURCE_PROGRAM, root / "_lib" / "program_cache.py")
        shutil.copyfile(SOURCE_SESSION, root / "preparation" / "session.py")
        return work, root

    def run_patcher(self, root, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "apply_b12x_preparation_bound.py"), *args, str(root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_patcher_applies_package_layout_twice_idempotently(self):
        work, root = self.make_package_root()
        with work:
            first = self.run_patcher(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("layout=package", first.stdout)
            self.assertIn("program_cache.py apply=4 existing=0", first.stdout)
            self.assertIn("session.py apply=2 existing=0", first.stdout)

            patched_program = (root / "_lib" / "program_cache.py").read_text()
            patched_session = (root / "preparation" / "session.py").read_text()

            second = self.run_patcher(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("program_cache.py apply=0 existing=4", second.stdout)
            self.assertIn("session.py apply=0 existing=2", second.stdout)
            self.assertEqual((root / "_lib" / "program_cache.py").read_text(), patched_program)
            self.assertEqual((root / "preparation" / "session.py").read_text(), patched_session)

    def test_patcher_rejects_malformed_input_without_writes(self):
        work, root = self.make_package_root()
        with work:
            program = root / "_lib" / "program_cache.py"
            session = root / "preparation" / "session.py"
            program_before = program.read_text()
            session_before = session.read_text()
            program.write_text(program_before.replace("import functools\n", "", 1))
            malformed_program = program.read_text()

            result = self.run_patcher(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("precondition", result.stderr)
            self.assertEqual(program.read_text(), malformed_program)
            self.assertEqual(session.read_text(), session_before)


if __name__ == "__main__":
    unittest.main()
