#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PROGRAM_CACHE_PATCHES = (
    (
        "program_cache import os",
        "import functools\nimport gc\nimport time\n",
        "import functools\nimport gc\nimport os\nimport time\n",
    ),
    (
        "program_cache env constants",
        '_PREPARATION_CACHE = ContextVar("b12x_preparation_program_cache", default=None)\n\n',
        '_PREPARATION_CACHE = ContextVar("b12x_preparation_program_cache", default=None)\n'
        '_FALSE_ENV_VALUES = {"0", "false", "no", "off"}\n'
        '_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}\n'
        '_FACTORY_CACHE_ENV = "LIL_B12X_PREPARATION_FACTORY_CACHE"\n\n',
    ),
    (
        "program_cache env helper",
        '    raise TypeError(f"program factory cache requires metadata, not {type(value).__name__}")\n'
        "\n\n"
        "class PreparationProgramCache:\n",
        '    raise TypeError(f"program factory cache requires metadata, not {type(value).__name__}")\n'
        "\n\n"
        "def _preparation_factory_cache_enabled():\n"
        "    value = os.environ.get(_FACTORY_CACHE_ENV)\n"
        "    if value is None or value == \"\":\n"
        "        return True\n"
        "    normalized = value.strip().lower()\n"
        "    if normalized in _FALSE_ENV_VALUES:\n"
        "        return False\n"
        "    if normalized in _TRUE_ENV_VALUES:\n"
        "        return True\n"
        "    raise ValueError(f\"{_FACTORY_CACHE_ENV} must be a boolean value\")\n"
        "\n\n"
        "class PreparationProgramCache:\n",
    ),
    (
        "program_cache uncached call",
        "    def call(self, factory, args, kwargs):\n"
        "        memo = self._caches.get(factory)\n"
        "        if memo is None:\n"
        "            memo = program_cache(factory._function)\n"
        "            memo._metadata = True\n"
        "            self._caches[factory] = memo\n"
        "        before = memo._misses\n"
        "        timing = self._timing\n",
        "    def call(self, factory, args, kwargs):\n"
        "        timing = self._timing\n"
        "        if not _preparation_factory_cache_enabled():\n"
        "            if timing is None or timing.path is None:\n"
        "                return factory._function(*args, **kwargs)\n"
        "            label = f\"factory.{factory.__module__}.{factory.__name__}\"\n"
        "            started = time.perf_counter()\n"
        "            try:\n"
        "                with timing.span(label):\n"
        "                    return factory._function(*args, **kwargs)\n"
        "            finally:\n"
        "                timing.add(label + \".uncached\", time.perf_counter() - started)\n"
        "        memo = self._caches.get(factory)\n"
        "        if memo is None:\n"
        "            memo = program_cache(factory._function)\n"
        "            memo._metadata = True\n"
        "            self._caches[factory] = memo\n"
        "        before = memo._misses\n",
    ),
)


SESSION_PATCHES = (
    (
        "session race env helpers",
        "def _default_compile_workers(device):\n"
        "    identity = device.identity\n"
        "    spark = (\n"
        "        identity is not None\n"
        "        and identity.vendor == \"nvidia\"\n"
        "        and identity.product_name in {\"gb10\", \"nvidia gb10\"}\n"
        "    )\n"
        "    return int(os.environ.get(\"B12X_COMPILE_WORKERS\", \"4\" if spark else \"8\"))\n"
        "\n\n"
        "class PreparationSession:\n",
        "def _default_compile_workers(device):\n"
        "    identity = device.identity\n"
        "    spark = (\n"
        "        identity is not None\n"
        "        and identity.vendor == \"nvidia\"\n"
        "        and identity.product_name in {\"gb10\", \"nvidia gb10\"}\n"
        "    )\n"
        "    return int(os.environ.get(\"B12X_COMPILE_WORKERS\", \"4\" if spark else \"8\"))\n"
        "\n\n"
        "_RACE_BATCH_ENV = \"LIL_B12X_PREPARATION_RACE_BATCH\"\n"
        "\n\n"
        "def _env_positive_int(name):\n"
        "    value = os.environ.get(name)\n"
        "    if value is None or value == \"\":\n"
        "        return None\n"
        "    try:\n"
        "        parsed = int(value)\n"
        "    except ValueError as error:\n"
        "        raise ValueError(f\"{name} must be a positive integer\") from error\n"
        "    if parsed <= 0:\n"
        "        raise ValueError(f\"{name} must be a positive integer\")\n"
        "    return parsed\n"
        "\n\n"
        "def _default_race_batch(race_batch):\n"
        "    override = _env_positive_int(_RACE_BATCH_ENV)\n"
        "    return race_batch if override is None else override\n"
        "\n\n"
        "class PreparationSession:\n",
    ),
    (
        "session apply race override",
        "        if compile_workers is None:\n"
        "            compile_workers = _default_compile_workers(self.device)\n"
        "        for name, value in (\n",
        "        if compile_workers is None:\n"
        "            compile_workers = _default_compile_workers(self.device)\n"
        "        race_batch = _default_race_batch(race_batch)\n"
        "        for name, value in (\n",
    ),
)


def apply_exact(text: str, patches: tuple[tuple[str, str, str], ...], *, path: Path) -> tuple[str, int, int]:
    applied = already_applied = 0
    for label, old, new in patches:
        old_count = text.count(old)
        new_count = text.count(new)
        if old_count == 1 and new_count == 0:
            text = text.replace(old, new, 1)
            applied += 1
            continue
        if old_count == 0 and new_count == 1:
            already_applied += 1
            continue
        raise SystemExit(
            f"{path}: precondition {label!r} matched old={old_count} new={new_count}, "
            "expected exactly one old or one new snippet"
        )
    return text, applied, already_applied


def resolve_sources(root: Path) -> tuple[Path, Path, str]:
    package_program_cache = root / "_lib" / "program_cache.py"
    package_session = root / "preparation" / "session.py"
    if package_program_cache.exists() or package_session.exists():
        if not package_program_cache.exists() or not package_session.exists():
            raise SystemExit(
                f"{root}: package layout requires both _lib/program_cache.py and preparation/session.py"
            )
        return package_program_cache, package_session, "package"

    flat_program_cache = root / "program_cache.py"
    flat_session = root / "session.py"
    if flat_program_cache.exists() and flat_session.exists():
        return flat_program_cache, flat_session, "flat"

    raise SystemExit(
        f"{root}: expected _lib/program_cache.py plus preparation/session.py, "
        "or flat program_cache.py plus session.py"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the bounded B12X preparation-memory source override."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="B12X package root, or a flat directory containing program_cache.py and session.py",
    )
    parser.add_argument("--check", action="store_true", help="validate preconditions without writing")
    args = parser.parse_args()

    program_cache, session, layout = resolve_sources(args.root)
    program_text, program_applied, program_existing = apply_exact(
        program_cache.read_text(), PROGRAM_CACHE_PATCHES, path=program_cache
    )
    session_text, session_applied, session_existing = apply_exact(
        session.read_text(), SESSION_PATCHES, path=session
    )

    if not args.check:
        program_cache.write_text(program_text)
        session.write_text(session_text)
    print(
        "preconditions matched: "
        f"layout={layout} "
        f"program_cache.py apply={program_applied} existing={program_existing} "
        f"session.py apply={session_applied} existing={session_existing}"
    )
    if args.check:
        print("check only: no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
