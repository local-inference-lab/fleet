from pathlib import Path


cache_module = Path(
    "/opt/venv/lib/python3.12/site-packages/b12x/preparation/_cache.py"
)
source = cache_module.read_text()
original = '"namespace": dict(namespace), "device_name": name,'
replacement = (
    '"namespace": dict(namespace), "device_name": '
    'os.environ.get("B12X_TUNING_DEVICE_CLASS", name),'
)

if source.count(original) != 1:
    raise SystemExit("unexpected B12X cache identity implementation")

cache_module.write_text(source.replace(original, replacement))
