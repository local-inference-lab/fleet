#!/bin/sh
# Apply only the opt-in vLLM /server_info sequence-capacity patch, then run the
# profile's original launcher unchanged ("$@" = original entrypoint + args).
# llmconduit mesh workers with capacity_source: vllm need /server_info to
# advertise a model. The vLLM package is located dynamically because images
# install it in different places; the patch refuses unexpected source.
set -eu
PY=/opt/venv/bin/python
VLLM_ROOT=$("$PY" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
"$PY" /opt/lil/fleet-overrides/apply_vllm_sequence_info.py "$VLLM_ROOT"
exec "$@"
