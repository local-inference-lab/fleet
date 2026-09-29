#!/bin/sh
set -eu

/opt/venv/bin/python /opt/lil/fleet-overrides/apply_b12x_preparation_bound.py \
  /opt/venv/lib/python3.12/site-packages/b12x

/opt/venv/bin/python /opt/lil/fleet-overrides/apply_vllm_sequence_info.py \
  /opt/venv/lib/python3.12/site-packages/vllm

exec /usr/local/bin/lil-serve "$@"
