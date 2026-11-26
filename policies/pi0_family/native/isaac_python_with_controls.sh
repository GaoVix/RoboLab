#!/bin/bash
# Invoked by the official python.sh through its supported PYTHONEXE hook.
# Keep the official environment and libcarb preload, then exec the bundled Python.
set -euo pipefail
: "${ROBOLAB_REPRO_NATIVE:?Missing verified NRD control library}"
: "${ROBOLAB_REPRO_PYTHON:?Missing Isaac Sim Python executable}"
export LD_PRELOAD="${ROBOLAB_REPRO_NATIVE}${LD_PRELOAD:+:${LD_PRELOAD}}"
unset PYTHONEXE
exec "${ROBOLAB_REPRO_PYTHON}" "$@"
