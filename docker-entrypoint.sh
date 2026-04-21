#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  case "$1" in
    bash|shell)
      shift
      exec bash "$@"
      ;;
    train)
      shift
      ;;
    *)
      exec "$@"
      ;;
  esac
fi

cd "${CODE_PATH}"

if [[ "${INSTALL_DEPS_ON_START}" == "1" ]]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
fi

cd "${TRAINING_DIR}"

if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: CONFIG_NAME is empty. Pass -e CONFIG_NAME=<your_config>."
  exit 1
fi

exec torchrun --nproc_per_node="${NPROC_PER_NODE}" launch.py --config "${CONFIG_NAME}"
