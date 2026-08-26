#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

backend="${1:-${ASIL_PYTHON_BACKEND:-conda}}"

setup_conda() {
  local env_name="${ASIL_CONDA_ENV:-asil-host}"
  local conda_base=""

  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
  elif command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  else
    echo "[setup-host-env] conda is not available." >&2
    exit 2
  fi

  local conda_sh="$conda_base/etc/profile.d/conda.sh"
  if [[ ! -f "$conda_sh" ]]; then
    echo "[setup-host-env] cannot find conda activation script: $conda_sh" >&2
    exit 2
  fi

  # shellcheck source=/dev/null
  source "$conda_sh"

  if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "[setup-host-env] updating conda env: $env_name"
    conda env update -n "$env_name" -f environment.yml --prune
  else
    echo "[setup-host-env] creating conda env: $env_name"
    if [[ "$env_name" == "asil-host" ]]; then
      conda env create -f environment.yml
    else
      conda env create -n "$env_name" -f environment.yml
    fi
  fi

  conda activate "$env_name"
  local pip_args=(--default-timeout=300 --retries 10)
  if [[ -f constraints-host.txt ]]; then
    pip_args+=(-c constraints-host.txt)
  fi
  python -m pip install "${pip_args[@]}" -e ".[dev,eval]"
  python -c 'import sys, asil, openai, pydantic; print(f"[setup-host-env] ready: {sys.executable} ({sys.version.split()[0]})")'
}

setup_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[setup-host-env] uv is not installed." >&2
    exit 2
  fi

  echo "[setup-host-env] syncing uv environment from uv.lock"
  uv sync --locked --extra dev --extra eval
  uv run --locked --extra dev --extra eval python -c 'import sys, asil, openai, pydantic; print(f"[setup-host-env] ready: {sys.executable} ({sys.version.split()[0]})")'
}

case "$backend" in
  conda)
    setup_conda
    ;;
  uv)
    setup_uv
    ;;
  *)
    echo "[setup-host-env] unsupported backend '$backend' (expected conda or uv)" >&2
    exit 2
    ;;
esac
