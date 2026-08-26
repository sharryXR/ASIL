#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF_DIR="$ROOT_DIR/singularity/defs"
SIF_DIR="$ROOT_DIR/singularity/images"
FORCE=0
USE_FAKEROOT=1
LOAD_MODULE=1
SUDO_BUILD=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/build_singularity_images.sh [options]

Options:
  --sif-dir DIR        Output directory for .sif images (default: singularity/images)
  --force              Rebuild existing images
  --no-fakeroot        Do not pass --fakeroot to singularity build
  --sudo-build         Run `singularity build` through sudo and chown outputs back
  --no-module-load     Do not run `module load apps/singularity/4.1.0`
  -h, --help           Show this help

Environment:
  SINGULARITY_BIN                 Override singularity/apptainer executable
  SINGULARITY_BUILD_EXTRA_ARGS    Extra args passed to `singularity build`
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif-dir)
      SIF_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-fakeroot)
      USE_FAKEROOT=0
      shift
      ;;
    --sudo-build)
      SUDO_BUILD=1
      USE_FAKEROOT=0
      shift
      ;;
    --no-module-load)
      LOAD_MODULE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$LOAD_MODULE" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load apps/singularity/4.1.0 || true
fi

SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [[ -z "$SINGULARITY_BIN" ]]; then
  if command -v singularity >/dev/null 2>&1; then
    SINGULARITY_BIN="singularity"
  elif command -v apptainer >/dev/null 2>&1; then
    SINGULARITY_BIN="apptainer"
  else
    echo "singularity/apptainer executable not found" >&2
    exit 127
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python/python3 executable not found" >&2
    exit 127
  fi
fi

mkdir -p "$SIF_DIR"
OWNER_UID="${SUDO_UID:-$(id -u)}"
OWNER_GID="${SUDO_GID:-$(id -g)}"

run_as_root() {
  if [[ "$(id -u)" == "0" ]]; then
    "$@"
  else
    sudo -E "$@"
  fi
}

build_one() {
  local name="$1"
  local def="$DEF_DIR/${name}.def"
  local sif="$SIF_DIR/${name}.sif"
  if [[ ! -f "$def" ]]; then
    echo "Definition not found: $def" >&2
    exit 1
  fi
  if [[ -f "$sif" && "$FORCE" != "1" ]]; then
    echo "[singularity-build] keeping existing $sif"
    return
  fi
  local args=()
  if [[ "$USE_FAKEROOT" == "1" ]]; then
    args+=(--fakeroot)
  fi
  if [[ -n "${SINGULARITY_BUILD_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=(${SINGULARITY_BUILD_EXTRA_ARGS})
  fi
  echo "[singularity-build] building $name -> $sif"
  if [[ "$SUDO_BUILD" == "1" ]]; then
    (cd "$ROOT_DIR" && run_as_root "$SINGULARITY_BIN" build "${args[@]}" "$sif" "$def")
    run_as_root chown "$OWNER_UID:$OWNER_GID" "$sif"
  else
    (cd "$ROOT_DIR" && "$SINGULARITY_BIN" build "${args[@]}" "$sif" "$def")
  fi
}

images=(asil_eval gitea obs_mock code_server jupyterlab drawio)
for image in "${images[@]}"; do
  build_one "$image"
done

manifest="$SIF_DIR/manifest.json"
"$PYTHON_BIN" - "$SINGULARITY_BIN" "$SIF_DIR" "${images[@]}" > "$manifest" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

singularity_bin = sys.argv[1]
sif_dir = Path(sys.argv[2])
names = sys.argv[3:]

try:
    version = subprocess.run(
        [singularity_bin, "--version"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
except Exception:
    version = ""

items = []
for name in names:
    path = sif_dir / f"{name}.sif"
    digest = ""
    if path.exists():
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
    items.append({"name": name, "path": str(path), "sha256": digest})

print(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "singularity_bin": singularity_bin,
    "singularity_version": version,
    "images": items,
}, indent=2))
PY

if [[ "$SUDO_BUILD" == "1" ]]; then
  run_as_root chown "$OWNER_UID:$OWNER_GID" "$manifest"
fi

echo "[singularity-build] wrote $manifest"
