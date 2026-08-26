#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${ASIL_BOOTSTRAP_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_NAME=${ASIL_BOOTSTRAP_PROJECT_NAME:-asil-bootstrap}
REPORT_PATH=${ASIL_BOOTSTRAP_REPORT_PATH:-$PROJECT_ROOT/results/setup/docker_bootstrap_report.json}
LOG_DIR=${ASIL_BOOTSTRAP_LOG_DIR:-$PROJECT_ROOT/results/setup/logs}
DOCKER_BIN=${ASIL_BOOTSTRAP_DOCKER_BIN:-docker}
PYTHON_BIN=${ASIL_BOOTSTRAP_PYTHON_BIN:-python3}
COMPOSE_FILE=$PROJECT_ROOT/docker/docker-compose.yml
ENV_FILE=$PROJECT_ROOT/.env
ENV_EXAMPLE=$PROJECT_ROOT/.env.example
VERIFIER=$PROJECT_ROOT/scripts/verify_rebuttal_docker.py
LOCK_FILE=$PROJECT_ROOT/results/setup/docker_bootstrap.lock

MODE=all
NO_CACHE=0
REQUIRE_OPENAI=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_rebuttal_docker.sh [OPTIONS]

Options:
  --check-only      Run host and repository preflight only.
  --build-only      Run preflight, pull, and build without runtime verification.
  --verify-only     Verify existing images/runtime without rebuilding.
  --require-openai  Require a non-empty OPENAI_API_KEY before any build.
  --no-cache        Disable Docker cache for local image builds.
  -h, --help        Show this help.
EOF
}

mode_seen=0
while (($#)); do
  case "$1" in
    --check-only|--build-only|--verify-only)
      ((mode_seen += 1))
      case "$1" in
        --check-only) MODE=check ;;
        --build-only) MODE=build ;;
        --verify-only) MODE=verify ;;
      esac
      ;;
    --require-openai) REQUIRE_OPENAI=1 ;;
    --no-cache) NO_CACHE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((mode_seen > 1)); then
  echo "--check-only, --build-only, and --verify-only are mutually exclusive" >&2
  exit 2
fi

mkdir -p "$(dirname "$REPORT_PATH")" "$LOG_DIR"
if [[ ${ASIL_BOOTSTRAP_SKIP_FLOCK:-0} != 1 ]]; then
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required (normally supplied by util-linux on Ubuntu)." >&2
    exit 2
  }
  exec 9>"$LOCK_FILE"
  flock -n 9 || {
    echo "Another ASIL Docker bootstrap is already running." >&2
    exit 2
  }
fi

if [[ ! -f "$ENV_FILE" ]]; then
  [[ -f "$ENV_EXAMPLE" ]] || {
    echo "Missing $ENV_EXAMPLE" >&2
    exit 2
  }
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  chmod 0600 "$ENV_FILE"
  echo "[bootstrap] created secret-free $ENV_FILE"
else
  echo "[bootstrap] preserving existing $ENV_FILE"
fi

compose() {
  "$DOCKER_BIN" compose \
    --env-file "$ENV_FILE" \
    -f "$COMPOSE_FILE" \
    -p "$PROJECT_NAME" \
    "$@"
}

cleanup() {
  local exit_code=$?
  set +e
  compose down --remove-orphans >>"$LOG_DIR/cleanup.log" 2>&1
  set -e
  return "$exit_code"
}
trap cleanup EXIT INT TERM

has_openai_key() {
  if [[ -n ${OPENAI_API_KEY:-} ]]; then
    return 0
  fi
  awk -F= '
    $1 == "OPENAI_API_KEY" {
      value = substr($0, index($0, "=") + 1)
      gsub(/^[[:space:]"]+|[[:space:]"]+$/, "", value)
      if (length(value) > 0) found = 1
    }
    END { exit found ? 0 : 1 }
  ' "$ENV_FILE"
}

if ((REQUIRE_OPENAI)) && ! has_openai_key; then
  echo "OPENAI_API_KEY is required by --require-openai; set it in .env or the environment." >&2
  exit 2
fi

run_verifier() {
  local phase=$1
  "$PYTHON_BIN" "$VERIFIER" \
    --root "$PROJECT_ROOT" \
    --phase "$phase" \
    --project-name "$PROJECT_NAME" \
    --report "$REPORT_PATH"
}

echo "[bootstrap] preflight"
run_verifier preflight

if [[ $MODE == check ]]; then
  echo "[bootstrap] preflight complete"
  exit 0
fi

git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
if [[ -f "$PROJECT_ROOT/constraints-host.txt" ]]; then
  constraints_sha=$(sha256sum "$PROJECT_ROOT/constraints-host.txt" | awk '{print $1}')
else
  constraints_sha=missing
fi
if [[ -d "$PROJECT_ROOT/src" ]]; then
  source_sha=$(cd "$PROJECT_ROOT" && find src -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
else
  source_sha=missing
fi
export ASIL_GIT_COMMIT=$git_commit
export ASIL_CONSTRAINTS_SHA256=$constraints_sha
export ASIL_SOURCE_SHA256=$source_sha

if [[ $MODE != verify ]]; then
  echo "[bootstrap] pulling immutable third-party services"
  compose pull gitea gitea-init code-server jupyterlab drawio \
    2>&1 | tee "$LOG_DIR/pull.log"
  echo "[bootstrap] building local images"
  build_args=(build)
  if ((NO_CACHE)); then
    build_args+=(--no-cache)
  fi
  build_args+=(obs-mock eval)
  compose "${build_args[@]}" 2>&1 | tee "$LOG_DIR/build.log"
fi

if [[ $MODE == build ]]; then
  echo "[bootstrap] build complete"
  exit 0
fi

echo "[bootstrap] verifying images, services, runtime, smoke, readiness, and cleanup"
run_verifier all
echo "[bootstrap] environment ready: $REPORT_PATH"
