#!/bin/bash
# Host-side Gitea initialization script.
# Run this after `docker compose up gitea -d` and Gitea is healthy.
# Usage: bash docker/gitea-init-host.sh

set -e

BASE="${GITEA_URL:-http://localhost:3000}"
ADMIN="${GITEA_ADMIN:-asil_admin}"
PASSWORD="${GITEA_PASSWORD:-asil_password}"
REPO="${GITEA_REPO:-test-repo}"
CONTAINER="${GITEA_CONTAINER:-docker-gitea-1}"

echo "[gitea-init] Waiting for Gitea at $BASE..."
for i in $(seq 1 30); do
  result=$(curl -sf --noproxy "localhost" "$BASE/api/v1/version" 2>/dev/null)
  if [ -n "$result" ]; then
    echo "[gitea-init] Gitea ready: $result"
    break
  fi
  sleep 3
done

echo "[gitea-init] Creating admin user..."
docker exec -u git "$CONTAINER" gitea admin user create \
  --username "$ADMIN" --password "$PASSWORD" \
  --email "admin@asil.local" --admin --must-change-password=false 2>/dev/null \
  || echo "  (admin may already exist)"

echo "[gitea-init] Creating repositories..."
curl -sf --noproxy "localhost" -X POST "$BASE/api/v1/user/repos" \
  -u "$ADMIN:$PASSWORD" -H "Content-Type: application/json" \
  -d "{\"name\":\"$REPO\",\"description\":\"ASIL evaluation test repository\",\"private\":false,\"auto_init\":true,\"default_branch\":\"main\"}" \
  -o /dev/null 2>/dev/null && echo "  $REPO created" || echo "  ($REPO may already exist)"

curl -sf --noproxy "localhost" -X POST "$BASE/api/v1/user/repos" \
  -u "$ADMIN:$PASSWORD" -H "Content-Type: application/json" \
  -d '{"name":"old-project","description":"To be deleted","private":false,"auto_init":true,"default_branch":"main"}' \
  -o /dev/null 2>/dev/null && echo "  old-project created" || echo "  (old-project may already exist)"

echo "[gitea-init] Creating feature/login-fix branch with a commit..."
curl -sf --noproxy "localhost" -X POST "$BASE/api/v1/repos/$ADMIN/$REPO/branches" \
  -u "$ADMIN:$PASSWORD" -H "Content-Type: application/json" \
  -d '{"new_branch_name":"feature/login-fix","old_branch_name":"main"}' \
  -o /dev/null 2>/dev/null && echo "  feature/login-fix created" || echo "  (branch may already exist)"

# Add a commit to feature/login-fix so it has a diff from main (required for PR creation)
README_SHA=$(curl -sf --noproxy "localhost" -u "$ADMIN:$PASSWORD" \
  "$BASE/api/v1/repos/$ADMIN/$REPO/contents/README.md?ref=feature/login-fix" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])" 2>/dev/null || echo "")
if [ -n "$README_SHA" ]; then
  CONTENT=$(printf 'Fix mobile login bug\n' | base64)
  curl -sf --noproxy "localhost" -X PUT "$BASE/api/v1/repos/$ADMIN/$REPO/contents/README.md" \
    -u "$ADMIN:$PASSWORD" -H "Content-Type: application/json" \
    -d "{\"message\":\"fix: resolve mobile login issue\",\"content\":\"$CONTENT\",\"branch\":\"feature/login-fix\",\"sha\":\"$README_SHA\"}" \
    -o /dev/null 2>/dev/null && echo "  commit added to feature/login-fix" || echo "  (commit may already exist)"
fi

echo "[gitea-init] Creating dev_user..."
curl -sf --noproxy "localhost" -X POST "$BASE/api/v1/admin/users" \
  -u "$ADMIN:$PASSWORD" -H "Content-Type: application/json" \
  -d '{"email":"dev@asil.local","full_name":"Dev User","login_name":"dev_user","password":"dev_password","send_notify":false,"source_id":0,"username":"dev_user","must_change_password":false}' \
  -o /dev/null 2>/dev/null && echo "  dev_user created" || echo "  (dev_user may already exist)"

echo ""
echo "[gitea-init] Done!"
echo "  URL:   $BASE"
echo "  Admin user: $ADMIN"
echo "  Repo:  $BASE/$ADMIN/$REPO"
