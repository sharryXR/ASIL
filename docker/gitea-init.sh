#!/bin/sh
# Gitea one-time initialization script.
# Creates admin user + test data via REST API only (no docker exec).
# Runs as a Docker sidecar after gitea service is healthy.

set -e

GITEA_URL="${GITEA_URL:-http://gitea:3000}"
ADMIN="${GITEA_ADMIN:-asil_admin}"
PASSWORD="${GITEA_PASSWORD:-asil_password}"
REPO="${GITEA_REPO:-test-repo}"

echo "[gitea-init] Gitea is up at $GITEA_URL"

AUTH="-u $ADMIN:$PASSWORD"

# ── 1. Create admin user via Gitea admin API (first-run bootstrap)
# On a fresh Gitea with INSTALL_LOCK=true and DISABLE_REGISTRATION=false,
# the first registered user becomes admin.
echo "[gitea-init] Registering admin user..."
curl -sf -X POST "$GITEA_URL/api/v1/admin/users" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$ADMIN\",\"password\":\"$PASSWORD\",\"email\":\"admin@asil.local\",\"must_change_password\":false,\"send_notify\":false}" \
  -o /dev/null 2>/dev/null && echo "  admin created" || true

# Fallback: try user/sign-up endpoint (Gitea < 1.22 without admin API bootstrap)
curl -sf -X POST "$GITEA_URL/user/sign_up" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "user_name=$ADMIN&password=$PASSWORD&retype=$PASSWORD&email=admin@asil.local" \
  -o /dev/null 2>/dev/null || true

# Verify we can authenticate
echo "[gitea-init] Verifying authentication..."
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" $AUTH "$GITEA_URL/api/v1/user" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
  echo "  WARNING: Cannot authenticate as $ADMIN (HTTP $HTTP_CODE). Trying to continue..."
fi

# ── 2. Create API token
echo "[gitea-init] Creating API token..."
# Delete existing token first (idempotent)
curl -sf -X DELETE "$GITEA_URL/api/v1/users/$ADMIN/tokens/eval-token" \
  $AUTH -o /dev/null 2>/dev/null || true

TOKEN_RESP=$(curl -sf -X POST "$GITEA_URL/api/v1/users/$ADMIN/tokens" \
  $AUTH \
  -H "Content-Type: application/json" \
  -d '{"name":"eval-token","scopes":["all"]}' 2>/dev/null || echo "{}")

TOKEN=$(echo "$TOKEN_RESP" | grep -o '"sha1":"[^"]*"' | cut -d'"' -f4)
if [ -z "$TOKEN" ]; then
  echo "  WARNING: Could not create token, using basic auth for remaining setup"
  TOKEN_HEADER=""
else
  echo "  API token created"
  TOKEN_HEADER="-H \"Authorization: token $TOKEN\""
fi

# ── 3. Create test-repo
echo "[gitea-init] Creating repository $REPO..."
curl -sf -X POST "$GITEA_URL/api/v1/user/repos" \
  $AUTH \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$REPO\",\"description\":\"ASIL evaluation test repository\",\"private\":false,\"auto_init\":true,\"default_branch\":\"main\"}" \
  -o /dev/null 2>/dev/null || echo "  (repo may already exist)"

# Wait for repo to be fully initialized
sleep 2

# ── 4. Create feature/login-fix branch with a diff from main
echo "[gitea-init] Creating feature/login-fix branch..."
curl -sf -X POST "$GITEA_URL/api/v1/repos/$ADMIN/$REPO/branches" \
  $AUTH \
  -H "Content-Type: application/json" \
  -d '{"new_branch_name":"feature/login-fix","old_branch_name":"main"}' \
  -o /dev/null 2>/dev/null || echo "  (branch may already exist)"

# Add a commit so the branch has a diff
README_SHA=$(curl -sf $AUTH "$GITEA_URL/api/v1/repos/$ADMIN/$REPO/contents/README.md?ref=feature/login-fix" 2>/dev/null | \
  grep -o '"sha":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -n "$README_SHA" ]; then
  CONTENT=$(printf 'Fix mobile login bug\n' | base64 | tr -d '\n')
  curl -sf -X PUT "$GITEA_URL/api/v1/repos/$ADMIN/$REPO/contents/README.md" \
    $AUTH \
    -H "Content-Type: application/json" \
    -d "{\"message\":\"fix: resolve mobile login issue\",\"content\":\"$CONTENT\",\"branch\":\"feature/login-fix\",\"sha\":\"$README_SHA\"}" \
    -o /dev/null 2>/dev/null && echo "  commit added to feature/login-fix" || echo "  (commit may already exist)"
fi

# ── 5. Create old-project repo (used by gitea_04 delete task)
echo "[gitea-init] Creating old-project repository..."
curl -sf -X POST "$GITEA_URL/api/v1/user/repos" \
  $AUTH \
  -H "Content-Type: application/json" \
  -d '{"name":"old-project","description":"To be deleted","private":false,"auto_init":true,"default_branch":"main"}' \
  -o /dev/null 2>/dev/null || echo "  (old-project may already exist)"

# ── 6. Create dev_user
echo "[gitea-init] Creating dev_user..."
curl -sf -X POST "$GITEA_URL/api/v1/admin/users" \
  $AUTH \
  -H "Content-Type: application/json" \
  -d '{"email":"dev@asil.local","full_name":"Dev User","login_name":"dev_user","password":"dev_password","send_notify":false,"source_id":0,"username":"dev_user","must_change_password":false}' \
  -o /dev/null 2>/dev/null || echo "  (dev_user may already exist)"

# ── 7. Write token to shared volume so eval container can read it
if [ -n "$TOKEN" ]; then
  echo "$TOKEN" > /shared/gitea_token.txt
  chmod 0444 /shared/gitea_token.txt
  test -s /shared/gitea_token.txt
fi

echo "[gitea-init] Done. Gitea initialized at $GITEA_URL"
echo "[gitea-init]   Admin user: $ADMIN"
echo "[gitea-init]   Repo:  $GITEA_URL/$ADMIN/$REPO"
if [ -n "$TOKEN" ]; then
  echo "[gitea-init]   Token file: /shared/gitea_token.txt"
fi
