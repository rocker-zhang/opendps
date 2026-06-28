#!/usr/bin/env bash
# Security and privacy compliance gate — runs in CI and locally before push.
# Blocks commit if any pattern is found in public-facing files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
FAIL=0

check() {
  local label="$1"; shift
  local results
  results=$(grep -rn "$@" \
    --include="*.py" --include="*.rs" --include="*.md" \
    --include="*.yml" --include="*.yaml" --include="*.toml" \
    --include="*.json" --include="*.txt" \
    --exclude-dir=".git" --exclude-dir="target" \
    --exclude-dir=".venv" --exclude-dir="__pycache__" \
    --exclude-dir="scripts" \
    --exclude="AGENTS.md" \
    . 2>/dev/null || true)
  if [ -n "$results" ]; then
    echo -e "${RED}FAIL${NC} [$label]"
    echo "$results" | head -10
    FAIL=1
  fi
}

echo "=== opendps security-check ==="

# 1. Hardcoded secrets / credentials
check "hardcoded-password"   -iE 'password\s*=\s*"[^"]+"' \
  --exclude="redfish_scraper.py"   # empty default is intentional
check "hardcoded-api-key"    -iE '(api_key|apikey|secret_key)\s*=\s*"[A-Za-z0-9/_-]{20,}"'
check "private-key-material" -E '-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----'

# 2. Internal infrastructure
check "ssh-hostname"         -iE '(jump-proxy|jump\.tiktok|\.internal\.|corp\.|vpn\.)' \
  --exclude="AGENTS.md"   # rule doc itself is exempt
check "internal-ip-private"  -E '(10\.[0-9]+\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' \
  --exclude-dir="deploy"   # docker compose private ranges are OK
check "ipv6-fd-prefix"       -E 'fd[0-9a-f]{2}:[0-9a-f]{4}:'
check "ssh-key-path"         -E '18-glide|id_rsa.*glide'

# 3. Employer / organization reveals
check "employer-name"        -iE '\b(tiktok|bytedance)\b' \
  --exclude="AGENTS.md"

# 4. Sensitive hardware naming (SKU suffixes that narrow supplier)
check "b300-sku-suffix"      -E 'B300 SXM6 AC|B300 NVL'

# 5. Claude attribution in committed files
check "claude-coauthor"      -iE 'Co-Authored-By: Claude|Claude-Session:'

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo -e "${GREEN}All checks passed.${NC}"
else
  echo -e "${RED}Security check FAILED — fix the issues above before pushing.${NC}"
  exit 1
fi
