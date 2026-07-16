#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_URL="${FUGUE_COMPAT_BASE_URL:-http://127.0.0.1:18765}"
OPENRESPONSES_COMMIT="92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c"
OPENRESPONSES_DIR="${TMPDIR:-/tmp}/fugue-openresponses-${OPENRESPONSES_COMMIT}"
BUN="${BUN:-$ROOT/tests/compat/node_modules/.bin/bun}"

cd "$ROOT"
npm --prefix tests/compat ci
python -m uvicorn compat.server:app --app-dir tests --host 127.0.0.1 --port 18765 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  if curl --fail --silent "$BASE_URL/healthz" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl --fail --silent "$BASE_URL/healthz" >/dev/null

python tests/compat/openai_client.py
npm --prefix tests/compat test

if [[ ! -d "$OPENRESPONSES_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/openresponses/openresponses.git "$OPENRESPONSES_DIR"
fi
git -C "$OPENRESPONSES_DIR" fetch origin "$OPENRESPONSES_COMMIT" --depth=1
git -C "$OPENRESPONSES_DIR" checkout --detach "$OPENRESPONSES_COMMIT"
"$BUN" install --cwd "$OPENRESPONSES_DIR" --frozen-lockfile
"$BUN" run --cwd "$OPENRESPONSES_DIR" bin/compliance-test.ts \
  --base-url "$BASE_URL/v1" \
  --api-key fugue-compatibility-key \
  --model fugue-candidate \
  --filter basic-response,streaming-response
