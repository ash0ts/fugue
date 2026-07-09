#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m fugue.bench.cli preflight --repo-root "$ROOT" "$@"
