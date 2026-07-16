#!/bin/sh
set -eu

runtime=/opt/fugue-agent-runtime
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$runtime/libexec/codex" "$@"
