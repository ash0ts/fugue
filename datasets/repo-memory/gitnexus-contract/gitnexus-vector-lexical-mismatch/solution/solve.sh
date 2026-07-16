#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' 'src/relay/amber_lantern.py' > /logs/artifacts/fugue-answer.txt
