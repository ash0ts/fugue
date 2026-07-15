#!/bin/sh
set -eu
cd /testbed
python -m pytest -q tests/test_job_config.py
