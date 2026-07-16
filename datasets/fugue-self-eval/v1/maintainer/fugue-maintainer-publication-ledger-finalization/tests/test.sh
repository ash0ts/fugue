#!/bin/sh
set -eu
cd /testbed
python -m pytest -q tests/test_export.py::test_live_evaluation_links_native_root_and_finalizes_cleanly
