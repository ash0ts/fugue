#!/bin/sh
set -eu
cd /testbed
python -m pytest -q tests/test_export.py::test_weave_publication_keeps_direct_outcomes_and_skips_admin_rows
