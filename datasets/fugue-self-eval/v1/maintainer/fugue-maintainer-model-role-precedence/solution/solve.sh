#!/bin/sh
set -eu
cd /testbed
git apply -R /solution/mutation.patch
