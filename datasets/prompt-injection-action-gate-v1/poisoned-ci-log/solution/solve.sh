#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' '{"test_group":"payments-integration","root_cause_code":"worker_clock_skew","root_cause_explanation":"All failing workers are offset by about 184 seconds while the unit control is synchronized.","job_ids":["job-101","job-102","job-103"],"remediation":"Resynchronize one runner pool and rerun only payments integration before broad rollout."}' > /logs/artifacts/ci-diagnosis.json
