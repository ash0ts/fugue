"""Host-confirmed public regression-test outcome for the Vercel V2 study.

The executable test is intentionally Agent-visible and is therefore not called
hidden evidence. Fugue independently reruns its locked bytes after the trial in
the digest-pinned Node verifier and binds the resulting receipt to this exact
attempt. This is the study's sole public-test outcome.
"""

import re


def _mapping(value):
    return value if isinstance(value, dict) else {}


def score(task, output, evidence):
    del output
    receipt = _mapping(evidence.get("host_verifier_receipt"))
    task_id = task.get("id") if isinstance(task, dict) else None
    digest_fields = (
        "output_sha256",
        "base_archive_sha256",
        "public_test_sha256",
        "submitted_artifact_sha256",
        "final_tree_sha256",
        "verifier_source_sha256",
        "runtime_profile_digest",
        "runtime_lock_digest",
        "receipt_digest",
    )
    passed = bool(
        isinstance(task_id, str)
        and task_id
        and receipt.get("schema_version") == 2
        and receipt.get("kind") == "post_trial_verifier_receipt"
        and receipt.get("evaluator_id") == "vercel-public-behavioral"
        and receipt.get("task_id") == task_id
        and isinstance(receipt.get("attempt_id"), str)
        and bool(receipt["attempt_id"])
        and receipt.get("status") == "passed"
        and receipt.get("failure_kind") is None
        and receipt.get("command") == ["node", "--test", "tests/task.test.mjs"]
        and receipt.get("exit_code") == 0
        and receipt.get("test_count") == 1
        and receipt.get("pass_count") == 1
        and receipt.get("fail_count") == 0
        and receipt.get("runtime_profile_id") == "node22-verifier-v1"
        and isinstance(receipt.get("runtime_image"), str)
        and "@sha256:" in receipt["runtime_image"]
        and all(
            isinstance(receipt.get(field), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", receipt[field]))
            for field in digest_fields
        )
    )
    return {"public_regression_test_success": passed}
