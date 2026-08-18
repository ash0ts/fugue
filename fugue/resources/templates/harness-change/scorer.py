def score(task, output, evidence):
    def contains(actual, expected):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and contains(actual[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return isinstance(actual, list) and all(
                any(contains(candidate, item) for candidate in actual)
                for item in expected
            )
        return actual == expected

    return {
        "answer_present": output is not None
        and (not isinstance(output, str) or bool(output.strip())),
        "expected_values": contains(output, evidence.get("expected")),
    }
