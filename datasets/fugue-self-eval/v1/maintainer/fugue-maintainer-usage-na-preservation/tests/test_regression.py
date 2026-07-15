from fugue.bench.export import _summarize_spans


def test_missing_span_usage_remains_unavailable():
    summary = _summarize_spans([])

    assert summary["weave_usage_status"] == "unavailable"
    assert summary["weave_input_tokens"] is None
    assert summary["weave_output_tokens"] is None
    assert summary["weave_total_cost_usd"] is None
