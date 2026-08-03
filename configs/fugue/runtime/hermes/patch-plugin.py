from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path("/opt/fugue-agent-runtime/hermes-otel")
tracer_path = root / "tracer.py"
tracer_source = tracer_path.read_text()
tracer_expected_sha256 = (
    "d01e2ca71330fbca465e73d3bb117377ae8a79191d7edfd5441cc9bb47ea0194"
)
if hashlib.sha256(tracer_source.encode()).hexdigest() != tracer_expected_sha256:
    raise SystemExit("hermes-otel tracer source digest mismatch")
attribute_needle = "attrs = dict(attributes or {})"
attribute_replacement = (
    "attrs = {**(self.config.resource_attributes or {}), **dict(attributes or {})}"
)
if tracer_source.count(attribute_needle) != 2:
    raise SystemExit("hermes-otel tracer patch target mismatch")
tracer_source = tracer_source.replace(attribute_needle, attribute_replacement)

trace_import_needle = (
    "    from opentelemetry.trace import INVALID_SPAN, set_span_in_context\n"
)
trace_import_replacement = """    from opentelemetry.trace import (
        INVALID_SPAN,
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
        set_span_in_context,
    )
"""
if tracer_source.count(trace_import_needle) != 1:
    raise SystemExit("hermes-otel span-context import patch target mismatch")
tracer_source = tracer_source.replace(trace_import_needle, trace_import_replacement)

parent_needle = '''            span_ctx = None
            if effective_parent is not None and hasattr(effective_parent, "get_span_context"):
                span_ctx = set_span_in_context(effective_parent)

            span = self.tracer.start_span(name, attributes=attrs, context=span_ctx, links=links)
'''
parent_replacement = '''            span_ctx = None
            if effective_parent is not None and hasattr(effective_parent, "get_span_context"):
                span_ctx = set_span_in_context(effective_parent)
            else:
                traceparent = os.environ.get("FUGUE_WEAVE_TRACEPARENT", "").strip()
                if traceparent:
                    # Start only a native root under Fugue's remote Evaluation span.
                    parts = traceparent.split("-")
                    if (
                        len(parts) != 4
                        or parts[0] != "00"
                        or len(parts[1]) != 32
                        or len(parts[2]) != 16
                        or len(parts[3]) != 2
                    ):
                        raise ValueError("invalid FUGUE_WEAVE_TRACEPARENT")
                    trace_id = int(parts[1], 16)
                    span_id = int(parts[2], 16)
                    if trace_id == 0 or span_id == 0:
                        raise ValueError("invalid FUGUE_WEAVE_TRACEPARENT")
                    parent = SpanContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        is_remote=True,
                        trace_flags=TraceFlags(int(parts[3], 16) & 1),
                        trace_state=TraceState(),
                    )
                    span_ctx = set_span_in_context(NonRecordingSpan(parent))

            span = self.tracer.start_span(name, attributes=attrs, context=span_ctx, links=links)
'''
if tracer_source.count(parent_needle) != 1:
    raise SystemExit("hermes-otel evaluation parent patch target mismatch")
tracer_source = tracer_source.replace(parent_needle, parent_replacement)

atexit_needle = "        atexit.register(self._force_flush)\n"
atexit_replacement = """        atexit.register(self._force_flush)
        if os.environ.get("FUGUE_WEAVE_SINGLE_TURN_KEY", "").strip():
            # atexit is LIFO: finish the one trial root before flushing it.
            atexit.register(self._finalize_fugue_single_turns)
"""
if tracer_source.count(atexit_needle) != 1:
    raise SystemExit("hermes-otel atexit patch target mismatch")
tracer_source = tracer_source.replace(atexit_needle, atexit_replacement)

flush_needle = "    def _force_flush(self):\n"
finalizer = '''    def _finalize_fugue_single_turns(self) -> None:
        """Close Fugue's one trial root after every Hermes continuation turn."""
        if not os.environ.get("FUGUE_WEAVE_SINGLE_TURN_KEY", "").strip():
            return
        for session_id in list(self._turn_started_at):
            session_key = f"session:{session_id}"
            keys = self._session_keys.pop(session_id, set())
            for key in [item for item in keys if item != session_key]:
                self.end_span(key, status="ok")
            self.spans.pop_parent(session_id=session_id)
            if session_key in self.spans._active_spans:
                self.end_span(
                    session_key,
                    attributes={"hermes.turn.final_status": "completed"},
                    status="ok",
                )
            self.unregister_turn(session_id)

'''
if tracer_source.count(flush_needle) != 1:
    raise SystemExit("hermes-otel finalizer patch target mismatch")
tracer_source = tracer_source.replace(flush_needle, finalizer + flush_needle)
tracer_path.write_text(tracer_source)

hooks_path = root / "hooks.py"
hooks_source = hooks_path.read_text()
hooks_expected_sha256 = (
    "8843ce683f5016fa71c80266f6b0abd7399143d314221d3c69fe35c9aefebf62"
)
if hashlib.sha256(hooks_source.encode()).hexdigest() != hooks_expected_sha256:
    raise SystemExit("hermes-otel hooks source digest mismatch")
turn_end_needle = '''    tracer.spans.pop_parent(session_id=session_id)
    tracer.end_span(key, attributes=attributes, status=status)
    tracer.unregister_turn(session_id)

    # End of a user-visible unit of work. Flush so the trace is visible in
'''
turn_end_replacement = '''    if os.environ.get("FUGUE_WEAVE_SINGLE_TURN_KEY", "").strip():
        # Hermes ends every internal continuation as a turn. Fugue measures the
        # Harbor trial, so keep one root open and close it at process exit.
        root = tracer.spans.get_span(key)
        if root is not None and hasattr(root, "set_attribute"):
            for name, value in attributes.items():
                root.set_attribute(name, value)
        tracer.register_turn(session_id)
        if tracer.config.force_flush_on_session_end:
            tracer._force_flush()
        debug_log(f"  Fugue trial root retained: key={key}, status={status}")
        return

    tracer.spans.pop_parent(session_id=session_id)
    tracer.end_span(key, attributes=attributes, status=status)
    tracer.unregister_turn(session_id)

    # End of a user-visible unit of work. Flush so the trace is visible in
'''
if hooks_source.count(turn_end_needle) != 1:
    raise SystemExit("hermes-otel turn-end patch target mismatch")
hooks_path.write_text(hooks_source.replace(turn_end_needle, turn_end_replacement))

(root / "fugue-patch-lock.json").write_text(
    json.dumps(
        {
            "hooks.py": hooks_expected_sha256,
            "tracer.py": tracer_expected_sha256,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
