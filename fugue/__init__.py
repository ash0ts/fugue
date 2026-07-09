"""Fugue: comparative evaluation of Weave-instrumented agent harnesses.

Harbor-based runner that executes any set of agent harnesses (Hermes,
OpenClaw, Claude Code, Codex CLI) on the same task suite — provider-routed
model calls, one trace plane (W&B Weave), fully containerized.

The flagship study is the repo-memory benchmark: does giving an agent a
wiki of the repository (OpenWiki et al.) improve it, per harness?

See README.md for usage.
"""
