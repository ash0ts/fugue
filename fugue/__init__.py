"""Fugue: comparative evaluation of Weave-instrumented agent experiments.

Fugue composes Harbor tasks, harnesses, prompts, skills, and context systems
into isolated trial cells. Provider-routed model calls share one W&B Weave
trace plane, while direct retrieval and sequence diagnostics remain explicitly
separate from harness outcomes.
"""
