# Source-use replay

This no-key example replays immutable attempt rows through Fugue's public
comparison and result contracts:

```bash
uv run fugue demo source-use
```

It demonstrates strict task/private-label separation, baseline and candidate
identity, deterministic rescoring, aligned pairs, and result export. It is a
replay, not a new live experiment or evidence that a Skill is generally useful.
