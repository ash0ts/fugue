# Dead-code boundaries

Fugue runs Vulture at both 80% and 60% confidence. The 60% gate includes
`vulture_whitelist.py`; every entry in that file is an explicitly reviewed
dynamic or public surface. New findings fail CI until code is removed, given a
real caller, or added to that file with a reason.

The whitelist is limited to four mechanisms Vulture cannot see statically:

- serialized dataclass and Pydantic fields;
- import-string provider and materializer registries;
- FastAPI, FastMCP, Textual, protocol, and standard-library callback dispatch;
- supported public analysis and reconstruction helpers.

The MCP qualification functions in this category are imported by the
checked-in zero-model entrypoints under
`examples/comparisons/wandb-mcp-maintenance/`; Vulture does not follow those
scripts when the gate scans the installable `fugue` package.

It must not be used for ordinary private helpers.

## Retained compatibility surfaces

Three compatibility surfaces were reviewed during the cleanup and deliberately
retained:

- `_legacy_dataset_fingerprint` reads existing dataset markers produced before
  source metadata joined the fingerprint. `tests/test_datasets.py` locks the
  migration behavior. Remove it only with an explicit marker-format migration.
- `/v1/research/*` aliases preserve persisted Research/Study artifact access
  while clients move to the canonical Study routes.
  `tests/test_research_transports.py` verifies artifact equality across the
  boundary. Remove the aliases only after a versioned API deprecation.
- `trace_api_key()` accepts `WANDB_API_KEY` as an evidence credential fallback.
  It is never an inference-route fallback: model routing still requires the
  provider-specific credential. This lets one W&B credential publish Weave
  evidence without weakening split inference/evidence routing.
