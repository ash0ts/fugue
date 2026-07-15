# Fugue maintainer task

W&B Inference requests are no longer carrying the configured project header. Restore the provider-neutral request metadata contract without exposing credentials or changing non-W&B providers.

Work only in the checked-out Fugue repository. Do not modify tests. Run focused
checks before broader checks, and keep the repair scoped to the reported contract.
