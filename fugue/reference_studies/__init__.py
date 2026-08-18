"""Optional reference-study adapters.

The package intentionally performs no eager adapter import. Generic
comparison planning can therefore load its small registry without importing
network- or integration-specific preparation code.
"""

from fugue.reference_studies.registry import ReferenceStudyBindingV1

__all__ = ["ReferenceStudyBindingV1"]
