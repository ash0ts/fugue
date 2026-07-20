"""Governed research memory and experiment execution for outer loops."""

from fugue.research.agent_contracts import (
    CandidateRefV1,
    ExecutionApprovalV1,
    TraceAuditDraftV1,
    TraceAuditPreviewV1,
    TraceAuditV1,
    TraceSourceRefV1,
    build_trace_audit_draft,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.candidate_sources import CandidateSourceRegistry
from fugue.research.client import FugueResearchClient
from fugue.research.contracts import (
    EvidenceRefV1,
    ExperimentDraftV1,
    ExperimentEventV1,
    ExperimentPreviewV1,
    ExperimentRecordV1,
    ResearchError,
    StudyBriefV1,
    StudyContextV1,
    StudyNoteV1,
    StudyResourceV1,
    StudyResultV1,
    StudyV1,
)
from fugue.research.service import ExperimentHandle, ResearchService, ResearchWorker
from fugue.research.traces import TraceAuditService, TraceSourceRegistry

__all__ = [
    "ApprovalLedger",
    "CandidateRefV1",
    "CandidateSourceRegistry",
    "EvidenceRefV1",
    "ExecutionApprovalV1",
    "ExperimentDraftV1",
    "ExperimentEventV1",
    "ExperimentHandle",
    "ExperimentPreviewV1",
    "ExperimentRecordV1",
    "FugueResearchClient",
    "ResearchError",
    "ResearchService",
    "ResearchWorker",
    "StudyBriefV1",
    "StudyContextV1",
    "StudyNoteV1",
    "StudyResourceV1",
    "StudyResultV1",
    "StudyV1",
    "TraceAuditDraftV1",
    "TraceAuditPreviewV1",
    "TraceAuditService",
    "TraceAuditV1",
    "TraceSourceRefV1",
    "TraceSourceRegistry",
    "build_trace_audit_draft",
]
