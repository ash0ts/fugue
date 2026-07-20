"""Stable public campaign-orchestration surface.

The implementation lives in focused internal modules so callers do not need to
track storage, reconciliation, or lifecycle refactors.
"""

from fugue.bench.campaign_contracts import (
    CAMPAIGN_SCHEMA_VERSION,
    AdmissionReceiptV1,
    CampaignCatalogSnapshotV1,
    CampaignError,
    CampaignEventV1,
    CampaignLimitsV1,
    CampaignStagePolicyV1,
    CampaignStatusV1,
    ExperimentProposalV1,
    OutcomePacketV1,
    PlanReceiptV1,
    PreparedPlanV1,
    ResearchCampaignSpecV1,
)
from fugue.bench.campaign_lifecycle import (
    CampaignService,
    admission_receipt_from_dict,
    build_experiment_proposal,
    campaign_catalog_snapshot_from_dict,
    campaign_error_from_dict,
    campaign_event_from_dict,
    campaign_spec_from_dict,
    campaign_status_from_dict,
    experiment_proposal_from_dict,
    get_campaign,
    list_campaigns,
    outcome_packet_from_dict,
    plan_receipt_from_dict,
    prepared_plan_from_dict,
)

__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "AdmissionReceiptV1",
    "CampaignCatalogSnapshotV1",
    "CampaignError",
    "CampaignEventV1",
    "CampaignLimitsV1",
    "CampaignService",
    "CampaignStagePolicyV1",
    "CampaignStatusV1",
    "ExperimentProposalV1",
    "OutcomePacketV1",
    "PlanReceiptV1",
    "PreparedPlanV1",
    "ResearchCampaignSpecV1",
    "admission_receipt_from_dict",
    "build_experiment_proposal",
    "campaign_catalog_snapshot_from_dict",
    "campaign_error_from_dict",
    "campaign_event_from_dict",
    "campaign_spec_from_dict",
    "campaign_status_from_dict",
    "experiment_proposal_from_dict",
    "get_campaign",
    "list_campaigns",
    "outcome_packet_from_dict",
    "plan_receipt_from_dict",
    "prepared_plan_from_dict",
]
