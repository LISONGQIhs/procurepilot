from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def short_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8].upper()}"


AgentState = Literal[
    "INPUT_RECEIVED",
    "EXTRACTING_REQUIREMENTS",
    "NEED_INFO",
    "RETRIEVING_POLICY",
    "PLANNING_TOOLS",
    "CALLING_TOOLS",
    "ASSESSING_RISK",
    "GENERATING_RECOMMENDATION",
    "WAITING_HUMAN_APPROVAL",
    "COMPLETED",
    "FAILED",
]

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "BLOCKED"]


class PurchaseRequest(BaseModel):
    department: str | None = None
    requester: str | None = None
    purchase_category: str | None = None
    item_name: str | None = None
    quantity: int | None = None
    amount: float | None = None
    currency: str = "CNY"
    purpose: str | None = None
    vendor_name: str | None = None
    delivery_requirement: str | None = None
    budget_category: str | None = None
    budget_category_confirmed: bool = False
    is_urgent: bool = False
    specified_brand_or_model: bool = False
    source_fields: dict[str, str] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class MissingInfoQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: short_id("Q"))
    field_name: str
    question_text: str
    required: bool = True
    reason: str
    status: str = "PENDING"
    answer: str | None = None


class PolicyCitation(BaseModel):
    citation_id: str = Field(default_factory=lambda: short_id("CIT"))
    doc_id: str
    doc_title: str
    section_id: str
    section_title: str
    content_excerpt: str
    policy_type: str
    risk_type: str
    relevance_score: float
    retrieval_source: Literal["retrieved", "rule_injected", "fallback"] = "retrieved"
    supports_conclusion: bool = False


class ToolCallRecord(BaseModel):
    tool_call_id: str = Field(default_factory=lambda: short_id("TOOL"))
    tool_name: str
    call_reason: str
    input_args: dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    output_data: dict[str, Any] = Field(default_factory=dict)
    status: Literal["SUCCESS", "FAILED", "SKIPPED"] = "SUCCESS"
    error_message: str | None = None
    risk_impact: str | None = None
    called_at: datetime = Field(default_factory=utc_now)


class RiskFinding(BaseModel):
    risk_id: str = Field(default_factory=lambda: short_id("RISK"))
    risk_type: str
    risk_level: RiskLevel
    title: str
    description: str
    evidence_type: str
    evidence_refs: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    suggested_action: str


class ApprovalRecommendation(BaseModel):
    recommendation_type: Literal[
        "SUBMIT_RECOMMENDED",
        "SUBMIT_AFTER_SUPPLEMENT",
        "HUMAN_REVIEW_REQUIRED",
        "HUMAN_APPROVED_TO_CONTINUE",
        "PAUSE_RECOMMENDED",
        "REJECT_RECOMMENDED",
        "NEED_MORE_INFO",
        "OUT_OF_SCOPE",
    ]
    summary: str
    risk_level: RiskLevel
    reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    policy_citation_ids: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    target_reviewer_roles: list[str] = Field(default_factory=list)


class HumanReview(BaseModel):
    review_id: str = Field(default_factory=lambda: short_id("HR"))
    required: bool = True
    reason: str
    reviewer_role: str
    status: Literal["PENDING", "APPROVED", "REJECTED", "MORE_INFO_REQUIRED", "TRANSFERRED"] = "PENDING"
    decision: str | None = None
    comment: str | None = None
    reviewed_at: datetime | None = None


class AgentTraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: short_id("EVT"))
    run_id: str
    state: AgentState
    event_type: str
    title: str
    detail: str
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentRun(BaseModel):
    run_id: str
    original_input: str
    current_state: AgentState = "INPUT_RECEIVED"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    purchase_request: PurchaseRequest | None = None
    missing_questions: list[MissingInfoQuestion] = Field(default_factory=list)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    recommendation: ApprovalRecommendation | None = None
    human_review: HumanReview | None = None
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)


class HumanReviewRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REQUEST_MORE_INFO", "TRANSFER"]
    reviewer_role: str = "采购专员"
    comment: str | None = None


class PrecheckRequest(BaseModel):
    message: str = Field(min_length=1)
    run_id: str | None = None
