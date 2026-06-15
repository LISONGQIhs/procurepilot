from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.extractor import extract_purchase_request
from app.llm.provider import LLMCallRecord, LLMProvider, LLMPurchaseExtraction
from app.models import PurchaseRequest


ALLOWED_LLM_EXTRACTION_FLAGS = {
    "USER_COMPLIANCE_RISK",
    "QUANTITY_AMBIGUOUS",
    "DEPARTMENT_AMBIGUOUS",
    "POSSIBLE_AMOUNT_CONFLICT",
}


@dataclass
class HybridExtractionResult:
    purchase: PurchaseRequest
    llm_records: list[LLMCallRecord] = field(default_factory=list)
    fallback_used: bool = False
    source: str = "rule"


class HybridExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def extract(self, message: str) -> HybridExtractionResult:
        rule_purchase = extract_purchase_request(message)
        llm_purchase, record = self.provider.extract_purchase_request(message)
        records = [record]
        if llm_purchase is None:
            return HybridExtractionResult(
                purchase=rule_purchase,
                llm_records=records,
                fallback_used=True,
                source="rule_fallback",
            )

        merged = merge_purchase(rule_purchase, llm_purchase)
        return HybridExtractionResult(
            purchase=merged,
            llm_records=records,
            fallback_used=record.fallback_used,
            source="llm_hybrid" if record.success else "rule_fallback",
        )


def merge_purchase(rule_purchase: PurchaseRequest, llm_purchase: LLMPurchaseExtraction) -> PurchaseRequest:
    merged = rule_purchase.model_copy(deep=True)
    llm_data = llm_purchase.model_dump()

    for field_name, value in llm_data.items():
        if field_name == "flags":
            continue
        if value is None or value == "":
            continue
        if field_name in {"quantity", "amount"} and not is_positive_number(value):
            continue
        current = getattr(merged, field_name, None)
        if current in {None, "", False}:
            setattr(merged, field_name, value)
            merged.source_fields.setdefault(field_name, "LLM 结构化抽取")
            merged.confidence.setdefault(field_name, 0.75)

    for flag in llm_purchase.flags:
        if isinstance(flag, str) and flag and flag not in merged.flags:
            if flag in ALLOWED_LLM_EXTRACTION_FLAGS:
                merged.flags.append(flag)

    # Preserve explicit rule detections for compliance and known numeric fields.
    for field_name in ["amount", "quantity"]:
        rule_value = getattr(rule_purchase, field_name)
        if rule_value is not None:
            setattr(merged, field_name, rule_value)
            merged.source_fields[field_name] = rule_purchase.source_fields.get(field_name, "规则抽取")
            merged.confidence[field_name] = rule_purchase.confidence.get(field_name, 0.9)

    if "USER_COMPLIANCE_RISK" in rule_purchase.flags and "USER_COMPLIANCE_RISK" not in merged.flags:
        merged.flags.append("USER_COMPLIANCE_RISK")

    return merged


def is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False
