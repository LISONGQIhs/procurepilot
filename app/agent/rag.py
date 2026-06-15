from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models import PolicyCitation, PurchaseRequest


class PolicyRetriever:
    def __init__(self, policy_path: Path | None = None) -> None:
        self.policy_path = policy_path or Path(__file__).resolve().parents[1] / "data" / "policies.json"
        self._chunks: list[dict[str, Any]] = json.loads(self.policy_path.read_text(encoding="utf-8"))

    def retrieve(self, purchase: PurchaseRequest, raw_text: str, limit: int = 8) -> list[PolicyCitation]:
        terms = self._build_terms(purchase, raw_text)
        mandatory_keys = self._mandatory_keys(purchase)
        scored_by_key: dict[tuple[str, str], tuple[float, dict[str, Any], str]] = {}

        for chunk in self._chunks:
            key = (chunk["doc_id"], chunk["section_id"])
            haystack = " ".join(
                [
                    chunk["doc_title"],
                    chunk["section_title"],
                    chunk["policy_type"],
                    chunk["category"],
                    chunk["risk_type"],
                    " ".join(chunk.get("tags", [])),
                    chunk["content"],
                ]
            )
            score = 0.0
            for term in terms:
                if term and term in haystack:
                    score += 1.0
            if purchase.purchase_category and purchase.purchase_category == chunk.get("category"):
                score += 2.0
            if chunk.get("category") == "通用":
                score += 0.3
            if purchase.vendor_name and chunk.get("policy_type") == "vendor":
                score += 2.5
            if purchase.amount is not None and chunk.get("risk_type") in {"AMOUNT_RISK", "APPROVAL_PATH_RISK"}:
                score += 1.5
            if purchase.is_urgent and chunk.get("doc_id") == "URGENT-001":
                score += 3.0
            if "USER_COMPLIANCE_RISK" in purchase.flags and chunk.get("risk_type") == "USER_COMPLIANCE_RISK":
                score += 4.0
            if score > 0:
                scored_by_key[key] = (score, chunk, "retrieved")

            if key in mandatory_keys:
                source = "retrieved" if score > 0 else "rule_injected"
                boosted_score = score + 6.0 if score > 0 else 6.0
                previous = scored_by_key.get(key)
                if previous is None or boosted_score > previous[0]:
                    scored_by_key[key] = (boosted_score, chunk, source)

        scored = sorted(scored_by_key.values(), key=lambda item: item[0], reverse=True)
        citations: list[PolicyCitation] = []
        for score, chunk, retrieval_source in scored:
            supports = self._supports_conclusion(chunk, purchase)
            citations.append(
                PolicyCitation(
                    doc_id=chunk["doc_id"],
                    doc_title=chunk["doc_title"],
                    section_id=chunk["section_id"],
                    section_title=chunk["section_title"],
                    content_excerpt=chunk["content"],
                    policy_type=chunk["policy_type"],
                    risk_type=chunk["risk_type"],
                    relevance_score=round(min(score / 8, 1.0), 2),
                    retrieval_source=retrieval_source,  # type: ignore[arg-type]
                    supports_conclusion=supports,
                )
            )
            if len(citations) >= limit:
                break
        return citations

    def _mandatory_keys(self, purchase: PurchaseRequest) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        if purchase.amount is not None:
            keys.add(("BUDGET-001", "3.2"))
            if purchase.amount < 10000:
                keys.add(("PROC-001", "4.1"))
            elif purchase.amount >= 100000:
                keys.add(("PROC-001", "4.2"))
            else:
                keys.add(("PROC-001", "4.2"))
        if purchase.purchase_category == "办公用品":
            keys.add(("OFFICE-001", "2.1"))
        if purchase.purchase_category == "IT设备":
            keys.add(("IT-001", "3.1"))
            keys.add(("IT-001", "3.3"))
        if purchase.vendor_name:
            keys.add(("VENDOR-001", "2.1"))
            keys.add(("VENDOR-001", "2.4"))
        if purchase.is_urgent:
            keys.add(("URGENT-001", "2.2"))
        if "USER_COMPLIANCE_RISK" in purchase.flags:
            keys.add(("PROC-001", "7.2"))
        return keys

    def _supports_conclusion(self, chunk: dict[str, Any], purchase: PurchaseRequest) -> bool:
        risk_type = chunk.get("risk_type")
        doc_id = chunk.get("doc_id")
        section_id = chunk.get("section_id")

        if risk_type == "AMOUNT_RISK":
            if purchase.amount is None:
                return False
            if section_id == "4.1":
                return purchase.amount < 10000
            if section_id == "4.2":
                return 10000 <= purchase.amount <= 500000
            return False

        if risk_type == "APPROVAL_PATH_RISK":
            return purchase.amount is not None and purchase.amount > 500000

        if risk_type == "CATEGORY_POLICY_RISK":
            return chunk.get("category") == purchase.purchase_category

        if risk_type == "PRICE_ANOMALY_RISK":
            return purchase.purchase_category == "IT设备" and purchase.amount is not None

        if risk_type in {"VENDOR_QUALIFICATION_RISK", "VENDOR_BLACKLIST_RISK"}:
            return bool(purchase.vendor_name)

        if risk_type == "BUDGET_RISK":
            return bool(purchase.department and purchase.amount is not None and purchase.budget_category)

        if risk_type == "MISSING_INFO":
            return bool(purchase.is_urgent and doc_id == "URGENT-001")

        if risk_type == "USER_COMPLIANCE_RISK":
            return "USER_COMPLIANCE_RISK" in purchase.flags

        return False

    def _build_terms(self, purchase: PurchaseRequest, raw_text: str) -> list[str]:
        terms = ["采购", "审批", "预算"]
        if purchase.purchase_category:
            terms.append(purchase.purchase_category)
        if purchase.item_name:
            terms.append(purchase.item_name)
        if purchase.vendor_name:
            terms.extend(["供应商", "指定供应商", purchase.vendor_name])
        if purchase.amount is not None:
            if purchase.amount < 10000:
                terms.extend(["低额", "1万元", "简化审批"])
            elif purchase.amount >= 100000:
                terms.extend(["10万", "50万", "比价", "采购专员"])
        if purchase.is_urgent:
            terms.extend(["紧急", "尽快", "到货"])
        if "USER_COMPLIANCE_RISK" in purchase.flags or any(word in raw_text for word in ["拆单", "拆分", "规避"]):
            terms.extend(["拆单", "规避"])
        return terms
