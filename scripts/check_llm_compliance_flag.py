from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.hybrid_extractor import HybridExtractor  # noqa: E402
from app.agent.workflow import ProcurePilotAgent  # noqa: E402
from app.llm.provider import LLMCallRecord, LLMIntentResult, LLMPurchaseExtraction  # noqa: E402
from app.services.store import RunStore  # noqa: E402


class FakeLLMProvider:
    @property
    def enabled(self) -> bool:
        return True

    def classify_intent(self, message: str) -> tuple[LLMIntentResult, LLMCallRecord]:
        return (
            LLMIntentResult(intent="procurement", reason="测试桩：意图识别不直接标记合规风险。"),
            LLMCallRecord(
                purpose="intent_classification",
                model="fake-llm",
                enabled=True,
                success=True,
                elapsed_ms=0,
                output_summary={"intent": "procurement"},
            ),
        )

    def extract_purchase_request(self, message: str) -> tuple[LLMPurchaseExtraction, LLMCallRecord]:
        return (
            LLMPurchaseExtraction(
                department="市场部",
                purchase_category="办公用品",
                item_name="采购服务",
                quantity=1,
                amount=90000,
                purpose="测试 LLM 抽取合规风险 flag",
                flags=["USER_COMPLIANCE_RISK"],
            ),
            LLMCallRecord(
                purpose="purchase_extraction",
                model="fake-llm",
                enabled=True,
                success=True,
                elapsed_ms=0,
                output_summary={"flags": ["USER_COMPLIANCE_RISK"]},
            ),
        )

    def polish_recommendation(self, context: dict[str, object]) -> tuple[None, LLMCallRecord]:
        return (
            None,
            LLMCallRecord(
                purpose="recommendation_polish",
                model="fake-llm",
                enabled=True,
                success=False,
                elapsed_ms=0,
                fallback_used=True,
                error_message="test disables polish",
            ),
        )


def main() -> int:
    provider = FakeLLMProvider()
    agent = ProcurePilotAgent(RunStore())
    agent.llm_provider = provider  # type: ignore[assignment]
    agent.extractor = HybridExtractor(provider)  # type: ignore[arg-type]

    run = agent.precheck("市场部想采购一项服务，预算 9 万。")
    result = {
        "state": run.current_state,
        "recommendation_type": run.recommendation.recommendation_type if run.recommendation else None,
        "human_review_required": run.recommendation.human_review_required if run.recommendation else None,
        "flags": run.purchase_request.flags if run.purchase_request else [],
        "risk_types": [risk.risk_type for risk in run.risk_findings],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    failures: list[str] = []
    if "USER_COMPLIANCE_RISK" not in result["flags"]:
        failures.append("LLM extraction flag was not merged into purchase.flags")
    if "USER_COMPLIANCE_RISK" not in result["risk_types"]:
        failures.append("USER_COMPLIANCE_RISK did not enter risk findings")
    if result["recommendation_type"] != "REJECT_RECOMMENDED":
        failures.append(f"recommendation expected=REJECT_RECOMMENDED actual={result['recommendation_type']}")
    if result["state"] != "WAITING_HUMAN_APPROVAL":
        failures.append(f"state expected=WAITING_HUMAN_APPROVAL actual={result['state']}")
    if result["human_review_required"] is not True:
        failures.append("human_review_required expected=True")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
