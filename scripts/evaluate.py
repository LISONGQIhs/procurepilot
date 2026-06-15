from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.workflow import ProcurePilotAgent  # noqa: E402
from app.services.store import RunStore  # noqa: E402


CASES_PATH = ROOT / "app" / "data" / "evaluation_cases.json"
REPORT_PATH = ROOT / "reports" / "evaluation-report.md"

POLICY_BACKED_RISKS = {
    "AMOUNT_RISK",
    "CATEGORY_POLICY_RISK",
    "VENDOR_QUALIFICATION_RISK",
    "VENDOR_BLACKLIST_RISK",
    "BUDGET_RISK",
    "PRICE_ANOMALY_RISK",
    "USER_COMPLIANCE_RISK",
}


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 0.01
        except (TypeError, ValueError):
            return False
    return actual == expected


def expected_recommendation_type(case: dict[str, Any]) -> str:
    if "expected_recommendation_type" in case:
        return case["expected_recommendation_type"]
    if case["expected_state"] == "NEED_INFO":
        return "NEED_MORE_INFO"
    if case["expected_human_review"]:
        if "USER_COMPLIANCE_RISK" in case.get("expected_risks", []):
            return "REJECT_RECOMMENDED"
        return "HUMAN_REVIEW_REQUIRED"

    fields = case.get("expected_fields", {})
    amount = fields.get("amount") or 0
    if amount >= 10000 or fields.get("vendor_name"):
        return "SUBMIT_AFTER_SUPPLEMENT"
    return "SUBMIT_RECOMMENDED"


def add_fraction(metrics: dict[str, int], prefix: str, numerator: int, denominator: int) -> None:
    metrics[f"{prefix}_num"] = metrics.get(f"{prefix}_num", 0) + numerator
    metrics[f"{prefix}_den"] = metrics.get(f"{prefix}_den", 0) + denominator


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.1%}"


def main() -> int:
    agent = ProcurePilotAgent(RunStore())
    cases = load_cases()
    metrics: dict[str, int] = {}
    failures: list[str] = []

    for case in cases:
        run = agent.precheck(case["input"])
        purchase = run.purchase_request
        actual_fields = purchase.model_dump() if purchase else {}
        actual_tools = [call.tool_name for call in run.tool_calls]
        successful_tools = [call.tool_name for call in run.tool_calls if call.status == "SUCCESS"]
        actual_tool_set = set(actual_tools)
        expected_tool_set = set(case.get("expected_tools", []))
        actual_risk_set = {risk.risk_type for risk in run.risk_findings}
        expected_risk_set = set(case.get("expected_risks", []))
        actual_missing_set = {question.field_name for question in run.missing_questions}
        expected_missing_set = set(case.get("expected_missing_questions", []))
        actual_human = bool(run.recommendation and run.recommendation.human_review_required)
        actual_recommendation = run.recommendation.recommendation_type if run.recommendation else None

        check_fields(case, actual_fields, metrics, failures)
        check_missing_questions(case, actual_missing_set, expected_missing_set, metrics, failures)
        check_rag(case, run, metrics, failures)
        check_tools(case, actual_tool_set, expected_tool_set, successful_tools, metrics, failures)
        check_risks(case, actual_risk_set, expected_risk_set, metrics, failures)

        add_fraction(metrics, "human_accuracy", int(actual_human == case["expected_human_review"]), 1)
        if actual_human != case["expected_human_review"]:
            failures.append(f"{case['id']} 人工升级: expected={case['expected_human_review']}, actual={actual_human}")

        expected_state = case["expected_state"]
        add_fraction(metrics, "state_accuracy", int(run.current_state == expected_state), 1)
        if run.current_state != expected_state:
            failures.append(f"{case['id']} 状态: expected={expected_state}, actual={run.current_state}")

        expected_rec = expected_recommendation_type(case)
        add_fraction(metrics, "recommendation_accuracy", int(actual_recommendation == expected_rec), 1)
        if actual_recommendation != expected_rec:
            failures.append(f"{case['id']} 建议类型: expected={expected_rec}, actual={actual_recommendation}")

    report = build_report(len(cases), metrics, failures)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    return 0 if not failures else 1


def check_fields(
    case: dict[str, Any],
    actual_fields: dict[str, Any],
    metrics: dict[str, int],
    failures: list[str],
) -> None:
    for field, expected in case.get("expected_fields", {}).items():
        matched = value_matches(actual_fields.get(field), expected)
        add_fraction(metrics, "field_accuracy", int(matched), 1)
        if not matched:
            failures.append(f"{case['id']} 字段 {field}: expected={expected!r}, actual={actual_fields.get(field)!r}")


def check_missing_questions(
    case: dict[str, Any],
    actual_missing_set: set[str],
    expected_missing_set: set[str],
    metrics: dict[str, int],
    failures: list[str],
) -> None:
    if not expected_missing_set:
        return
    hit = len(actual_missing_set & expected_missing_set)
    add_fraction(metrics, "missing_recall", hit, len(expected_missing_set))
    add_fraction(metrics, "missing_precision", hit, len(actual_missing_set) or 1)
    missing = sorted(expected_missing_set - actual_missing_set)
    extra = sorted(actual_missing_set - expected_missing_set)
    if missing:
        failures.append(f"{case['id']} 缺失追问漏识别: {', '.join(missing)}")
    if extra:
        failures.append(f"{case['id']} 缺失追问多余项: {', '.join(extra)}")


def check_rag(case: dict[str, Any], run: Any, metrics: dict[str, int], failures: list[str]) -> None:
    supported_citations = [citation for citation in run.policy_citations if citation.supports_conclusion]
    supported_docs = {citation.doc_title for citation in supported_citations}
    supported_risks = {citation.risk_type for citation in supported_citations}
    citations_by_id = {citation.citation_id: citation for citation in run.policy_citations}

    for doc in case.get("expected_policy_docs", []):
        ok = doc in supported_docs
        add_fraction(metrics, "rag_correctness", int(ok), 1)
        if not ok:
            failures.append(f"{case['id']} 未命中可支撑结论的制度文档: {doc}")

    expected_policy_risks = set(case.get("expected_policy_risks", []))
    if not expected_policy_risks:
        expected_policy_risks = set(case.get("expected_risks", [])) & POLICY_BACKED_RISKS
    for risk_type in expected_policy_risks:
        ok = risk_type in supported_risks
        add_fraction(metrics, "rag_correctness", int(ok), 1)
        if not ok:
            failures.append(f"{case['id']} 未命中可支撑 {risk_type} 的制度引用")

    for risk in run.risk_findings:
        if risk.evidence_type != "制度":
            continue
        refs_ok = bool(risk.evidence_refs) and all(
            citations_by_id.get(ref) and citations_by_id[ref].supports_conclusion and citations_by_id[ref].risk_type == risk.risk_type
            for ref in risk.evidence_refs
        )
        add_fraction(metrics, "rag_correctness", int(refs_ok), 1)
        if not refs_ok:
            failures.append(f"{case['id']} 风险 {risk.risk_type} 引用了不能支撑结论的制度依据")


def check_tools(
    case: dict[str, Any],
    actual_tool_set: set[str],
    expected_tool_set: set[str],
    successful_tools: list[str],
    metrics: dict[str, int],
    failures: list[str],
) -> None:
    selected = actual_tool_set & expected_tool_set
    precision_num, precision_den = set_score(len(selected), len(actual_tool_set), len(expected_tool_set))
    recall_num, recall_den = set_score(len(selected), len(expected_tool_set), len(actual_tool_set))
    add_fraction(metrics, "tool_precision", precision_num, precision_den)
    add_fraction(metrics, "tool_recall", recall_num, recall_den)
    exact = actual_tool_set == expected_tool_set
    add_fraction(metrics, "tool_selection_accuracy", int(exact), 1)
    add_fraction(metrics, "tool_success", len(successful_tools), len(actual_tool_set))

    missing = sorted(expected_tool_set - actual_tool_set)
    extra = sorted(actual_tool_set - expected_tool_set)
    if missing:
        failures.append(f"{case['id']} 未调用工具: {', '.join(missing)}")
    if extra:
        failures.append(f"{case['id']} 多调用工具: {', '.join(extra)}")
    failed = sorted(actual_tool_set - set(successful_tools))
    if failed:
        failures.append(f"{case['id']} 工具执行失败: {', '.join(failed)}")


def check_risks(
    case: dict[str, Any],
    actual_risk_set: set[str],
    expected_risk_set: set[str],
    metrics: dict[str, int],
    failures: list[str],
) -> None:
    identified = actual_risk_set & expected_risk_set
    precision_num, precision_den = set_score(len(identified), len(actual_risk_set), len(expected_risk_set))
    recall_num, recall_den = set_score(len(identified), len(expected_risk_set), len(actual_risk_set))
    add_fraction(metrics, "risk_precision", precision_num, precision_den)
    add_fraction(metrics, "risk_recall", recall_num, recall_den)
    missing = sorted(expected_risk_set - actual_risk_set)
    extra = sorted(actual_risk_set - expected_risk_set)
    if missing:
        failures.append(f"{case['id']} 未识别风险: {', '.join(missing)}")
    if extra:
        failures.append(f"{case['id']} 多识别风险: {', '.join(extra)}")


def set_score(overlap: int, primary_size: int, secondary_size: int) -> tuple[int, int]:
    if primary_size == 0 and secondary_size == 0:
        return 1, 1
    if primary_size == 0:
        return 0, 1
    return overlap, primary_size


def metric(metrics: dict[str, int], prefix: str) -> str:
    return pct(metrics.get(f"{prefix}_num", 0), metrics.get(f"{prefix}_den", 0))


def build_report(total_cases: int, metrics: dict[str, int], failures: list[str]) -> str:
    lines = [
        "# ProcurePilot Regression Report",
        "",
        "This report is generated from local fixture cases. It is intended as an MVP regression check, not an external benchmark.",
        "",
        f"- Fixture cases: {total_cases}",
        f"- Field extraction accuracy: {metric(metrics, 'field_accuracy')}",
        f"- Missing-field recall: {metric(metrics, 'missing_recall')}",
        f"- Missing-field precision: {metric(metrics, 'missing_precision')}",
        f"- Policy citation correctness: {metric(metrics, 'rag_correctness')}",
        f"- Tool selection precision: {metric(metrics, 'tool_precision')}",
        f"- Tool selection recall: {metric(metrics, 'tool_recall')}",
        f"- Tool selection exact-match accuracy: {metric(metrics, 'tool_selection_accuracy')}",
        f"- Tool execution success rate: {metric(metrics, 'tool_success')}",
        f"- Risk precision: {metric(metrics, 'risk_precision')}",
        f"- Risk recall: {metric(metrics, 'risk_recall')}",
        f"- Human-review escalation accuracy: {metric(metrics, 'human_accuracy')}",
        f"- Recommendation type accuracy: {metric(metrics, 'recommendation_accuracy')}",
        f"- Final state accuracy: {metric(metrics, 'state_accuracy')}",
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures[:80])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Follow-up Work",
            "",
            "- Add a separate LLM-enabled evaluation set for structured extraction and intent classification.",
            "- Add labeled retrieval relevance checks if vector or hybrid retrieval is introduced.",
            "- Expand dedicated cases for tool failures, conflicting inputs, and uncertain budget categories.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
