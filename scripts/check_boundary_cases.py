from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.workflow import ProcurePilotAgent  # noqa: E402
from app.services.store import RunStore  # noqa: E402


def main() -> int:
    cases = json.loads((ROOT / "app" / "data" / "boundary_cases.json").read_text(encoding="utf-8"))
    agent = ProcurePilotAgent(RunStore())
    failures: list[str] = []
    rows: list[dict[str, object]] = []

    for case in cases:
        run = agent.precheck(case["input"])
        recommendation_type = run.recommendation.recommendation_type if run.recommendation else None
        tools = {call.tool_name for call in run.tool_calls}
        risks = {risk.risk_type for risk in run.risk_findings}
        human = bool(run.recommendation and run.recommendation.human_review_required)
        rows.append(
            {
                "id": case["id"],
                "state": run.current_state,
                "recommendation_type": recommendation_type,
                "tools": sorted(tools),
                "risks": sorted(risks),
                "human_review_required": human,
                "llm_trace_events": len([event for event in run.trace_events if event.event_type == "LLM_CALLED"]),
            }
        )

        if run.current_state != case["expected_state"]:
            failures.append(f"{case['id']} state expected={case['expected_state']} actual={run.current_state}")
        if recommendation_type != case["expected_recommendation_type"]:
            failures.append(
                f"{case['id']} recommendation expected={case['expected_recommendation_type']} actual={recommendation_type}"
            )
        if human != case["expected_human_review"]:
            failures.append(f"{case['id']} human expected={case['expected_human_review']} actual={human}")
        if "expected_tools" in case:
            expected_tools = set(case.get("expected_tools", []))
            if expected_tools and not expected_tools.issubset(tools):
                failures.append(f"{case['id']} tools missing={sorted(expected_tools - tools)}")
            if not expected_tools and tools:
                failures.append(f"{case['id']} expected no tools actual={sorted(tools)}")
        expected_risks = set(case.get("expected_risks", []))
        if expected_risks and not expected_risks.issubset(risks):
            failures.append(f"{case['id']} risks missing={sorted(expected_risks - risks)}")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
