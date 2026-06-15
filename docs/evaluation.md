# Evaluation

ProcurePilot includes lightweight regression scripts that verify the MVP workflow without requiring external services.
The fixture scores should be read as regression coverage for the included scenarios, not as an external benchmark.

## Commands

```powershell
.\.venv\Scripts\python scripts\evaluate.py
.\.venv\Scripts\python scripts\check_boundary_cases.py
.\.venv\Scripts\python scripts\check_llm_compliance_flag.py
```

The scripts run in rule-fallback mode by default and do not require an API key.

## Main Evaluation Set

`scripts/evaluate.py` loads `app/data/evaluation_cases.json`, runs each case through `ProcurePilotAgent`, and checks:

- Field extraction accuracy.
- Missing-question recall and precision.
- Policy citation correctness.
- Tool selection precision and recall.
- Tool execution success.
- Risk precision and recall.
- Human-review escalation accuracy.
- Recommendation type accuracy.
- Final state accuracy.

The generated report is written to `reports/evaluation-report.md`.

## Boundary Cases

`scripts/check_boundary_cases.py` verifies that out-of-scope, casual, and compliance-risk inputs do not enter unsupported tool or policy flows.

## LLM Compliance Flag Check

`scripts/check_llm_compliance_flag.py` uses a fake LLM provider to ensure `USER_COMPLIANCE_RISK` from LLM extraction is merged into the agent workflow and leads to the expected high-risk handling.

## CI

The GitHub Actions workflow runs:

```text
python -m compileall app scripts
python scripts/check_boundary_cases.py
python scripts/check_llm_compliance_flag.py
python scripts/evaluate.py
```

This keeps the public repository reproducible without external API credentials.
