# Contributing

This repository is an MVP codebase. Keep changes small, testable, and aligned with the existing explicit workflow design.

## Local Checks

```powershell
python -m compileall app scripts
python scripts/check_boundary_cases.py
python scripts/check_llm_compliance_flag.py
python scripts/evaluate.py
```

## Guidelines

- Do not commit `.env`, API keys, real procurement data, customer data, or internal documents.
- Keep policy evidence, tool evidence, and LLM output boundaries explicit.
- Prefer deterministic workflow rules for final risk and approval decisions.
- Add or update evaluation cases when changing extraction, RAG, tool planning, or risk rules.
