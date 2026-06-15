# ProcurePilot Regression Report

This report is generated from local fixture cases. It is intended as an MVP regression check, not an external benchmark.

- Fixture cases: 30
- Field extraction accuracy: 100.0%
- Missing-field recall: 100.0%
- Missing-field precision: 100.0%
- Policy citation correctness: 100.0%
- Tool selection precision: 100.0%
- Tool selection recall: 100.0%
- Tool selection exact-match accuracy: 100.0%
- Tool execution success rate: 100.0%
- Risk precision: 100.0%
- Risk recall: 100.0%
- Human-review escalation accuracy: 100.0%
- Recommendation type accuracy: 100.0%
- Final state accuracy: 100.0%

## Failures

- None

## Follow-up Work

- Add a separate LLM-enabled evaluation set for structured extraction and intent classification.
- Add labeled retrieval relevance checks if vector or hybrid retrieval is introduced.
- Expand dedicated cases for tool failures, conflicting inputs, and uncertain budget categories.
