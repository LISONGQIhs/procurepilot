# Architecture

ProcurePilot is organized around a controlled procurement pre-check workflow. The agent does not free-form approve requests; it moves through explicit states and records intermediate evidence for review.

## Workflow

```text
Natural-language purchase request
-> intent classification
-> field extraction
-> missing information check
-> policy retrieval
-> tool planning
-> business tool calls
-> risk assessment
-> recommendation generation
-> human review gate or completion
-> trace and audit output
```

## Main Components

| Component | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI app, static UI, and REST endpoints |
| `app/models.py` | Pydantic schemas for runs, purchase fields, citations, tools, risks, and recommendations |
| `app/agent/workflow.py` | Agent state flow, risk rules, recommendation rules, and human-review transitions |
| `app/agent/extractor.py` | Rule-based purchase field extraction |
| `app/agent/hybrid_extractor.py` | LLM-first extraction with rule fallback |
| `app/agent/rag.py` | Local policy retrieval and citation support tagging |
| `app/llm/provider.py` | OpenAI-compatible JSON chat calls and validation safeguards |
| `app/tools/business_tools.py` | Simulated budget, vendor, price, and approval-chain tools |
| `app/services/store.py` | In-memory run storage |

## State Model

The main run states are:

- `INPUT_RECEIVED`
- `EXTRACTING_REQUIREMENTS`
- `NEED_INFO`
- `RETRIEVING_POLICY`
- `PLANNING_TOOLS`
- `CALLING_TOOLS`
- `ASSESSING_RISK`
- `GENERATING_RECOMMENDATION`
- `WAITING_HUMAN_APPROVAL`
- `COMPLETED`
- `FAILED`

Each transition can add an `AgentTraceEvent`, allowing the UI and API consumers to inspect what happened during a run.

## Evidence Boundaries

ProcurePilot keeps three evidence sources separate:

- User input and extracted procurement fields.
- Policy citations retrieved from local policy fixtures.
- Business tool outputs from simulated operational data.

Policy text is only used as formal evidence when the citation has `supports_conclusion=true`. Business tool outputs are referenced by tool call IDs and can affect budget, vendor, price, and approval-path risks.

## LLM Safety Boundary

The LLM provider is optional. When disabled or unavailable, the system falls back to rule-based behavior. When enabled, LLM output is validated before it can affect the workflow:

- Intent classification is restricted to a small schema.
- Field extraction must match a Pydantic schema.
- Recommendation polishing cannot contradict the structured recommendation.
- Boundary replies cannot claim unsupported procurement facts or tool results.

Final risk level, recommendation type, and human-review requirement are still controlled by deterministic workflow logic.
