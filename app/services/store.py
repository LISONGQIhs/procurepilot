from __future__ import annotations

from datetime import datetime

from app.models import AgentRun, short_id, utc_now


class RunStore:
    """In-memory store for the MVP demo.

    The workflow is intentionally isolated behind this store so the MVP can
    later move to SQLite/PostgreSQL without changing the agent contract.
    """

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    def create(self, original_input: str) -> AgentRun:
        now = utc_now()
        run = AgentRun(
            run_id=f"RUN-{datetime.now().strftime('%Y%m%d')}-{short_id('')[-8:]}",
            original_input=original_input,
            created_at=now,
            updated_at=now,
        )
        self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def save(self, run: AgentRun) -> AgentRun:
        run.updated_at = utc_now()
        self._runs[run.run_id] = run
        return run

    def list(self) -> list[AgentRun]:
        return sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)
