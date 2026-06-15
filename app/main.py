from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.workflow import ProcurePilotAgent
from app.models import AgentRun, HumanReviewRequest, PrecheckRequest
from app.services.store import RunStore


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

store = RunStore()
agent = ProcurePilotAgent(store)

app = FastAPI(
    title="ProcurePilot API",
    version="0.1.0",
    description="企业采购合规预审智能体 MVP",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ProcurePilot"}


@app.post("/api/precheck", response_model=AgentRun)
def precheck(request: PrecheckRequest) -> AgentRun:
    try:
        return agent.precheck(request.message, request.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs", response_model=list[AgentRun])
def list_runs() -> list[AgentRun]:
    return store.list()


@app.get("/api/runs/{run_id}", response_model=AgentRun)
def get_run(run_id: str) -> AgentRun:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.post("/api/runs/{run_id}/human-review", response_model=AgentRun)
def human_review(run_id: str, request: HumanReviewRequest) -> AgentRun:
    try:
        return agent.apply_human_review(run_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/demo-cases")
def demo_cases() -> list[dict[str, str]]:
    return json.loads((DATA_DIR / "demo_cases.json").read_text(encoding="utf-8"))
