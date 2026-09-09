"""Read-oriented API plus an operational rerun trigger; no manual stock ingestion."""
from fastapi import APIRouter, HTTPException, Query

from .domain import ACTIVE_CANDIDATE_STATES, TERMINAL_CANDIDATE_STATES
from .service import get_service

router = APIRouter(prefix="/api/custom/ai-research", tags=["ai-research"])


@router.get("/status")
def status() -> dict:
    try:
        return get_service().status()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/candidates")
def candidates(
    state: str | None = Query(default=None),
    limit: int = Query(default=300, ge=1, le=1000),
) -> dict:
    allowed = set(ACTIVE_CANDIDATE_STATES + TERMINAL_CANDIDATE_STATES)
    if state is not None and state not in allowed:
        raise HTTPException(status_code=400, detail="unknown candidate state")
    rows = get_service().store.candidates(state=state, limit=limit)
    return {"items": rows, "count": len(rows)}


@router.get("/portfolio")
def portfolio() -> dict:
    return get_service().store.portfolio()


@router.get("/evaluation")
def evaluation() -> dict:
    return get_service().store.evaluation()


@router.post("/run")
def run_now() -> dict:
    """Rerun automatic discovery; intentionally accepts no symbols or discretionary input."""
    return get_service().enqueue("operational_rerun")
