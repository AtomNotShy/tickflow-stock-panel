"""Single-agent daily orchestration and background run manager."""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from app.extensions import ExtensionContext
from app.services.ai_provider import ai_configured, current_ai_model, current_ai_provider

from .agent import attach_financials, discover_inputs, run_agent
from .domain import METHODOLOGY_VERSION, ResearchConfig
from .lifecycle import apply_agent_decisions
from .paper import PaperBroker
from .store import ResearchStore

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(self, data_dir: Path, repository: Any, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()
        self.repository = repository
        self.store = ResearchStore(data_dir / "user_data" / "ai_research.sqlite3", self.config)
        self.paper = PaperBroker(self.store, self.config)
        self._guard = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        with self._guard:
            return self._running

    def enqueue(self, trigger: str) -> dict[str, Any]:
        with self._guard:
            if self._running:
                return {"enqueued": False, "reused": True}
            self._running = True
        thread = threading.Thread(
            target=self._background_run,
            args=(trigger,),
            daemon=True,
            name="ai-research-agent",
        )
        thread.start()
        return {"enqueued": True, "reused": False}

    def _background_run(self, trigger: str) -> None:
        try:
            self.run_once(trigger)
        except Exception:
            logger.exception("AI research run failed")
        finally:
            with self._guard:
                self._running = False

    def run_once(self, trigger: str = "manual") -> dict[str, Any]:
        latest, latest_date = self.repository.get_enriched_latest()
        if latest_date is None or latest.is_empty():
            raise RuntimeError("enriched market data is not ready")
        as_of = latest_date if isinstance(latest_date, date) else date.fromisoformat(str(latest_date))
        run_id, is_new = self.store.begin_run(as_of, trigger, METHODOLOGY_VERSION)
        if not is_new:
            return {"run_id": run_id, "reused": True}

        try:
            self.paper.process_market_day(as_of, latest)
            evaluated = self.paper.evaluate_selection(self.repository, as_of)
            if not ai_configured():
                self.store.finish_run(
                    run_id, "skipped", model=current_ai_model(),
                    error="AI provider is not configured; paper risk controls were still processed",
                )
                return {"run_id": run_id, "status": "skipped", "evaluated": evaluated}

            active = self.store.active_candidates()
            inputs = discover_inputs(latest, active, self.config)
            if not inputs:
                self.store.finish_run(
                    run_id, "skipped", model=current_ai_model(), error="no eligible market inputs"
                )
                return {"run_id": run_id, "status": "skipped", "evaluated": evaluated}
            attach_financials(self.store.path.parents[1], as_of, inputs)
            self.store.record_run_input(
                run_id,
                {
                    "as_of": as_of.isoformat(),
                    "methodology_version": METHODOLOGY_VERSION,
                    "config": asdict(self.config),
                    "candidates": inputs,
                },
            )
            response = asyncio.run(run_agent(as_of, inputs))
            names = self.repository.get_name_map([item["symbol"] for item in inputs])
            prompted = {(item["symbol"], item["anomaly_key"]) for item in inputs}
            lifecycle = apply_agent_decisions(
                self.store, run_id, as_of, response, names, prompted, self.config
            )
            orders = self.paper.reconcile_signals(as_of)
            self.store.finish_run(
                run_id,
                "completed",
                model=f"{current_ai_provider()}:{current_ai_model()}",
                input_count=len(inputs),
                decision_count=len(response.decisions),
                market_summary=response.market_summary,
                output_payload=response.model_dump(mode="json"),
            )
            return {
                "run_id": run_id,
                "status": "completed",
                "as_of": as_of.isoformat(),
                "lifecycle": lifecycle,
                "orders": orders,
                "evaluated": evaluated,
            }
        except Exception as exc:
            self.store.finish_run(
                run_id,
                "failed",
                model=current_ai_model(),
                error=str(exc)[:2000],
                output_payload=(
                    {"raw": exc.raw_output}
                    if hasattr(exc, "raw_output") and exc.raw_output
                    else None
                ),
            )
            raise

    def status(self) -> dict[str, Any]:
        counts = self.store.fetch_all(
            "SELECT state, COUNT(*) AS count FROM candidates GROUP BY state ORDER BY state"
        )
        latest_snapshot = self.store.fetch_one(
            "SELECT * FROM account_snapshots ORDER BY as_of DESC LIMIT 1"
        )
        return {
            "configured": ai_configured(),
            "running": self.running,
            "methodology_version": METHODOLOGY_VERSION,
            "latest_run": self.store.latest_run(),
            "candidate_counts": {row["state"]: row["count"] for row in counts},
            "account_snapshot": latest_snapshot,
            "config": {
                "entry_score": self.config.entry_score,
                "exit_score": self.config.exit_score,
                "min_confidence": self.config.min_confidence,
                "max_positions": self.config.max_positions,
                "max_exposure": self.config.max_exposure,
                "stop_loss_pct": self.config.stop_loss_pct,
                "max_hold_days": self.config.max_hold_days,
                "execution": "next_open_100_share_lots",
            },
        }


_service: ResearchService | None = None


def startup_service(context: ExtensionContext) -> ResearchService:
    global _service
    _service = ResearchService(context.data_dir, context.repository)
    return _service


def get_service() -> ResearchService:
    if _service is None:
        raise RuntimeError("AI research extension is not initialized")
    return _service
