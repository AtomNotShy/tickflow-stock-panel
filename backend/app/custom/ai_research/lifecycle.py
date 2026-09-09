"""Candidate-pool lifecycle with entry/exit hysteresis and immutable snapshots."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta

from .domain import AgentDecision, AgentResponse, ResearchConfig
from .store import ResearchStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def apply_agent_decisions(
    store: ResearchStore,
    run_id: str,
    as_of: date,
    response: AgentResponse,
    names: dict[str, str],
    prompted: set[tuple[str, str]],
    config: ResearchConfig,
) -> dict[str, int]:
    counts = {"created": 0, "eligible": 0, "watching": 0, "terminal": 0, "missed": 0}
    returned = {(item.symbol, item.anomaly_key) for item in response.decisions}
    with store.transaction() as conn:
        for decision in response.decisions:
            row = conn.execute(
                "SELECT * FROM candidates WHERE symbol=? AND anomaly_key=?",
                (decision.symbol, decision.anomaly_key),
            ).fetchone()
            previous_state = str(row["state"]) if row else None
            candidate_id = str(row["id"]) if row else str(uuid.uuid4())
            confirm_count = int(row["confirm_count"]) if row else 0
            cooldown_until = date.fromisoformat(row["cooldown_until"]) if row and row["cooldown_until"] else None
            state, confirm_count, next_cooldown = _next_state(
                decision, as_of, previous_state, confirm_count, cooldown_until, config
            )
            expires_on = as_of + timedelta(days=max(decision.horizon_days * 2, 3))
            payload = decision.model_dump(mode="json")
            values = (
                candidate_id, decision.anomaly_key, decision.symbol,
                names.get(decision.symbol), state, decision.score, decision.confidence,
                decision.horizon_days, decision.thesis,
                json.dumps(decision.evidence, ensure_ascii=False),
                json.dumps(decision.impact_path, ensure_ascii=False),
                decision.fundamental_view, decision.expectation_view,
                decision.technical_view, json.dumps(decision.risks, ensure_ascii=False),
                int(decision.technical_confirmed), int(decision.risk_veto),
                row["discovered_on"] if row else as_of.isoformat(), as_of.isoformat(),
                expires_on.isoformat(), confirm_count, 0,
                next_cooldown.isoformat() if next_cooldown else None,
                run_id, _now(),
            )
            conn.execute(
                """
                INSERT INTO candidates(
                    id, anomaly_key, symbol, name, state, score, confidence,
                    horizon_days, thesis, evidence_json, impact_path_json, fundamental_view,
                    expectation_view, technical_view, risks_json,
                    technical_confirmed, risk_veto, discovered_on, last_seen_on,
                    expires_on, confirm_count, miss_count, cooldown_until,
                    last_run_id, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(anomaly_key, symbol) DO UPDATE SET
                    name=excluded.name, state=excluded.state, score=excluded.score,
                    confidence=excluded.confidence, horizon_days=excluded.horizon_days,
                    thesis=excluded.thesis, evidence_json=excluded.evidence_json,
                    impact_path_json=excluded.impact_path_json,
                    fundamental_view=excluded.fundamental_view,
                    expectation_view=excluded.expectation_view,
                    technical_view=excluded.technical_view, risks_json=excluded.risks_json,
                    technical_confirmed=excluded.technical_confirmed,
                    risk_veto=excluded.risk_veto, last_seen_on=excluded.last_seen_on,
                    expires_on=excluded.expires_on, confirm_count=excluded.confirm_count,
                    miss_count=0, cooldown_until=excluded.cooldown_until,
                    last_run_id=excluded.last_run_id, updated_at=excluded.updated_at
                """,
                values,
            )
            conn.execute(
                "INSERT OR REPLACE INTO candidate_snapshots"
                "(run_id,candidate_id,as_of,state,score,confidence,payload_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    run_id, candidate_id, as_of.isoformat(), state,
                    decision.score, decision.confidence,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            if state == "eligible" and previous_state != "eligible":
                for horizon in config.evaluation_horizons:
                    conn.execute(
                        "INSERT OR IGNORE INTO selection_outcomes"
                        "(id,candidate_id,signal_date,horizon_days,status) "
                        "VALUES (?,?,?,?, 'pending')",
                        (str(uuid.uuid4()), candidate_id, as_of.isoformat(), horizon),
                    )
            if not row:
                counts["created"] += 1
            if state == "eligible":
                counts["eligible"] += 1
            elif state == "watching":
                counts["watching"] += 1
            elif state in ("invalidated", "rejected", "expired", "stale"):
                counts["terminal"] += 1

        missing = conn.execute(
            "SELECT * FROM candidates WHERE state IN "
            "('discovered','analyzed','watching','eligible')"
        ).fetchall()
        for row in missing:
            identity = (str(row["symbol"]), str(row["anomaly_key"]))
            if identity in returned:
                continue
            misses = int(row["miss_count"]) + 1
            expired = as_of > date.fromisoformat(row["expires_on"])
            if identity not in prompted and not expired:
                continue
            state = "expired" if expired else (
                "stale" if misses >= config.stale_after_misses else str(row["state"])
            )
            conn.execute(
                "UPDATE candidates SET state=?, miss_count=?, last_run_id=?, updated_at=? "
                "WHERE id=?",
                (state, misses, run_id, _now(), row["id"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO candidate_snapshots"
                "(run_id,candidate_id,as_of,state,score,confidence,payload_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    run_id, row["id"], as_of.isoformat(), state,
                    row["score"], row["confidence"],
                    json.dumps({"reason": "omitted_by_agent", "miss_count": misses}),
                ),
            )
            counts["missed"] += 1
            if state in ("expired", "stale"):
                counts["terminal"] += 1
    return counts


def _next_state(
    decision: AgentDecision,
    as_of: date,
    previous_state: str | None,
    confirm_count: int,
    cooldown_until: date | None,
    config: ResearchConfig,
) -> tuple[str, int, date | None]:
    if decision.invalidated or decision.risk_veto or decision.action == "exit":
        return "invalidated", 0, as_of + timedelta(days=config.cooldown_days)
    if decision.action == "reject":
        return "rejected", 0, as_of + timedelta(days=config.cooldown_days)
    if decision.action == "watch":
        return "watching", 0, cooldown_until

    qualified = (
        decision.score >= config.entry_score
        and decision.confidence >= config.min_confidence
        and decision.technical_confirmed
    )
    if cooldown_until and as_of < cooldown_until:
        qualified = False
    if not qualified:
        # Separate entry and exit thresholds avoid daily boundary churn.
        if previous_state == "eligible" and decision.score >= config.exit_score:
            return "eligible", confirm_count, cooldown_until
        return "watching", 0, cooldown_until
    confirmations = confirm_count + 1
    state = "eligible" if confirmations >= config.confirmations_required else "analyzed"
    return state, confirmations, None
