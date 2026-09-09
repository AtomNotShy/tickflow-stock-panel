"""Transactional SQLite ledger for candidates, orders, paper holdings and evaluations."""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .domain import ResearchConfig


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ResearchStore:
    def __init__(self, path: Path, config: ResearchConfig) -> None:
        self.path = path
        self.config = config
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    model TEXT,
                    methodology_version TEXT NOT NULL,
                    input_count INTEGER NOT NULL DEFAULT 0,
                    decision_count INTEGER NOT NULL DEFAULT 0,
                    market_summary TEXT,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_runs_as_of ON runs(as_of, started_at DESC);

                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    anomaly_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    state TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    thesis TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    impact_path_json TEXT NOT NULL,
                    fundamental_view TEXT NOT NULL,
                    expectation_view TEXT NOT NULL,
                    technical_view TEXT NOT NULL,
                    risks_json TEXT NOT NULL,
                    technical_confirmed INTEGER NOT NULL,
                    risk_veto INTEGER NOT NULL,
                    discovered_on TEXT NOT NULL,
                    last_seen_on TEXT NOT NULL,
                    expires_on TEXT NOT NULL,
                    confirm_count INTEGER NOT NULL DEFAULT 0,
                    miss_count INTEGER NOT NULL DEFAULT 0,
                    cooldown_until TEXT,
                    last_run_id TEXT NOT NULL REFERENCES runs(id),
                    updated_at TEXT NOT NULL,
                    UNIQUE(anomaly_key, symbol)
                );
                CREATE INDEX IF NOT EXISTS ix_candidates_state ON candidates(state, score DESC);

                CREATE TABLE IF NOT EXISTS candidate_snapshots (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    as_of TEXT NOT NULL,
                    state TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, candidate_id)
                );

                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    cash REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    quantity INTEGER NOT NULL,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    last_price REAL NOT NULL,
                    highest_close REAL NOT NULL,
                    hold_days INTEGER NOT NULL,
                    candidate_id TEXT REFERENCES candidates(id),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                    state TEXT NOT NULL CHECK(state IN ('pending', 'filled', 'cancelled')),
                    signal_date TEXT NOT NULL,
                    fill_date TEXT,
                    fill_price REAL,
                    quantity INTEGER,
                    budget REAL,
                    reason TEXT NOT NULL,
                    candidate_id TEXT REFERENCES candidates(id),
                    blocked_reason TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_pending_order
                    ON orders(symbol, side) WHERE state = 'pending';

                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    gross_pnl REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    duration INTEGER NOT NULL,
                    exit_reason TEXT NOT NULL,
                    candidate_id TEXT REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS account_snapshots (
                    as_of TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    equity REAL NOT NULL,
                    exposure REAL NOT NULL,
                    positions INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS selection_outcomes (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    signal_date TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    entry_date TEXT,
                    entry_price REAL,
                    evaluated_date TEXT,
                    exit_price REAL,
                    return_pct REAL,
                    benchmark_return_pct REAL,
                    excess_return_pct REAL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'complete')),
                    UNIQUE(candidate_id, signal_date, horizon_days)
                );
                """
            )
            candidate_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
            }
            if "evidence_json" not in candidate_columns:
                conn.execute(
                    "ALTER TABLE candidates ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '[]'"
                )
            run_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            for column in ("input_json", "output_json"):
                if column not in run_columns:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
            outcome_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(selection_outcomes)").fetchall()
            }
            for column in ("benchmark_return_pct", "excess_return_pct"):
                if column not in outcome_columns:
                    conn.execute(f"ALTER TABLE selection_outcomes ADD COLUMN {column} REAL")
            conn.execute(
                "UPDATE schema_meta SET value='4' WHERE key='schema_version'"
            )
            conn.execute(
                "INSERT OR IGNORE INTO account(id, cash, initial_capital, updated_at) "
                "VALUES (1, ?, ?, ?)",
                (self.config.initial_capital, self.config.initial_capital, _now()),
            )
            conn.execute(
                "UPDATE runs SET status='failed', completed_at=?, "
                "error=COALESCE(error, 'process restarted during run') WHERE status='running'",
                (_now(),),
            )

    def begin_run(self, as_of: date, trigger: str, methodology_version: str) -> tuple[str, bool]:
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id, status FROM runs WHERE as_of=? ORDER BY started_at DESC LIMIT 1",
                (as_of.isoformat(),),
            ).fetchone()
            if existing and existing["status"] in ("running", "completed"):
                return str(existing["id"]), False
            run_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO runs(id, as_of, started_at, status, trigger, methodology_version) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, as_of.isoformat(), _now(), trigger, methodology_version),
            )
            return run_id, True

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        model: str | None = None,
        input_count: int = 0,
        decision_count: int = 0,
        market_summary: str | None = None,
        error: str | None = None,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET completed_at=?, status=?, model=?, input_count=?, "
                "decision_count=?, market_summary=?, error=?, "
                "input_json=COALESCE(?, input_json), output_json=COALESCE(?, output_json) "
                "WHERE id=?",
                (
                    _now(), status, model, input_count, decision_count,
                    market_summary, error,
                    json.dumps(input_payload, ensure_ascii=False) if input_payload is not None else None,
                    json.dumps(output_payload, ensure_ascii=False) if output_payload is not None else None,
                    run_id,
                ),
            )

    def record_run_input(self, run_id: str, payload: Any) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET input_json=? WHERE id=?",
                (json.dumps(payload, ensure_ascii=False), run_id),
            )

    def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None

    def latest_run(self) -> dict[str, Any] | None:
        return self.fetch_one(
            "SELECT id,as_of,started_at,completed_at,status,trigger,model,"
            "methodology_version,input_count,decision_count,market_summary,error "
            "FROM runs ORDER BY started_at DESC LIMIT 1"
        )

    def active_candidates(self) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM candidates WHERE state IN "
            "('discovered','analyzed','watching','eligible') ORDER BY score DESC, symbol"
        )

    def candidates(self, state: str | None = None, limit: int = 300) -> list[dict[str, Any]]:
        if state:
            rows = self.fetch_all(
                "SELECT * FROM candidates WHERE state=? ORDER BY updated_at DESC LIMIT ?",
                (state, limit),
            )
        else:
            rows = self.fetch_all(
                "SELECT * FROM candidates ORDER BY "
                "CASE state WHEN 'eligible' THEN 0 WHEN 'watching' THEN 1 ELSE 2 END, "
                "score DESC, updated_at DESC LIMIT ?",
                (limit,),
            )
        for row in rows:
            row["evidence"] = json.loads(row.pop("evidence_json"))
            row["impact_path"] = json.loads(row.pop("impact_path_json"))
            row["risks"] = json.loads(row.pop("risks_json"))
            row["technical_confirmed"] = bool(row["technical_confirmed"])
            row["risk_veto"] = bool(row["risk_veto"])
        return rows

    def portfolio(self) -> dict[str, Any]:
        return {
            "account": self.fetch_one("SELECT * FROM account WHERE id=1"),
            "positions": self.fetch_all("SELECT * FROM positions ORDER BY updated_at DESC"),
            "orders": self.fetch_all(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT 200"
            ),
            "trades": self.fetch_all(
                "SELECT * FROM trades ORDER BY exit_date DESC, id DESC LIMIT 300"
            ),
            "equity_curve": self.fetch_all(
                "SELECT * FROM account_snapshots ORDER BY as_of"
            ),
        }

    def evaluation(self) -> dict[str, Any]:
        summary = self.fetch_all(
            "SELECT horizon_days, COUNT(*) AS sample_count, "
            "SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) AS wins, "
            "AVG(return_pct) AS avg_return_pct, "
            "AVG(excess_return_pct) AS avg_excess_return_pct "
            "FROM selection_outcomes WHERE status='complete' GROUP BY horizon_days "
            "ORDER BY horizon_days"
        )
        trades = self.fetch_one(
            "SELECT COUNT(*) AS trade_count, "
            "SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            "SUM(net_pnl) AS net_pnl, AVG(return_pct) AS avg_return_pct FROM trades"
        )
        return {
            "selection_summary": summary,
            "execution_summary": trades,
            "outcomes": self.fetch_all(
                "SELECT o.*, c.symbol, c.name, c.thesis FROM selection_outcomes o "
                "JOIN candidates c ON c.id=o.candidate_id "
                "ORDER BY o.signal_date DESC, o.horizon_days LIMIT 500"
            ),
        }
