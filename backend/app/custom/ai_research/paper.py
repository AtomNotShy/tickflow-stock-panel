"""Deterministic next-open paper broker and fixed-horizon selection evaluator."""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import polars as pl

from .domain import ResearchConfig
from .store import ResearchStore, _now


def _raw_prices(row: dict[str, Any]) -> tuple[float, float, float, float]:
    close = float(row.get("close") or 0)
    raw_close = float(row.get("raw_close") or close)
    factor = raw_close / close if close > 0 and raw_close > 0 else 1.0
    raw_open = float(row.get("open") or close) * factor
    raw_high = float(row.get("raw_high") or row.get("high") or raw_open)
    raw_low = float(row.get("raw_low") or row.get("low") or raw_open)
    return raw_open, raw_high, raw_low, raw_close


def _locked(row: dict[str, Any], side: str) -> bool:
    raw_open, raw_high, raw_low, raw_close = _raw_prices(row)
    same_price = max(raw_open, raw_high, raw_low, raw_close) - min(
        raw_open, raw_high, raw_low, raw_close
    ) < 0.005
    flag = bool(row.get("signal_limit_up" if side == "buy" else "signal_limit_down"))
    return flag and same_price


class PaperBroker:
    def __init__(self, store: ResearchStore, config: ResearchConfig) -> None:
        self.store = store
        self.config = config

    def process_market_day(self, as_of: date, latest: pl.DataFrame) -> dict[str, Any]:
        """Settle older orders, mark holdings, enforce deterministic risk, snapshot once."""
        if self.store.fetch_one(
            "SELECT as_of FROM account_snapshots WHERE as_of=?", (as_of.isoformat(),)
        ):
            return {"reused": True}
        market = {
            str(row["symbol"]): row
            for row in latest.to_dicts()
            if row.get("symbol")
        }
        with self.store.transaction() as conn:
            pending = conn.execute(
                "SELECT * FROM orders WHERE state='pending' AND signal_date < ? "
                "ORDER BY CASE side WHEN 'sell' THEN 0 ELSE 1 END, created_at",
                (as_of.isoformat(),),
            ).fetchall()
            for order in pending:
                row = market.get(str(order["symbol"]))
                if row is None:
                    attempts = int(order["attempt_count"]) + 1
                    state = (
                        "cancelled"
                        if order["side"] == "buy" and attempts >= self.config.pending_buy_max_attempts
                        else "pending"
                    )
                    conn.execute(
                        "UPDATE orders SET state=?, blocked_reason='suspended_or_missing', "
                        "attempt_count=?, updated_at=? WHERE id=?",
                        (state, attempts, _now(), order["id"]),
                    )
                    continue
                attempts = int(order["attempt_count"]) + 1
                if _locked(row, str(order["side"])):
                    blocked = "one_price_limit_up" if order["side"] == "buy" else "one_price_limit_down"
                    state = (
                        "cancelled"
                        if order["side"] == "buy" and attempts >= self.config.pending_buy_max_attempts
                        else "pending"
                    )
                    conn.execute(
                        "UPDATE orders SET state=?, blocked_reason=?, attempt_count=?, updated_at=? "
                        "WHERE id=?",
                        (state, blocked, attempts, _now(), order["id"]),
                    )
                    continue
                if order["side"] == "sell":
                    self._fill_sell(conn, order, row, as_of, attempts)
                else:
                    self._fill_buy(conn, order, row, as_of, attempts)

            positions = conn.execute("SELECT * FROM positions").fetchall()
            for position in positions:
                row = market.get(str(position["symbol"]))
                if row is None:
                    continue
                raw_close = _raw_prices(row)[3]
                hold_days = int(position["hold_days"]) + 1
                highest = max(float(position["highest_close"]), raw_close)
                conn.execute(
                    "UPDATE positions SET last_price=?, highest_close=?, hold_days=?, updated_at=? "
                    "WHERE symbol=?",
                    (raw_close, highest, hold_days, _now(), position["symbol"]),
                )
                loss = raw_close / float(position["entry_price"]) - 1
                if loss <= -self.config.stop_loss_pct:
                    self._ensure_order(
                        conn, str(position["symbol"]), position["name"], "sell", as_of,
                        "stop_loss", position["candidate_id"], None,
                    )
                elif hold_days >= self.config.max_hold_days:
                    self._ensure_order(
                        conn, str(position["symbol"]), position["name"], "sell", as_of,
                        "max_hold", position["candidate_id"], None,
                    )

            account = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
            marked = conn.execute("SELECT quantity, last_price FROM positions").fetchall()
            market_value = sum(float(p["quantity"]) * float(p["last_price"]) for p in marked)
            cash = float(account["cash"])
            equity = cash + market_value
            conn.execute(
                "INSERT INTO account_snapshots"
                "(as_of,cash,market_value,equity,exposure,positions,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    as_of.isoformat(), cash, market_value, equity,
                    market_value / equity if equity > 0 else 0,
                    len(marked), _now(),
                ),
            )
        return {"reused": False}

    def reconcile_signals(self, as_of: date) -> dict[str, int]:
        """Create next-open exits first, then rank eligible entries within account limits."""
        created = {"buy_orders": 0, "sell_orders": 0}
        with self.store.transaction() as conn:
            pending_buys = conn.execute(
                "SELECT * FROM orders WHERE side='buy' AND state='pending'"
            ).fetchall()
            for order in pending_buys:
                eligible = conn.execute(
                    "SELECT 1 FROM candidates WHERE symbol=? AND state='eligible' LIMIT 1",
                    (order["symbol"],),
                ).fetchone()
                if not eligible:
                    conn.execute(
                        "UPDATE orders SET state='cancelled', blocked_reason='signal_withdrawn', "
                        "updated_at=? WHERE id=?",
                        (_now(), order["id"]),
                    )

            positions = conn.execute("SELECT * FROM positions").fetchall()
            for position in positions:
                active = conn.execute(
                    "SELECT COUNT(*) AS n FROM candidates WHERE symbol=? "
                    "AND (state='eligible' OR (state='watching' AND score>=? AND risk_veto=0))",
                    (position["symbol"], self.config.exit_score),
                ).fetchone()["n"]
                if not active and self._ensure_order(
                    conn, str(position["symbol"]), position["name"], "sell", as_of,
                    "thesis_invalidated", position["candidate_id"], None,
                ):
                    created["sell_orders"] += 1

            held = {row["symbol"] for row in positions}
            pending_buys = conn.execute(
                "SELECT symbol, budget FROM orders WHERE side='buy' AND state='pending'"
            ).fetchall()
            reserved = held | {row["symbol"] for row in pending_buys}
            slots = max(self.config.max_positions - len(reserved), 0)
            if slots <= 0:
                return created
            latest_snapshot = conn.execute(
                "SELECT equity, market_value FROM account_snapshots ORDER BY as_of DESC LIMIT 1"
            ).fetchone()
            account = conn.execute("SELECT cash, initial_capital FROM account WHERE id=1").fetchone()
            equity = float(latest_snapshot["equity"]) if latest_snapshot else float(account["initial_capital"])
            market_value = float(latest_snapshot["market_value"]) if latest_snapshot else 0.0
            pending_budget = sum(float(row["budget"] or 0) for row in pending_buys)
            available_exposure = max(
                equity * self.config.max_exposure - market_value - pending_budget, 0.0
            )
            unit_budget = equity * self.config.max_exposure / self.config.max_positions
            candidates = conn.execute(
                "SELECT * FROM candidates WHERE state='eligible' "
                "ORDER BY score DESC, confidence DESC, symbol"
            ).fetchall()
            used_symbols: set[str] = set()
            for candidate in candidates:
                symbol = str(candidate["symbol"])
                if symbol in reserved or symbol in used_symbols or slots <= 0:
                    continue
                budget = min(unit_budget, available_exposure)
                if budget <= 0:
                    break
                if self._ensure_order(
                    conn, symbol, candidate["name"], "buy", as_of,
                    "eligible_rank", candidate["id"], budget,
                ):
                    created["buy_orders"] += 1
                    used_symbols.add(symbol)
                    slots -= 1
                    available_exposure -= budget
        return created

    def evaluate_selection(self, repo: Any, as_of: date) -> int:
        pending = self.store.fetch_all(
            "SELECT o.*, c.symbol FROM selection_outcomes o JOIN candidates c "
            "ON c.id=o.candidate_id WHERE o.status='pending' ORDER BY o.signal_date"
        )
        if not pending:
            return 0
        completed = 0
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for item in pending:
            by_symbol.setdefault(str(item["symbol"]), []).append(item)
        benchmark: dict[date, dict[str, Any]] = {}
        try:
            earliest = min(date.fromisoformat(item["signal_date"]) for item in pending)
            benchmark_frame = repo.get_index_daily(
                "000300.SH", earliest, as_of, columns=["date", "open", "close"]
            )
            benchmark = {row["date"]: row for row in benchmark_frame.to_dicts()}
        except Exception:  # benchmark is optional; raw outcomes remain valid
            benchmark = {}

        with self.store.transaction() as conn:
            for symbol, outcomes in by_symbol.items():
                start = min(date.fromisoformat(item["signal_date"]) for item in outcomes)
                frame = repo.get_daily(
                    symbol, start, as_of,
                    columns=["symbol", "date", "open", "close", "raw_close"],
                )
                if frame.is_empty():
                    continue
                rows = frame.sort("date").to_dicts()
                for outcome in outcomes:
                    signal_date = date.fromisoformat(outcome["signal_date"])
                    future = [row for row in rows if row["date"] > signal_date]
                    if not future:
                        continue
                    entry_price = _raw_prices(future[0])[0]
                    horizon = int(outcome["horizon_days"])
                    if len(future) < horizon:
                        conn.execute(
                            "UPDATE selection_outcomes SET entry_date=?, entry_price=? WHERE id=?",
                            (future[0]["date"].isoformat(), entry_price, outcome["id"]),
                        )
                        continue
                    terminal = future[horizon - 1]
                    exit_price = _raw_prices(terminal)[3]
                    return_pct = exit_price / entry_price - 1 if entry_price > 0 else None
                    benchmark_entry = benchmark.get(future[0]["date"])
                    benchmark_exit = benchmark.get(terminal["date"])
                    benchmark_return = None
                    if benchmark_entry and benchmark_exit:
                        benchmark_open = float(benchmark_entry.get("open") or 0)
                        benchmark_close = float(benchmark_exit.get("close") or 0)
                        if benchmark_open > 0 and benchmark_close > 0:
                            benchmark_return = benchmark_close / benchmark_open - 1
                    excess_return = (
                        return_pct - benchmark_return
                        if return_pct is not None and benchmark_return is not None
                        else None
                    )
                    conn.execute(
                        "UPDATE selection_outcomes SET entry_date=?, entry_price=?, "
                        "evaluated_date=?, exit_price=?, return_pct=?, benchmark_return_pct=?, "
                        "excess_return_pct=?, status='complete' WHERE id=?",
                        (
                            future[0]["date"].isoformat(), entry_price,
                            terminal["date"].isoformat(), exit_price, return_pct,
                            benchmark_return, excess_return,
                            outcome["id"],
                        ),
                    )
                    completed += 1
        return completed

    def _fill_buy(self, conn, order, row, as_of: date, attempts: int) -> None:
        if conn.execute("SELECT 1 FROM positions WHERE symbol=?", (order["symbol"],)).fetchone():
            conn.execute(
                "UPDATE orders SET state='cancelled', blocked_reason='already_held', updated_at=? "
                "WHERE id=?", (_now(), order["id"]),
            )
            return
        raw_open = _raw_prices(row)[0]
        execution_price = raw_open * (1 + self.config.slippage_bps / 10_000.0)
        cash = float(conn.execute("SELECT cash FROM account WHERE id=1").fetchone()["cash"])
        budget = min(float(order["budget"] or cash), cash)
        quantity = int(budget / (execution_price * (1 + self.config.commission_pct)) // 100 * 100)
        if quantity <= 0:
            conn.execute(
                "UPDATE orders SET state='cancelled', blocked_reason='insufficient_cash', "
                "attempt_count=?, updated_at=? WHERE id=?", (attempts, _now(), order["id"]),
            )
            return
        cost = quantity * execution_price * (1 + self.config.commission_pct)
        conn.execute("UPDATE account SET cash=cash-?, updated_at=? WHERE id=1", (cost, _now()))
        conn.execute(
            "INSERT INTO positions(symbol,name,quantity,entry_date,entry_price,cost_basis,"
            "last_price,highest_close,hold_days,candidate_id,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                order["symbol"], order["name"], quantity, as_of.isoformat(), execution_price,
                cost, _raw_prices(row)[3], _raw_prices(row)[3], 0,
                order["candidate_id"], _now(),
            ),
        )
        conn.execute(
            "UPDATE orders SET state='filled', fill_date=?, fill_price=?, quantity=?, "
            "attempt_count=?, blocked_reason=NULL, updated_at=? WHERE id=?",
            (as_of.isoformat(), execution_price, quantity, attempts, _now(), order["id"]),
        )

    def _fill_sell(self, conn, order, row, as_of: date, attempts: int) -> None:
        position = conn.execute(
            "SELECT * FROM positions WHERE symbol=?", (order["symbol"],)
        ).fetchone()
        if position is None:
            conn.execute(
                "UPDATE orders SET state='cancelled', blocked_reason='not_held', updated_at=? "
                "WHERE id=?", (_now(), order["id"]),
            )
            return
        raw_open = _raw_prices(row)[0]
        execution_price = raw_open * (1 - self.config.slippage_bps / 10_000.0)
        quantity = int(position["quantity"])
        gross_proceeds = quantity * execution_price
        proceeds = gross_proceeds * (1 - self.config.commission_pct - self.config.stamp_tax_pct)
        gross_pnl = gross_proceeds - quantity * float(position["entry_price"])
        net_pnl = proceeds - float(position["cost_basis"])
        conn.execute("UPDATE account SET cash=cash+?, updated_at=? WHERE id=1", (proceeds, _now()))
        conn.execute(
            "INSERT INTO trades(id,symbol,name,entry_date,exit_date,entry_price,exit_price,"
            "quantity,gross_pnl,net_pnl,return_pct,duration,exit_reason,candidate_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), position["symbol"], position["name"],
                position["entry_date"], as_of.isoformat(), position["entry_price"],
                execution_price, quantity, gross_pnl, net_pnl,
                net_pnl / float(position["cost_basis"]), position["hold_days"],
                order["reason"], position["candidate_id"],
            ),
        )
        conn.execute("DELETE FROM positions WHERE symbol=?", (position["symbol"],))
        conn.execute(
            "UPDATE orders SET state='filled', fill_date=?, fill_price=?, quantity=?, "
            "attempt_count=?, blocked_reason=NULL, updated_at=? WHERE id=?",
            (as_of.isoformat(), execution_price, quantity, attempts, _now(), order["id"]),
        )

    @staticmethod
    def _ensure_order(
        conn, symbol: str, name: str | None, side: str, signal_date: date,
        reason: str, candidate_id: str | None, budget: float | None,
    ) -> bool:
        exists = conn.execute(
            "SELECT 1 FROM orders WHERE symbol=? AND side=? AND state='pending'",
            (symbol, side),
        ).fetchone()
        if exists:
            return False
        conn.execute(
            "INSERT INTO orders(id,symbol,name,side,state,signal_date,budget,reason,"
            "candidate_id,created_at,updated_at) VALUES (?,?,?,?, 'pending',?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), symbol, name, side, signal_date.isoformat(), budget,
                reason, candidate_id, _now(), _now(),
            ),
        )
        return True
