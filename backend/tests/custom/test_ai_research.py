from __future__ import annotations

# ruff: noqa: RUF001
import json
from datetime import date

import polars as pl
import pytest

from app.custom.ai_research.agent import discover_inputs, run_agent
from app.custom.ai_research.domain import (
    METHODOLOGY_VERSION,
    AgentOutputError,
    AgentResponse,
    ResearchConfig,
    parse_agent_response,
)
from app.custom.ai_research.lifecycle import apply_agent_decisions
from app.custom.ai_research.paper import PaperBroker
from app.custom.ai_research.service import ResearchService
from app.custom.ai_research.store import ResearchStore


def _response(*, action: str = "enter", score: float = 82, risk_veto: bool = False) -> AgentResponse:
    return AgentResponse.model_validate(
        {
            "market_summary": "本地量价异常候选",
            "decisions": [
                {
                    "symbol": "000001.SZ",
                    "anomaly_key": "local:ma20-breakout",
                    "action": action,
                    "score": score,
                    "confidence": 0.8,
                    "horizon_days": 10,
                    "thesis": "价格突破中期均线并获得成交量确认，后续利润弹性需要财务数据验证。",
                    "evidence": ["momentum_20d=0.12", "signal_ma20_breakout=true"],
                    "impact_path": ["量价异常", "市场预期改善", "估值变化"],
                    "fundamental_view": "财务数据不足，暂不作额外推断。",
                    "expectation_view": "放量意味着部分预期可能已被价格反映。",
                    "technical_view": "收盘突破 MA20 且量能同步扩大。",
                    "risks": ["跌回 MA20 且放量"],
                    "technical_confirmed": True,
                    "risk_veto": risk_veto,
                    "invalidated": False,
                }
            ],
        }
    )


def _market(day: date, *, open_price: float, close: float, locked_up: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [day],
            "open": [open_price],
            "high": [open_price if locked_up else max(open_price, close)],
            "low": [open_price if locked_up else min(open_price, close)],
            "close": [close],
            "raw_close": [close],
            "raw_high": [open_price if locked_up else max(open_price, close)],
            "raw_low": [open_price if locked_up else min(open_price, close)],
            "signal_limit_up": [locked_up],
            "signal_limit_down": [False],
        }
    )


def _store(tmp_path, config: ResearchConfig | None = None) -> ResearchStore:
    return ResearchStore(tmp_path / "ai-research.sqlite3", config or ResearchConfig())


def _apply(store: ResearchStore, as_of: date, response: AgentResponse) -> str:
    run_id, is_new = store.begin_run(as_of, "test", METHODOLOGY_VERSION)
    assert is_new
    apply_agent_decisions(
        store,
        run_id,
        as_of,
        response,
        {"000001.SZ": "平安银行"},
        {("000001.SZ", "local:ma20-breakout")},
        store.config,
    )
    store.finish_run(run_id, "completed")
    return run_id


def test_parser_rejects_hallucinated_candidate() -> None:
    payload = _response().model_dump(mode="json")
    payload["decisions"][0]["symbol"] = "600000.SH"

    with pytest.raises(AgentOutputError, match="unprovided"):
        parse_agent_response(
            json.dumps(payload), {("000001.SZ", "local:ma20-breakout")}
        )


def test_parser_accepts_json_after_reasoning_preamble() -> None:
    payload = json.dumps(_response().model_dump(mode="json"), ensure_ascii=False)

    parsed = parse_agent_response(
        f"分析完成，结果如下：\n```json\n{payload}\n```",
        {("000001.SZ", "local:ma20-breakout")},
    )

    assert parsed.decisions[0].symbol == "000001.SZ"


def test_parser_reports_empty_visible_content() -> None:
    with pytest.raises(AgentOutputError, match="empty visible content"):
        parse_agent_response("   ", set())


@pytest.mark.asyncio
async def test_agent_batches_candidates_and_sends_configured_output_limit(monkeypatch) -> None:
    inputs = [
        {
            "symbol": f"{index:06d}.SZ",
            "anomaly_key": "local:ma20-breakout",
            "market_data": {"close": 10 + index},
        }
        for index in range(1, 14)
    ]
    calls: list[dict] = []

    async def fake_generate(messages, **kwargs):
        marker = "候选输入(JSON)：\n"
        batch = json.loads(messages[1]["content"].split(marker, 1)[1])
        calls.append({"size": len(batch), **kwargs})
        return json.dumps(
            {
                "market_summary": f"批次包含 {len(batch)} 个候选",
                "decisions": [
                    {
                        "symbol": item["symbol"],
                        "anomaly_key": item["anomaly_key"],
                        "action": "watch",
                        "score": 60,
                        "confidence": 0.7,
                        "horizon_days": 10,
                        "thesis": "量价信号需要后续行情与财务数据共同确认。",
                        "evidence": ["close 已提供"],
                        "impact_path": ["量价异常", "市场预期", "估值变化"],
                        "fundamental_view": "财务数据不足。",
                        "expectation_view": "部分预期可能已经反映。",
                        "technical_view": "等待后续价格确认。",
                        "risks": ["信号失效"],
                        "technical_confirmed": False,
                        "risk_veto": False,
                        "invalidated": False,
                    }
                    for item in batch
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("app.custom.ai_research.agent.generate_ai_text", fake_generate)
    monkeypatch.setattr(
        "app.custom.ai_research.agent.current_ai_max_output_tokens", lambda: 12_000
    )

    response = await run_agent(date(2026, 9, 9), inputs)

    assert [call["size"] for call in calls] == [6, 6, 1]
    assert all(call["max_tokens"] == 12_000 for call in calls)
    assert all(call["prefer_final_answer"] is True for call in calls)
    assert len(response.decisions) == 13


def test_discovery_keeps_existing_thesis_and_adds_new_anomaly() -> None:
    latest = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "name": ["平安银行"],
            "amount": [1_000_000_000.0],
            "momentum_20d": [0.12],
            "vol_ratio_5d": [2.2],
            "turnover_rate": [0.03],
            "signal_ma20_breakout": [True],
            "signal_n_day_high": [False],
            "signal_volume_surge": [True],
        }
    )
    active = [
        {
            "symbol": "000001.SZ",
            "anomaly_key": "local:older-thesis",
            "state": "watching",
            "score": 60,
            "confidence": 0.7,
            "thesis": "旧论点仍需继续复核",
            "risks_json": "[]",
        }
    ]

    inputs = discover_inputs(latest, active, ResearchConfig())

    assert {(item["symbol"], item["anomaly_key"]) for item in inputs} == {
        ("000001.SZ", "local:older-thesis"),
        ("000001.SZ", "local:ma20-breakout"),
    }


def test_lifecycle_uses_separate_entry_exit_thresholds(tmp_path) -> None:
    store = _store(tmp_path)
    first = date(2026, 9, 1)
    _apply(store, first, _response(score=82))
    assert store.active_candidates()[0]["state"] == "eligible"

    second = date(2026, 9, 2)
    _apply(store, second, _response(score=55))

    candidate = store.active_candidates()[0]
    assert candidate["state"] == "eligible"
    assert candidate["score"] == 55


def test_risk_veto_invalidates_candidate(tmp_path) -> None:
    store = _store(tmp_path)
    _apply(store, date(2026, 9, 1), _response())

    _apply(store, date(2026, 9, 2), _response(risk_veto=True))

    candidate = store.candidates()[0]
    assert candidate["state"] == "invalidated"
    assert candidate["cooldown_until"] is not None


def test_paper_order_fills_at_next_open_and_is_idempotent(tmp_path) -> None:
    config = ResearchConfig(max_positions=1, max_exposure=0.9)
    store = _store(tmp_path, config)
    broker = PaperBroker(store, config)
    first = date(2026, 9, 1)
    broker.process_market_day(first, _market(first, open_price=10, close=10))
    _apply(store, first, _response())
    assert broker.reconcile_signals(first)["buy_orders"] == 1

    second = date(2026, 9, 2)
    broker.process_market_day(second, _market(second, open_price=10.5, close=10.8))
    broker.process_market_day(second, _market(second, open_price=99, close=99))

    portfolio = store.portfolio()
    assert len(portfolio["positions"]) == 1
    assert portfolio["positions"][0]["entry_date"] == second.isoformat()
    assert portfolio["positions"][0]["entry_price"] == pytest.approx(10.5 * 1.0005)
    assert len(portfolio["equity_curve"]) == 2


def test_one_price_limit_up_keeps_buy_pending(tmp_path) -> None:
    config = ResearchConfig(max_positions=1, pending_buy_max_attempts=3)
    store = _store(tmp_path, config)
    broker = PaperBroker(store, config)
    first = date(2026, 9, 1)
    broker.process_market_day(first, _market(first, open_price=10, close=10))
    _apply(store, first, _response())
    broker.reconcile_signals(first)

    second = date(2026, 9, 2)
    broker.process_market_day(
        second, _market(second, open_price=11, close=11, locked_up=True)
    )

    order = store.portfolio()["orders"][0]
    assert order["state"] == "pending"
    assert order["blocked_reason"] == "one_price_limit_up"
    assert store.portfolio()["positions"] == []


def test_invalidated_thesis_exits_at_following_open(tmp_path) -> None:
    config = ResearchConfig(max_positions=1)
    store = _store(tmp_path, config)
    broker = PaperBroker(store, config)
    first = date(2026, 9, 1)
    broker.process_market_day(first, _market(first, open_price=10, close=10))
    _apply(store, first, _response())
    broker.reconcile_signals(first)

    second = date(2026, 9, 2)
    broker.process_market_day(second, _market(second, open_price=10.5, close=10.4))
    _apply(store, second, _response(action="exit"))
    assert broker.reconcile_signals(second)["sell_orders"] == 1
    assert len(store.portfolio()["positions"]) == 1

    third = date(2026, 9, 3)
    broker.process_market_day(third, _market(third, open_price=10.2, close=10.1))

    portfolio = store.portfolio()
    assert portfolio["positions"] == []
    assert len(portfolio["trades"]) == 1
    assert portfolio["trades"][0]["exit_date"] == third.isoformat()
    assert portfolio["trades"][0]["exit_reason"] == "thesis_invalidated"


def test_invalidated_signal_cancels_blocked_pending_buy(tmp_path) -> None:
    config = ResearchConfig(max_positions=1)
    store = _store(tmp_path, config)
    broker = PaperBroker(store, config)
    first = date(2026, 9, 1)
    broker.process_market_day(first, _market(first, open_price=10, close=10))
    _apply(store, first, _response())
    broker.reconcile_signals(first)

    second = date(2026, 9, 2)
    broker.process_market_day(
        second, _market(second, open_price=11, close=11, locked_up=True)
    )
    _apply(store, second, _response(action="exit"))
    broker.reconcile_signals(second)

    order = store.portfolio()["orders"][0]
    assert order["state"] == "cancelled"
    assert order["blocked_reason"] == "signal_withdrawn"
    assert store.portfolio()["positions"] == []


def test_selection_book_computes_fixed_horizon_excess_return(tmp_path) -> None:
    store = _store(tmp_path)
    broker = PaperBroker(store, store.config)
    signal_day = date(2026, 9, 1)
    evaluation_day = date(2026, 9, 2)
    _apply(store, signal_day, _response())

    class EvaluationRepository:
        def get_daily(self, symbol, start, end, columns=None):
            del symbol, start, end, columns
            return pl.DataFrame(
                {
                    "symbol": ["000001.SZ", "000001.SZ"],
                    "date": [signal_day, evaluation_day],
                    "open": [9.8, 10.0],
                    "close": [10.0, 11.0],
                    "raw_close": [10.0, 11.0],
                }
            )

        def get_index_daily(self, symbol, start, end, columns=None):
            del symbol, start, end, columns
            return pl.DataFrame(
                {"date": [evaluation_day], "open": [100.0], "close": [101.0]}
            )

    assert broker.evaluate_selection(EvaluationRepository(), evaluation_day) == 1
    summary = store.evaluation()["selection_summary"][0]
    assert summary["horizon_days"] == 1
    assert summary["avg_return_pct"] == pytest.approx(0.10)
    assert summary["avg_excess_return_pct"] == pytest.approx(0.09)


def test_service_runs_complete_automatic_flow(tmp_path, monkeypatch) -> None:
    as_of = date(2026, 9, 1)
    latest = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "name": ["平安银行"],
            "date": [as_of],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "raw_close": [10.1],
            "raw_high": [10.2],
            "raw_low": [9.9],
            "amount": [1_000_000_000.0],
            "momentum_20d": [0.12],
            "vol_ratio_5d": [2.2],
            "turnover_rate": [0.03],
            "signal_ma20_breakout": [True],
            "signal_n_day_high": [False],
            "signal_volume_surge": [True],
            "signal_limit_up": [False],
            "signal_limit_down": [False],
        }
    )

    class FakeRepository:
        def get_enriched_latest(self):
            return latest, as_of

        def get_name_map(self, symbols=None):
            del symbols
            return {"000001.SZ": "平安银行"}

        def get_daily(self, symbol, start, end, columns=None):
            del symbol, start, end, columns
            return latest

        def get_index_daily(self, symbol, start, end, columns=None):
            del symbol, start, end, columns
            return pl.DataFrame()

    async def fake_agent(run_date, inputs):
        assert run_date == as_of
        assert inputs[0]["financials"] == {}
        return _response()

    monkeypatch.setattr("app.custom.ai_research.service.ai_configured", lambda: True)
    monkeypatch.setattr("app.custom.ai_research.service.current_ai_model", lambda: "test-model")
    monkeypatch.setattr("app.custom.ai_research.service.current_ai_provider", lambda: "test")
    monkeypatch.setattr("app.custom.ai_research.service.run_agent", fake_agent)
    service = ResearchService(tmp_path, FakeRepository(), ResearchConfig(max_positions=1))

    result = service.run_once("test")

    assert result["status"] == "completed"
    assert service.store.candidates()[0]["state"] == "eligible"
    assert service.store.portfolio()["orders"][0]["side"] == "buy"
    run = service.store.fetch_one("SELECT input_json, output_json FROM runs LIMIT 1")
    assert run is not None
    assert run["input_json"]
    assert run["output_json"]
