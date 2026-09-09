"""Local anomaly discovery and one-call AI research orchestration."""
# ruff: noqa: RUF001
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from app.services.ai_provider import current_ai_max_output_tokens, generate_ai_text
from app.services.financial_sync import get_financial_df

from .domain import AgentResponse, ResearchConfig, parse_agent_response

_FEATURE_COLUMNS = [
    "symbol", "name", "date", "open", "high", "low", "close", "raw_close",
    "volume", "amount", "turnover_rate", "change_pct", "amplitude", "ma5",
    "ma20", "ma60", "vol_ratio_5d", "momentum_5d", "momentum_20d",
    "momentum_60d", "rsi_14", "macd_dif", "macd_dea", "annual_vol_20d",
    "signal_ma20_breakout", "signal_ma20_breakdown", "signal_n_day_high",
    "signal_volume_surge", "signal_limit_up", "signal_limit_down",
]

# A single response containing every field for the full 24-stock shortlist can
# exceed otherwise generous provider defaults.  Keep one research methodology
# and one model, but request decisions in bounded batches so every JSON document
# remains small enough to validate and retry independently.
_AGENT_BATCH_SIZE = 6


def _safe(value: Any) -> Any:
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


def _anomaly_key(row: dict[str, Any]) -> str:
    for column, label in (
        ("signal_limit_up", "limit-up"),
        ("signal_n_day_high", "sixty-day-high"),
        ("signal_ma20_breakout", "ma20-breakout"),
        ("signal_volume_surge", "volume-surge"),
    ):
        if row.get(column):
            return f"local:{label}"
    return "local:momentum-liquidity"


def discover_inputs(
    latest: pl.DataFrame,
    active: list[dict[str, Any]],
    config: ResearchConfig,
) -> list[dict[str, Any]]:
    """Build a deterministic, liquid shortlist and always retain active theses."""
    if latest.is_empty() or "symbol" not in latest.columns:
        return []
    cols = [column for column in _FEATURE_COLUMNS if column in latest.columns]
    frame = latest.select(cols)
    if "amount" in frame.columns:
        frame = frame.filter(pl.col("amount").fill_null(0) > 0)
    if "name" in frame.columns:
        frame = frame.filter(~pl.col("name").fill_null("").str.contains("ST"))

    score_parts: list[pl.Expr] = []
    for column, weight in (
        ("momentum_20d", 0.30),
        ("amount", 0.25),
        ("vol_ratio_5d", 0.20),
        ("turnover_rate", 0.15),
    ):
        if column in frame.columns:
            score_parts.append(
                pl.col(column).fill_null(float("-inf")).rank("average")
                / max(len(frame), 1) * weight
            )
    signal_cols = [
        c for c in ("signal_ma20_breakout", "signal_n_day_high", "signal_volume_surge")
        if c in frame.columns
    ]
    if signal_cols:
        signal_points = sum(
            (pl.col(column).fill_null(False).cast(pl.Int8) for column in signal_cols),
            start=pl.lit(0),
        )
        score_parts.append((signal_points.clip(0, 2) / 2) * 0.10)
    if not score_parts:
        return []
    composite = sum(score_parts[1:], start=score_parts[0]).alias("_local_score")
    frame = frame.with_columns(composite).sort(
        ["_local_score", "symbol"], descending=[True, False]
    )

    active_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for candidate in active:
        active_by_symbol.setdefault(str(candidate["symbol"]), []).append(candidate)
    wanted_symbols = set(active_by_symbol)
    rows = frame.head(config.shortlist_size).to_dicts()
    selected_symbols = {str(row["symbol"]) for row in rows}
    if wanted_symbols - selected_symbols:
        rows.extend(
            frame.filter(pl.col("symbol").is_in(sorted(wanted_symbols - selected_symbols))).to_dicts()
        )

    inputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        symbol = str(row["symbol"])
        base = {key: _safe(value) for key, value in row.items() if key != "_local_score"}
        active_for_symbol = active_by_symbol.get(symbol, [])
        identities = [(str(item["anomaly_key"]), item) for item in active_for_symbol]
        current_key = _anomaly_key(row)
        if not any(key == current_key for key, _previous in identities):
            identities.append((current_key, None))
        for anomaly_key, previous in identities:
            identity = (symbol, anomaly_key)
            if identity in seen:
                continue
            seen.add(identity)
            inputs.append(
                {
                    "symbol": symbol,
                    "name": row.get("name"),
                    "anomaly_key": anomaly_key,
                    "local_rank_score": round(float(row.get("_local_score") or 0) * 100, 2),
                    "market_data": base,
                    "previous_thesis": {
                        key: previous.get(key)
                        for key in ("state", "score", "confidence", "thesis", "risks_json")
                    } if previous else None,
                }
            )
    return inputs


def attach_financials(data_dir: Path, as_of: date, inputs: list[dict[str, Any]]) -> None:
    symbols = {item["symbol"] for item in inputs}
    by_symbol: dict[str, dict[str, list[dict[str, Any]]]] = {
        symbol: {} for symbol in symbols
    }
    for table in ("metrics", "income", "balance_sheet"):
        frame = get_financial_df(data_dir, table)
        if frame.is_empty() or "symbol" not in frame.columns:
            continue
        frame = frame.filter(pl.col("symbol").is_in(sorted(symbols)))
        if "announce_date" in frame.columns:
            frame = frame.filter(
                pl.col("announce_date").is_null()
                | (pl.col("announce_date").cast(pl.Date, strict=False) <= as_of)
            )
        sort_col = "period_end" if "period_end" in frame.columns else None
        if sort_col:
            frame = frame.sort(sort_col, descending=True)
        for symbol in symbols:
            rows = frame.filter(pl.col("symbol") == symbol).head(2).to_dicts()
            by_symbol[symbol][table] = [
                {key: _safe(value) for key, value in row.items() if key != "symbol"}
                for row in rows
            ]
    for item in inputs:
        item["financials"] = by_symbol[item["symbol"]]


_SYSTEM_PROMPT = """你是一个严谨的 A 股事件驱动研究 Agent。一次完成以下链路：
1. Event：只识别输入中明确存在的本地行情异常；没有新闻源，不得编造新闻、政策、公告或宏观事件。
2. Impact：给出从该异常到产业/公司收入、成本、库存、利润的影响路径。
3. Stock mapping：只能分析输入给你的股票及 anomaly_key，不能新增标的。
4. Fundamental：使用给定的点时财务数据；缺失时必须明确写“数据不足”。
5. Expectation：判断价格和成交量是否可能已经反映预期，并明确这是行情数据推断。
6. Technical：判断突破、回踩、缩量、支撑等技术确认。
7. Risk：列出可以被后续数据验证的证伪条件；硬风险时 risk_veto=true。
8. Portfolio：给出 enter/watch/exit/reject 和 0-100 分排序分。

这是研究与模拟实验，不是真实投资建议。必须严格输出一个 JSON 对象，不要 Markdown，不要代码围栏，结构如下：
{"market_summary":"...","decisions":[{"symbol":"000001.SZ","anomaly_key":"local:...","action":"enter|watch|exit|reject","score":0,"confidence":0.0,"horizon_days":20,"thesis":"...","evidence":["输入中的字段与数值"],"impact_path":["异常","产业节点","公司利润"],"fundamental_view":"...","expectation_view":"...","technical_view":"...","risks":["可证伪条件"],"technical_confirmed":true,"risk_veto":false,"invalidated":false}]}

每个输入候选必须且只能返回一条 decision。每个结论必须能追溯到输入数据。
保持内容紧凑：thesis 和三个 view 各不超过 120 个汉字；evidence、impact_path、risks 各不超过 3 项，每项不超过 60 个汉字。
对仍有效的 previous_thesis 要复核而不是自动重复；明确证伪时 action=exit 且 invalidated=true。"""


async def _run_agent_batch(
    as_of: date,
    inputs: list[dict[str, Any]],
    *,
    batch_number: int,
    batch_count: int,
) -> AgentResponse:
    allowed = {(item["symbol"], item["anomaly_key"]) for item in inputs}
    user_prompt = (
        f"研究日期：{as_of.isoformat()}。以下数据均截至该日收盘，禁止使用未来数据。\n"
        f"这是本次研究的第 {batch_number}/{batch_count} 批，共 {len(inputs)} 个候选；"
        "必须逐一返回 decision。\n"
        "候选输入(JSON)：\n" + json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    )
    raw = await generate_ai_text(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        # Thinking is disabled for this structured report, so the configured
        # output allowance can safely be sent instead of relying on the
        # provider's smaller default.
        max_tokens=current_ai_max_output_tokens(),
        timeout=240,
        prefer_final_answer=True,
    )
    response = parse_agent_response(raw, allowed)
    returned = {(item.symbol, item.anomaly_key) for item in response.decisions}
    missing = allowed - returned
    if missing:
        preview = ", ".join(f"{symbol}/{key}" for symbol, key in sorted(missing)[:3])
        raise ValueError(f"AI 未返回本批次全部候选决策: {preview}")
    return response


async def run_agent(
    as_of: date,
    inputs: list[dict[str, Any]],
) -> AgentResponse:
    batches = [
        inputs[offset:offset + _AGENT_BATCH_SIZE]
        for offset in range(0, len(inputs), _AGENT_BATCH_SIZE)
    ]
    responses = [
        await _run_agent_batch(
            as_of,
            batch,
            batch_number=index,
            batch_count=len(batches),
        )
        for index, batch in enumerate(batches, start=1)
    ]
    summaries = [response.market_summary.strip() for response in responses]
    market_summary = "；".join(summary for summary in summaries if summary)[:1200]
    decisions = [decision for response in responses for decision in response.decisions]
    return AgentResponse(market_summary=market_summary, decisions=decisions)
