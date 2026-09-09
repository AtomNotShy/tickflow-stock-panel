"""Domain configuration and strict AI decision contract for the research extension."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

METHODOLOGY_VERSION = "ai-research-v1"
ACTIVE_CANDIDATE_STATES = ("discovered", "analyzed", "watching", "eligible")
TERMINAL_CANDIDATE_STATES = ("expired", "invalidated", "archived", "rejected", "stale")


@dataclass(frozen=True)
class ResearchConfig:
    shortlist_size: int = 24
    entry_score: float = 70.0
    exit_score: float = 50.0
    min_confidence: float = 0.65
    confirmations_required: int = 1
    stale_after_misses: int = 3
    cooldown_days: int = 5
    initial_capital: float = 1_000_000.0
    max_positions: int = 5
    max_exposure: float = 0.90
    stop_loss_pct: float = 0.08
    max_hold_days: int = 20
    pending_buy_max_attempts: int = 3
    commission_pct: float = 0.0002
    stamp_tax_pct: float = 0.0005
    slippage_bps: float = 5.0
    evaluation_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)

    @property
    def buy_cost_pct(self) -> float:
        return self.commission_pct + self.slippage_bps / 10_000.0

    @property
    def sell_cost_pct(self) -> float:
        return self.commission_pct + self.stamp_tax_pct + self.slippage_bps / 10_000.0


class AgentDecision(BaseModel):
    """One thesis-level decision. Unknown fields are rejected to prevent prompt drift."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(pattern=r"^\d{6}\.(?:SH|SZ|BJ)$")
    anomaly_key: str = Field(min_length=3, max_length=160)
    action: Literal["enter", "watch", "exit", "reject"]
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    horizon_days: int = Field(ge=1, le=120)
    thesis: str = Field(min_length=8, max_length=1200)
    evidence: list[str] = Field(min_length=1, max_length=12)
    impact_path: list[str] = Field(min_length=1, max_length=8)
    fundamental_view: str = Field(min_length=2, max_length=800)
    expectation_view: str = Field(min_length=2, max_length=800)
    technical_view: str = Field(min_length=2, max_length=800)
    risks: list[str] = Field(min_length=1, max_length=8)
    technical_confirmed: bool
    risk_veto: bool
    invalidated: bool

    @field_validator("evidence", "impact_path", "risks")
    @classmethod
    def _non_empty_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty item is required")
        return cleaned


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    market_summary: str = Field(min_length=2, max_length=1200)
    decisions: list[AgentDecision] = Field(max_length=80)


class AgentOutputError(ValueError):
    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


def parse_agent_response(raw: str, allowed: set[tuple[str, str]]) -> AgentResponse:
    """Parse a JSON-only response and reject hallucinated symbols or anomaly identities."""
    text = raw.strip()
    if not text:
        raise AgentOutputError("AI service returned empty visible content", raw)

    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
        )
    )
    decoder = json.JSONDecoder()
    candidates.extend(text[index:] for index, char in enumerate(text) if char == "{")
    last_error: Exception | None = None
    response: AgentResponse | None = None
    for candidate in candidates:
        try:
            payload, _end = decoder.raw_decode(candidate.lstrip())
            response = AgentResponse.model_validate(payload)
            break
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
    if response is None:
        preview = re.sub(r"\s+", " ", text)[:240]
        raise AgentOutputError(
            f"AI output does not match the decision schema: {last_error}; preview={preview!r}",
            raw,
        ) from last_error

    seen: set[tuple[str, str]] = set()
    for decision in response.decisions:
        identity = (decision.symbol, decision.anomaly_key)
        if identity not in allowed:
            raise AgentOutputError(f"AI returned an unprovided candidate: {identity}", raw)
        if identity in seen:
            raise AgentOutputError(f"AI returned a duplicate candidate: {identity}", raw)
        seen.add(identity)
    return response
