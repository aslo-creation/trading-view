"""
agents/risk_structurer.py
=========================
The Structurer / Risk Manager. Goes LAST. Consumes the Macro and Quant
reports, computes invalidation (stop) zones from volume-profile liquidity
levels, and produces a position-sizing recommendation capped by both
quarter-Kelly and a portfolio VaR budget. Holds veto power: if macro and
quant disagree with high confidence on both sides, the verdict is NO TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from agents.base import AgentReport, CommitteeAgent
from core.math_engine import position_size


@dataclass(frozen=True)
class InvalidationZone:
    symbol: str
    side: str                # 'long' | 'short'
    entry_ref: float
    stop: float
    stop_distance_pct: float
    basis: str


def volume_profile_levels(df: pd.DataFrame, bins: int = 24) -> list[float]:
    """High-volume nodes: prices where the most volume transacted. These act
    as liquidity pools — logical invalidation anchors beyond local extremes."""
    prices = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()
    vols = df["volume"].to_numpy(dtype=float)
    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    top = np.argsort(hist)[-4:]
    return sorted(float((edges[i] + edges[i + 1]) / 2.0) for i in top)


def build_invalidation(symbol: str, df: pd.DataFrame, side: str) -> InvalidationZone:
    last = float(df["close"].iloc[-1])
    nodes = volume_profile_levels(df)
    if side == "long":
        below = [n for n in nodes if n < last]
        stop = (max(below) if below else float(df["low"].tail(20).min())) * 0.997
        basis = "Below nearest high-volume node (liquidity pool) under price."
    else:
        above = [n for n in nodes if n > last]
        stop = (min(above) if above else float(df["high"].tail(20).max())) * 1.003
        basis = "Above nearest high-volume node (liquidity pool) over price."
    return InvalidationZone(symbol, side, last, round(stop, 4),
                            round(abs(stop - last) / last * 100.0, 2), basis)


class RiskStructurer(CommitteeAgent):
    name = "Structurer / Risk Manager"
    system_prompt = (
        "You are the risk manager with veto power on an institutional committee. "
        "You receive the macro and quant agents' reports plus computed stops and "
        "sizing. Your job is to challenge the trade, not to sell it. If the "
        "evidence conflicts, say NO TRADE."
    )

    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        macro: AgentReport = context["macro_report"]
        quant: AgentReport = context["quant_report"]
        ohlcv: dict[str, pd.DataFrame] = context["ohlcv"]

        # Pick candidate: most extreme |z20| flagged by the quant agent.
        zmap = quant.data.get("zscores", {})
        extremes = {s: v for s, v in zmap.items() if v.get("extreme")}
        if not extremes:
            return AgentReport(
                agent=self.name, stance="NO TRADE", confidence=0.8,
                key_points=["No statistical extreme passed the quant filter; "
                            "capital preservation is the trade."],
                data={"verdict": "no_trade"},
            )

        symbol = max(extremes, key=lambda s: abs(extremes[s]["z20"]))
        side = "short" if extremes[symbol]["direction"] == "overextended_up" else "long"
        zone = build_invalidation(symbol, ohlcv[symbol], side)

        # Conservative priors; calibrate from your own signal backtests.
        sizing = position_size(win_prob=0.52, win_loss_ratio=1.8,
                               closes=ohlcv[symbol]["close"])

        parsed = await self._ask_llm_json(
            "Committee inputs:\n"
            f"MACRO REPORT:\n{macro.to_prompt_block()}\n"
            f"QUANT REPORT:\n{quant.to_prompt_block()}\n"
            f"Proposed: {side.upper()} {symbol}, entry≈{zone.entry_ref}, "
            f"stop={zone.stop} ({zone.stop_distance_pct}% away, basis: {zone.basis}), "
            f"max size {sizing.max_position_pct:.2%} of portfolio.\n"
            "Stress-test this. Approve, shrink, or veto — and say why."
        )

        return AgentReport(
            agent=self.name,
            stance=parsed["stance"],
            confidence=parsed["confidence"],
            key_points=[
                f"Setup: {side.upper()} {symbol} | entry ref {zone.entry_ref} | "
                f"stop {zone.stop} ({zone.stop_distance_pct}%)",
                f"Stop basis: {zone.basis}",
                *sizing.rationale,
                *parsed.get("key_points", []),
            ],
            data={
                "verdict": "sized_trade",
                "symbol": symbol, "side": side,
                "invalidation": zone.__dict__,
                "sizing": {
                    "kelly_full": sizing.kelly_full,
                    "kelly_used": sizing.kelly_fraction_used,
                    "var_95": sizing.var_95_pct,
                    "max_position_pct": sizing.max_position_pct,
                },
            },
        )
