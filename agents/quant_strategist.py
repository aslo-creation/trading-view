"""
agents/quant_strategist.py
==========================
The Quantitative Strategist. Runs core/math_engine over Gold, Crude Oil,
S&P 500 and Bitcoin: VWAP-distance Z-scores, volatility structural breaks,
and cross-asset correlation breakdowns. All numbers are computed locally;
the LLM only synthesizes a narrative over verified figures.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agents.base import AgentReport, CommitteeAgent
from core.math_engine import (
    detect_divergences,
    mean_reversion_scan,
    volatility_break,
)

CORE_ASSETS = ("GOLD", "WTI", "SPX", "BTC")


class QuantStrategist(CommitteeAgent):
    name = "Quantitative Strategist"
    system_prompt = (
        "You are the quant strategist on an institutional committee. You receive "
        "verified statistics (z-scores, vol ratios, correlation deltas). Identify "
        "which anomalies are tradeable and which are noise. Never invent numbers."
    )

    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        ohlcv: dict[str, pd.DataFrame] = context["ohlcv"]   # symbol -> OHLCV df

        signals = []
        vol_flags = []
        for sym, df in ohlcv.items():
            try:
                mr = mean_reversion_scan(sym, df)
                vb = volatility_break(sym, df["close"])
            except (KeyError, IndexError, ValueError):
                continue
            signals.append(mr)
            if vb.structural_break:
                vol_flags.append(vb)

        closes = pd.DataFrame({s: d["close"] for s, d in ohlcv.items()}).dropna()
        vols = pd.DataFrame({s: d["volume"] for s, d in ohlcv.items()}).dropna()
        divergences = detect_divergences(closes, vols)[:5]

        facts = {
            "zscores": {s.symbol: {"z20": round(s.z20, 2), "z50": round(s.z50, 2),
                                   "extreme": s.extreme, "direction": s.direction}
                        for s in signals},
            "vol_breaks": [{"symbol": v.symbol, "ratio": round(v.ratio, 2)}
                           for v in vol_flags],
            "divergences": [{"pair": list(d.pair), "delta": round(d.delta, 2),
                             "volume_surge": d.volume_surge} for d in divergences],
        }

        parsed = await self._ask_llm_json(
            "Verified statistics from the math engine (ground truth):\n"
            f"{facts}\n\nWhich of these are actionable anomalies vs noise, "
            "and on which of GOLD/WTI/SPX/BTC?"
        )

        extremes = [s for s in signals if s.extreme]
        return AgentReport(
            agent=self.name,
            stance=parsed["stance"],
            confidence=parsed["confidence"],
            key_points=(
                [f"{s.symbol}: z20={s.z20:+.2f} ({s.direction})" for s in extremes]
                + [f"VOL BREAK {v.symbol}: recent/baseline vol ratio {v.ratio:.1f}x"
                   for v in vol_flags]
                + [f"DIVERGENCE {d.pair[0]}/{d.pair[1]}: corr shift {d.delta:+.2f}"
                   + (" + volume surge" if d.volume_surge else "")
                   for d in divergences]
                + list(parsed.get("key_points", []))
            ),
            data=facts,
        )
