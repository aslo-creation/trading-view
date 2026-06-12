"""
agents/orchestrator.py
======================
Committee orchestration with async coroutines (no heavyweight framework).

Flow:
  1. Macro Economist + Quant Strategist run CONCURRENTLY (asyncio.gather).
  2. Risk Structurer consumes both reports and the raw OHLCV.
  3. One bounded "debate" round: if structurer confidence < threshold, the
     dissent is sent back to macro+quant for a single rebuttal each.
  4. Emit a CommitteeDebrief: ordered transcript + final verdict.

Bounded by design: max ONE rebuttal round, every step rate-limited upstream,
every LLM failure degrades to abstention instead of crashing the session.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.base import AgentReport
from agents.macro_analyst import MacroAnalyst
from agents.quant_strategist import QuantStrategist
from agents.risk_structurer import RiskStructurer
from services.api_client import ClaudeClient, SecureHttpClient

logger = logging.getLogger("orchestrator")

DEBATE_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class CommitteeDebrief:
    started_at: str
    transcript: list[AgentReport] = field(default_factory=list)
    rebuttals: list[AgentReport] = field(default_factory=list)
    final_verdict: dict[str, Any] = field(default_factory=dict)

    def render_lines(self) -> list[tuple[str, str]]:
        """(speaker, text) pairs for the UI debrief panel."""
        lines: list[tuple[str, str]] = []
        for rep in self.transcript + self.rebuttals:
            lines.append((rep.agent,
                          f"[conf {rep.confidence:.0%}] {rep.stance}"))
            for kp in rep.key_points:
                lines.append((rep.agent, f"  • {kp}"))
        return lines


class CommitteeOrchestrator:
    def __init__(self, http: SecureHttpClient | None = None) -> None:
        self.http = http or SecureHttpClient()
        llm = ClaudeClient(self.http)
        self.macro = MacroAnalyst(llm)
        self.quant = QuantStrategist(llm)
        self.risk = RiskStructurer(llm)

    async def convene(self, market_context: dict[str, Any]) -> CommitteeDebrief:
        debrief = CommitteeDebrief(
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
        )

        # Round 1 — independent analysis in parallel.
        macro_task = asyncio.create_task(self._safe(self.macro.analyze, market_context))
        quant_task = asyncio.create_task(self._safe(self.quant.analyze, market_context))
        macro_report, quant_report = await asyncio.gather(macro_task, quant_task)
        debrief.transcript += [macro_report, quant_report]

        # Round 2 — risk structurer with full context.
        risk_ctx = {**market_context,
                    "macro_report": macro_report, "quant_report": quant_report}
        risk_report = await self._safe(self.risk.analyze, risk_ctx)
        debrief.transcript.append(risk_report)

        # Optional single debate round when the risk desk is unconvinced.
        if (risk_report.data.get("verdict") == "sized_trade"
                and risk_report.confidence < DEBATE_CONFIDENCE_THRESHOLD):
            logger.info("Risk confidence %.2f < %.2f — opening one rebuttal round.",
                        risk_report.confidence, DEBATE_CONFIDENCE_THRESHOLD)
            rebut_ctx = {**market_context, "risk_dissent": risk_report.key_points}
            r1, r2 = await asyncio.gather(
                self._safe(self.macro.analyze, rebut_ctx),
                self._safe(self.quant.analyze, rebut_ctx),
            )
            debrief.rebuttals += [r1, r2]
            risk_report = await self._safe(
                self.risk.analyze,
                {**risk_ctx, "macro_report": r1, "quant_report": r2},
            )
            debrief.rebuttals.append(risk_report)

        debrief.final_verdict = {
            "verdict": risk_report.data.get("verdict", "no_trade"),
            "detail": risk_report.data,
            "stance": risk_report.stance,
            "confidence": risk_report.confidence,
            "disclaimer": ("Analytical output for research purposes — not "
                           "investment advice. Validate independently before "
                           "risking capital."),
        }
        return debrief

    @staticmethod
    async def _safe(coro_fn, ctx) -> AgentReport:
        """Any agent failure becomes an explicit abstention, never a crash."""
        try:
            return await coro_fn(ctx)
        except Exception as exc:  # noqa: BLE001 — boundary: degrade gracefully
            logger.error("Agent failure: %s", exc)
            return AgentReport(agent=getattr(coro_fn, "__qualname__", "agent"),
                               stance="ABSTAIN (error)", confidence=0.0,
                               key_points=[f"Agent error: {type(exc).__name__}"],
                               data={"verdict": "abstain"})

    async def aclose(self) -> None:
        await self.http.aclose()
