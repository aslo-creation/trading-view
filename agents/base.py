"""
agents/base.py
==============
Shared contract for committee members. Each agent produces a structured
`AgentReport` so the orchestrator can run a deterministic debate loop and the
UI can render a clean "Investment Committee Debrief".

LLM usage discipline:
- Quantitative facts are computed in core/math_engine.py and passed to the
  model as ground truth. The model interprets; it never invents numbers.
- All LLM calls flow through services.api_client.ClaudeClient, which is
  rate-limited and HTTPS-enforced.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from services.api_client import ClaudeClient

logger = logging.getLogger("agents")


@dataclass
class AgentReport:
    agent: str
    stance: str                      # e.g. 'Risk-On', 'Fade the move', 'No trade'
    confidence: float                # 0.0 - 1.0, self-reported & bounded
    key_points: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)   # machine-readable facts

    def to_prompt_block(self) -> str:
        return json.dumps(
            {"agent": self.agent, "stance": self.stance,
             "confidence": self.confidence, "key_points": self.key_points},
            indent=2,
        )


class CommitteeAgent(ABC):
    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm: ClaudeClient) -> None:
        self.llm = llm

    @abstractmethod
    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        """Consume market context (and prior reports), return a report."""

    async def _ask_llm_json(self, user_prompt: str) -> dict[str, Any]:
        """Request strict JSON; tolerate fenced output; fail safe to neutral."""
        raw = await self.llm.complete(
            system=self.system_prompt + (
                "\nRespond with ONLY a JSON object, no markdown fences, with keys: "
                "stance (string), confidence (number 0-1), key_points (array of strings)."
            ),
            user=user_prompt,
        )
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
            parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("%s: LLM returned non-JSON; defaulting to neutral.", self.name)
            return {"stance": "No view (parse failure)", "confidence": 0.0,
                    "key_points": ["Model output unparseable — treated as abstention."]}
