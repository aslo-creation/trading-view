"""
services/rate_limiter.py
========================
Sliding-window rate limiting to protect the LLM endpoint (financial drain)
and market-data endpoints (vendor bans) from abusive or runaway clients.

Architecture: an abstract `RateLimiterBackend` so the default thread-safe
in-memory deque implementation can be swapped for Redis (ZADD/ZREMRANGEBYSCORE)
in a multi-replica deployment with zero changes at call sites.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Deque, Dict


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class RateLimitExceeded(Exception):
    def __init__(self, decision: RateLimitDecision, scope: str):
        self.decision = decision
        super().__init__(
            f"Rate limit exceeded for '{scope}'. "
            f"Retry in {decision.retry_after_seconds:.1f}s."
        )


class RateLimiterBackend(ABC):
    @abstractmethod
    def check(self, key: str, max_calls: int, window_seconds: float) -> RateLimitDecision:
        ...


class InMemorySlidingWindow(RateLimiterBackend):
    """Thread-safe sliding window. O(1) amortized per check.
    Suitable for a single-process Streamlit deployment behind a reverse proxy."""

    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, max_calls: int, window_seconds: float) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            cutoff = now - window_seconds
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= max_calls:
                retry_after = q[0] + window_seconds - now
                return RateLimitDecision(False, 0, max(retry_after, 0.0))
            q.append(now)
            return RateLimitDecision(True, max_calls - len(q), 0.0)


# class RedisSlidingWindow(RateLimiterBackend):
#     """Production multi-replica backend (sketch):
#     pipeline: ZREMRANGEBYSCORE key 0 (now-window) ; ZCARD key ;
#               ZADD key now now ; EXPIRE key window
#     Implement when scaling horizontally — interface stays identical."""


class RateLimiter:
    """Facade with named scopes so different resources get different budgets."""

    DEFAULT_POLICIES: dict[str, tuple[int, float]] = {
        # scope:            (max_calls, window_seconds)
        "llm_committee":    (5,   60.0),    # Claude calls are the costly resource
        "market_data":      (60,  60.0),
        "auth_attempts":    (5,   300.0),   # brute-force mitigation
        "ticker_mutation":  (10,  60.0),
    }

    def __init__(self, backend: RateLimiterBackend | None = None) -> None:
        self.backend = backend or InMemorySlidingWindow()

    def acquire(self, scope: str, identity: str) -> RateLimitDecision:
        max_calls, window = self.DEFAULT_POLICIES.get(scope, (30, 60.0))
        decision = self.backend.check(f"{scope}:{identity}", max_calls, window)
        if not decision.allowed:
            raise RateLimitExceeded(decision, scope)
        return decision

    def guard(self, scope: str, identity_fn: Callable[..., str] = lambda *a, **k: "global"):
        """Decorator form:  @limiter.guard('llm_committee', lambda user: user)"""
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                self.acquire(scope, identity_fn(*args, **kwargs))
                return fn(*args, **kwargs)
            return wrapper
        return decorator


# Process-wide singleton used by app.py and services.
GLOBAL_LIMITER = RateLimiter()
