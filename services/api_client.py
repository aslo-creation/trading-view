"""
services/api_client.py
======================
Single hardened gateway for ALL outbound network calls.

Guarantees:
- HTTPS only (any http:// URL raises before a socket opens; TLS 1.3 is
  negotiated automatically by modern OpenSSL — see docs/DEPLOYMENT.md for
  enforcing minimums at the proxy).
- Hard connect/read timeouts so a hung vendor cannot freeze the event loop.
- Bounded retries with exponential backoff + jitter on transient failures only.
- Typed exceptions; raw vendor errors (which may embed our query params)
  are scrubbed before logging.
- Every call passes through the global rate limiter.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from typing import Any, Optional

import httpx

from config.security import get_secret, scrub_for_logging
from services.rate_limiter import GLOBAL_LIMITER, RateLimitExceeded

logger = logging.getLogger("api_client")

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 20.0
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ApiError(Exception):
    """Base for all outbound-call failures (already scrubbed of secrets)."""


class ApiTimeout(ApiError):
    pass


class ApiUpstreamError(ApiError):
    def __init__(self, status: int, body_preview: str):
        self.status = status
        super().__init__(f"Upstream returned {status}: {body_preview[:200]}")


def _tls_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # 1.3 preferred & auto-negotiated
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


class SecureHttpClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT,
                                  write=10.0, pool=5.0),
            verify=_tls_context(),
            follow_redirects=False,          # redirects can downgrade/leak
            headers={"User-Agent": "quant-terminal/1.0"},
        )

    async def get_json(self, url: str, params: Optional[dict[str, Any]] = None,
                       headers: Optional[dict[str, str]] = None,
                       rate_scope: str = "market_data") -> Any:
        if not url.lower().startswith("https://"):
            raise ApiError("Refused: non-HTTPS URL blocked by policy.")
        GLOBAL_LIMITER.acquire(rate_scope, identity="backend")

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.get(url, params=params, headers=headers)
                if resp.status_code in RETRYABLE_STATUS:
                    raise ApiUpstreamError(resp.status_code,
                                           scrub_for_logging(resp.text))
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_exc = ApiTimeout(f"Timeout contacting upstream (attempt {attempt}).")
            except ApiUpstreamError as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                # 4xx other than 429: do NOT retry — it's our fault or auth.
                raise ApiUpstreamError(exc.response.status_code,
                                       scrub_for_logging(exc.response.text)) from None
            backoff = min(2 ** attempt, 8) + random.uniform(0, 0.5)
            logger.warning("Retry %d/%d after error: %s (sleep %.1fs)",
                           attempt, MAX_RETRIES, last_exc, backoff)
            await asyncio.sleep(backoff)
        raise last_exc or ApiError("Exhausted retries.")

    async def post_json(self, url: str, json_body: dict[str, Any],
                        headers: Optional[dict[str, str]] = None,
                        rate_scope: str = "market_data") -> Any:
        if not url.lower().startswith("https://"):
            raise ApiError("Refused: non-HTTPS URL blocked by policy.")
        GLOBAL_LIMITER.acquire(rate_scope, identity="backend")
        try:
            resp = await self._client.post(url, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            raise ApiTimeout("Timeout on POST.") from None
        except httpx.HTTPStatusError as exc:
            raise ApiUpstreamError(exc.response.status_code,
                                   scrub_for_logging(exc.response.text)) from None

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Vendor wrappers — keys injected per-request, never stored on instances
# ---------------------------------------------------------------------------

class ClaudeClient:
    """Minimal Anthropic Messages API wrapper for the agent committee."""

    URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-6"

    def __init__(self, http: SecureHttpClient) -> None:
        self.http = http

    async def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        headers = {
            "x-api-key": get_secret("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = await self.http.post_json(self.URL, body, headers=headers,
                                         rate_scope="llm_committee")
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")


class FredClient:
    URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, http: SecureHttpClient) -> None:
        self.http = http

    async def series(self, series_id: str, limit: int = 120) -> list[dict[str, Any]]:
        params = {
            "series_id": series_id,
            "api_key": get_secret("FRED_API_KEY"),
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        data = await self.http.get_json(self.URL, params=params)
        return data.get("observations", [])


class AlphaVantageClient:
    URL = "https://www.alphavantage.co/query"

    def __init__(self, http: SecureHttpClient) -> None:
        self.http = http

    async def daily(self, symbol: str) -> dict[str, Any]:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,                 # MUST already be sanitized upstream
            "apikey": get_secret("ALPHAVANTAGE_API_KEY"),
            "outputsize": "compact",
        }
        return await self.http.get_json(self.URL, params=params)
