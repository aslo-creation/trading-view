"""
services/market_data.py
=======================
Live market data via Yahoo Finance (yfinance — free, no API key) with an
automatic fallback to the synthetic demo generator when offline or when a
symbol fails. The UI always knows which symbols are LIVE vs DEMO.

Friendly aliases keep the UX simple while real Yahoo tickers do the work:
    GOLD -> GC=F (Gold futures)      WTI -> CL=F (Crude Oil WTI futures)
    SPX  -> ^GSPC (S&P 500 index)    BTC -> BTC-USD
User-added tickers MUST already be sanitized by config.security.sanitize_ticker.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from services.demo_data import load_demo_fred, load_demo_ohlcv

logger = logging.getLogger("market_data")

FRIENDLY_TO_YAHOO = {"GOLD": "GC=F", "WTI": "CL=F", "SPX": "^GSPC", "BTC": "BTC-USD"}
LABELS = {
    "GOLD": "Or (Gold futures)",
    "WTI": "Pétrole brut (WTI)",
    "SPX": "S&P 500",
    "BTC": "Bitcoin",
}

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def to_yahoo(symbol: str) -> str:
    return FRIENDLY_TO_YAHOO.get(symbol, symbol)


def label_of(symbol: str) -> str:
    return LABELS.get(symbol, symbol)


def fetch_history(symbol: str, period: str = "6mo",
                  interval: str = "1d") -> Optional[pd.DataFrame]:
    """One symbol -> normalized lowercase OHLCV DataFrame, or None on failure."""
    try:
        import yfinance as yf  # imported lazily so the app still boots without it
        df = yf.Ticker(to_yahoo(symbol)).history(
            period=period, interval=interval, auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        if missing:
            return None
        out = df[OHLCV_COLS].copy()
        out.index = pd.to_datetime(out.index)
        # Indices (^GSPC) sometimes report 0 volume on partial days — keep rows,
        # the math engine is NaN/zero-volume safe.
        return out.dropna(subset=["close"])
    except Exception as exc:  # noqa: BLE001 — network boundary, degrade only
        logger.warning("Live fetch failed for %s: %s", symbol, type(exc).__name__)
        return None


def fetch_many(symbols: list[str], period: str = "6mo",
               interval: str = "1d") -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Returns ({symbol: df}, demo_fallback_symbols).
    Symbols are fetched IN PARALLEL — total latency ≈ slowest single symbol
    instead of the sum of all of them."""
    from concurrent.futures import ThreadPoolExecutor
    out: dict[str, pd.DataFrame] = {}
    demo: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(symbols)))) as pool:
        fetched = dict(zip(symbols,
                           pool.map(lambda s: fetch_history(s, period, interval),
                                    symbols)))
    for s in symbols:
        df = fetched.get(s)
        if df is None or len(df) < 30:
            demo.add(s)
            df = load_demo_ohlcv([s])[s]
        out[s] = df
    return out, demo


def fetch_intraday(symbol: str) -> Optional[pd.DataFrame]:
    """5-minute bars over the last few days — used by the news-latency engine."""
    return fetch_history(symbol, period="5d", interval="5m")


# ---------------------------------------------------------------------------
# FRED (macro series) — live when FRED_API_KEY is configured, demo otherwise
# ---------------------------------------------------------------------------

FRED_SERIES = ("DGS10", "DGS2", "DTWEXBGS", "T10YIE", "NFCI", "DFF", "CPIAUCSL")


def fetch_fred() -> tuple[dict[str, pd.Series], bool]:
    """Returns ({series_id: Series}, is_live)."""
    key = os.getenv("FRED_API_KEY")
    if not key:
        return load_demo_fred(), False
    try:
        import httpx
        out: dict[str, pd.Series] = {}
        with httpx.Client(timeout=10.0) as client:
            for sid in FRED_SERIES:
                r = client.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": sid, "api_key": key, "file_type": "json",
                            "sort_order": "asc", "limit": 200},
                )
                r.raise_for_status()
                obs = r.json().get("observations", [])
                s = pd.Series({
                    pd.Timestamp(o["date"]): float(o["value"])
                    for o in obs if o.get("value") not in (".", "", None)
                }).sort_index()
                if not s.empty:
                    out[sid] = s
        if len(out) >= 2:
            return out, True
        return load_demo_fred(), False
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED live fetch failed: %s", type(exc).__name__)
        return load_demo_fred(), False


def fed_funds_implied() -> float | None:
    """Taux implicite des Fed Funds via le contrat à terme ZQ=F (100 - prix).
    Estimation simplifiée du marché — None si indisponible."""
    df = fetch_history("ZQ=F", period="5d", interval="1d")
    if df is None or df.empty:
        return None
    try:
        return round(100.0 - float(df["close"].iloc[-1]), 3)
    except (KeyError, ValueError, TypeError):
        return None
