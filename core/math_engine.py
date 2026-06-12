"""
core/math_engine.py
===================
Pure, dependency-light quantitative kernels. No I/O, no secrets, fully
unit-testable. Everything returns plain dataclasses/np arrays so the agents
layer can reason over results without touching pandas internals.

Implements:
- VWAP + rolling Z-Score of price distance from VWAP (mean-reversion engine)
- Rolling correlation matrix + inter-market divergence (correlation breakdown)
- Realized-volatility structural-break detection (variance-ratio test)
- News→volume latency metric (sentiment momentum arbitrage)
- Kelly criterion (fractional) and historical VaR position sizing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

Z_EXTREME = 2.5


# ---------------------------------------------------------------------------
# VWAP & Mean reversion
# ---------------------------------------------------------------------------

def vwap(df: pd.DataFrame) -> pd.Series:
    """df must contain columns: high, low, close, volume."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / std.replace(0, np.nan)


@dataclass(frozen=True)
class MeanReversionSignal:
    symbol: str
    z20: float
    z50: float
    z_intraday: float
    extreme: bool
    direction: str  # 'overextended_up' | 'overextended_down' | 'neutral'


def mean_reversion_scan(symbol: str, df: pd.DataFrame) -> MeanReversionSignal:
    """Z-scores of the close's distance from VWAP over multiple horizons."""
    dist = df["close"] - vwap(df)
    z20 = float(rolling_zscore(dist, 20).iloc[-1])
    z50 = float(rolling_zscore(dist, 50).iloc[-1]) if len(df) >= 50 else float("nan")
    zid = float(rolling_zscore(dist, min(len(df), 12)).iloc[-1])

    extreme = any(abs(z) >= Z_EXTREME for z in (z20, z50) if not np.isnan(z))
    if extreme and z20 > 0:
        direction = "overextended_up"
    elif extreme and z20 < 0:
        direction = "overextended_down"
    else:
        direction = "neutral"
    return MeanReversionSignal(symbol, z20, z50, zid, extreme, direction)


# ---------------------------------------------------------------------------
# Cross-asset correlation & divergence detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DivergenceFlag:
    pair: tuple[str, str]
    baseline_corr: float
    recent_corr: float
    delta: float
    volume_surge: bool
    note: str


def correlation_matrix(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return returns.tail(window).corr()


def detect_divergences(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None,
                       baseline_window: int = 120, recent_window: int = 20,
                       threshold: float = 0.45) -> list[DivergenceFlag]:
    """Flag pairs whose recent correlation has broken away from its baseline.
    Classic example: BTC decoupling from SPX on a volume surge, or US10Y
    spiking while DXY drops (regime-inconsistent — often forced flows)."""
    rets = prices.pct_change().dropna()
    if len(rets) < baseline_window:
        baseline_window = max(recent_window * 2, len(rets) // 2)

    base = rets.tail(baseline_window).corr()
    recent = rets.tail(recent_window).corr()
    flags: list[DivergenceFlag] = []
    cols = list(rets.columns)

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            delta = float(recent.loc[a, b] - base.loc[a, b])
            if abs(delta) >= threshold:
                surge = False
                if volumes is not None:
                    v = volumes[[c for c in (a, b) if c in volumes]].tail(5).mean()
                    v_base = volumes[[c for c in (a, b) if c in volumes]].tail(60).mean()
                    surge = bool((v / v_base.replace(0, np.nan) > 1.5).any())
                flags.append(DivergenceFlag(
                    pair=(a, b),
                    baseline_corr=float(base.loc[a, b]),
                    recent_corr=float(recent.loc[a, b]),
                    delta=delta,
                    volume_surge=surge,
                    note=("Correlation breakdown" + (" with volume surge — "
                          "likely real flow, not noise" if surge else "")),
                ))
    return sorted(flags, key=lambda f: -abs(f.delta))


# ---------------------------------------------------------------------------
# Volatility structural breaks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolBreak:
    symbol: str
    vol_recent: float      # annualized
    vol_baseline: float
    ratio: float
    structural_break: bool


def volatility_break(symbol: str, closes: pd.Series,
                     recent: int = 10, baseline: int = 60,
                     ratio_threshold: float = 2.0) -> VolBreak:
    rets = closes.pct_change().dropna()
    ann = np.sqrt(252.0)
    v_r = float(rets.tail(recent).std(ddof=1) * ann)
    v_b = float(rets.tail(baseline).std(ddof=1) * ann)
    ratio = v_r / v_b if v_b > 0 else float("nan")
    return VolBreak(symbol, v_r, v_b, ratio,
                    structural_break=bool(ratio >= ratio_threshold))


# ---------------------------------------------------------------------------
# News → volume latency (momentum arbitrage metric)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsLatencyResult:
    headline_ts: pd.Timestamp
    first_response_ts: Optional[pd.Timestamp]
    latency_seconds: Optional[float]
    priced_in: bool
    note: str


def news_volume_latency(headline_ts: pd.Timestamp,
                        intraday: pd.DataFrame,
                        vol_z_threshold: float = 2.0,
                        priced_in_after: float = 300.0) -> NewsLatencyResult:
    """Measure seconds between a headline and the first abnormal volume bar.
    If the surge happened at/before the headline (or never), assume priced in
    or ignored; small positive latency = residual momentum may remain."""
    vol_z = rolling_zscore(intraday["volume"].astype(float), 20)
    after = vol_z[vol_z.index >= headline_ts]
    spikes = after[after >= vol_z_threshold]
    if spikes.empty:
        return NewsLatencyResult(headline_ts, None, None, True,
                                 "No abnormal volume after headline — inert or pre-priced.")
    first = spikes.index[0]
    latency = float((first - headline_ts).total_seconds())
    priced_in = latency > priced_in_after
    note = ("Fast reaction; if trend persists, residual momentum possible."
            if not priced_in else
            "Slow/late surge — informational edge likely gone.")
    return NewsLatencyResult(headline_ts, first, latency, priced_in, note)


# ---------------------------------------------------------------------------
# Position sizing: fractional Kelly + historical VaR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizingRecommendation:
    kelly_full: float
    kelly_fraction_used: float   # we cap at quarter-Kelly by policy
    var_95_pct: float            # 1-day 95% historical VaR of the asset (fraction)
    max_position_pct: float      # of portfolio, after both constraints
    rationale: list[str] = field(default_factory=list)


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """f* = p - (1-p)/R. Negative => no edge => size zero."""
    if win_loss_ratio <= 0:
        return 0.0
    f = win_prob - (1.0 - win_prob) / win_loss_ratio
    return max(f, 0.0)


def historical_var(closes: pd.Series, confidence: float = 0.95) -> float:
    rets = closes.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float(-np.percentile(rets, (1.0 - confidence) * 100.0))


def position_size(win_prob: float, win_loss_ratio: float, closes: pd.Series,
                  portfolio_var_budget: float = 0.02,
                  kelly_cap: float = 0.25) -> SizingRecommendation:
    """Final size = min(quarter-Kelly, VaR-budget / asset-VaR)."""
    f_full = kelly_fraction(win_prob, win_loss_ratio)
    f_used = f_full * kelly_cap
    asset_var = historical_var(closes)
    var_constrained = (portfolio_var_budget / asset_var) if asset_var > 0 else 0.0
    final = round(min(f_used, var_constrained), 4)
    return SizingRecommendation(
        kelly_full=round(f_full, 4),
        kelly_fraction_used=round(f_used, 4),
        var_95_pct=round(asset_var, 4),
        max_position_pct=final,
        rationale=[
            f"Full Kelly {f_full:.2%} capped to quarter-Kelly {f_used:.2%} (drawdown control).",
            f"Asset 1-day 95% VaR {asset_var:.2%}; portfolio VaR budget "
            f"{portfolio_var_budget:.2%} caps size at {var_constrained:.2%}.",
            f"Binding constraint: {'VaR budget' if var_constrained < f_used else 'quarter-Kelly'}.",
        ],
    )
