"""
core/signal_engine.py
=====================
"Quand acheter / quand vendre" — composite heuristic score per asset that the
UI renders as a gauge and the committee uses as quantitative input.

Honest scope: this is a TRANSPARENT, EXPLAINABLE heuristic over verified
statistics — not a crystal ball. Every score ships with its reasons so a
beginner learns WHY, and an advanced user can audit the components.

Score in [-100, +100]:
    >= +45 Achat fort | >= +15 Achat | (-15,+15) Neutre | <= -15 Vente | <= -45 Vente forte

Components (weights sum to 100):
    Mean reversion (40): rolling Z-score of distance from 20d rolling VWAP,
                         contrarian (oversold -> buy).
    Trend (35):          MA20 vs MA50 spread, trend-following.
    RSI 14 (25):         contrarian at extremes (<30 buy, >70 sell).
Volatility regime: a structural vol break (10d/60d ratio >= 2) multiplies the
score by 0.7 — abnormal volatility makes ANY signal less reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.math_engine import rolling_zscore, volatility_break


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window, min_periods=5).sum()
    vv = df["volume"].rolling(window, min_periods=5).sum().replace(0, np.nan)
    out = pv / vv
    # Zero-volume instruments (some indices): fall back to a simple MA.
    return out.fillna(df["close"].rolling(window, min_periods=5).mean())


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


@dataclass(frozen=True)
class CompositeSignal:
    symbol: str
    score: float                      # -100 .. +100
    label: str
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    vol_warning: bool = False


def _label(score: float) -> str:
    if score >= 45:
        return "🟢 Achat fort"
    if score >= 15:
        return "🟢 Achat"
    if score <= -45:
        return "🔴 Vente forte"
    if score <= -15:
        return "🔴 Vente"
    return "⚪ Neutre / Attendre"


def composite_signal(symbol: str, df: pd.DataFrame) -> CompositeSignal:
    close = df["close"]
    reasons: list[str] = []

    # 1) Mean reversion vs rolling VWAP (contrarian) — weight 40
    dist = close - rolling_vwap(df, 20)
    z = rolling_zscore(dist, 20)
    z_now = float(z.iloc[-1]) if not np.isnan(z.iloc[-1]) else 0.0
    mr = float(np.clip(-z_now / 3.0, -1, 1) * 40)
    if z_now <= -2.0:
        reasons.append(f"Prix très étiré SOUS sa moyenne pondérée (Z={z_now:+.1f}) "
                       "→ rebond statistique possible.")
    elif z_now >= 2.0:
        reasons.append(f"Prix très étiré AU-DESSUS de sa moyenne pondérée "
                       f"(Z={z_now:+.1f}) → repli statistique possible.")
    else:
        reasons.append(f"Écart à la moyenne pondérée normal (Z={z_now:+.1f}).")

    # 2) Trend MA20 vs MA50 (trend-following) — weight 35
    ma20 = close.rolling(20, min_periods=10).mean().iloc[-1]
    ma50 = close.rolling(50, min_periods=25).mean().iloc[-1]
    spread_pct = float((ma20 / ma50 - 1.0) * 100) if pd.notna(ma50) and ma50 else 0.0
    tr = float(np.clip(spread_pct / 3.0, -1, 1) * 35)
    reasons.append(
        f"Tendance {'haussière' if spread_pct > 0 else 'baissière' if spread_pct < 0 else 'plate'} : "
        f"moyenne 20j {'au-dessus' if spread_pct >= 0 else 'en dessous'} de la 50j "
        f"({spread_pct:+.1f} %)."
    )

    # 3) RSI 14 (contrarian at extremes) — weight 25
    r_now = float(rsi(close).iloc[-1])
    mom = float(np.clip((50.0 - r_now) / 20.0, -1, 1) * 25)
    if r_now >= 70:
        reasons.append(f"RSI {r_now:.0f} : zone de surachat.")
    elif r_now <= 30:
        reasons.append(f"RSI {r_now:.0f} : zone de survente.")
    else:
        reasons.append(f"RSI {r_now:.0f} : zone neutre.")

    score = mr + tr + mom

    # Volatility regime damping
    vb = volatility_break(symbol, close)
    vol_warning = bool(vb.structural_break)
    if vol_warning:
        score *= 0.7
        reasons.append(f"⚠️ Volatilité anormale ({vb.ratio:.1f}× la normale) : "
                       "fiabilité du signal réduite, prudence sur la taille.")

    score = float(np.clip(score, -100, 100))
    return CompositeSignal(
        symbol=symbol,
        score=round(score, 1),
        label=_label(score),
        reasons=reasons,
        components={"mean_reversion": round(mr, 1), "trend": round(tr, 1),
                    "rsi": round(mom, 1)},
        vol_warning=vol_warning,
    )


def historical_extremes(df: pd.DataFrame, window: int = 20,
                        threshold: float = 2.5) -> pd.DataFrame:
    """Past statistical extremes for chart markers: where Z crossed ±threshold.
    Educational overlay — shows where the engine WOULD have flagged."""
    dist = df["close"] - rolling_vwap(df, window)
    z = rolling_zscore(dist, window)
    out = pd.DataFrame({"close": df["close"], "z": z})
    out["buy_flag"] = out["z"] <= -threshold
    out["sell_flag"] = out["z"] >= threshold
    return out
