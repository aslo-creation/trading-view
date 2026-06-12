"""
services/demo_data.py
=====================
Seeded synthetic OHLCV + FRED-like series so the full committee pipeline and
UI can be exercised offline / in CI before live vendors are wired in.
Replace the calls in app.py with your real data layer when keys are live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)

_BASE_PRICE = {"GOLD": 2400.0, "WTI": 78.0, "SPX": 5500.0, "BTC": 68000.0}


def _synthetic_ohlcv(start_price: float, periods: int = 180) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=periods, freq="D")
    rets = _RNG.normal(0.0003, 0.015, periods)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(_RNG.normal(0, 0.006, periods)))
    low = close * (1 - np.abs(_RNG.normal(0, 0.006, periods)))
    open_ = np.concatenate([[start_price], close[:-1]])
    volume = _RNG.lognormal(mean=10, sigma=0.4, size=periods)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def load_demo_ohlcv(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s: _synthetic_ohlcv(_BASE_PRICE.get(s, 100.0)) for s in symbols}


def load_demo_fred() -> dict[str, pd.Series]:
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=120, freq="D")
    return {
        "DGS10": pd.Series(4.0 + np.cumsum(_RNG.normal(0, 0.02, 120)), index=idx),
        "DTWEXBGS": pd.Series(120 + np.cumsum(_RNG.normal(0, 0.1, 120)), index=idx),
        "T10YIE": pd.Series(2.3 + np.cumsum(_RNG.normal(0, 0.01, 120)), index=idx),
        "NFCI": pd.Series(_RNG.normal(-0.3, 0.1, 120), index=idx),
        "DGS2": pd.Series(3.9 + np.cumsum(_RNG.normal(0, 0.02, 120)), index=idx),
        "DFF": pd.Series(4.33 + _RNG.normal(0, 0.01, 120), index=idx),
        "CPIAUCSL": pd.Series(310 * np.exp(np.cumsum(_RNG.normal(0.0002, 0.0003, 120))), index=idx),
    }
