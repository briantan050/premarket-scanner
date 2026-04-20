"""
backtesting.py wrapper — runs the S/R bounce strategy and returns summary stats.

Strategy logic:
  - Long entry: a candle's low touches within the zone band and the candle
    closes above the zone centre (bullish bounce confirmation).
  - Stop loss: 1% below the zone centre.
  - Take profit: 2× the risk distance above entry (2:1 R/R).
  - Only one position open at a time; skip zone if already in a trade.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

from scanner.support_resistance import SRZone


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(ticker: str, zones: list[SRZone], config: dict) -> dict:
    """
    Download historical OHLCV, run the S/R bounce strategy, and return stats.

    Returns dict:
        win_rate   (float 0–1)
        total_return (float, e.g. 0.34 = 34%)
        sharpe     (float)
        num_trades (int)
        max_drawdown (float 0–1)

    Returns a zeroed dict if there is insufficient data or no trades fired.
    """
    bt_cfg = config["backtest"]
    lookback_years: int = bt_cfg["lookback_years"]
    commission: float = bt_cfg["commission"]

    data = _fetch_daily(ticker, lookback_years)
    if data is None or len(data) < 60:
        return _empty_stats()

    zone_prices = [z.price for z in zones]
    if not zone_prices:
        return _empty_stats()

    # Suppress backtesting.py plotting warnings in headless runs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = Backtest(
            data,
            SRBounceStrategy,
            cash=100_000,
            commission=commission,
            exclusive_orders=True,
        )
        stats = bt.run(zone_prices=zone_prices)

    return _parse_stats(stats)


# ---------------------------------------------------------------------------
# Strategy definition
# ---------------------------------------------------------------------------

class SRBounceStrategy(Strategy):
    # Injected by Backtest.run(zone_prices=[...])
    zone_prices: list[float] = []

    # How close price must get to the zone to trigger (% of zone price)
    touch_band_pct: float = 0.005   # 0.5 %
    # Stop loss below zone centre
    stop_pct: float = 0.01          # 1 %
    # Risk/reward ratio (TP = entry + rr * risk)
    rr: float = 2.0

    def init(self):
        # Pre-compute zone band tuples once
        self._bands: list[tuple[float, float]] = [
            (p * (1 - self.touch_band_pct), p * (1 + self.touch_band_pct))
            for p in self.zone_prices
        ]

    def next(self):
        if self.position:
            return  # manage via sl/tp; no manual exit needed

        low = self.data.Low[-1]
        close = self.data.Close[-1]

        for zone_price, (band_lo, band_hi) in zip(self.zone_prices, self._bands):
            # Low dipped into the zone band and close recovered above zone centre
            if low <= band_hi and close > zone_price:
                stop = zone_price * (1 - self.stop_pct)
                risk = close - stop
                if risk <= 0:
                    continue
                tp = close + self.rr * risk
                self.buy(sl=stop, tp=tp)
                break  # one trade at a time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_daily(ticker: str, lookback_years: int) -> pd.DataFrame | None:
    period = f"{lookback_years}y"
    df = yf.download(
        ticker, period=period, interval="1d",
        auto_adjust=True, progress=False,
    )
    df = _flatten(df)
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _parse_stats(stats: pd.Series) -> dict:
    def _get(key: str, default=0.0):
        val = stats.get(key, default)
        return default if val is None or (isinstance(val, float) and np.isnan(val)) else val

    return {
        "win_rate": round(_get("Win Rate [%]", 0.0) / 100, 4),
        "total_return": round(_get("Return [%]", 0.0) / 100, 4),
        "sharpe": round(float(_get("Sharpe Ratio", 0.0)), 3),
        "num_trades": int(_get("# Trades", 0)),
        "max_drawdown": round(abs(_get("Max. Drawdown [%]", 0.0)) / 100, 4),
    }


def _empty_stats() -> dict:
    return {"win_rate": 0.0, "total_return": 0.0, "sharpe": 0.0,
            "num_trades": 0, "max_drawdown": 0.0}
