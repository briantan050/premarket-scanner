"""tvscreener volume scan — returns list of tickers above volume threshold."""
from __future__ import annotations

import logging

from tvscreener import StockScreener, StockField, Market

log = logging.getLogger(__name__)


def scan(config: dict) -> list[str]:
    """
    Query TradingView screener for US stocks meeting volume and price filters.

    Returns up to config['scan']['max_results'] ticker symbols sorted by
    relative volume descending.

    tvscreener 0.3.0 returns a DataFrame with columns:
      Symbol | Price | Volume | Relative Volume | Average Volume (30 day) | ...
    Symbol values are formatted as "EXCHANGE:TICKER" (e.g. "NYSE:BML/PG").
    Preferred share slashes are converted to dashes for Yahoo Finance compatibility.
    """
    cfg = config["scan"]
    volume_multiplier: float = cfg["volume_multiplier"]
    min_price: float = cfg["min_price"]
    max_results: int = cfg["max_results"]

    ss = StockScreener()
    ss.set_markets(Market.AMERICA)

    ss.where(StockField.PRICE >= min_price)
    ss.where(StockField.RELATIVE_VOLUME >= volume_multiplier)

    ss.select(
        StockField.PRICE,
        StockField.VOLUME,
        StockField.RELATIVE_VOLUME,
        StockField.AVERAGE_VOLUME_30_DAY,
    )

    df = ss.get()

    if df is None or df.empty:
        return []

    # Sort by relative volume descending
    rel_vol_col = _find_col(df, ["Relative Volume", "relative_volume_10d_calc", "RELATIVE_VOLUME"])
    if rel_vol_col:
        df = df.sort_values(rel_vol_col, ascending=False)

    df = df.head(max_results)

    # Extract tickers from the Symbol column ("EXCHANGE:TICKER" format)
    sym_col = _find_col(df, ["Symbol", "ticker", "name", "symbol", "Ticker", "Name"])
    if sym_col:
        raw = df[sym_col].tolist()
    else:
        raw = df.index.tolist()

    tickers = []
    for t in raw:
        t = str(t).split(":")[-1]   # strip exchange prefix
        t = t.replace("/", "-")     # preferred shares: BML/PG → BML-PG
        tickers.append(t)

    log.info("  %d tickers from screener: %s", len(tickers), tickers[:10])
    return tickers


def _find_col(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
