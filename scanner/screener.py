"""tvscreener volume scan — returns list of tickers above volume threshold."""
from __future__ import annotations

from tvscreener import StockScreener, StockField, Market


def scan(config: dict) -> list[str]:
    """
    Query TradingView screener for US stocks meeting volume and price filters.

    Relative volume uses TradingView's 10-day calculation (closest available to
    the config's conceptual 20-day average threshold).

    Returns up to config['scan']['max_results'] ticker symbols, sorted by
    relative volume descending.
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

    # Debug: log actual structure so we can fix extraction if needed
    import logging
    log = logging.getLogger(__name__)
    log.info("Screener columns: %s", df.columns.tolist())
    log.info("Screener index sample: %s", df.index[:3].tolist())
    log.info("Screener first row: %s", df.iloc[0].to_dict())

    # Sort by relative volume descending and cap results
    rel_vol_col = _find_col(df, ["relative_volume_10d_calc", "Relative Volume", "RELATIVE_VOLUME"])
    if rel_vol_col:
        df = df.sort_values(rel_vol_col, ascending=False)

    df = df.head(max_results)

    # Extract ticker symbols — try all known locations
    for col in ("ticker", "name", "symbol", "Ticker", "Name", "Symbol"):
        if col in df.columns:
            tickers = df[col].tolist()
            return [str(t).split(":")[-1] for t in tickers]

    # Fall back to index — strip exchange prefix if present (e.g. "NASDAQ:AAPL")
    return [str(t).split(":")[-1] for t in df.index]


def _find_col(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
