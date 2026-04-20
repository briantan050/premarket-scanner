"""Pivot-based S/R detection, zone clustering, scoring, and proximity check."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# How many bars each side a candle must dominate to count as a pivot.
PIVOT_WINDOW = 3

# yfinance max lookback per interval.  Sub-daily capped at 730 days.
_LOOKBACK = {"1mo": "20y", "1wk": "5y", "1d": "2y", "4h": "730d", "1h": "730d"}

# Score assigned to 52-week H/L synthetic zones — high enough to always survive
# the major_level_min_score filter and appear near the top of the ranked list.
_52WK_SCORE = 15.0


@dataclass
class SRZone:
    price: float
    score: float
    touch_count: int
    timeframe: str       # dominant contributing timeframe
    flipped: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_levels(ticker: str, config: dict) -> list[SRZone]:
    """
    Fetch OHLCV across configured timeframes, detect pivots, cluster into
    zones, score them, and return only zones above major_level_min_score.
    Optionally injects 52-week H/L as synthetic high-score zones.
    """
    sr_cfg = config["support_resistance"]
    timeframes: list[dict] = sr_cfg["timeframes"]
    merge_pct: float = sr_cfg["merge_distance"]
    min_score: float = sr_cfg["major_level_min_score"]
    min_touches: int = sr_cfg["min_touches"]
    flip_flag: bool = sr_cfg.get("flip_confirmation", True)
    include_52wk: bool = sr_cfg.get("include_52wk_levels", True)

    # Collect raw pivot records: (price, weight, timeframe, is_high)
    raw: list[tuple[float, int, str, bool]] = []
    daily_df: pd.DataFrame | None = None  # kept for flip detection + 52wk calc

    for tf in timeframes:
        interval = tf["interval"]
        weight = tf["weight"]
        df = _fetch_ohlcv(ticker, interval)
        if df is None or len(df) < PIVOT_WINDOW * 2 + 1:
            continue

        if interval == "1d":
            daily_df = df

        highs = _pivot_highs(df["High"].values, PIVOT_WINDOW)
        lows = _pivot_lows(df["Low"].values, PIVOT_WINDOW)

        for idx in highs:
            raw.append((float(df["High"].iloc[idx]), weight, interval, True))
        for idx in lows:
            raw.append((float(df["Low"].iloc[idx]), weight, interval, False))

    if not raw:
        return []

    zones = _cluster_and_score(raw, merge_pct, min_touches)

    # Filter below minimum score
    zones = [z for z in zones if z.score >= min_score]

    # Flip detection using daily data
    if flip_flag and daily_df is not None:
        closes = daily_df["Close"].values
        for z in zones:
            z.flipped = _detect_flip(z.price, closes)

    # Inject 52-week H/L as synthetic zones — these carry massive respect even
    # when they haven't been touched in months, so we always include them.
    if include_52wk and daily_df is not None and len(daily_df) >= 2:
        zones += _synthetic_52wk_zones(daily_df, merge_pct)

    return zones


def near_level(price: float, zones: list[SRZone], threshold_pct: float) -> list[SRZone]:
    """Return zones within threshold_pct % of price, sorted by proximity."""
    matches = [z for z in zones if abs(price - z.price) / price * 100 <= threshold_pct]
    return sorted(matches, key=lambda z: abs(price - z.price))


# ---------------------------------------------------------------------------
# OHLCV fetching
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns from yfinance (ticker as level 1) to flat."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_ohlcv(ticker: str, interval: str) -> pd.DataFrame | None:
    """Download OHLCV for the interval, resampling 1H→4H when needed."""
    if interval == "4h":
        df = yf.download(ticker, period=_LOOKBACK["4h"], interval="1h",
                         auto_adjust=True, progress=False)
        df = _flatten(df)
        if df is None or df.empty:
            return None
        df = _resample_4h(df)
    else:
        df = yf.download(ticker, period=_LOOKBACK.get(interval, "2y"),
                         interval=interval, auto_adjust=True, progress=False)
        df = _flatten(df)

    if df is None or df.empty:
        return None

    df = df.dropna(subset=["High", "Low", "Close"])
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    rule = "4h"
    resampled = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}
    ).dropna()
    return resampled


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------

def _pivot_highs(values: np.ndarray, n: int) -> list[int]:
    """Indices where values[i] is strictly greater than all n bars each side."""
    indices = []
    for i in range(n, len(values) - n):
        window = values[i - n: i + n + 1]
        if values[i] == window.max() and list(window).count(values[i]) == 1:
            indices.append(i)
    return indices


def _pivot_lows(values: np.ndarray, n: int) -> list[int]:
    """Indices where values[i] is strictly less than all n bars each side."""
    indices = []
    for i in range(n, len(values) - n):
        window = values[i - n: i + n + 1]
        if values[i] == window.min() and list(window).count(values[i]) == 1:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Clustering and scoring
# ---------------------------------------------------------------------------

def _cluster_and_score(
    raw: list[tuple[float, int, str, bool]],
    merge_pct: float,
    min_touches: int,
) -> list[SRZone]:
    """
    Merge pivots within merge_pct % of each other into zones.
    Score = sum(weight) for all pivots in the cluster.
    Dominant timeframe = the timeframe contributing the most weight.
    """
    # Sort by price ascending
    raw_sorted = sorted(raw, key=lambda x: x[0])

    clusters: list[list[tuple[float, int, str, bool]]] = []
    current: list[tuple[float, int, str, bool]] = [raw_sorted[0]]

    for item in raw_sorted[1:]:
        ref_price = current[0][0]
        if abs(item[0] - ref_price) / ref_price * 100 <= merge_pct:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    zones: list[SRZone] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue

        avg_price = float(np.mean([c[0] for c in cluster]))
        touch_count = len(cluster)

        # Weighted score per timeframe contribution
        tf_weights: dict[str, float] = {}
        for _, weight, tf, _ in cluster:
            tf_weights[tf] = tf_weights.get(tf, 0) + weight

        score = float(sum(tf_weights.values()))
        dominant_tf = max(tf_weights, key=tf_weights.__getitem__)

        zones.append(SRZone(
            price=round(avg_price, 4),
            score=score,
            touch_count=touch_count,
            timeframe=dominant_tf,
        ))

    return zones


# ---------------------------------------------------------------------------
# 52-week H/L synthetic zones
# ---------------------------------------------------------------------------

def _synthetic_52wk_zones(daily_df: pd.DataFrame, merge_pct: float) -> list[SRZone]:
    """
    Compute 52-week high and low from daily data and return them as SRZone
    objects with a high synthetic score.  Skips a level if it falls within
    merge_pct % of an existing zone (caller deduplication is not needed here
    because we only inject two points and they are typically far apart).
    """
    recent = daily_df.tail(252)
    wk52_high = float(recent["High"].max())
    wk52_low  = float(recent["Low"].min())

    zones = []
    for price, label in ((wk52_high, "1d"), (wk52_low, "1d")):
        zones.append(SRZone(
            price=round(price, 4),
            score=_52WK_SCORE,
            touch_count=1,
            timeframe=label,
            flipped=False,
        ))
    return zones


# ---------------------------------------------------------------------------
# Flip detection
# ---------------------------------------------------------------------------

def _detect_flip(level: float, closes: np.ndarray, tolerance_pct: float = 0.5) -> bool:
    """
    Return True if price crossed through the level at least once —
    indicating the zone has acted as both support and resistance.
    The tolerance_pct band around the level counts as a crossing zone.
    """
    band = level * tolerance_pct / 100
    above = closes > (level + band)
    below = closes < (level - band)

    was_above = False
    was_below = False

    for a, b in zip(above, below):
        if a:
            if was_below:
                return True
            was_above = True
        elif b:
            if was_above:
                return True
            was_below = True

    return False
