"""
Generate self-contained interactive HTML charts using lightweight-charts v5.

Each chart is a single .html file — no backend required once hosted on
GitHub Pages.  All data is embedded as JSON; rendering is 100% client-side.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from scanner.support_resistance import SRZone

CHARTS_DIR = Path("charts")
TEMPLATE_PATH = Path("templates/chart_template.html")

# TP/stop expressed as multiples of the zone's implied risk
_STOP_PCT = 0.012    # 1.2 % below zone centre
_TP1_PCT  = 0.018    # 1.8 % above zone centre  (~1.5× R)
_TP2_PCT  = 0.035    # 3.5 % above zone centre  (~3× R)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    ticker: str,
    zones: list[SRZone],
    stats: dict,
    trade: dict | None = None,
    config: dict | None = None,
) -> Path:
    """
    Build and write charts/{TICKER}-{DATE}.html.

    trade (optional): {ticker, entry_price, shares, date_entered, pnl_pct}
    config (optional): full config dict; used for chart display filters.
    Returns the Path of the written file.
    """
    CHARTS_DIR.mkdir(exist_ok=True)

    df = _fetch_2y(ticker)
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker}")

    current_price = float(df["Close"].iloc[-1])

    # Derive TP/stop from the nearest qualifying zone
    best = max(zones, key=lambda z: z.score) if zones else None
    tp1 = round(best.price * (1 + _TP1_PCT), 2) if best else None
    tp2 = round(best.price * (1 + _TP2_PCT), 2) if best else None
    stop = round(best.price * (1 - _STOP_PCT), 2) if best else None

    # Apply chart display filters: proximity band + top-N cap
    chart_zones = _filter_chart_zones(zones, current_price, config)

    payload = {
        "ticker": ticker,
        "date": date.today().isoformat(),
        "current_price": current_price,
        "ohlcv": _candle_data(df),
        "volume": _volume_data(df),
        "ema20": _line_data(df, _ema(df["Close"], 20)),
        "ema50": _line_data(df, _ema(df["Close"], 50)),
        "rsi": _line_data(df, _rsi(df["Close"], 14)),
        "zones": [
            {
                "price": z.price,
                "score": z.score,
                "touch_count": z.touch_count,
                "timeframe": z.timeframe,
                "flipped": z.flipped,
            }
            for z in chart_zones
        ],
        "stats": stats,
        "trade": trade,
        "tp1": tp1,
        "tp2": tp2,
        "stop": stop,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DATA_JSON__", json.dumps(payload, allow_nan=False))
    html = html.replace("__TICKER__", ticker)

    out_path = CHARTS_DIR / f"{ticker}-{payload['date']}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def pages_url(path: Path, repo: str | None = None) -> str:
    """
    Derive the GitHub Pages URL for a chart file.

    repo: e.g. 'username/premarket-scanner-bot'  (read from git remote if omitted)
    """
    if repo is None:
        try:
            remote = subprocess.check_output(
                ["git", "remote", "get-url", "origin"], text=True
            ).strip()
            # https://github.com/user/repo.git  or  git@github.com:user/repo.git
            repo = remote.rstrip(".git").split("github.com")[-1].lstrip(":/")
        except Exception:
            return str(path)

    parts = repo.split("/")
    user, repo_name = parts[-2], parts[-1]
    return f"https://{user}.github.io/{repo_name}/{path.name}"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_2y(ticker: str) -> pd.DataFrame | None:
    df = yf.download(
        ticker, period="2y", interval="1d",
        auto_adjust=True, progress=False,
    )
    df = _flatten(df)
    if df is None or df.empty:
        return None
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def _filter_chart_zones(
    zones: list[SRZone],
    current_price: float,
    config: dict | None,
) -> list[SRZone]:
    """
    Reduce the zone list to the most actionable levels for the chart:
      1. Keep only zones within chart_proximity_pct % of the current price.
      2. Cap to max_zones_on_chart, ranked by score (highest first).
    The full zone list (including distant monthly levels) is still used
    elsewhere (backtest, analysis, Telegram) — this only controls what lines
    are drawn on the chart.
    """
    proximity_pct = 6.0
    max_zones = 6
    if config:
        sr_cfg = config.get("support_resistance", {})
        proximity_pct = sr_cfg.get("chart_proximity_pct", proximity_pct)
        max_zones = sr_cfg.get("max_zones_on_chart", max_zones)

    nearby = [
        z for z in zones
        if abs(z.price - current_price) / current_price * 100 <= proximity_pct
    ]
    nearby.sort(key=lambda z: z.score, reverse=True)
    return nearby[:max_zones]


def _ts(idx: pd.DatetimeIndex) -> list[str]:
    """Convert DatetimeIndex to 'YYYY-MM-DD' strings for lightweight-charts."""
    return [d.strftime("%Y-%m-%d") for d in idx]


def _candle_data(df: pd.DataFrame) -> list[dict]:
    times = _ts(df.index)
    return [
        {
            "time": t,
            "open": round(float(o), 4),
            "high": round(float(h), 4),
            "low": round(float(lo), 4),
            "close": round(float(c), 4),
        }
        for t, o, h, lo, c in zip(
            times, df["Open"], df["High"], df["Low"], df["Close"]
        )
    ]


def _volume_data(df: pd.DataFrame) -> list[dict]:
    times = _ts(df.index)
    closes = df["Close"].values
    opens = df["Open"].values
    return [
        {
            "time": t,
            "value": int(v),
            "color": "#26a69a" if c >= o else "#ef5350",
        }
        for t, v, c, o in zip(times, df["Volume"], closes, opens)
    ]


def _line_data(df: pd.DataFrame, series: pd.Series) -> list[dict]:
    times = _ts(df.index)
    return [
        {"time": t, "value": round(float(v), 4)}
        for t, v in zip(times, series)
        if not (isinstance(v, float) and np.isnan(v))
    ]


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
