"""
Fundamental analysis scorer — returns a 0–100 composite score per ticker.

Methodology mirrors faizancodes/Automated-Fundamental-Analysis:
four categories (Valuation, Profitability, Growth, Performance), each scored
0–100, averaged into a final composite.  Data sourced from yfinance .info.
"""
from __future__ import annotations

import yfinance as yf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(ticker: str) -> float:
    """
    Return a 0–100 fundamental score for ticker.

    Returns 0.0 if insufficient data is available rather than raising.
    """
    info = _fetch_info(ticker)
    if not info:
        return 0.0

    categories = [
        _valuation_score(info),
        _profitability_score(info),
        _growth_score(info),
        _performance_score(info),
    ]

    valid = [s for s in categories if s is not None]
    if not valid:
        return 0.0

    return round(sum(valid) / len(valid), 1)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_info(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # yfinance returns {'trailingPegRatio': None} for bad tickers
        if not info or info.get("quoteType") not in ("EQUITY", "ETF", None):
            return None
        return info
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Category scorers
# Each returns 0–100 or None if data unavailable.
# ---------------------------------------------------------------------------

def _valuation_score(info: dict) -> float | None:
    """Lower P/E, P/B, P/S = better value."""
    scores: list[float] = []

    # Trailing P/E: < 10 = 100, 10–20 = 75, 20–30 = 50, 30–50 = 25, > 50 = 0
    pe = info.get("trailingPE")
    if pe and pe > 0:
        scores.append(_score_lower(pe, [(10, 100), (20, 75), (30, 50), (50, 25)], 0))

    # Forward P/E: same scale, slightly more generous
    fpe = info.get("forwardPE")
    if fpe and fpe > 0:
        scores.append(_score_lower(fpe, [(10, 100), (18, 75), (25, 50), (40, 25)], 0))

    # P/B: < 1 = 100, 1–2 = 80, 2–4 = 60, 4–8 = 30, > 8 = 0
    pb = info.get("priceToBook")
    if pb and pb > 0:
        scores.append(_score_lower(pb, [(1, 100), (2, 80), (4, 60), (8, 30)], 0))

    # P/S: < 1 = 100, 1–3 = 75, 3–6 = 50, 6–12 = 25, > 12 = 0
    ps = info.get("priceToSalesTrailing12Months")
    if ps and ps > 0:
        scores.append(_score_lower(ps, [(1, 100), (3, 75), (6, 50), (12, 25)], 0))

    return _avg(scores)


def _profitability_score(info: dict) -> float | None:
    """Higher ROE, ROA, margins = better profitability."""
    scores: list[float] = []

    # ROE: > 20% = 100, 10–20% = 75, 5–10% = 50, 0–5% = 25, < 0 = 0
    roe = info.get("returnOnEquity")
    if roe is not None:
        scores.append(_score_higher(roe * 100, [(20, 100), (10, 75), (5, 50), (0, 25)], 0))

    # ROA: > 10% = 100, 5–10% = 75, 2–5% = 50, 0–2% = 25, < 0 = 0
    roa = info.get("returnOnAssets")
    if roa is not None:
        scores.append(_score_higher(roa * 100, [(10, 100), (5, 75), (2, 50), (0, 25)], 0))

    # Operating margin: > 20% = 100, 10–20% = 75, 5–10% = 50, 0–5% = 25, < 0 = 0
    op_margin = info.get("operatingMargins")
    if op_margin is not None:
        scores.append(_score_higher(op_margin * 100, [(20, 100), (10, 75), (5, 50), (0, 25)], 0))

    # Net margin: > 15% = 100, 8–15% = 75, 3–8% = 50, 0–3% = 25, < 0 = 0
    net_margin = info.get("profitMargins")
    if net_margin is not None:
        scores.append(_score_higher(net_margin * 100, [(15, 100), (8, 75), (3, 50), (0, 25)], 0))

    return _avg(scores)


def _growth_score(info: dict) -> float | None:
    """Higher revenue and earnings growth = better."""
    scores: list[float] = []

    # Revenue growth (YoY): > 20% = 100, 10–20% = 75, 5–10% = 50, 0–5% = 25, < 0 = 0
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        scores.append(_score_higher(rev_growth * 100, [(20, 100), (10, 75), (5, 50), (0, 25)], 0))

    # Earnings growth (YoY)
    earn_growth = info.get("earningsGrowth")
    if earn_growth is not None:
        scores.append(_score_higher(earn_growth * 100, [(25, 100), (15, 75), (5, 50), (0, 25)], 0))

    # EPS forward vs trailing — positive surprise direction
    eps_trailing = info.get("trailingEps")
    eps_forward = info.get("forwardEps")
    if eps_trailing and eps_forward and eps_trailing > 0:
        implied_growth = (eps_forward - eps_trailing) / eps_trailing * 100
        scores.append(_score_higher(implied_growth, [(20, 100), (10, 75), (0, 50), (-10, 25)], 0))

    return _avg(scores)


def _performance_score(info: dict) -> float | None:
    """Balance sheet health: low debt, good current ratio, positive cash flow."""
    scores: list[float] = []

    # Debt/Equity: < 0.3 = 100, 0.3–0.6 = 75, 0.6–1.0 = 50, 1.0–2.0 = 25, > 2.0 = 0
    de = info.get("debtToEquity")
    if de is not None and de >= 0:
        scores.append(_score_lower(de / 100, [(0.3, 100), (0.6, 75), (1.0, 50), (2.0, 25)], 0))

    # Current ratio: > 2 = 100, 1.5–2 = 75, 1–1.5 = 50, 0.8–1 = 25, < 0.8 = 0
    cr = info.get("currentRatio")
    if cr is not None:
        scores.append(_score_higher(cr, [(2.0, 100), (1.5, 75), (1.0, 50), (0.8, 25)], 0))

    # Free cash flow yield — positive FCF is enough, magnitude matters more than sign
    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap")
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100
        scores.append(_score_higher(fcf_yield, [(5, 100), (2, 75), (0, 50), (-2, 25)], 0))

    return _avg(scores)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_lower(value: float, thresholds: list[tuple[float, float]], floor: float) -> float:
    """Map value to score where lower is better. Thresholds: [(cutoff, score), ...]."""
    for cutoff, s in thresholds:
        if value <= cutoff:
            return s
    return floor


def _score_higher(value: float, thresholds: list[tuple[float, float]], floor: float) -> float:
    """Map value to score where higher is better. Thresholds: [(cutoff, score), ...]."""
    for cutoff, s in sorted(thresholds, key=lambda x: -x[0]):
        if value >= cutoff:
            return s
    return floor


def _avg(scores: list[float]) -> float | None:
    return round(sum(scores) / len(scores), 1) if scores else None
