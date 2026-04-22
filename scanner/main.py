"""
Entry point — orchestrates the full premarket scan pipeline.

Run modes:
  python scanner/main.py                   full morning scan
  python scanner/main.py --ticker NVDA     single-ticker (skips screener)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scanner")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    import re
    with open(path) as f:
        raw = f.read()
    # Replace ${VAR_NAME} placeholders with environment variable values
    raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Technical indicator helpers (used for open-trade analysis)
# ---------------------------------------------------------------------------

def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    rsi  = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _slope_pct(series: pd.Series, lookback: int = 5) -> float | None:
    if len(series) < lookback:
        return None
    return float((series.iloc[-1] - series.iloc[-lookback]) / series.iloc[-lookback] * 100)


def _tp_stop(best_zone_price: float) -> tuple[float, float, float]:
    """Derive TP1, TP2, and stop from zone centre price."""
    stop = round(best_zone_price * 0.988, 2)
    tp1  = round(best_zone_price * 1.018, 2)
    tp2  = round(best_zone_price * 1.035, 2)
    return tp1, tp2, stop


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(ticker: str | None = None) -> None:
    config = load_config()
    log.info("=== Premarket scanner starting. Mode: %s ===",
             f"single:{ticker}" if ticker else "full scan")

    from scanner import (
        screener,
        support_resistance as sr,
        fundamentals,
        backtest as bt_module,
        chart as chart_module,
        sheets,
        analysis as analysis_module,
        telegram_bot,
    )

    # ── 1. Screener ──────────────────────────────────────────────────────────
    if ticker:
        tickers = [ticker.upper()]
    else:
        log.info("Step 1/7 — Screener")
        tickers = screener.scan(config)
        log.info("  %d tickers passed volume/price filter: %s", len(tickers), tickers)

    proximity_pct  = config["support_resistance"]["proximity_threshold"]
    min_fa_score   = config["fundamentals"]["min_score"]
    min_win_rate   = config["backtest"]["min_win_rate"]

    # ── 2–5. S/R → Fundamentals → Backtest filter ────────────────────────────
    log.info("Step 2-5/7 — S/R detection, fundamentals, backtest")
    final: list[dict] = []

    for t in tickers:
        try:
            # S/R detection
            zones = sr.get_levels(t, config)
            price = float(yf.Ticker(t).fast_info["last_price"])
            nearby = sr.near_level(price, zones, proximity_pct)
            if not nearby:
                log.info("  %-6s skipped — not near any major S/R level", t)
                continue
            log.info("  %-6s @ $%.2f — %d zone(s) nearby", t, price, len(nearby))

            # Fundamentals
            fa = fundamentals.score(t)
            log.info("  %-6s fundamental score: %.0f/100", t, fa)
            if fa < min_fa_score:
                log.info("  %-6s skipped — FA score below threshold (%.0f)", t, min_fa_score)
                continue

            # Backtest
            stats = bt_module.run_backtest(t, nearby, config)
            log.info("  %-6s backtest: win rate %.0f%% (%d trades)",
                     t, stats["win_rate"] * 100, stats["num_trades"])
            if stats["win_rate"] < min_win_rate:
                log.info("  %-6s skipped — win rate below threshold (%.0f%%)",
                         t, min_win_rate * 100)
                continue

            # Compute TP/stop from best zone
            best = max(nearby, key=lambda z: z.score)
            tp1, tp2, stop = _tp_stop(best.price)

            # Volume ratio (relative volume vs 30d avg)
            hist = yf.Ticker(t).history(period="35d")
            avg_vol = float(hist["Volume"].iloc[:-1].mean()) if len(hist) > 1 else 0
            today_vol = float(hist["Volume"].iloc[-1]) if len(hist) > 0 else 0
            vol_ratio = round(today_vol / avg_vol, 2) if avg_vol > 0 else None

            min_vol_ratio = config["scan"].get("min_vol_ratio", 1.5)
            if vol_ratio is not None and vol_ratio < min_vol_ratio:
                log.info("  %-6s skipped — vol ratio %.1fx below threshold (%.1fx)",
                         t, vol_ratio, min_vol_ratio)
                continue

            final.append({
                "ticker":            t,
                "price":             price,
                "zones":             nearby,
                "fundamental_score": fa,
                "backtest":          stats,
                "tp1":               tp1,
                "tp2":               tp2,
                "stop":              stop,
                "volume_ratio":      vol_ratio,
            })

        except Exception:
            log.warning("  %-6s ERROR during scan pipeline:\n%s", t, traceback.format_exc())

    log.info("Step 2-5 complete — %d setup(s) passed all filters", len(final))

    # ── 6. Chart generation ──────────────────────────────────────────────────
    log.info("Step 6/7 — Chart generation")
    chart_urls: dict[str, str] = {}
    for c in final:
        t = c["ticker"]
        try:
            path = chart_module.generate(t, c["zones"], c["backtest"])
            chart_urls[t] = chart_module.pages_url(path)
            log.info("  %-6s chart → %s", t, chart_urls[t])
        except Exception:
            log.warning("  %-6s chart generation failed:\n%s", t, traceback.format_exc())
            chart_urls[t] = ""

    # ── 7a. Open trade analysis ───────────────────────────────────────────────
    log.info("Step 7/7 — Open trade analysis + alerts")
    trade_updates: list[dict] = []
    try:
        open_trades = sheets.get_open_trades(config)
        log.info("  %d open trade(s): %s", len(open_trades),
                 [t["ticker"] for t in open_trades])
    except Exception:
        log.warning("  Could not fetch Google Sheet — skipping trade updates:\n%s",
                    traceback.format_exc())
        open_trades = []

    for trade in open_trades:
        t = trade["ticker"]
        try:
            ticker_obj    = yf.Ticker(t)
            current_price = float(ticker_obj.fast_info["last_price"])
            trade["current_price"] = current_price
            trade["pnl_pct"]       = (current_price - trade["entry_price"]) / trade["entry_price"]

            hist  = ticker_obj.history(period="60d")
            close = hist["Close"]
            ema20 = _ema(close, 20)
            ema50 = _ema(close, 50)

            indicators = {
                "rsi":          _rsi(close),
                "ema20_slope":  _slope_pct(ema20),
                "ema50_slope":  _slope_pct(ema50),
                "volume_trend": (
                    "rising"
                    if len(hist) >= 10 and
                       hist["Volume"].iloc[-3:].mean() > hist["Volume"].iloc[-10:-3].mean()
                    else "falling"
                ),
            }

            trade_zones = sr.get_levels(t, config)
            nearby      = sr.near_level(current_price, trade_zones, proximity_pct)

            summary = analysis_module.summarise(trade, nearby, indicators, config)
            log.info("  %-6s analysis: %s", t, summary[:80].replace("\n", " "))

            best = max(nearby, key=lambda z: z.score) if nearby else None
            tp1, tp2, stop = _tp_stop(best.price) if best else (None, None, None)

            chart_path = chart_module.generate(
                t, nearby, {},
                trade={
                    "entry_price": trade["entry_price"],
                    "pnl_pct":     trade["pnl_pct"],
                },
            )
            trade_updates.append({
                **trade,
                "summary":   summary,
                "chart_url": chart_module.pages_url(chart_path),
                "tp1": tp1, "tp2": tp2, "stop": stop,
            })

        except Exception:
            log.warning("  %-6s trade analysis failed:\n%s", t, traceback.format_exc())

    # ── 7b. Telegram alerts ───────────────────────────────────────────────────
    for c in final:
        t = c["ticker"]
        try:
            telegram_bot.send_new_setup(
                ticker=t,
                zones=c["zones"],
                stats={
                    **c["backtest"],
                    "fundamental_score": c["fundamental_score"],
                    "current_price":     c["price"],
                    "volume_ratio":      c["volume_ratio"],
                    "tp1": c["tp1"], "tp2": c["tp2"], "stop": c["stop"],
                },
                chart_url=chart_urls.get(t, ""),
            )
        except Exception:
            log.warning("  %-6s Telegram send failed:\n%s", t, traceback.format_exc())

    for tu in trade_updates:
        try:
            telegram_bot.send_trade_update(tu, tu["summary"], tu["chart_url"])
        except Exception:
            log.warning("  %s trade update Telegram send failed:\n%s",
                        tu["ticker"], traceback.format_exc())

    if not final and not trade_updates and config["alerts"].get("send_if_no_setups"):
        telegram_bot.send_no_setups()

    log.info("=== Run complete. %d setup(s), %d trade update(s) ===",
             len(final), len(trade_updates))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Premarket scanner bot")
    parser.add_argument(
        "--ticker",
        default=None,
        metavar="TICKER",
        help="Run for a single ticker only (skips screener)",
    )
    args = parser.parse_args()
    run(ticker=args.ticker)
