"""
End-to-end smoke test — Step 12 of the build order.

Usage:
    python test_e2e.py              # test all stages; skip credential-gated ones
    python test_e2e.py --ticker MSFT
    python test_e2e.py --open-browser   # open generated chart in browser when done

Credential-gated stages (Telegram, Google Sheets, Claude API, GitHub dispatch)
are skipped automatically if the required env vars are absent.  Set them to run
the full pipeline end-to-end.

Exit code:
    0  all attempted stages passed
    1  one or more stages failed
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import webbrowser
from pathlib import Path

# ── Minimal config so we don't need GitHub Actions env ───────────────────────
os.environ.setdefault("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")

TICKER = "AAPL"   # overridden by --ticker flag

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []   # (stage, status, detail)


def stage(name: str):
    """Decorator-style context wrapper for a test stage."""
    class _Stage:
        def __enter__(self):
            print(f"\n{'-'*60}")
            print(f"  {name}")
            print(f"{'-'*60}")
            return self

        def __exit__(self, exc_type, exc_val, tb):
            if exc_type is None:
                return False
            results.append((name, FAIL, traceback.format_exc(limit=3)))
            print(f"  {FAIL} FAILED: {exc_val}")
            return True   # suppress — continue to next stage

    return _Stage()


def ok(name: str, detail: str = "") -> None:
    results.append((name, PASS, detail))
    print(f"  {PASS} {detail}" if detail else f"  {PASS} OK")


def skip(name: str, reason: str) -> None:
    results.append((name, SKIP, reason))
    print(f"  {SKIP} SKIPPED — {reason}")


def need_env(*vars_) -> bool:
    missing = [v for v in vars_ if not os.environ.get(v)]
    return len(missing) == 0


# ─────────────────────────────────────────────────────────────────────────────

def main(ticker: str, open_browser: bool) -> int:
    global TICKER
    TICKER = ticker.upper()
    print(f"\n{'='*60}")
    print(f"  Premarket Scanner -- End-to-End Smoke Test")
    print(f"  Ticker: {TICKER}")
    print(f"{'='*60}")

    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # ── Stage 1: Config load ──────────────────────────────────────────────────
    name = "1. Config load"
    with stage(name):
        assert "scan" in config
        assert "support_resistance" in config
        ok(name, f"sections: {list(config.keys())}")

    # ── Stage 2: yfinance price fetch ─────────────────────────────────────────
    name = "2. yfinance price fetch"
    price: float = 0.0
    with stage(name):
        import yfinance as yf
        info = yf.Ticker(TICKER).fast_info
        price = float(info["last_price"])
        assert price > 0, f"Got price={price}"
        ok(name, f"{TICKER} @ ${price:.2f}")

    # ── Stage 3: S/R detection ────────────────────────────────────────────────
    name = "3. S/R detection"
    zones = []
    with stage(name):
        from scanner.support_resistance import get_levels, near_level
        t0 = time.time()
        zones = get_levels(TICKER, config)
        elapsed = time.time() - t0
        nearby = near_level(price, zones, config["support_resistance"]["proximity_threshold"])
        ok(name, f"{len(zones)} total zones, {len(nearby)} within proximity  [{elapsed:.1f}s]")
        for z in zones[:5]:
            print(f"     ${z.price:.2f}  score={z.score}  touches={z.touch_count}"
                  f"  tf={z.timeframe}  flipped={z.flipped}")

    # ── Stage 4: Fundamental scoring ──────────────────────────────────────────
    name = "4. Fundamental scoring"
    fa_score: float = 0.0
    with stage(name):
        from scanner.fundamentals import score
        t0 = time.time()
        fa_score = score(TICKER)
        elapsed = time.time() - t0
        assert 0 <= fa_score <= 100, f"Score out of range: {fa_score}"
        ok(name, f"{TICKER} FA score: {fa_score}/100  [{elapsed:.1f}s]")

    # ── Stage 5: Backtest ─────────────────────────────────────────────────────
    name = "5. Backtest"
    bt_stats: dict = {}
    with stage(name):
        from scanner.backtest import run_backtest
        t0 = time.time()
        bt_stats = run_backtest(TICKER, zones, config)
        elapsed = time.time() - t0
        ok(name, (
            f"win rate={bt_stats['win_rate']:.1%}  "
            f"return={bt_stats['total_return']:+.1%}  "
            f"sharpe={bt_stats['sharpe']:.2f}  "
            f"trades={bt_stats['num_trades']}  [{elapsed:.1f}s]"
        ))

    # ── Stage 6: Chart generation ─────────────────────────────────────────────
    name = "6. Chart generation"
    chart_path: Path | None = None
    with stage(name):
        from scanner.chart import generate, pages_url
        from scanner.support_resistance import near_level
        nearby = near_level(price, zones, config["support_resistance"]["proximity_threshold"])
        t0 = time.time()
        chart_path = generate(TICKER, nearby or zones[:3], bt_stats)
        elapsed = time.time() - t0
        assert chart_path.exists(), f"Chart file not written: {chart_path}"
        size_kb = chart_path.stat().st_size // 1024
        url = pages_url(chart_path)
        ok(name, f"{chart_path}  ({size_kb} KB)  [{elapsed:.1f}s]\n     URL: {url}")

    # ── Stage 7: Claude API analysis ──────────────────────────────────────────
    name = "7. Claude API analysis"
    if need_env("ANTHROPIC_API_KEY"):
        with stage(name):
            from scanner.analysis import summarise
            import yfinance as yf
            import numpy as np
            import pandas as pd

            hist  = yf.Ticker(TICKER).history(period="60d")
            close = hist["Close"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            def _slope(s):
                return float((s.iloc[-1] - s.iloc[-5]) / s.iloc[-5] * 100) if len(s) >= 5 else None

            trade = {
                "ticker":        TICKER,
                "entry_price":   price * 0.97,
                "shares":        10,
                "date_entered":  "2026-04-01",
                "current_price": price,
                "pnl_pct":       0.03,
                "days_held":     5,
            }
            from scanner.support_resistance import near_level
            nearby = near_level(price, zones, config["support_resistance"]["proximity_threshold"])
            indicators = {
                "rsi":          None,
                "ema20_slope":  _slope(ema20),
                "ema50_slope":  _slope(ema50),
                "volume_trend": "rising",
            }
            summary = summarise(trade, nearby, indicators, config)
            assert len(summary) > 20
            ok(name, f"\n     {summary}")
    else:
        skip(name, "ANTHROPIC_API_KEY not set")

    # ── Stage 8: Google Sheets (read) ─────────────────────────────────────────
    name = "8. Google Sheets read"
    if need_env("GOOGLE_SERVICE_ACCOUNT_JSON") and \
       os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] != "{}":
        with stage(name):
            from scanner.sheets import get_open_trades
            trades = get_open_trades(config)
            ok(name, f"{len(trades)} open trade(s) found")
    else:
        skip(name, "GOOGLE_SERVICE_ACCOUNT_JSON not set")

    # ── Stage 9: Telegram send ────────────────────────────────────────────────
    name = "9. Telegram alert send"
    if need_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        with stage(name):
            from scanner.telegram_bot import send_new_setup
            from scanner.support_resistance import near_level
            nearby = near_level(price, zones, config["support_resistance"]["proximity_threshold"])
            tp1 = round(price * 1.018, 2)
            tp2 = round(price * 1.035, 2)
            stop = round(price * 0.988, 2)
            send_new_setup(
                ticker=TICKER,
                zones=nearby or zones[:2],
                stats={
                    **bt_stats,
                    "fundamental_score": fa_score,
                    "current_price":     price,
                    "volume_ratio":      2.1,
                    "tp1": tp1, "tp2": tp2, "stop": stop,
                },
                chart_url=pages_url(chart_path) if chart_path else "https://example.com",
            )
            ok(name, "Message sent — check Telegram")
    else:
        skip(name, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    failed = 0
    for s_name, status, detail in results:
        icon = status
        print(f"  {icon}  {s_name}")
        if status == FAIL and detail:
            for line in detail.strip().splitlines()[-4:]:
                print(f"       {line}")
        failed += status == FAIL

    print(f"\n  {len(results) - failed} passed  |  {failed} failed  |  "
          f"{sum(1 for _, s, _ in results if s == SKIP)} skipped")

    if chart_path and chart_path.exists() and open_browser:
        print(f"\n  Opening chart in browser: {chart_path.resolve()}")
        webbrowser.open(chart_path.resolve().as_uri())

    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E smoke test")
    parser.add_argument("--ticker",       default="AAPL", metavar="TICKER")
    parser.add_argument("--open-browser", action="store_true",
                        help="Open generated chart in default browser")
    args = parser.parse_args()
    sys.exit(main(args.ticker, args.open_browser))
