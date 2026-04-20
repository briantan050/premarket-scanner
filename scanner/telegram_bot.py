"""
Telegram alert sender and button-callback webhook handler.

Two roles:
  1. Imported by main.py during a GitHub Actions run → send_new_setup() /
     send_trade_update() fire one-shot messages.
  2. Run directly on Render (python -m scanner.telegram_bot) → runs an
     Application in webhook mode to handle inline-keyboard button callbacks.

Button callback data format:
  refresh_all            → trigger full workflow_dispatch
  refresh_TICKER         → single-ticker workflow_dispatch
  backtest_TICKER        → run backtest, reply with stats
  view_trades            → read Google Sheet, reply with table
  close_TICKER           → mark ticker closed in Google Sheet
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import requests
import yaml
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _bot_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or _cfg()["telegram"]["bot_token"]


def _chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID") or str(_cfg()["telegram"]["chat_id"])


# ---------------------------------------------------------------------------
# One-shot senders (called from main.py during GHA run)
# ---------------------------------------------------------------------------

def send_new_setup(ticker: str, zones: list, stats: dict, chart_url: str) -> None:
    """Send a morning alert for a new scanner setup."""
    asyncio.run(_async_send_new_setup(ticker, zones, stats, chart_url))


def send_trade_update(trade: dict, summary: str, chart_url: str) -> None:
    """Send a morning update for an open position."""
    asyncio.run(_async_send_trade_update(trade, summary, chart_url))


def send_no_setups() -> None:
    """Optionally notify that no setups were found today."""
    asyncio.run(_async_send_text("📭 No qualifying setups found this morning."))


# ---------------------------------------------------------------------------
# Async message builders
# ---------------------------------------------------------------------------

async def _async_send_new_setup(
    ticker: str, zones: list, stats: dict, chart_url: str
) -> None:
    bot = Bot(token=_bot_token())

    best = max(zones, key=lambda z: z.score) if zones else None
    level_line = f"Level: ${best.price:.2f} | Score: {best.score}/10 | Tested {best.touch_count}× " \
                 f"{'(flipped)' if best.flipped else ''}" if best else "No major level"

    wr    = stats.get("win_rate", 0)
    ret   = stats.get("total_return", 0)
    sh    = stats.get("sharpe", 0)
    fscore = stats.get("fundamental_score", "–")

    stop  = stats.get("stop")
    tp1   = stats.get("tp1")
    tp2   = stats.get("tp2")
    vol   = stats.get("volume_ratio", "–")

    zone_type = "Major Support" + (" (flipped resistance)" if best and best.flipped else "") \
        if best and best.price < stats.get("current_price", best.price) else "Major Resistance"

    text = (
        f"🔔 *{ticker}* — {zone_type}\n"
        f"{level_line}\n"
        f"Volume: {vol}x avg | Fundamental score: {fscore}/100\n"
        f"Backtest (2yr): Win rate {wr:.0%} | Return {ret:+.1%} | Sharpe {sh:.2f}\n"
    )
    if stop:
        text += f"Stop: ${stop:.2f}"
    if tp1:
        text += f" | TP1: ${tp1:.2f}"
    if tp2:
        text += f" | TP2: ${tp2:.2f}"
    text += f"\n👉 [View chart]({chart_url})"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"🔄 Refresh {ticker}", callback_data=f"refresh_{ticker}"),
            InlineKeyboardButton("🔍 Backtest",          callback_data=f"backtest_{ticker}"),
        ],
        [InlineKeyboardButton("📋 View All Trades", callback_data="view_trades")],
    ])

    await bot.send_message(
        chat_id=_chat_id(),
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


async def _async_send_trade_update(
    trade: dict, summary: str, chart_url: str
) -> None:
    bot = Bot(token=_bot_token())

    ticker      = trade["ticker"]
    entry       = trade["entry_price"]
    shares      = trade["shares"]
    current     = trade.get("current_price", entry)
    pnl_pct     = trade.get("pnl_pct", 0)
    pnl_dollar  = (current - entry) * shares
    days_held   = trade.get("days_held", 0)
    pnl_sign    = "+" if pnl_pct >= 0 else ""

    direction = "HOLD" if abs(pnl_pct) < 0.03 else ("▲ PROFIT" if pnl_pct > 0 else "▼ LOSS")

    text = (
        f"📊 *{ticker}* — {direction} (day {days_held})\n"
        f"Entry: ${entry:.2f} × {shares} shares | Current: ${current:.2f}\n"
        f"P&L: {pnl_sign}${pnl_dollar:.2f} ({pnl_sign}{pnl_pct * 100:.1f}%)\n\n"
        f"{summary}\n"
        f"👉 [View chart]({chart_url})"
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"🔄 Refresh {ticker}", callback_data=f"refresh_{ticker}"),
            InlineKeyboardButton("✅ Mark Closed",        callback_data=f"close_{ticker}"),
        ],
    ])

    await bot.send_message(
        chat_id=_chat_id(),
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


async def _async_send_text(text: str) -> None:
    bot = Bot(token=_bot_token())
    await bot.send_message(chat_id=_chat_id(), text=text)


# ---------------------------------------------------------------------------
# Webhook / callback server (run on Render)
# ---------------------------------------------------------------------------

# In-memory cooldown for full-refresh button (key → epoch seconds of last trigger)
_last_refresh: dict[str, float] = {}


def _gh_dispatch(workflow_file: str, ticker: str | None = None) -> bool:
    """Trigger a GitHub Actions workflow_dispatch. Returns True on success."""
    token  = os.environ.get("GITHUB_TOKEN", "")
    repo   = os.environ.get("GITHUB_REPOSITORY", "")   # e.g. "user/premarket-scanner-bot"
    ref    = os.environ.get("GITHUB_REF_NAME", "main")
    if not token or not repo:
        logger.warning("GITHUB_TOKEN or GITHUB_REPOSITORY not set — cannot dispatch")
        return False

    inputs: dict = {}
    if ticker:
        inputs["ticker"] = ticker

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": inputs},
        timeout=10,
    )
    return resp.status_code == 204


async def _handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data    = query.data or ""
    cfg     = _cfg()
    chat_id = str(query.message.chat_id)

    # ── Refresh full scan ──
    if data == "refresh_all":
        cooldown = cfg["alerts"]["cooldown_minutes"] * 60
        last     = _last_refresh.get("all", 0)
        remaining = cooldown - (time.time() - last)
        if remaining > 0:
            await query.edit_message_text(
                f"⏳ Full refresh on cooldown — {int(remaining / 60)} min remaining."
            )
            return
        ok = _gh_dispatch("scanner.yml")
        if ok:
            _last_refresh["all"] = time.time()
            await query.edit_message_text("🔄 Full scan triggered. Results in ~2 min.")
        else:
            await query.edit_message_text("❌ Could not trigger workflow. Check GITHUB_TOKEN.")
        return

    # ── Refresh single ticker ──
    if data.startswith("refresh_"):
        ticker = data[len("refresh_"):]
        ok = _gh_dispatch("scanner.yml", ticker=ticker)
        msg = f"🔄 Refreshing {ticker}..." if ok else f"❌ Could not trigger workflow for {ticker}."
        await query.edit_message_text(msg)
        return

    # ── Backtest ──
    if data.startswith("backtest_"):
        ticker = data[len("backtest_"):]
        await query.edit_message_text(f"⏳ Running backtest for {ticker}...")
        try:
            from scanner import support_resistance as sr, backtest as bt_module
            import yfinance as yf
            price = float(yf.Ticker(ticker).fast_info["last_price"])
            zones = sr.get_levels(ticker, cfg)
            stats = bt_module.run_backtest(ticker, zones, cfg)
            reply = (
                f"📈 *{ticker}* backtest (2yr)\n"
                f"Win rate: {stats['win_rate']:.1%} | Trades: {stats['num_trades']}\n"
                f"Return: {stats['total_return']:+.1%} | Sharpe: {stats['sharpe']:.2f}\n"
                f"Max drawdown: {stats['max_drawdown']:.1%}"
            )
        except Exception as exc:
            reply = f"❌ Backtest failed: {exc}"
        await query.edit_message_text(reply, parse_mode=ParseMode.MARKDOWN)
        return

    # ── View all trades ──
    if data == "view_trades":
        try:
            from scanner import sheets
            trades = sheets.get_open_trades(cfg)
            if not trades:
                await query.edit_message_text("📋 No open trades found.")
                return
            lines = ["📋 *Open Trades*\n"]
            for t in trades:
                pnl = t.get("pnl_pct")
                pnl_str = f" ({pnl:+.1%})" if pnl is not None else ""
                lines.append(
                    f"• *{t['ticker']}* — {t['shares']} @ ${t['entry_price']:.2f} "
                    f"(day {t['days_held']}){pnl_str}"
                )
            await query.edit_message_text(
                "\n".join(lines), parse_mode=ParseMode.MARKDOWN
            )
        except Exception as exc:
            await query.edit_message_text(f"❌ Could not load trades: {exc}")
        return

    # ── Mark closed ──
    if data.startswith("close_"):
        ticker = data[len("close_"):]
        try:
            from scanner import sheets
            updated = sheets.mark_closed(ticker, cfg)
            msg = f"✅ {ticker} marked as closed." if updated else f"⚠️ No open position found for {ticker}."
        except Exception as exc:
            msg = f"❌ Could not update sheet: {exc}"
        await query.edit_message_text(msg)
        return

    await query.edit_message_text(f"Unknown action: {data}")


async def _handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Premarket scanner bot 📈\n"
        "Alerts are sent automatically each morning at 8:30am ET."
    )


def run_webhook(
    webhook_url: str,
    port: int = 8443,
    path: str = "/webhook",
) -> None:
    """
    Start the bot in webhook mode (for Render deployment).

    webhook_url: public HTTPS URL of the Render service,
                 e.g. https://your-app.onrender.com
    """
    token = _bot_token()
    app = (
        Application.builder()
        .token(token)
        .build()
    )
    app.add_handler(CommandHandler("start", _handle_start))
    app.add_handler(CallbackQueryHandler(_handle_callback))

    full_url = webhook_url.rstrip("/") + path
    logger.info("Starting webhook on %s (port %d)", full_url, port)

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=path,
        webhook_url=full_url,
    )


# ---------------------------------------------------------------------------
# Entry point for Render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    public_url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not public_url:
        raise SystemExit("RENDER_EXTERNAL_URL env var must be set on Render")
    run_webhook(webhook_url=public_url, port=int(os.environ.get("PORT", 8443)))
