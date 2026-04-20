"""
Claude API trade analysis — returns a 2–3 sentence summary + recommendation.

Called once per open trade per morning run.  Uses claude-haiku-4-5 (cheapest
model) with prompt caching on the stable system prompt so the ~300-token
system block is only billed once per batch of trade calls (~0.1× on repeats).
"""
from __future__ import annotations

import os

import anthropic

_SYSTEM_PROMPT = """\
You are a concise pre-market trading analyst. You receive data about an open \
stock position and provide a short, actionable morning briefing.

Rules:
- Respond in exactly 2–3 sentences of plain English. No markdown, no bullet points.
- End with one of these recommendations on its own line: \
BUY MORE | HOLD | REDUCE | SELL
- Be direct. Cite the most important factor driving your recommendation.
- Never repeat the ticker in the summary body (it appears in the header).\
"""


def summarise(
    trade: dict,
    zones: list,
    indicators: dict,
    config: dict,
) -> str:
    """
    Call Claude Haiku with trade context and return a plain-English summary.

    Args:
        trade: {ticker, entry_price, shares, date_entered, current_price,
                pnl_pct, days_held}
        zones: list of SRZone objects near current price
        indicators: {rsi (float), ema20_slope (float), ema50_slope (float),
                     volume_trend (str: 'rising'|'falling'|'flat')}
        config: full config dict

    Returns:
        Plain-text summary string ending with a recommendation line.
    """
    api_cfg = config.get("claude_api", {})
    model = api_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = int(api_cfg.get("max_tokens", 400))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_prompt = _build_prompt(trade, zones, indicators)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Cache the system prompt — it's identical for every trade
                # in the same morning run, so only the first call pays full
                # write price; subsequent calls are ~0.1× input cost.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )

    return _extract_text(response)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(trade: dict, zones: list, indicators: dict) -> str:
    ticker        = trade["ticker"]
    entry         = trade["entry_price"]
    shares        = trade["shares"]
    current       = trade.get("current_price", entry)
    pnl_pct       = trade.get("pnl_pct", (current - entry) / entry)
    days_held     = trade.get("days_held", 0)
    date_entered  = trade.get("date_entered", "unknown")

    pnl_dollar    = (current - entry) * shares
    pnl_sign      = "+" if pnl_pct >= 0 else ""

    # Summarise nearest zones
    zone_lines: list[str] = []
    for z in sorted(zones, key=lambda z: abs(current - z.price))[:3]:
        relation = "below" if z.price < current else "above"
        flip_tag = " (flipped)" if z.flipped else ""
        zone_lines.append(
            f"  • ${z.price:.2f} {relation}, score {z.score}, "
            f"{z.touch_count}× touched{flip_tag}"
        )
    zones_text = "\n".join(zone_lines) if zone_lines else "  • None detected nearby"

    rsi           = indicators.get("rsi")
    ema20_slope   = indicators.get("ema20_slope")
    ema50_slope   = indicators.get("ema50_slope")
    volume_trend  = indicators.get("volume_trend", "unknown")

    def _fmt_slope(s) -> str:
        if s is None:
            return "unknown"
        return f"{'rising' if s > 0 else 'falling'} ({s:+.2f}%/day)"

    return f"""\
Position: {ticker}
Entry: ${entry:.2f} × {shares} shares on {date_entered} ({days_held} day(s) held)
Current price: ${current:.2f}
P&L: {pnl_sign}{pnl_pct * 100:.2f}% (${pnl_dollar:+.2f})

Nearby S/R zones:
{zones_text}

Technical indicators:
  RSI(14): {f'{rsi:.1f}' if rsi is not None else 'unknown'}
  EMA20 slope: {_fmt_slope(ema20_slope)}
  EMA50 slope: {_fmt_slope(ema50_slope)}
  Volume trend: {volume_trend}

Provide your 2–3 sentence morning briefing and recommendation for {ticker}.\
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_text(response: anthropic.types.Message) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text.strip()
    return "No analysis available."
