"""Google Sheets read/write for the open trade log."""
from __future__ import annotations

import json
import os
from datetime import date, datetime

import gspread
from google.oauth2.service_account import Credentials

# Columns expected in the worksheet (case-insensitive match on header row)
_COL_TICKER  = "ticker"
_COL_ENTRY   = "entry price"
_COL_SHARES  = "shares"
_COL_DATE    = "date entered"
_COL_STATUS  = "status"

_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_open_trades(config: dict) -> list[dict]:
    """
    Return all rows where Status == 'open'.

    Each row dict has keys:
        ticker, entry_price (float), shares (int),
        date_entered (str YYYY-MM-DD), days_held (int), row_index (int)
    """
    ws = _worksheet(config)
    rows = ws.get_all_records(head=1)

    open_trades = []
    today = date.today()

    for i, row in enumerate(rows, start=2):   # row 1 is the header
        normalised = {k.strip().lower(): v for k, v in row.items()}
        if str(normalised.get(_COL_STATUS, "")).strip().lower() != "open":
            continue

        try:
            entry_price = float(str(normalised[_COL_ENTRY]).replace(",", ""))
            shares      = int(float(str(normalised[_COL_SHARES]).replace(",", "")))
            date_str    = str(normalised[_COL_DATE]).strip()
            ticker      = str(normalised[_COL_TICKER]).strip().upper()
        except (KeyError, ValueError):
            continue

        try:
            entered = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_held = (today - entered).days
        except ValueError:
            days_held = 0

        open_trades.append({
            "ticker":       ticker,
            "entry_price":  entry_price,
            "shares":       shares,
            "date_entered": date_str,
            "days_held":    days_held,
            "row_index":    i,
        })

    return open_trades


def mark_closed(ticker: str, config: dict) -> bool:
    """
    Set the Status cell to 'closed' for the most recent open row of ticker.

    Returns True if a row was updated, False if no matching open row was found.
    """
    ws = _worksheet(config)
    rows = ws.get_all_records(head=1)

    # Find header row to locate the Status column index (1-based for gspread)
    headers = [h.strip().lower() for h in ws.row_values(1)]
    try:
        status_col = headers.index(_COL_STATUS) + 1
    except ValueError:
        raise ValueError(f"No '{_COL_STATUS}' column found in worksheet header row")

    for i, row in enumerate(rows, start=2):
        normalised = {k.strip().lower(): v for k, v in row.items()}
        if (str(normalised.get(_COL_TICKER, "")).strip().upper() == ticker.upper()
                and str(normalised.get(_COL_STATUS, "")).strip().lower() == "open"):
            ws.update_cell(i, status_col, "closed")
            return True

    return False


def append_trade(trade: dict, config: dict) -> None:
    """
    Append a new trade row (used in tests / manual seeding).

    trade keys: ticker, entry_price, shares, date_entered
    """
    ws = _worksheet(config)
    ws.append_row([
        trade["ticker"].upper(),
        trade["entry_price"],
        trade["shares"],
        trade.get("date_entered", date.today().isoformat()),
        "open",
    ], value_input_option="USER_ENTERED")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client() -> gspread.Client:
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        raise EnvironmentError("GOOGLE_SERVICE_ACCOUNT_JSON env var is not set")
    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=_SCOPES
    )
    return gspread.authorize(creds)


def _worksheet(config: dict) -> gspread.Worksheet:
    sheets_cfg = config["google_sheets"]
    spreadsheet_id  = sheets_cfg["spreadsheet_id"]
    worksheet_name  = sheets_cfg["worksheet_name"]
    gc = _client()
    sh = gc.open_by_key(spreadsheet_id)
    return sh.worksheet(worksheet_name)
