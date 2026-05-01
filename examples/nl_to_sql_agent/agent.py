from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import anthropic

_MODEL = "claude-sonnet-4-6"
_DB_PATH = Path(__file__).parent / "chinook.db"

_SCHEMA = """
Tables: albums(AlbumId, Title, ArtistId), artists(ArtistId, Name),
tracks(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice),
invoices(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingCountry, Total),
invoice_items(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity),
customers(CustomerId, FirstName, LastName, Company, Country, Email),
genres(GenreId, Name), media_types(MediaTypeId, Name),
playlists(PlaylistId, Name), playlist_track(PlaylistId, TrackId),
employees(EmployeeId, LastName, FirstName, Title, ReportsTo, HireDate).
"""

_SYSTEM = f"""You are a SQL expert. Convert natural language questions to SQLite queries.
Schema: {_SCHEMA}
Respond ONLY with the SQL query, no explanation, no markdown fences."""


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _execute_sql(sql: str) -> list[dict[str, Any]] | str:
    if not _DB_PATH.exists():
        return "chinook.db not found — run download_chinook.py first"
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchmany(50)]
        conn.close()
        return rows
    except sqlite3.Error as e:
        return f"SQL error: {e}"


async def run_agent(input_data: dict[str, Any]) -> dict[str, Any]:
    question = input_data["question"]
    response = _get_client().messages.create(
        model=_MODEL,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    sql = response.content[0].text.strip()
    result = _execute_sql(sql)
    return {
        "sql": sql,
        "result": result,
        "error": isinstance(result, str),
    }
