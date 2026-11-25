"""SQLite storage for collected market data. One file, safe to re-run (idempotent upserts)."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS eq_daily (          -- NSE full bhavcopy: prices + delivery
    date TEXT, symbol TEXT, series TEXT,
    prev_close REAL, open REAL, high REAL, low REAL, last REAL, close REAL, avg_price REAL,
    volume INTEGER, turnover_lacs REAL, trades INTEGER, deliv_qty INTEGER, deliv_pct REAL,
    PRIMARY KEY (date, symbol, series)
);
CREATE INDEX IF NOT EXISTS eq_daily_symbol ON eq_daily(symbol, date);

CREATE TABLE IF NOT EXISTS deals (             -- bulk and block deals
    date TEXT, kind TEXT, symbol TEXT, security TEXT, client TEXT, side TEXT,
    qty INTEGER, price REAL, remarks TEXT,
    PRIMARY KEY (date, kind, symbol, client, side, qty, price)
);

CREATE TABLE IF NOT EXISTS fii_dii (           -- daily cash-market flows, Rs crore
    date TEXT, category TEXT, buy REAL, sell REAL, net REAL,
    PRIMARY KEY (date, category)
);

CREATE TABLE IF NOT EXISTS announcements (     -- NSE corporate announcements / results RSS
    link TEXT PRIMARY KEY, feed TEXT, published TEXT, company TEXT, symbol TEXT,
    subject TEXT, description TEXT, fetched_at TEXT,
    tags TEXT                                   -- filled later by the LLM tagger (JSON)
);
CREATE INDEX IF NOT EXISTS announcements_symbol ON announcements(symbol, published);

CREATE TABLE IF NOT EXISTS collect_log (
    run_at TEXT, source TEXT, ref TEXT, status TEXT, rows INTEGER, detail TEXT
);
"""


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def upsert(con, table, rows, replace=False):
    """Insert a list of dicts; existing primary keys are ignored (or replaced)."""
    if not rows:
        return 0
    cols = list(rows[0])
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    sql = f"{verb} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    before = con.total_changes
    con.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    con.commit()
    return con.total_changes - before


def log(con, source, ref, status, rows=0, detail=""):
    con.execute("INSERT INTO collect_log VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, ref, status,
                 rows, detail[:500]))
    con.commit()
