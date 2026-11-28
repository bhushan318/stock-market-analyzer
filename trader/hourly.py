"""Hourly price bars (Yahoo Finance, free) for today's Nifty 500 - collected on the Pi.

    python -m trader collect hourly

Yahoo serves 1-hour bars for roughly the last 730 days only, so the first run fills ~2 years and
every later run (weekday evenings) adds the last few days; overlapping bars are ignored.
Bars are labelled by their start time in IST (09:15, 10:15, ... 15:15; the last one is short).
Use: which of target/stop was touched first on a day, and entry timing (open vs later hours).
Caveats: today's Nifty 500 only (survivorship), Yahoo's intraday data has gaps and odd prints,
prices are not split-adjusted backwards consistently - check before trusting a result.
"""

import csv

import numpy as np
import pandas as pd

from . import ROOT
from .store import log, upsert

SCHEMA = """
CREATE TABLE IF NOT EXISTS eq_hourly (
    ts TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (ts, symbol)
);
"""
CHUNK = 50


def nifty500_symbols():
    p = ROOT / "universes" / "nifty500_2026-11.csv"
    return sorted({r["Symbol"].strip() for r in csv.DictReader(open(p, encoding="utf-8"))})


def frame_to_rows(df, symbol):
    """Yahoo bars (DatetimeIndex, Open/High/Low/Close/Volume) -> eq_hourly rows, IST timestamps."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert("Asia/Kolkata")
    vol = df["Volume"].fillna(0).astype("int64") if "Volume" in df else np.zeros(len(df), "int64")
    return [{"ts": t.strftime("%Y-%m-%dT%H:%M"), "symbol": symbol, "open": float(o), "high": float(h),
             "low": float(lo), "close": float(c), "volume": int(v)}
            for t, o, h, lo, c, v in zip(idx, df["Open"], df["High"], df["Low"], df["Close"], vol)]


def collect_hourly(con):
    import yfinance as yf
    con.executescript(SCHEMA)
    have = con.execute("SELECT COUNT(*) FROM eq_hourly").fetchone()[0]
    period = "5d" if have else "730d"
    syms = nifty500_symbols()
    total, missing = 0, []
    for k in range(0, len(syms), CHUNK):
        part = syms[k:k + CHUNK]
        tick = [s + ".NS" for s in part]
        try:
            raw = yf.download(tick, period=period, interval="60m", group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
        except Exception as e:  # noqa: BLE001 - logged, next chunk continues
            log(con, "hourly", f"chunk {k}", "error", 0, f"{type(e).__name__}: {str(e)[:200]}")
            missing += part
            continue
        rows = []
        for s, t in zip(part, tick):
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                missing.append(s)
                continue
            r = frame_to_rows(df, s)
            if not r:
                missing.append(s)
            rows += r
        total += upsert(con, "eq_hourly", rows)
    log(con, "hourly", period, "ok", total, f"{len(missing)} symbols without bars" if missing else "")
    return {"period": period, "new_bars": total, "symbols_missing": len(missing)}
