"""Price download with a local cache. Only completed daily sessions are kept."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
COLS = ["Open", "High", "Low", "Close", "Volume"]
MARKET_CLOSE_DONE_IST = (15, 45)  # after this, today's bar is treated as final


def _drop_incomplete_today(df, now=None):
    """Yahoo returns a partial bar for today while the market is open; drop it."""
    now = now or datetime.now(IST)
    today = pd.Timestamp(now.date())
    if (now.hour, now.minute) < MARKET_CLOSE_DONE_IST:
        df = df[df.index < today]
    return df


def clean_bad_ticks(df, spike=0.5, bad_open=0.15):
    """Repair obvious data-vendor errors (Yahoo has some for Indian ETFs).

    1. Spikes: a Close more than 2x / less than 0.5x the median of the surrounding 11 days is a
       mis-scaled print (e.g. NIFTYBEES at 1/10th on 2019-12-19). Those rows are dropped.
       This uses neighbouring days, which is fine for cleaning history; live data gets only
       the backward-looking part of the check.
    2. Phantom opens: an Open more than 15% away from BOTH the previous close and the same
       day's close (and the High equal to that Open) is a bad print. Open/High are reset
       conservatively so a fake price can never trigger a target in the backtest.
    """
    df = df.copy()
    med = df["Close"].rolling(11, center=True, min_periods=3).median()
    ratio = df["Close"] / med
    bad = (ratio < 1 - spike) | (ratio > 1 / (1 - spike))
    df = df[~bad]
    prev = df["Close"].shift()
    o = df["Open"]
    phantom = ((o / prev - 1).abs() > bad_open) & ((o / df["Close"] - 1).abs() > bad_open)
    if phantom.any():
        df.loc[phantom, "Open"] = prev[phantom]
        df.loc[phantom, "High"] = np.maximum(prev[phantom], df.loc[phantom, "Close"])
        df.loc[phantom, "Low"] = np.minimum(df.loc[phantom, "Low"], df.loc[phantom, "Open"])
    return df, int(bad.sum()), int(phantom.sum())


def download_prices(tickers, start, cache_dir="data/cache", max_age_hours=12, refresh=False):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} of split/dividend-adjusted prices."""
    import yfinance as yf

    import hashlib
    tag = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    cache = Path(cache_dir) / f"prices_{tag}.pkl"
    if cache.exists() and not refresh:
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        cached = pd.read_pickle(cache)
        if age_h < max_age_hours and set(tickers) <= set(cached):
            return {t: cached[t] for t in tickers}

    raw = yf.download(list(tickers), start=start, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t][COLS].dropna(subset=["Open", "High", "Low", "Close"])
        except KeyError:
            print(f"  ! no data for {t}")
            continue
        df = df[df["Close"] > 0]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = _drop_incomplete_today(df)
        df, n_spike, n_open = clean_bad_ticks(df)
        if n_spike or n_open:
            print(f"  cleaned {t}: {n_spike} mis-scaled rows dropped, {n_open} phantom opens fixed")
        if len(df):
            out[t] = df.astype(float)
        else:
            print(f"  ! empty data for {t}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, cache)
    return out


def wide(prices, field, calendar):
    """date x ticker matrix of one OHLCV field, aligned to `calendar` (NaN where not traded)."""
    return pd.DataFrame({t: df[field] for t, df in prices.items()}).reindex(calendar)
