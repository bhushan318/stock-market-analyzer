"""Price download with a local cache. Only completed daily sessions are kept."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def download_prices(tickers, start, cache_dir="data/cache", max_age_hours=12, refresh=False):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} of split/dividend-adjusted prices."""
    import yfinance as yf

    cache = Path(cache_dir) / "prices.pkl"
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
