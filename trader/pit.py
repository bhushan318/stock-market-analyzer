"""Point-in-time stock panel from data/market.db (NSE full bhavcopy, every EQ/BE stock).

Why: Yahoo + today's Nifty 500 only contains stocks that survived and grew into the index.
The bhavcopy holds every stock that traded on each day, including ones that later collapsed,
were suspended or merged away, so a universe built from it on date t knows nothing after t.

Steps:
  1. Symbol renames (NSE symbolchange.csv) are mapped to today's symbol, effective-dated, so a
     renamed stock keeps one continuous history.
  2. Splits / bonuses / demergers. NSE's PREV_CLOSE is NOT adjusted on the ex-date (checked:
     MAZDOCK 1:2 split 2024-12-27 shows PREV_CLOSE 4729.75, open 2375), so they are detected from
     the overnight gap OPEN(t) / CLOSE(t-1): price bands (<= 20%) make an opening gap beyond
     SPLIT_LO / SPLIT_HI practically impossible without a corporate action, while real crashes
     (INDUSINDBK 2025-03-11, ADANIENT 2023-02-01) open near the old price and fall intraday.
     The factor snaps to the nearest clean ratio (1/2, 2/3, 1/5, 1/10 ...) if within 4%,
     otherwise the gap itself is used (demergers, e.g. TATAMOTORS 2025-10-14 at 0.605).
     Prices before the event are multiplied by the factor (volume divided). Dividends are NOT
     added back (Yahoo does), so returns are ~1-1.5%/yr lower for every strategy and control alike.
     (Until 2026-11-26 this used PREV_CLOSE and found only 14 events in 7 years - every split
     looked like a crash. The first momentum-pit run was invalid because of it.)
  3. A stock that stops trading before the end of the data ("delisted") gets its last close as
     the open on every later day, so a holding can be exited there. That is optimistic for
     collapses (real holders may be locked in) - events are counted and reported.
"""

import csv
import sqlite3

import numpy as np
import pandas as pd

from . import ROOT

SPLIT_LO, SPLIT_HI = 0.78, 1.30
SNAP_TOL = 0.04  # log distance to a clean ratio (CDSL's 1:1 bonus opened at 0.518)
# bonus a:b -> b/(a+b) and splits 1/k; combinations (split + bonus) give 1/10, 1/20 ...
_FRACS = sorted({a / b for b in range(2, 7) for a in range(1, b)}
                | {1 / k for k in (7, 8, 9, 10, 15, 20, 25, 50, 100)})
CLEAN_RATIOS = np.array(_FRACS + [1 / f for f in _FRACS])
UNI = ROOT / "universes"


def load_symbol_changes(path=UNI / "nse_symbol_changes.csv"):
    """[(old, new, effective Timestamp)] sorted by date."""
    out = []
    with open(path, encoding="latin1") as fh:
        for r in csv.reader(fh):
            if len(r) < 4 or r[1].strip().upper() == "OLD SYMBOL":
                continue
            try:
                out.append((r[1].strip(), r[2].strip(), pd.to_datetime(r[3].strip(), format="%d-%b-%Y")))
            except ValueError:
                continue
    return sorted(out, key=lambda x: x[2])


def apply_symbol_changes(df, changes):
    """Rows of `old` dated before the effective date become `new`; chains (A->B->C) resolve
    because changes are applied in date order. A later reuse of an old symbol is left alone."""
    sym = df["symbol"].to_numpy(dtype=object).copy()
    dates = df["date"].to_numpy()
    for old, new, eff in changes:
        m = (sym == old) & (dates < np.datetime64(eff))
        if m.any():
            sym[m] = new
    df = df.copy()
    df["symbol"] = sym
    return df


def snap_ratio(x):
    """Nearest clean split/bonus ratio if within SNAP_TOL (log), else x itself."""
    x = np.asarray(x, dtype=float)
    d = np.log(x[:, None] / CLEAN_RATIOS[None, :])
    k = np.abs(d).argmin(1)
    near = np.abs(d[np.arange(len(x)), k]) < SNAP_TOL
    return np.where(near, CLEAN_RATIOS[k], x)


def split_events(df):
    """DataFrame[date, symbol, factor] of detected splits/bonuses/demergers
    (df: long, one row per day/symbol, columns date, symbol, open, close)."""
    d = df.sort_values(["symbol", "date"])
    g = d.groupby("symbol")
    last_close = g["close"].shift()
    gap_days = (d["date"] - g["date"].shift()).dt.days
    base = d["open"].where(d["open"] > 0, d["close"])
    f = base / last_close
    consistent = (d["close"] / base - 1).abs() < 0.3  # the day trades near the new base
    recent = gap_days <= 10  # previous row is the last session, not the far side of a data hole
    ev = d.loc[((f < SPLIT_LO) | (f > SPLIT_HI)) & consistent & recent & last_close.gt(0), ["date", "symbol"]]
    ev["factor"] = snap_ratio(f[ev.index].to_numpy())
    return ev.reset_index(drop=True)


def adjustment_multiplier(events, index, columns):
    """date x symbol multiplier: product of the factors of all events strictly after each date."""
    F = pd.DataFrame(1.0, index=index, columns=columns)
    for r in events.itertuples(index=False):
        if r.symbol in F.columns and r.date in F.index:
            F.at[r.date, r.symbol] *= r.factor
    M = F.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return M


def etf_symbols():
    p = UNI / "nse_etfs.csv"
    if not p.exists():
        return set()
    return {r["Symbol"].strip() for r in csv.DictReader(open(p, encoding="latin1"))}


def load_panel(db=None, start="2020-01-01"):
    """db: path or list of paths. Wide, split-adjusted frames {O, H, L, C, V, TO(Rs cr), DELIV(%)} + events + delisted info."""
    dbs = db if isinstance(db, (list, tuple)) else [db or ROOT / "data" / "market.db"]
    parts = []
    for d in dbs:  # several DBs (history.db 2005-19 + market.db 2020+) form one continuous panel
        con = sqlite3.connect(d)
        parts.append(pd.read_sql(
            "SELECT date, symbol, series, prev_close, open, high, low, close, volume, turnover_lacs, deliv_pct "
            "FROM eq_daily WHERE date >= ?", con, params=(start,), parse_dates=["date"]))
        con.close()
    df = pd.concat(parts, ignore_index=True)
    df = df[~df["symbol"].isin(etf_symbols())]
    # rights entitlements (XYZ-RE, XYZ-RE1) trade for a few days in BE; they are not shares
    df = df[~df["symbol"].str.contains(r"-RE\d*$", regex=True)]
    # one row per day/symbol; EQ preferred over BE if both exist
    df = df.sort_values("series", key=lambda s: s.ne("EQ")).drop_duplicates(["date", "symbol"])
    p = UNI / "nse_symbol_changes.csv"
    if p.exists():
        df = apply_symbol_changes(df, load_symbol_changes(p))
        df = df.sort_values("series", key=lambda s: s.ne("EQ")).drop_duplicates(["date", "symbol"])
    df = df[df["close"] > 0]
    # BE (trade-for-trade) shares must be delivered, so their 100% delivery says nothing
    df.loc[df["series"] != "EQ", "deliv_pct"] = np.nan
    ev = split_events(df)
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    wide = {k: df.pivot(index="date", columns="symbol", values=c).reindex(cal)
            for k, c in [("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"),
                         ("V", "volume"), ("TO", "turnover_lacs"), ("DELIV", "deliv_pct")]}
    M = adjustment_multiplier(ev, cal, wide["C"].columns)
    for k in "OHLC":
        wide[k] = wide[k] * M
    wide["V"] = wide["V"] / M
    wide["TO"] = wide["TO"] / 100.0  # lakh -> crore
    wide["O"] = wide["O"].where(wide["O"] > 0)
    # delisted: stopped trading > 10 sessions before the end -> exit possible at the last close
    C = wide["C"]
    last = C.apply(pd.Series.last_valid_index)
    gone = last[last < cal[-11]]
    O = wide["O"]
    for s, d in gone.items():
        after = O.index > d
        O.loc[after, s] = C.at[d, s]
    return {**wide, "events": ev, "delisted": gone, "cal": cal}


def liquid_universe(C, TO, top=500, min_price=10.0, window=126):
    """Boolean date x symbol: the `top` stocks by median daily turnover over the past `window`
    sessions (point-in-time proxy for the Nifty 500, which ranks by free-float market cap)."""
    med = TO.fillna(0).rolling(window, min_periods=window // 2).median()
    live = C.notna() & (C >= min_price)
    med = med.where(live)
    rank = med.rank(axis=1, ascending=False, method="first")
    return rank <= top
