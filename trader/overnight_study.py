"""Owner's idea: use last night's US market (S&P 500) to judge the Indian day.

    python -m trader overnight

The S&P 500 closes ~01:30-02:30 IST, before NSE opens at 09:15. For each NSE day: the most recent
S&P 500 daily return with a US date BEFORE that day (Monday uses Friday's US session). Indian side,
equal-weight average over the point-in-time top 200 stocks (from our own bhavcopy panel):
  gap       open / previous close - 1   (already decided before you can buy at the open)
  intraday  close / open - 1            (what a buy at the open earns that day)
  week      close 4 sessions later / open - 1
Bucketed by the S&P move; two halves (2006-15, 2016-26) to see if a pattern is stable.
(GIFT/SGX Nifty history is not freely available; it mostly mirrors this overnight US move.)
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .pit import liquid_universe, load_panel

BINS = [-1, -0.02, -0.01, -0.003, 0.003, 0.01, 0.02, 1]
LABELS = ["< -2%", "-2..-1%", "-1..-0.3%", "flat", "+0.3..1%", "+1..2%", "> +2%"]


def align_us(us_ret, india_days):
    """For each India date, the latest US return dated strictly before it."""
    us = us_ret.dropna().sort_index()
    a = pd.DataFrame({"d": pd.DatetimeIndex(india_days).astype("datetime64[ns]")})
    b = pd.DataFrame({"d": pd.DatetimeIndex(us.index).astype("datetime64[ns]"), "sp": us.values})
    m = pd.merge_asof(a, b, on="d", allow_exact_matches=False)
    return pd.Series(m["sp"].values, index=india_days)


def run_overnight(cfg):
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, O, cal = p["C"], p["O"], p["cal"]
    uni = liquid_universe(C, p["TO"], top=200)
    Cp = C.shift(1)
    gap = (O / Cp - 1).where(uni & (O / Cp - 1).abs().lt(0.2))
    intra = (C / O - 1).where(uni & (C / O - 1).abs().lt(0.2))
    week = (C.shift(-4) / O - 1).where(uni)
    sp = download_prices(["^GSPC"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^GSPC"]["Close"]
    D = pd.DataFrame({"gap": gap.mean(axis=1), "intra": intra.mean(axis=1), "week": week.mean(axis=1),
                      "up_share": (intra > 0).sum(axis=1) / intra.notna().sum(axis=1)}, index=cal)
    D["sp"] = align_us(sp.pct_change(), cal)
    D = D[D.index >= pd.Timestamp("2006-01-01")].dropna(subset=["sp", "gap", "intra"])
    D["bucket"] = pd.cut(D.sp, BINS, labels=LABELS)
    D["half"] = np.where(D.index < pd.Timestamp("2016-01-01"), "2006-15", "2016-26")
    D["monday"] = D.index.dayofweek == 0
    out = ROOT / cfg["backtest"]["output_dir"]
    D.to_csv(out / "overnight_days.csv")
    pct = lambda v: f"{v:+.2%}"  # noqa: E731
    print(f"\n{len(D)} NSE days. Correlation with last night's S&P 500 move: gap {D.sp.corr(D.gap):+.2f}, "
          f"intraday {D.sp.corr(D.intra):+.2f}, open->week {D.sp.corr(D.week):+.2f}")
    for half, g in [("all", D)] + list(D.groupby("half")):
        t = g.groupby("bucket", observed=True).agg(days=("sp", "size"), gap=("gap", "mean"), intraday=("intra", "mean"),
                                                   stocks_up=("up_share", "mean"), week_from_open=("week", "mean"))
        for c in ("gap", "intraday", "week_from_open"):
            t[c] = t[c].map(pct)
        t["stocks_up"] = t["stocks_up"].map("{:.0%}".format)
        print(f"\n--- {half}: average top-200 stock, by last night's S&P 500 move ---")
        print(t.to_string())
    m = D[D.monday]
    t = m.groupby("bucket", observed=True).agg(days=("sp", "size"), intraday=("intra", "mean"), week=("week", "mean"))
    print("\n--- Mondays only (our buy day), by Friday's S&P 500 move ---")
    print(t.assign(intraday=t.intraday.map(pct), week=t.week.map(pct)).to_string())
    print(f"\nCSV: {out}/overnight_days.csv")
