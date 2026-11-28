"""News impact corrected for the shortlist bias.

    python -m trader news baseline

The shortlist is TODAY's top-momentum stocks, so they beat the index on almost ANY day of the
past years. The fair question is: after a given announcement type, did the stock do better than
the SAME stock did on an ordinary day of the same period? abnormal = event's 20-day excess return
minus that stock's average 20-day excess return over all sessions of that period (learn 2019-23,
check 2024+). A category counts only if both periods agree in sign with |t| > 2.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .pit import load_panel


def run_news_baseline(cfg, min_events=30):
    out = ROOT / cfg["backtest"]["output_dir"]
    E = pd.read_csv(out / "news_events.csv", parse_dates=["date"])
    syms = sorted(E.symbol.unique())
    p = load_panel(start="2019-01-01")
    O, C, cal = p["O"][syms], p["C"].ffill()[syms], p["cal"]
    n5 = download_prices(["^CRSLDX"], "2019-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    x20 = (C.shift(-19) / O - 1).sub(n5.shift(-19) / n5.shift(1) - 1, axis=0)
    base = x20.stack().rename("x20").reset_index()
    base.columns = ["date", "symbol", "x20"]
    base["period"] = np.where(base.date < pd.Timestamp("2024-01-01"), "learn 2019-23", "check 2024+")
    print("Same 50 stocks on ANY day, avg 20-day return vs Nifty 500: "
          + ", ".join(f"{k} {v:+.1%}" for k, v in base.groupby("period").x20.mean().items()))
    bs = base.groupby(["symbol", "period"]).x20.mean().rename("base")
    E = E.join(bs, on=["symbol", "period"])
    E["abn"] = E.x20 - E["base"]

    def summ(g):
        se = g.abn.std(ddof=1) / np.sqrt(len(g)) if len(g) > 2 else np.nan
        return pd.Series({"n": len(g), "abn": g.abn.mean(), "t": g.abn.mean() / se if se else np.nan})
    S = E.groupby(["category", "period"]).apply(summ).unstack("period")
    S = S[S[("n", "learn 2019-23")].fillna(0) >= min_events]
    L, K = "learn 2019-23", "check 2024+"
    S["holds"] = ((np.sign(S[("abn", L)]) == np.sign(S[("abn", K)]))
                  & (S[("t", L)].abs() > 2) & (S[("t", K)].abs() > 2))
    S = S.sort_values(("n", L), ascending=False)
    S.to_csv(out / "news_baseline.csv")
    pd.set_option("display.width", 250)
    show = S.copy()
    for c in show.columns:
        if c[0] == "abn":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
        elif c[0] == "t":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}")
        elif c[0] == "n":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    print(f"\nAbnormal 20-day return after each announcement type (vs the same stock's ordinary days), "
          f">= {min_events} events in 2019-23:")
    print(show.to_string())
    print(f"\nCategories that hold in both periods (|t| > 2, same sign): {int(S['holds'].sum())} of {len(S)}")
    print(f"CSV: {out}/news_baseline.csv")
