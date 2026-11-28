"""Owner's idea: don't fill all 5 slots at once - add one stock at a time, only the strongest.

    python -m trader stagger

Gate 1 rule (combo ranking, weekly, sell when out of the top 50, no new buys while Nifty 500 is
below its 200-day average, 5 slots, Rs 25k) on 2006-2026 (history.db + market.db), changing only
how slots are filled:
  max_new     at most N new stocks per week (N=1: one at a time)
  entry_rank  only buy a stock ranked in the top K that week (K=1: only the single best)
Same random control (random picks under the same limits). Unused slots sit in cash at 0%.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .momentum_pit import _orders, pit_signals
from .momentum_study import simulate, stats, weekly_dates
from .pit import liquid_universe, load_panel

START = "2006-07-01"
PERIODS = [("2006-19", "2006-07-01", "2019-12-31"), ("2020-26", "2020-01-01", "2026-12-31"),
           ("full", "2006-07-01", "2026-12-31")]
V = [("fill all 5 (Gate 1 rule)", None, None), ("1 new stock per week", 1, None), ("2 new per week", 2, None),
     ("1 per week, top 3 only", 1, 3), ("1 per week, only the #1", 1, 1), ("fill all, top 5 only", None, 5)]
SEEDS = 5


def run_stagger(cfg):
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, cal = p["C"], p["cal"]
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    regime = (n5 > n5.rolling(200, min_periods=150).mean()).to_numpy() | n5.isna().to_numpy()
    sig, _ = pit_signals(C, p["TO"], liquid_universe(C, p["TO"]))
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(START)) & (rebal < cal[-1])]
    od = _orders(sig["combo"], [cal.get_loc(d) for d in rebal])
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    rows = []
    for name, mx, er in V:
        print(f"  {name} ...", flush=True)
        kw = dict(exit_rank=50, regime=regime, max_new=mx, entry_rank=er)
        r = {"variant": name}
        for pn, a, b in PERIODS:
            i0 = cal.searchsorted(pd.Timestamp(a))
            i1 = min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
            res = simulate(Oa, Cf, od, i0, i1, 25_000, cfg["costs"], **kw)
            st = stats(res, 25_000)
            hd = np.array([t[0] for t in res["trades"]] or [20])
            rnd = [stats(simulate(Oa, Cf, od, i0, i1, 25_000, cfg["costs"], **kw, rng=np.random.default_rng(s),
                                  hold_dist=hd), 25_000)["cagr"] for s in range(SEEDS)]
            r[f"{pn}_cagr"], r[f"{pn}_random"] = st["cagr"], float(np.median(rnd))
            if pn == "full":
                r.update(max_dd=st["max_dd"], trades_yr=st["trades_yr"], win_rate=st["win_rate"],
                         avg_cash=float(res["cash_share"].mean()), final_rs=res["equity"][-1])
        bench = {pn: (n5.iloc[min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal)) - 1]
                      / n5.iloc[cal.searchsorted(pd.Timestamp(a))]) for pn, a, b in PERIODS}
        rows.append(r)
    R = pd.DataFrame(rows)
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "stagger_variants.csv", index=False)
    show = R.copy()
    for c in show.columns:
        if c.endswith(("cagr", "random", "dd", "rate", "cash")):
            show[c] = show[c].map(lambda v: f"{v:+.0%}")
    show["trades_yr"] = R["trades_yr"].map("{:.0f}".format)
    show["final_rs"] = R["final_rs"].map("{:,.0f}".format)
    yrs = {pn: (min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
                - cal.searchsorted(pd.Timestamp(a))) / 248 for pn, a, b in PERIODS}
    print("\nNifty 500 per year: " + ", ".join(f"{pn} {bench[pn] ** (1 / yrs[pn]) - 1:+.0%}" for pn in bench))
    print("\n=== Filling slots: all at once vs one at a time (Rs 25k, after charges; 'random' = same limits) ===")
    print(show[["variant", "final_rs", "full_cagr", "full_random", "2006-19_cagr", "2006-19_random",
                "2020-26_cagr", "2020-26_random", "max_dd", "avg_cash", "trades_yr", "win_rate"]].to_string(index=False))
