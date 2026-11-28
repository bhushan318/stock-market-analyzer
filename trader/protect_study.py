"""Crash protection for the whole portfolio, 2006-2026 (history.db + market.db as one panel).

    python -m trader protect

Results 11: in 2008 the index fell 64% and the strategy 78%; the 200-day filter only stopped new
buys. Two protections, tested on the full 20 years (both were proposed before looking):
  1. index part ("core") held only while Nifty 500 > its 200-day average, else a liquid fund
     (6%/yr assumed - roughly the 2006-2026 average of short-term rates). Checked on Fridays,
     switched at Monday's close, 0.1% cost per switch. Every switch out is a taxable sale (not
     modelled - reported as switches/yr).
  2. strategy stocks all sold when the filter turns off (regime_exit), instead of only blocking
     new buys. Results 8 (2021-26) said this whipsaws; here it also meets 2008, 2011, 2018.
Mixes: 25% strategy (Rs 25k) / 75% core (Rs 75k), reset each January. Charges included for the
strategy; before tax. Nifty 500 = price index (no dividends) for the core and the filter.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .momentum_pit import _orders, pit_signals
from .momentum_study import simulate, weekly_dates
from .pit import liquid_universe, load_panel
from .robust_study import mix_equity

START, CHECK_OLD_END, NEW_START = "2006-07-01", "2019-12-31", "2021-01-01"
CASH_RATE, SWITCH_COST, LIMIT_DD = 0.06, 0.001, -0.25
FALLS = [("2008 crash", "2008-01-08", "2009-03-09"), ("2010-11 fall", "2010-11-05", "2011-12-20"),
         ("2015-16 fall", "2015-03-03", "2016-02-29"), ("2018 mid/small crash", "2018-01-29", "2018-10-26"),
         ("2020 Covid crash", "2020-01-14", "2020-03-24"), ("2024-25 fall", "2024-09-26", "2025-03-04")]


def timed_core(idx, above, check, cash_rate=CASH_RATE, cost=SWITCH_COST):
    """Value (from 1) of holding the index while the last checked signal was True, else cash.
    idx: index levels; above: bool per day (index > its average at that close); check: bool per
    day (signal is read at that close); the switch happens at the NEXT day's close.
    Returns (values, number of switches)."""
    idx, above, check = np.asarray(idx, float), np.asarray(above, bool), np.asarray(check, bool)
    inside, want, v, sw = bool(above[0]), bool(above[0]), [1.0], 0
    daily_cash = (1 + cash_rate) ** (1 / 248)
    for k in range(1, len(idx)):
        val = v[-1] * (idx[k] / idx[k - 1] if inside else daily_cash)
        if want != inside:  # decided at an earlier check, executed at today's close
            val *= 1 - cost
            inside, sw = want, sw + 1
        if check[k]:
            want = bool(above[k])
        v.append(val)
    return np.array(v), sw


def describe(x, i_new, windows):
    """x: pd.Series of values over the full period."""
    def cagr(s):
        return (s.iloc[-1] / s.iloc[0]) ** (248 / len(s)) - 1
    dd = lambda s: float((s / s.cummax() - 1).min())  # noqa: E731
    w1 = [(x.iloc[a:b]) for a, b in windows]
    return {"cagr": cagr(x), "old_cagr": cagr(x[x.index <= pd.Timestamp(CHECK_OLD_END)]),
            "new_cagr": cagr(x.iloc[i_new:]), "max_dd": dd(x),
            "worst_1y": min(s.iloc[-1] / s.iloc[0] - 1 for s in w1),
            "w1_fall_over_25": np.mean([dd(s) < LIMIT_DD for s in w1]),
            "losing_years": int(sum(g.iloc[-1] / g.iloc[0] < 1 for _, g in x.groupby(x.index.year)))}


def run_protect(cfg):
    print("Loading 2005-2026 panel (history.db + market.db) ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, cal = p["C"], p["cal"]
    print(f"  {len(cal)} sessions {cal[0].date()} -> {cal[-1].date()}, {C.shape[1]} stocks, "
          f"{len(p['events'])} split/bonus events")
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    above = (n5 > n5.rolling(200, min_periods=150).mean())
    regime = above.to_numpy() | n5.isna().to_numpy()
    sig, _ = pit_signals(C, p["TO"], liquid_universe(C, p["TO"]))
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(START)) & (rebal < cal[-1])]
    ridx = [cal.get_loc(d) for d in rebal]
    od = _orders(sig["combo"], ridx)
    i0, i1 = cal.searchsorted(pd.Timestamp(START)), len(cal)
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    days = cal[i0:i1]
    idx = n5.iloc[i0:i1]

    strat = {}
    for name, ex in (("block buys", False), ("sell all", True)):
        print(f"  strategy, filter = {name} ...", flush=True)
        r = simulate(Oa, Cf, od, i0, i1, 25_000, cfg["costs"], exit_rank=50, regime=regime, regime_exit=ex)
        strat[name] = pd.Series(r["equity"] / 25_000, index=days)
        strat[name + " trades"] = len(r["trades"])
    check = np.zeros(len(days), bool)
    check[[k - i0 for k in ridx if k >= i0]] = True
    tc, switches = timed_core(idx.values, above.iloc[i0:i1].values, check)
    cores = {"index fund, always": pd.Series(idx.values / idx.values[0], index=days),
             "index fund, timed": pd.Series(tc, index=days)}
    jan = [k for k in range(1, len(days)) if days[k].year != days[k - 1].year]
    S = {}
    for cname, cv in cores.items():
        S[f"100% {cname}"] = cv
    for sname in ("block buys", "sell all"):
        S[f"100% strategy ({sname})"] = strat[sname]
        for cname, cv in cores.items():
            S[f"75% {cname} + 25% strategy ({sname})"] = pd.Series(
                mix_equity(strat[sname].values, cv.values, 0.25, jan, fee=0.0), index=days)

    i_new = days.searchsorted(pd.Timestamp(NEW_START))
    win = []
    for s0 in pd.date_range(START, days[-1] - pd.DateOffset(years=1), freq="3MS"):
        a, b = days.searchsorted(s0), days.searchsorted(s0 + pd.DateOffset(years=1))
        win.append((a, b))
    R = pd.DataFrame([{"portfolio": k, "rs_1lakh_to": 100_000 * v.iloc[-1], **describe(v, i_new, win)}
                      for k, v in S.items()])
    F = []
    for lab, a, b in FALLS:
        row = {"period": lab}
        for k in ("100% index fund, always", "100% index fund, timed", "100% strategy (block buys)",
                  "100% strategy (sell all)", "75% index fund, always + 25% strategy (block buys)",
                  "75% index fund, timed + 25% strategy (sell all)", "75% index fund, timed + 25% strategy (block buys)"):
            s = S[k][(S[k].index >= pd.Timestamp(a)) & (S[k].index <= pd.Timestamp(b))]
            row[k] = float((s / s.cummax() - 1).min())
        F.append(row)
    F = pd.DataFrame(F)
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "protect_portfolios.csv", index=False)
    F.to_csv(out / "protect_falls.csv", index=False)
    pd.DataFrame(S).to_csv(out / "protect_curves.csv")

    yrs = len(days) / 248
    print(f"\nTimed index fund: {switches} switches in {yrs:.0f} years ({switches / yrs:.1f}/yr); "
          f"strategy trades: block {strat['block buys trades']}, sell-all {strat['sell all trades']}")
    show = R.copy()
    for c in ("cagr", "old_cagr", "new_cagr", "max_dd", "worst_1y", "w1_fall_over_25"):
        show[c] = show[c].map(lambda v: f"{v:+.0%}" if "fall" not in c else f"{v:.0%}")
    show["rs_1lakh_to"] = R["rs_1lakh_to"].map("{:,.0f}".format)
    print(f"\n=== Rs 1 lakh, Jul 2006 -> Nov 2026 (before tax; liquid fund {CASH_RATE:.0%}/yr) ===")
    print(show.to_string(index=False))
    print("\n=== Worst fall inside each market fall ===")
    fs = F.copy()
    for c in fs.columns[1:]:
        fs[c] = fs[c].map("{:.0%}".format)
    fs.columns = ["period", "index", "index timed", "strat block", "strat sell-all", "75/25 now",
                  "75 timed/25 sell-all", "75 timed/25 block"]
    print(fs.to_string(index=False))
    print(f"\nCSV: {out}/protect_portfolios.csv, protect_falls.csv, protect_curves.csv")
