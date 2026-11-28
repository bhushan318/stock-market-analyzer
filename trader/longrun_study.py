"""Out-of-sample test on 2006-2019 (data/history.db): the Gate 1 rule, unchanged, in older markets.

    python -m trader longrun

Everything was chosen on 2021-2026; nothing here is tuned. 2006-2019 holds the 2008 crash (Nifty
500 about -64%), the 2009 rebound (a known trap for momentum), the flat 2010-2013, the 2015-16
fall and the 2018 mid/small-cap crash. Same point-in-time top-500 universe (by turnover, from the
old NSE bhavcopies, including stocks that later died), same signals, charges, Rs 2 cr turnover
floor (nominal - fewer stocks pass it in 2006), same random control. Market filter and benchmark:
Nifty 500 price index from Yahoo (^CRSLDX, starts Sep 2005, no dividends - same as the strategy).
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .momentum_pit import SEEDS, _orders, concentration, pit_signals
from .momentum_study import simulate, stats, weekly_dates
from .pit import liquid_universe, load_panel
from .robust_study import mix_equity, path_stats

DB = ROOT / "data" / "history.db"
START, END = "2006-07-01", "2019-12-31"
CAP, TOTAL, SHARE = 25_000, 100_000, 0.25
FALLS = [("2008 crash", "2008-01-08", "2009-03-09"), ("2010-11 fall", "2010-11-05", "2011-12-20"),
         ("2015-16 fall", "2015-03-03", "2016-02-29"), ("2018 mid/small crash", "2018-01-29", "2018-10-26")]


def run_longrun(cfg):
    print("Loading 2005-2019 panel from data/history.db ...", flush=True)
    p = load_panel(db=DB, start="2005-01-01")
    C, cal = p["C"], p["cal"]
    print(f"  {len(cal)} sessions {cal[0].date()} -> {cal[-1].date()}, {C.shape[1]} stocks, "
          f"{len(p['events'])} split/bonus events, {len(p['delisted'])} stopped trading before the end")
    dc = cfg["data"]
    # refresh: the price cache is keyed by ticker only and may hold a shorter history
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / dc["cache_dir"], 24, refresh=True)["^CRSLDX"]["Close"]
    if n5.index[0] > pd.Timestamp("2006-01-01"):
        raise SystemExit(f"Nifty 500 history starts {n5.index[0].date()} - too short for 2006")
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    regime = (n5 > n5.rolling(200, min_periods=150).mean()).to_numpy() | n5.isna().to_numpy()
    unis = {"top500": liquid_universe(C, p["TO"]), "top200": liquid_universe(C, p["TO"], top=200)}
    sigs = {k: pit_signals(C, p["TO"], u) for k, u in unis.items()}
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(START)) & (rebal <= pd.Timestamp(END)) & (rebal < cal[-1])]
    ridx = [cal.get_loc(d) for d in rebal]
    elig = sigs["top500"][1].iloc[ridx].sum(axis=1)
    print(f"  eligible stocks per week: {elig.iloc[0]:.0f} (2006) -> {elig.iloc[-1]:.0f} (2019), median {elig.median():.0f}")
    i0, i1 = cal.searchsorted(pd.Timestamp(START)), len(cal)
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    costs = cfg["costs"]

    V = [("Gate 1 rule: W50 + 200d filter", "top500", 50, True),
         ("W50, no filter", "top500", 50, False),
         ("W20, no filter (Results 7)", "top500", 20, False),
         ("top 200 only, W50 + filter", "top200", 50, True)]
    windows = []
    for y in (1, 3):
        for s0 in pd.date_range(START, pd.Timestamp(END) - pd.DateOffset(years=y), freq="3MS"):
            a, b = cal.searchsorted(s0), cal.searchsorted(s0 + pd.DateOffset(years=y))
            windows.append((y, s0, a, b))

    def bcagr(a, b):
        return (n5.iloc[b - 1] / n5.iloc[a]) ** (248 / (b - a)) - 1

    rows, curves = [], {}
    for name, uni, ex, filt in V:
        print(f"  running {name} ...", flush=True)
        od = _orders(sigs[uni][0]["combo"], ridx)
        kw = dict(exit_rank=ex, regime=regime if filt else None)
        full = simulate(Oa, Cf, od, i0, i1, CAP, costs, **kw)
        curves[name] = pd.Series(full["equity"], index=cal[i0:i1])
        hd = np.array([t[0] for t in full["trades"]] or [20])

        def run(a, b, sd):
            return stats(simulate(Oa, Cf, od, a, b, CAP, costs, **kw,
                                  rng=None if sd is None else np.random.default_rng(sd), hold_dist=hd), CAP)
        st = stats(full, CAP)
        conc = concentration(full["trades"], C.columns)
        r = {"variant": name, "final_rs": full["equity"][-1], "cagr": st["cagr"], "max_dd": st["max_dd"],
             "random": float(np.median([run(i0, i1, sd)["cagr"] for sd in range(SEEDS)])),
             "trades_yr": st["trades_yr"], "win_rate": st["win_rate"], "charges_yr": st["charges_pct_yr"],
             "top10_share": conc["top10_pnl"] / conc["total_pnl"] if conc.get("total_pnl", 0) > 0 else np.nan}
        for y in (1, 3):
            res = [(run(a, b, None)["cagr"], float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)])),
                    bcagr(a, b)) for yy, _, a, b in windows if yy == y]
            res = np.array(res)
            r[f"w{y}_beat_random"] = (res[:, 0] > res[:, 1]).mean()
            r[f"w{y}_beat_n500"] = (res[:, 0] > res[:, 2]).mean()
            r[f"w{y}_worst"] = res[:, 0].min()
            r[f"w{y}_losing"] = (res[:, 0] < 0).mean()
        rows.append(r)
    R = pd.DataFrame(rows)

    # 75/25 mix with the Gate 1 rule, by calendar year and in the big falls
    idx = n5.iloc[i0:i1]
    jan = [k for k in range(1, len(idx)) if idx.index[k].year != idx.index[k - 1].year]
    g1 = curves[V[0][0]]
    mix = pd.Series(mix_equity(g1.values, idx.values, SHARE, jan), index=idx.index)
    Y = []

    def seg(x, a, b):
        x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
        return x.iloc[-1] / x.iloc[0] - 1, float((x / x.cummax() - 1).min())
    for yr in range(2006, 2020):
        a, b = f"{yr}-01-01", f"{yr}-12-31"
        Y.append({"period": str(yr) + (" (from Jul)" if yr == 2006 else ""),
                  **dict(zip(["n500", "n500_fall"], seg(idx, a, b))),
                  **dict(zip(["strategy", "strategy_fall"], seg(g1, a, b))),
                  **dict(zip(["mix", "mix_fall"], seg(mix, a, b)))})
    for lab, a, b in FALLS:
        Y.append({"period": lab, **dict(zip(["n500", "n500_fall"], seg(idx, a, b))),
                  **dict(zip(["strategy", "strategy_fall"], seg(g1, a, b))),
                  **dict(zip(["mix", "mix_fall"], seg(mix, a, b)))})
    Y = pd.DataFrame(Y)
    ms = path_stats(mix.values)

    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "longrun_variants.csv", index=False)
    Y.to_csv(out / "longrun_years.csv", index=False)
    pct = lambda v: "" if pd.isna(v) else f"{v:+.0%}"  # noqa: E731
    print(f"\nNifty 500 {START} -> {END}: {bcagr(i0, i1):+.1%}/yr, Rs 25k -> Rs {25_000 * idx.iloc[-1] / idx.iloc[0]:,.0f}, "
          f"worst fall {float((idx / idx.cummax() - 1).min()):.0%}")
    print(f"75/25 mix (Rs 1 lakh, Gate 1 rule): Rs {TOTAL * mix.iloc[-1]:,.0f}, {ms['cagr']:+.1%}/yr, worst fall {ms['max_dd']:.0%}")
    show = R.copy()
    for c in show.columns[2:]:
        if c != "trades_yr":
            show[c] = show[c].map(pct)
    show["final_rs"] = R["final_rs"].map("{:,.0f}".format)
    show["trades_yr"] = R["trades_yr"].map("{:.0f}".format)
    print("\n=== 2006-2019, Rs 25k strategy alone, after charges ===")
    print(show[["variant", "final_rs", "cagr", "random", "max_dd", "trades_yr", "win_rate", "charges_yr",
                "top10_share"]].to_string(index=False))
    print(show[["variant", "w1_beat_random", "w1_beat_n500", "w1_losing", "w1_worst",
                "w3_beat_random", "w3_beat_n500", "w3_losing", "w3_worst"]].to_string(index=False))
    print("\n=== By year and in the big falls: Nifty 500 | Gate 1 strategy | 75/25 mix (return, worst fall) ===")
    ys = Y.copy()
    for c in ys.columns[1:]:
        ys[c] = ys[c].map(pct)
    print(ys.to_string(index=False))
    print(f"\nCSV: {out}/longrun_variants.csv, longrun_years.csv")
