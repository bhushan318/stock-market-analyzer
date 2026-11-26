"""Gate 0 follow-up: can the momentum "combo" ranking (Results 7) be made cheaper and calmer?

    python -m trader combo

combo = average percentile rank of 12-1 month return, 6-1 month return and closeness to the
52-week high, on the point-in-time top-500 universe (no survivorship bias). Four questions:

  A. turnover   weekly vs monthly ranking; sell when out of the top 20 vs top 50
  B. capital    the same at Rs 20k and Rs 1 lakh (the flat Rs 15.93 DP charge per sell matters less)
  C. market     no new buys while Nifty 500 is below its 200-day average ("block"),
                or also sell everything then ("exit")
  D. exits      rank exit vs never-sell-at-loss vs rank exit + volatility stop
                vs +15% target + volatility stop (no rank exit)

Every variant is compared with random picks from the same eligible stocks under the *same* rules
(same rebalance days, regime filter, stops/targets; holding periods drawn from the rule's own
trades) and with Nifty 500 buy-and-hold, on the full period, learn 2021-23, check 2024+ and
rolling 1y/3y windows. All charges included, dividends not, before 20% STCG.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .momentum_pit import SEEDS, _orders, bench_cagr, concentration, load_context, pit_signals
from .momentum_study import stats, simulate

UNI = "point-in-time top500"
STOP_K = 2.0      # stop = buy price x (1 - 2 x daily vol x sqrt(20)): about -15% for a 1.7%-vol stock
TARGET = 0.15     # +15% after charges

# name, ranking frequency, exit rank, capital, regime, exit rule
V = []
for cap in (20_000, 100_000):
    for freq in ("W", "M"):
        for ex in (20, 50):
            V.append((f"{freq}{ex} {cap // 1000}k", freq, ex, cap, None, "rank"))
for freq in ("W", "M"):
    for reg in ("block", "exit"):
        V.append((f"{freq}50 20k regime-{reg}", freq, 50, 20_000, reg, "rank"))
    for rule in ("noloss", "stop", "target+stop"):
        V.append((f"{freq}50 20k {rule}", freq, 50, 20_000, None, rule))


def monthly(rebal_idx, cal):
    """Last weekly rebalance day of each calendar month."""
    s = pd.Series(rebal_idx, index=cal[rebal_idx])
    return list(s.groupby([s.index.year, s.index.month]).max().values)


def run_combo(cfg):
    ctx = load_context(cfg)
    p, bench, cal = ctx["p"], ctx["bench"], ctx["p"]["cal"]
    costs = cfg["costs"]
    C = p["C"]
    sig, elig = pit_signals(C, p["TO"], ctx["universes"][UNI])
    print(f"Universe {UNI}: avg eligible per week {elig.loc[ctx['rebal']].sum(axis=1).mean():.0f}")
    rebal = {"W": ctx["rebal_idx"], "M": monthly(ctx["rebal_idx"], cal)}
    orders = {f: _orders(sig["combo"], idx) for f, idx in rebal.items()}
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    ret = C / C.ffill(limit=5).shift(1) - 1
    vol = ret.rolling(20, min_periods=15).std().to_numpy()
    n5 = bench["nifty500"]
    regime = (n5 > n5.rolling(200, min_periods=150).mean()).to_numpy() | n5.isna().to_numpy()
    print(f"Nifty 500 above its 200-day average on {regime[ctx['i_start']:].mean():.0%} of days since 2021")

    periods, windows, i_start, i_end = ctx["periods"], ctx["windows"], ctx["i_start"], ctx["i_end"]
    rows, W = [], []
    for name, freq, ex, cap, reg, rule in V:
        print(f"  running {name} ...", flush=True)
        kw = dict(exit_rank=10**9 if rule == "target+stop" else ex, no_loss=rule == "noloss",
                  regime=None if reg is None else regime, regime_exit=reg == "exit",
                  target=TARGET if rule == "target+stop" else None,
                  stop_k=STOP_K if rule in ("stop", "target+stop") else None, vol=vol)
        od = orders[freq]
        full = simulate(Oa, Cf, od, i_start, i_end, cap, costs, **kw)
        conc = concentration(full["trades"], C.columns)
        hold_dist = np.array([t[0] for t in full["trades"]] or [20])

        def run(a, b, sd):
            return stats(simulate(Oa, Cf, od, a, b, cap, costs, **kw,
                                  rng=None if sd is None else np.random.default_rng(sd),
                                  hold_dist=hold_dist), cap)
        r = {"variant": name, "capital": cap, "final_rs": full["equity"][-1]}
        for pname, (a, b) in periods.items():
            st = run(a, b, None)
            rnd = pd.DataFrame([run(a, b, sd) for sd in range(SEEDS)]).median()
            key = pname.split()[0]
            r[f"{key}_cagr"], r[f"{key}_random"] = st["cagr"], rnd["cagr"]
            if key == "full":
                r.update(max_dd=st["max_dd"], trades_yr=st["trades_yr"], median_hold=st["median_hold"],
                         win_rate=st["win_rate"], stuck=st["stuck_pct"], charges_yr=st["charges_pct_yr"])
        r["top10_share"] = conc["top10_pnl"] / conc["total_pnl"] if conc.get("total_pnl", 0) > 0 else np.nan
        for y, s0, a, b in windows:
            W.append({"variant": name, "years": y, "start": s0.date(), "rule": run(a, b, None)["cagr"],
                      "random": float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)])),
                      "nifty500": bench_cagr(bench, "nifty500", a, b)})
        rows.append(r)

    R, W = pd.DataFrame(rows), pd.DataFrame(W)
    for y in (1, 3):
        g = W[W.years == y].groupby("variant", sort=False)
        R[f"w{y}_beat_random"] = R.variant.map(g.apply(lambda d: (d.rule > d.random).mean()))
        R[f"w{y}_beat_n500"] = R.variant.map(g.apply(lambda d: (d.rule > d.nifty500).mean()))
        R[f"w{y}_vs_n500"] = R.variant.map(g.apply(lambda d: (d.rule - d.nifty500).median()))
        R[f"w{y}_worst"] = R.variant.map(g["rule"].min())
        R[f"w{y}_losing"] = R.variant.map(g.apply(lambda d: (d.rule < 0).mean()))
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "combo_variants.csv", index=False)
    W.to_csv(out / "combo_windows.csv", index=False)

    i_s = ctx["i_start"]
    print(f"\nNifty 500 CAGR: full {bench_cagr(bench, 'nifty500', i_s, i_end):+.1%}, "
          f"learn {bench_cagr(bench, 'nifty500', *periods['learn 2021-23']):+.1%}, "
          f"check {bench_cagr(bench, 'nifty500', *periods['check 2024+']):+.1%}; "
          f"max DD {float((n5.iloc[i_s:] / n5.iloc[i_s:].cummax() - 1).min()):.1%}")
    show = R.copy()
    pct = [c for c in show.columns if c.endswith(("cagr", "random", "_dd", "rate", "stuck", "charges_yr",
                                                  "share", "beat_random", "beat_n500", "vs_n500", "worst", "losing"))]
    for c in pct:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}" if "vs" in c else f"{v:.0%}")
    show["final_rs"] = show["final_rs"].map("{:,.0f}".format)
    show["trades_yr"] = show["trades_yr"].map("{:.0f}".format)
    cols1 = ["variant", "final_rs", "full_cagr", "full_random", "learn_cagr", "learn_random", "check_cagr",
             "check_random", "max_dd", "trades_yr", "charges_yr", "stuck", "top10_share"]
    cols2 = ["variant", "w1_beat_random", "w1_beat_n500", "w1_losing", "w1_worst",
             "w3_beat_random", "w3_beat_n500", "w3_vs_n500", "w3_worst"]
    print("\n=== Combo variants: CAGR rule vs random (same rules), after charges ===")
    print(show[cols1].to_string(index=False))
    print("\n=== Rolling windows (new start every quarter) ===")
    print(show[cols2].to_string(index=False))
    print(f"\nCSV: {out}/combo_variants.csv, combo_windows.csv. Survivorship-free universe; "
          "dividends not included; before 20% STCG.")
