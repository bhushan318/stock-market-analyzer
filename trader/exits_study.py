"""Exit rules with standing orders: targets and stops checked against each day's HIGH and LOW.

    python -m trader exits

Owner's point (2026-11-27): a real target/stop order (Zerodha GTT) sells the moment the price
touches it during the day; checking only the close (Results 8) misses those exits. NSE's daily
file has each day's high and low, so here target/stop are simulated as standing orders:
target fills at the target price (or a higher open), stop at the stop price (or a lower open) minus
slippage, from the day after the buy; if both are touched on one day the stop is assumed first.
Unknowable from daily data: the order within a day, so that assumption is the cautious one.

Every variant uses the Gate 1 entry rule (combo, weekly, 5 stocks, no new buys while Nifty 500 is
below its 200-day average, point-in-time top 500, Rs 20k) and differs only in the exit:
  rank     sell when out of the top 50 (decided Friday close, sold Monday open) - Gate 1 rule
  stop     -2 x 20-day volatility x sqrt(20) below the buy price (about -15% for a typical stock)
  target   +X% after all charges
  noloss   never sell below the buy price (owner's original preference)
Random control = random picks under the same exit rules. All charges, before 20% STCG.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .momentum_pit import SEEDS, _orders, bench_cagr, concentration, load_context, pit_signals
from .momentum_study import simulate, stats

UNI = "point-in-time top500"
CAP = 20_000
NO_RANK = 10**9
# name, exit_rank, target, stop_k, no_loss, intraday
V = [("rank exit (Gate 1 rule)", 50, None, None, False, False),
     ("rank + stop, close-checked", 50, None, 2.0, False, False),
     ("rank + stop, intraday", 50, None, 2.0, False, True),
     ("rank + target 15%, intraday", 50, 0.15, None, False, True),
     ("rank + target 25%, intraday", 50, 0.25, None, False, True),
     ("target 10% + stop, close-checked", NO_RANK, 0.10, 2.0, False, False),
     ("target 10% + stop, intraday", NO_RANK, 0.10, 2.0, False, True),
     ("target 15% + stop, close-checked", NO_RANK, 0.15, 2.0, False, False),
     ("target 15% + stop, intraday", NO_RANK, 0.15, 2.0, False, True),
     ("target 20% + stop, intraday", NO_RANK, 0.20, 2.0, False, True),
     ("no-loss + target 5%, intraday", NO_RANK, 0.05, None, True, True),
     ("no-loss + target 10%, intraday", NO_RANK, 0.10, None, True, True),
     ("no-loss + rank + target 10%, intraday", 50, 0.10, None, True, True)]


def run_exits(cfg):
    ctx = load_context(cfg)
    p, bench = ctx["p"], ctx["bench"]
    costs, C = cfg["costs"], p["C"]
    periods, windows, i_start, i_end = ctx["periods"], ctx["windows"], ctx["i_start"], ctx["i_end"]
    sig, _ = pit_signals(C, p["TO"], ctx["universes"][UNI])
    od = _orders(sig["combo"], ctx["rebal_idx"])
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    HL = (p["H"].to_numpy(), p["L"].to_numpy())
    ret = C / C.ffill(limit=5).shift(1) - 1
    vol = ret.rolling(20, min_periods=15).std().to_numpy()
    n5 = bench["nifty500"]
    regime = (n5 > n5.rolling(200, min_periods=150).mean()).to_numpy() | n5.isna().to_numpy()

    rows, W = [], []
    for name, ex, tgt, sk, nl, intra in V:
        print(f"  running {name} ...", flush=True)
        kw = dict(exit_rank=ex, target=tgt, stop_k=sk, no_loss=nl, vol=vol, regime=regime,
                  intraday=HL if intra else None)
        full = simulate(Oa, Cf, od, i_start, i_end, CAP, costs, **kw)
        hold_dist = np.array([t[0] for t in full["trades"]] or [20])
        conc = concentration(full["trades"], C.columns)

        def run(a, b, sd):
            return stats(simulate(Oa, Cf, od, a, b, CAP, costs, **kw,
                                  rng=None if sd is None else np.random.default_rng(sd),
                                  hold_dist=hold_dist), CAP)
        st = stats(full, CAP)
        r = {"variant": name, "final_rs": full["equity"][-1], "full_cagr": st["cagr"], "max_dd": st["max_dd"],
             "trades_yr": st["trades_yr"], "median_hold": st["median_hold"], "win_rate": st["win_rate"],
             "stuck": st["stuck_pct"], "charges_yr": st["charges_pct_yr"],
             "top10_share": conc["top10_pnl"] / conc["total_pnl"] if conc.get("total_pnl", 0) > 0 else np.nan,
             "open_underwater_end": full["open_underwater"]}
        r["full_random"] = float(np.median([run(i_start, i_end, sd)["cagr"] for sd in range(SEEDS)]))
        for pname in ("learn 2021-23", "check 2024+"):
            a, b = periods[pname]
            key = pname.split()[0]
            r[f"{key}_cagr"] = run(a, b, None)["cagr"]
            r[f"{key}_random"] = float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)]))
        for y, s0, a, b in windows:
            s = run(a, b, None)
            W.append({"variant": name, "years": y, "start": s0.date(), "rule": s["cagr"], "rule_dd": s["max_dd"],
                      "random": float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)])),
                      "nifty500": bench_cagr(bench, "nifty500", a, b)})
        rows.append(r)

    R, W = pd.DataFrame(rows), pd.DataFrame(W)
    for y in (1, 3):
        g = W[W.years == y].groupby("variant", sort=False)
        R[f"w{y}_beat_random"] = R.variant.map(g.apply(lambda d: (d.rule > d.random).mean()))
        R[f"w{y}_beat_n500"] = R.variant.map(g.apply(lambda d: (d.rule > d.nifty500).mean()))
        R[f"w{y}_worst"] = R.variant.map(g["rule"].min())
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "exits_variants.csv", index=False)
    W.to_csv(out / "exits_windows.csv", index=False)

    print(f"\nNifty 500 CAGR: full {bench_cagr(bench, 'nifty500', i_start, i_end):+.1%}, "
          f"check {bench_cagr(bench, 'nifty500', *periods['check 2024+']):+.1%}")
    show = R.copy()
    for c in show.columns:
        if c in ("variant", "final_rs", "trades_yr", "median_hold", "open_underwater_end"):
            continue
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    show["final_rs"] = show["final_rs"].map("{:,.0f}".format)
    show["trades_yr"] = show["trades_yr"].map("{:.0f}".format)
    show["median_hold"] = show["median_hold"].map("{:.0f}".format)
    print("\n=== Exits (entry = Gate 1 rule, Rs 20k), after charges ===")
    print(show[["variant", "final_rs", "full_cagr", "full_random", "learn_cagr", "check_cagr", "check_random",
                "max_dd", "trades_yr", "median_hold", "win_rate", "stuck", "charges_yr"]].to_string(index=False))
    print(show[["variant", "w1_beat_random", "w1_beat_n500", "w1_worst", "w3_beat_random", "w3_beat_n500",
                "w3_worst", "top10_share", "open_underwater_end"]].to_string(index=False))
    print(f"\nCSV: {out}/exits_variants.csv, exits_windows.csv")
