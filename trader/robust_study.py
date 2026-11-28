"""Gate 0 robustness: does "weekly combo + market filter" (Results 8) survive small changes?

    python -m trader robust

No new tuning - each line changes ONE thing from the base and we look for a plateau, not a peak.
Base: combo ranking, weekly, sell when out of the top 50, no new buys while Nifty 500 is below its
200-day average, 5 positions, Rs 20k, point-in-time top-500 universe.

  A. neighbours   exit rank 30/40/75, filter 150/250 days (and none), 3-4 positions,
                  decide on Wednesday instead of Friday (timing luck)
  B. universe     top 200 by turnover only (large caps) vs ranks 201-500 (mid/small)
  C. core+satellite  w in the strategy, (1-w) in a Nifty 500 index ETF (0.2%/yr fee),
                  rebalanced back to w every January; total Rs 20k and Rs 1 lakh. The strategy part is
                  simulated at its own smaller capital, so the flat charges bite harder.
                  Judged against the owner's limit: a fall of at most 25%.

Same random control (same rules, random picks) and rolling 1y/3y windows as combo_study.
All charges, no dividends (index too), before 20% STCG. Survivorship-free universe.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .momentum_pit import SEEDS, _orders, bench_cagr, load_context, pit_signals
from .momentum_study import simulate, stats
from .pit import liquid_universe

BASE = dict(uni="top500", day="Fri", exit=50, ma=200, n=5, cap=20_000)
V = [("base", {}),
     ("exit 30", {"exit": 30}), ("exit 40", {"exit": 40}), ("exit 75", {"exit": 75}),
     ("filter 150d", {"ma": 150}), ("filter 250d", {"ma": 250}), ("no filter", {"ma": None}),
     ("4 positions", {"n": 4}), ("3 positions", {"n": 3}),
     ("decide Wed", {"day": "Wed"}),
     ("top200 only", {"uni": "top200"}), ("ranks 201-500", {"uni": "mid"}),
     ("base 1L", {"cap": 100_000})]
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
ETF_FEE = 0.002
MIN_SLOT = 1500   # simulate() skips buys below this slot size
LIMIT_DD = -0.25  # the owner's pain limit


def mix_equity(strat, etf, w, rebal_at=(), fee=ETF_FEE):
    """Value (starting at 1) of w in `strat` + (1-w) in `etf`, both daily value series of equal
    length; reset to w at each position in `rebal_at` (relative to the start)."""
    strat, etf = np.asarray(strat, float), np.asarray(etf, float)
    rs, re = strat[1:] / strat[:-1], etf[1:] / etf[:-1] * (1 - fee) ** (1 / 248)
    a, b, out, reb = w, 1 - w, [1.0], set(rebal_at)
    for k in range(len(rs)):
        a, b = a * rs[k], b * re[k]
        t = a + b
        if k + 1 in reb:
            a, b = w * t, (1 - w) * t
        out.append(t)
    return np.array(out)


def path_stats(e):
    years = len(e) / 248
    return {"cagr": (e[-1] / e[0]) ** (1 / years) - 1, "max_dd": float((e / np.maximum.accumulate(e) - 1).min())}


def run_robust(cfg):
    ctx = load_context(cfg)
    p, bench, cal = ctx["p"], ctx["bench"], ctx["p"]["cal"]
    costs, C = cfg["costs"], ctx["p"]["C"]
    periods, windows, i_start, i_end = ctx["periods"], ctx["windows"], ctx["i_start"], ctx["i_end"]
    top500 = ctx["universes"]["point-in-time top500"]
    top200 = liquid_universe(C, p["TO"], top=200)
    unis = {"top500": top500, "top200": top200, "mid": top500 & ~top200}
    sigs = {k: pit_signals(C, p["TO"], u)[0]["combo"] for k, u in unis.items()}
    days = {"Fri": ctx["rebal_idx"], "Wed": [i - 2 for i in ctx["rebal_idx"]]}
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    ret = C / C.ffill(limit=5).shift(1) - 1
    vol = ret.rolling(20, min_periods=15).std().to_numpy()
    n5 = bench["nifty500"]
    regimes = {m: (n5 > n5.rolling(m, min_periods=int(m * 0.75)).mean()).to_numpy() | n5.isna().to_numpy()
               for m in (150, 200, 250)}
    orders = {}

    def setup(over):
        s = {**BASE, **over}
        key = (s["uni"], s["day"])
        if key not in orders:
            orders[key] = _orders(sigs[s["uni"]], days[s["day"]])
        kw = dict(n=s["n"], exit_rank=s["exit"], regime=None if s["ma"] is None else regimes[s["ma"]], vol=vol)
        return s, orders[key], kw

    rows, W, full_eq = [], [], {}
    for name, over in V:
        print(f"  running {name} ...", flush=True)
        s, od, kw = setup(over)
        cap = s["cap"]
        full = simulate(Oa, Cf, od, i_start, i_end, cap, costs, **kw)
        full_eq[name] = full["equity"]
        hold_dist = np.array([t[0] for t in full["trades"]] or [20])

        def run(a, b, sd):
            return stats(simulate(Oa, Cf, od, a, b, cap, costs, **kw,
                                  rng=None if sd is None else np.random.default_rng(sd),
                                  hold_dist=hold_dist), cap)
        r = {"variant": name, "final_rs": full["equity"][-1], "capital": cap}
        for pname, (a, b) in periods.items():
            st = run(a, b, None)
            key = pname.split()[0]
            r[f"{key}_cagr"] = st["cagr"]
            r[f"{key}_random"] = float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)]))
            if key == "full":
                r.update(max_dd=st["max_dd"], trades_yr=st["trades_yr"], charges_yr=st["charges_pct_yr"])
        for y, s0, a, b in windows:
            st = run(a, b, None)
            W.append({"variant": name, "years": y, "start": s0.date(), "rule": st["cagr"], "rule_dd": st["max_dd"],
                      "random": float(np.median([run(a, b, sd)["cagr"] for sd in range(SEEDS)])),
                      "nifty500": bench_cagr(bench, "nifty500", a, b)})
        rows.append(r)
    R, W = pd.DataFrame(rows), pd.DataFrame(W)
    for y in (1, 3):
        g = W[W.years == y].groupby("variant", sort=False)
        R[f"w{y}_beat_random"] = R.variant.map(g.apply(lambda d: (d.rule > d.random).mean()))
        R[f"w{y}_beat_n500"] = R.variant.map(g.apply(lambda d: (d.rule > d.nifty500).mean()))
        R[f"w{y}_worst"] = R.variant.map(g["rule"].min())
    g1 = W[W.years == 1].groupby("variant", sort=False)
    R["w1_dd_over_25"] = R.variant.map(g1.apply(lambda d: (d.rule_dd < LIMIT_DD).mean()))

    # ---- C. core + satellite
    etf = n5.to_numpy()[i_start:i_end]
    jan = [k for k in range(1, len(etf)) if cal[i_start + k].year != cal[i_start + k - 1].year]
    i_check = periods["check 2024+"][0]
    M = []
    for total in (20_000, 100_000):
        for w in MIXES:
            sc = w * total
            if 0 < w and sc / BASE["n"] < MIN_SLOT:
                M.append({"total": total, "strategy_share": w, "note": "too small for 5 slots"})
                continue
            if w == 0:
                strat = etf
            else:
                s, od, kw = setup({"cap": sc})
                strat = simulate(Oa, Cf, od, i_start, i_end, sc, costs, **kw)["equity"]
            e = mix_equity(strat, etf, w, jan)
            full, chk = path_stats(e), path_stats(e[i_check - i_start:])
            w1 = [path_stats(e[a - i_start:b - i_start]) for y, _, a, b in windows if y == 1]
            w3 = [path_stats(e[a - i_start:b - i_start]) for y, _, a, b in windows if y == 3]
            M.append({"total": total, "strategy_share": w, "final_rs": total * e[-1], "full_cagr": full["cagr"],
                      "check_cagr": chk["cagr"], "max_dd": full["max_dd"],
                      "w1_losing": np.mean([x["cagr"] < 0 for x in w1]), "w1_worst": min(x["cagr"] for x in w1),
                      "w1_dd_over_25": np.mean([x["max_dd"] < LIMIT_DD for x in w1]),
                      "w3_worst": min(x["cagr"] for x in w3)})
    M = pd.DataFrame(M)

    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "robust_variants.csv", index=False)
    W.to_csv(out / "robust_windows.csv", index=False)
    M.to_csv(out / "robust_mixes.csv", index=False)

    print(f"\nNifty 500 CAGR: full {bench_cagr(bench, 'nifty500', i_start, i_end):+.1%}, "
          f"check {bench_cagr(bench, 'nifty500', *periods['check 2024+']):+.1%}")

    def pct(df, cols):
        df = df.copy()
        for c in cols:
            if c in df:
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
        if "final_rs" in df:
            df["final_rs"] = df["final_rs"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
        return df
    rc = ["full_cagr", "full_random", "learn_cagr", "learn_random", "check_cagr", "check_random", "max_dd",
          "charges_yr", "w1_beat_random", "w1_beat_n500", "w1_worst", "w1_dd_over_25",
          "w3_beat_random", "w3_beat_n500", "w3_worst"]
    print("\n=== A/B. One change at a time from the base (W50 + 200d filter, 5 positions, Rs 20k) ===")
    print(pct(R, rc)[["variant", "final_rs", "full_cagr", "full_random", "learn_cagr", "check_cagr",
                      "check_random", "max_dd", "charges_yr"]].to_string(index=False))
    print(pct(R, rc)[["variant", "w1_beat_random", "w1_beat_n500", "w1_worst", "w1_dd_over_25",
                      "w3_beat_random", "w3_beat_n500", "w3_worst"]].to_string(index=False))
    print("\n=== C. Core + satellite (rest in a Nifty 500 ETF, rebalanced each January) ===")
    mc = ["strategy_share", "full_cagr", "check_cagr", "max_dd", "w1_losing", "w1_worst", "w1_dd_over_25", "w3_worst"]
    print(pct(M, mc).fillna("").to_string(index=False))
    print(f"\nCSV: {out}/robust_variants.csv, robust_windows.csv, robust_mixes.csv")
