"""Gate 0 step 1, redone without survivorship bias: momentum on a point-in-time universe.

    python -m trader momentum-pit

Same signals, portfolio rules, charges and random control as momentum_study.py, but:
  - prices = NSE full bhavcopy (every stock that traded, incl. later collapsed / delisted),
    split/bonus adjusted from NSE's own PREV_CLOSE (see pit.py; dividends not added back);
  - universe on each date = top 500 stocks by 6-month median turnover AT THAT DATE, price >= Rs 10
    (a point-in-time stand-in for the Nifty 500), plus the same Rs 2 cr 20-day turnover floor.
Side by side, the same code runs on "today's Nifty 500" restricted to the same data and period:
the gap between the two is the survivorship bias of the first study.

Data starts 2020-01 and signals need 12 months, so: learn 2021-2023, check 2024+.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices
from .momentum_study import BENCH, N_POS, MIN_TURNOVER_CR, signal_test, simulate, stats, weekly_dates
from .pit import liquid_universe, load_panel

START = "2021-01-01"
CHECK = "2024-01-01"
SEEDS = 5
VARIANTS = [("mom12_1", 20, False), ("mom6_1", 20, False), ("combo", 20, False), ("mom12_1", 20, True)]


def pit_signals(C, TO, universe):
    """Same definitions as momentum_study.signals (no sector RS: no point-in-time sectors)."""
    Cf = C.ffill(limit=5)
    mom12 = Cf.shift(21) / Cf.shift(252) - 1
    mom6 = Cf.shift(21) / Cf.shift(126) - 1
    high52 = Cf / Cf.rolling(252, min_periods=240).max()
    turn20 = TO.rolling(20, min_periods=15).median()
    elig = universe & mom12.notna() & high52.notna() & C.notna() & (turn20 >= MIN_TURNOVER_CR)
    pr = [x.where(elig).rank(axis=1, pct=True) for x in (mom12, mom6, high52)]
    sig = {"mom12_1": mom12, "mom6_1": mom6, "combo": sum(pr) / 3}
    return {k: v.where(elig) for k, v in sig.items()}, elig


def _orders(s, rebal_idx):
    a = s.to_numpy()
    out = {}
    for i in rebal_idx:
        ok = ~np.isnan(a[i])
        out[i] = np.where(ok)[0][np.argsort(-a[i][ok], kind="stable")]
    return out


def _label(name, ex, nl):
    return f"{name} exit>{ex}" + (" NO-LOSS" if nl else "")


def concentration(trades, cols, top=10):
    """How much of the profit came from the best `top` trades."""
    if not trades:
        return {}
    pnl = np.array([t[5] for t in trades])
    order = np.argsort(-pnl)
    best = [(cols[trades[k][2]], round(trades[k][1] * 100)) for k in order[:5]]
    return {"trades": len(pnl), "total_pnl": pnl.sum(), "top10_pnl": pnl[order[:top]].sum(),
            "rest_pnl": pnl[order[top:]].sum(), "best5": best}


def run_momentum_pit(cfg):
    costs, capital = cfg["costs"], cfg["capital"]
    print("Loading point-in-time panel from data/market.db ...", flush=True)
    p = load_panel()
    O, C, TO, cal = p["O"], p["C"], p["TO"], p["cal"]
    days_per_sym = C.notna().sum()
    print(f"  {len(cal)} sessions {cal[0].date()} -> {cal[-1].date()}, {C.shape[1]} stocks, "
          f"{len(p['events'])} split/bonus events, {len(p['delisted'])} stopped trading before the end")
    full_days = (C.notna().sum(axis=1) >= 1000).mean()
    if full_days < 0.95:
        print(f"  !! only {full_days:.0%} of sessions have all stocks - is the backfill finished?")

    n500 = pd.read_csv(ROOT / "universes" / "nifty500_2026-11.csv")["Symbol"].str.strip()
    universes = {"point-in-time top500": liquid_universe(C, TO),
                 "today's Nifty 500": C.notna() & C.columns.isin(n500)[None, :]}

    dc = cfg["data"]
    bp = download_prices(list(BENCH.values()), dc["start"], ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    bench = {k: bp[t]["Close"].reindex(cal).ffill() for k, t in BENCH.items() if t in bp}

    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(START)) & (rebal < cal[-21])]
    rebal_idx = [cal.get_loc(d) for d in rebal]
    i_start, i_check, i_end = cal.searchsorted(pd.Timestamp(START)), cal.searchsorted(pd.Timestamp(CHECK)), len(cal)
    periods = {"learn 2021-23": (i_start, i_check), "check 2024+": (i_check, i_end), "full 2021+": (i_start, i_end)}
    windows = []
    for y in (1, 3):
        for s0 in pd.date_range(START, cal[-1] - pd.DateOffset(years=y), freq="3MS"):
            a, b = cal.searchsorted(s0), cal.searchsorted(s0 + pd.DateOffset(years=y))
            if b - a >= 200 * y:  # skip windows that fall into a hole in the data
                windows.append((y, s0, a, b))

    def bench_cagr(key, a, b):
        x = bench.get(key)
        if x is None or np.isnan(x.iloc[a]):
            return np.nan
        return (x.iloc[b - 1] / x.iloc[a]) ** (248 / (b - a)) - 1

    out_dir = ROOT / cfg["backtest"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    Oa, Cf = O.to_numpy(), C.ffill().to_numpy()
    ST, P, W, CONC = [], [], [], []
    for uname, uni in universes.items():
        sig, elig = pit_signals(C, TO, uni)
        print(f"\n######## Universe: {uname}  (avg eligible per week: {elig.loc[rebal].sum(axis=1).mean():.0f})")
        st, _ = signal_test(sig, O, C, rebal)
        st = st.rename(index={"learn 2016-23": "learn 2021-23"})
        st["universe"] = uname
        ST.append(st)
        show = st.drop(columns="universe").copy()
        show["top_minus_all"] = show["top_minus_all"].map("{:+.2%}".format)
        show["top_beats_pct"] = show["top_beats_pct"].map("{:.0%}".format)
        show["rank_ic"] = show["rank_ic"].map("{:+.3f}".format)
        print("A. Signal test: top decile minus average eligible stock, forward return before costs")
        print(show.to_string())

        orders = {k: _orders(s, rebal_idx) for k, s in sig.items()}
        for name, ex, nl in VARIANTS:
            label = _label(name, ex, nl)
            print(f"  running {label} ...", flush=True)
            od = orders[name]
            full = simulate(Oa, Cf, od, i_start, i_end, capital, costs, exit_rank=ex, no_loss=nl)
            CONC.append({"universe": uname, "variant": label, **concentration(full["trades"], C.columns)})
            hold_dist = np.array([t[0] for t in full["trades"]] or [20])

            def run(a, b, sd):
                return stats(simulate(Oa, Cf, od, a, b, capital, costs, exit_rank=ex, no_loss=nl,
                                      rng=None if sd is None else np.random.default_rng(sd),
                                      hold_dist=hold_dist), capital)
            for kind, seeds in (("rule", [None]), ("random", range(SEEDS))):
                for pname, (a, b) in periods.items():
                    r = pd.DataFrame([run(a, b, sd) for sd in seeds]).median().to_dict()
                    P.append({"universe": uname, "variant": label, "kind": kind, "period": pname, **r})
                for y, s0, a, b in windows:
                    W.append({"universe": uname, "variant": label, "kind": kind, "years": y, "start": s0.date(),
                              "cagr": float(np.median([run(a, b, sd)["cagr"] for sd in seeds]))})
    for pname, (a, b) in periods.items():
        for k in bench:
            x = bench[k].iloc[a:b]
            P.append({"universe": "-", "variant": f"BENCH {k}", "kind": "bench", "period": pname,
                      "cagr": bench_cagr(k, a, b), "max_dd": float((x / x.cummax() - 1).min())})
    for y, s0, a, b in windows:
        for k in bench:
            W.append({"universe": "-", "variant": f"BENCH {k}", "kind": "bench", "years": y,
                      "start": s0.date(), "cagr": bench_cagr(k, a, b)})

    ST, P, W, CONC = pd.concat(ST), pd.DataFrame(P), pd.DataFrame(W), pd.DataFrame(CONC)
    ST.to_csv(out_dir / "mompit_signal_test.csv")
    P.to_csv(out_dir / "mompit_portfolio.csv", index=False)
    W.to_csv(out_dir / "mompit_windows.csv", index=False)
    CONC.to_csv(out_dir / "mompit_concentration.csv", index=False)
    p["events"].to_csv(out_dir / "mompit_split_events.csv", index=False)

    print(f"\n=== B. Portfolio: {N_POS} stocks, Rs {capital:,}, weekly rotation, after all charges "
          "(dividends not included) ===")
    for pname in periods:
        t = P[P["period"] == pname].drop(columns="period").set_index(["universe", "variant", "kind"])
        for c in ["cagr", "max_dd", "win_rate", "stuck_pct"]:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        for c in ["trades_yr", "median_hold"]:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        print(f"\n--- {pname} ---")
        print(t.drop(columns="charges_pct_yr").to_string())

    print("\n--- Rolling windows (new start every quarter): rule vs random vs Nifty 500 ---")
    summ = []
    n5 = W[W["variant"] == "BENCH nifty500"].set_index(["years", "start"])["cagr"]
    for (uname, label), g in W[W["kind"] != "bench"].groupby(["universe", "variant"], sort=False):
        piv = g.pivot_table(index=["years", "start"], columns="kind", values="cagr")
        for y in (1, 3):
            if y not in piv.index.get_level_values(0):
                continue
            r, rnd = piv.loc[y, "rule"], piv.loc[y, "random"]
            b = n5.loc[y].reindex(r.index)
            summ.append({"universe": uname, "variant": label, "years": y, "windows": len(r),
                         "median": r.median(), "worst": r.min(), "losing": (r < 0).mean(),
                         "beats_random": (r > rnd).mean(), "median_vs_random": (r - rnd).median(),
                         "beats_nifty500": (r > b).mean(), "median_vs_nifty500": (r - b).median()})
    S = pd.DataFrame(summ)
    S.to_csv(out_dir / "mompit_rolling_summary.csv", index=False)
    show = S.copy()
    for c in show.columns[4:]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    print(show.to_string(index=False))

    print("\n--- Where the profit came from (full period, single path) ---")
    for r in CONC.itertuples():
        print(f"  {r.universe:22s} {r.variant:24s} {r.trades:4d} trades | total Rs {r.total_pnl:>10,.0f} | "
              f"best 10 Rs {r.top10_pnl:>10,.0f} | other Rs {r.rest_pnl:>10,.0f} | best: {r.best5}")

    if "mom30_etf" in bench:
        e = bench["mom30_etf"].dropna()
        a = cal.searchsorted(e.index[0])
        print(f"\n--- vs Nifty 200 Momentum 30 ETF since {e.index[0].date()} ---")
        print(f"  ETF {bench_cagr('mom30_etf', a, i_end):+.1%} | Nifty 500 {bench_cagr('nifty500', a, i_end):+.1%}")
        for uname, uni in universes.items():
            sig, _ = pit_signals(C, TO, uni)
            r = stats(simulate(Oa, Cf, _orders(sig["mom12_1"], rebal_idx), a, i_end, capital, costs), capital)
            print(f"  mom12_1 exit>20 on {uname:22s} {r['cagr']:+.1%}  maxDD {r['max_dd']:.1%}")
    print(f"\nCSV files in {out_dir}/mompit_*.csv. Returns before 20% STCG and without dividends.")
