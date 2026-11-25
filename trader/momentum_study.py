"""Gate 0 step 1: momentum / relative strength on Nifty 500 stocks.

    python -m trader momentum

Two questions, answered separately:
  A. Signal test (no portfolio): every week rank the eligible stocks by each signal and measure
     the forward 5/10/20-day return of the top decile vs the average eligible stock (entry at the
     next open, exit at the close h days later, before costs). Rank IC = Spearman correlation of
     signal vs forward return across stocks, averaged over weeks.
  B. Portfolio test: 5 stocks, weekly rotation, Rs capital from config, all charges + DP charge +
     slippage. Buy the best-ranked stocks into free slots; sell a holding when it falls out of the
     top `exit_rank` (a buffer that cuts turnover). Controls:
       - RANDOM: same eligible set, random picks, each held for a holding period drawn from the
         strategy's own holding periods (same turnover => same charges), sold at a weekly rebalance.
       - Nifty 500 index (^CRSLDX, price index: excludes ~1.3%/yr dividends) and NIFTYBEES
         (dividend-adjusted) buy-and-hold.
       - Nifty 200 Momentum 30 ETF (MOMENTUM.NS, listed Aug 2022) where its history allows.

Learn on 2016-2023, check on 2024+. All parameters are fixed up front (classic textbook choices),
nothing was tuned on the check period.

Survivorship bias: the universe is TODAY's Nifty 500. Stocks that rose strongly in the past are
over-represented (they rose INTO the index) and that flatters momentum in particular, because
momentum buys exactly the stocks that were rising. Random picks share the general bias but not
the momentum-specific part, so even a gap over random is an upper bound on the real edge.
"""

import math

import numpy as np
import pandas as pd

from . import ROOT
from .costs import buy_charges, sell_charges
from .data import download_prices, wide

START = "2016-01-01"
CHECK = "2024-01-01"
HORIZONS = (5, 10, 20)
SEEDS = 5
N_POS = 5
MIN_TURNOVER_CR = 2.0
BENCH = {"nifty500": "^CRSLDX", "niftybees": "NIFTYBEES.NS", "mom30_etf": "MOMENTUM.NS"}


# ---------------------------------------------------------------- signals (no look-ahead)

def signals(C, volume, industry):
    """date x ticker frames. Every value at date t uses closes up to and including t only."""
    Cf = C.ffill(limit=5)
    mom12 = Cf.shift(21) / Cf.shift(252) - 1
    mom6 = Cf.shift(21) / Cf.shift(126) - 1
    high52 = Cf / Cf.rolling(252, min_periods=240).max()
    turnover = (volume * C).rolling(20, min_periods=15).median() / 1e7
    elig = mom12.notna() & high52.notna() & C.notna() & (turnover >= MIN_TURNOVER_CR)
    m6e = mom6.where(elig)
    ind = industry.reindex(C.columns).fillna("Other")
    sect_med = m6e.T.groupby(ind).transform("median").T
    sector_rs = m6e - sect_med
    pr = [x.where(elig).rank(axis=1, pct=True) for x in (mom12, mom6, high52)]
    combo = sum(pr) / 3
    sig = {"mom12_1": mom12, "mom6_1": mom6, "near_52w_high": high52,
           "sector_rs_6m": sector_rs, "combo": combo}
    return {k: v.where(elig) for k, v in sig.items()}, elig


def weekly_dates(cal):
    """Last trading day of each ISO week."""
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex(s.groupby([cal.isocalendar().year, cal.isocalendar().week]).max().values)


# ---------------------------------------------------------------- A. signal test

def signal_test(sig, O, C, rebal):
    rows = []
    for h in HORIZONS:
        fwd = C.shift(-h) / O.shift(-1) - 1
        for name, s in sig.items():
            for d in rebal:
                x = s.loc[d].dropna()
                f = fwd.loc[d].reindex(x.index)
                ok = f.notna()
                x, f = x[ok], f[ok]
                if len(x) < 50:
                    continue
                top = f[x >= x.quantile(0.9)]
                rows.append({"h": h, "signal": name, "date": d,
                             "top_minus_all": top.mean() - f.mean(),
                             "top": top.mean(), "all": f.mean(),
                             "ic": x.rank().corr(f.rank())})
    df = pd.DataFrame(rows)
    df["period"] = np.where(df["date"] < pd.Timestamp(CHECK), "learn 2016-23", "check 2024+")
    g = df.groupby(["h", "signal", "period"], sort=False)
    out = g.agg(top_minus_all=("top_minus_all", "mean"), top_beats_pct=("top_minus_all", lambda v: (v > 0).mean()),
                rank_ic=("ic", "mean"), weeks=("ic", "size"))
    return out, df


# ---------------------------------------------------------------- B. portfolio simulator

def simulate(O, C, order, i0, i1, capital, costs, n=N_POS, exit_rank=20, no_loss=False,
             rng=None, hold_dist=None, stuck_days=60, min_val=1500):
    """Weekly rotation. O, C: numpy (T x N); C forward-filled for marking.

    order: {day index -> np.array of column indices, best first (eligible only)} on rebalance days.
    Random mode (rng + hold_dist given): picks are random eligible stocks, each held for a period
    drawn from hold_dist and then sold at the next rebalance.
    Orders decided at a rebalance close fill at the next open; a missing open delays the order.
    """
    slip = costs["slippage"]
    rand = rng is not None
    cash = float(capital)
    pos = {}  # j -> [shares, cost_basis, buy_i, hold_days]
    sells, buys = set(), []
    eq, stuck, trades = [], [], []
    charges = 0.0
    prev_equity = cash
    for i in range(i0, i1):
        for j in list(sells):
            o = O[i, j]
            if np.isnan(o) or j not in pos:
                if j not in pos:
                    sells.discard(j)
                continue
            sh, basis, bi, _ = pos.pop(j)
            v = sh * o * (1 - slip)
            ch = sell_charges(v, costs)
            cash += v - ch
            charges += ch
            trades.append((i - bi, (v - ch) / basis - 1, j, bi, i, v - ch - basis))
            sells.discard(j)
        if buys:
            slot = prev_equity / n
            left = len(buys)
            for j, hold in buys:
                o = O[i, j]
                if np.isnan(o) or j in pos:
                    left -= 1
                    continue
                px = o * (1 + slip)
                budget = min(slot, cash / left)
                left -= 1
                sh = math.floor(budget / (px * 1.0013))
                while sh > 0 and sh * px + buy_charges(sh * px, costs) > cash:
                    sh -= 1
                if sh < 1 or sh * px < min_val * 0.5:
                    continue
                ch = buy_charges(sh * px, costs)
                cash -= sh * px + ch
                charges += ch
                pos[j] = [sh, sh * px + ch, i, hold]
            buys = []
        held, stk = 0.0, 0.0
        for j, (sh, basis, bi, _) in pos.items():
            v = sh * C[i, j]
            held += v
            if v < basis and i - bi >= stuck_days:
                stk += v
        equity = cash + held
        prev_equity = equity
        eq.append(equity)
        stuck.append(stk / equity)
        if i not in order or i + 1 >= i1:
            continue
        ranked = order[i]
        rank = {j: r for r, j in enumerate(ranked)}
        for j, (sh, basis, bi, hold) in pos.items():
            if j in sells:
                continue
            # sell at the rebalance whose next-open fill lands nearest the drawn holding period
            out = (i + 1 - bi >= hold - 2) if rand else rank.get(j, 10**9) >= exit_rank
            if out and no_loss:
                v = sh * C[i, j] * (1 - slip)
                out = v - sell_charges(v, costs) >= basis
            if out:
                sells.add(j)
        free = n - (len(pos) - len(sells))
        slot = equity / n
        if free <= 0 or slot < min_val:
            continue
        cands = rng.permutation(ranked) if rand else ranked
        for j in cands:
            if free == 0:
                break
            if j in pos or C[i, j] * 1.01 > slot:
                continue
            hold = int(rng.choice(hold_dist)) if rand else 0
            buys.append((j, hold))
            free -= 1
    eq = np.array(eq)
    return {"equity": eq, "stuck": np.array(stuck), "trades": trades, "charges": charges,
            "open_underwater": sum(1 for j, (sh, b, _, _) in pos.items() if sh * C[i1 - 1, j] < b)}


def stats(res, capital, ndays_year=248):
    eq = res["equity"]
    years = len(eq) / ndays_year
    tr = res["trades"]
    return {"cagr": (eq[-1] / capital) ** (1 / years) - 1,
            "max_dd": (eq / np.maximum.accumulate(eq) - 1).min(),
            "trades_yr": len(tr) / years,
            "median_hold": float(np.median([t[0] for t in tr])) if tr else np.nan,
            "win_rate": float(np.mean([t[1] > 0 for t in tr])) if tr else np.nan,
            "stuck_pct": float(res["stuck"].mean()),
            "charges_pct_yr": res["charges"] / capital / years}


# ---------------------------------------------------------------- driver

def load(cfg):
    uni = pd.read_csv(ROOT / "universes" / "nifty500_2026-11.csv")
    uni["t"] = uni["Symbol"].str.strip() + ".NS"
    dc = cfg["data"]
    sp = download_prices(list(uni["t"]), dc["start"], ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    bp = download_prices(list(BENCH.values()), dc["start"], ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    cal = bp[BENCH["nifty500"]].index
    O, C, V = (wide(sp, f, cal) for f in ["Open", "Close", "Volume"])
    industry = uni.set_index("t")["Industry"]
    bench = {k: bp[t]["Close"].reindex(cal).ffill() for k, t in BENCH.items() if t in bp}
    return O, C, V, industry, bench


def run_momentum(cfg):
    O, C, V, industry, bench = load(cfg)
    costs, capital = cfg["costs"], cfg["capital"]
    cal = C.index
    sig, elig = signals(C, V, industry)
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(START)) & (rebal < cal[-21])]
    out_dir = ROOT / cfg["backtest"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    print(f"Universe: today's Nifty 500 ({C.shape[1]} with prices) - SURVIVORSHIP BIAS, see notes.")
    print(f"Eligible: 1y of history, median 20-day turnover >= Rs {MIN_TURNOVER_CR:.0f} cr. "
          f"Avg eligible per week: {elig.loc[rebal].sum(axis=1).mean():.0f}")

    # A
    print("\n=== A. Signal test: top decile vs average eligible stock, forward return before costs ===")
    st, raw = signal_test(sig, O, C, rebal)
    raw.to_csv(out_dir / "momentum_signal_weekly.csv", index=False)
    st.to_csv(out_dir / "momentum_signal_test.csv")
    show = st.copy()
    for c in ["top_minus_all", "top_beats_pct"]:
        show[c] = show[c].map("{:+.2%}".format if c == "top_minus_all" else "{:.0%}".format)
    show["rank_ic"] = show["rank_ic"].map("{:+.3f}".format)
    print(show.to_string())

    # B
    Oa, Cf = O.to_numpy(), C.ffill().to_numpy()
    col_rebal = {cal.get_loc(d): d for d in rebal}
    orders = {}
    for name, s in sig.items():
        a = s.to_numpy()
        orders[name] = {i: np.where(~np.isnan(a[i]))[0][np.argsort(-a[i][~np.isnan(a[i])], kind="stable")]
                        for i in col_rebal}
    variants = [(name, 20, False) for name in sig] + [("mom12_1", 10, False), ("mom12_1", 50, False),
                                                      ("mom12_1", 20, True), ("combo", 20, True)]
    i_start = cal.searchsorted(pd.Timestamp(START))
    i_check = cal.searchsorted(pd.Timestamp(CHECK))
    i_end = len(cal)
    periods = {"learn 2016-23": (i_start, i_check), "check 2024+": (i_check, i_end),
               "full 2016+": (i_start, i_end)}
    windows = []
    for y in (1, 3):
        for s0 in pd.date_range(START, cal[-1] - pd.DateOffset(years=y), freq="3MS"):
            a, b = cal.searchsorted(s0), cal.searchsorted(s0 + pd.DateOffset(years=y))
            windows.append((y, s0, a, b))

    def bench_cagr(key, a, b):
        x = bench.get(key)
        if x is None or np.isnan(x.iloc[a]):
            return np.nan
        return (x.iloc[b - 1] / x.iloc[a]) ** (248 / (b - a)) - 1

    print(f"\n=== B. Portfolio: {N_POS} stocks, Rs {capital:,}, weekly rotation, after all charges ===")
    rows, wrows = [], []
    for name, ex, nl in variants:
        label = f"{name} exit>{ex}" + (" NO-LOSS" if nl else "")
        print(f"  running {label} ...", flush=True)
        od = orders[name]
        full = simulate(Oa, Cf, od, i_start, i_end, capital, costs, exit_rank=ex, no_loss=nl)
        hold_dist = np.array([t[0] for t in full["trades"]] or [20])
        # hold periods are in trading days; random holdings get the same distribution
        for kind in ("rule", "random"):
            seeds = [None] if kind == "rule" else range(SEEDS)
            for pname, (a, b) in periods.items():
                ss = [stats(simulate(Oa, Cf, od, a, b, capital, costs, exit_rank=ex, no_loss=nl,
                                     rng=None if sd is None else np.random.default_rng(sd),
                                     hold_dist=hold_dist), capital) for sd in seeds]
                r = pd.DataFrame(ss).median().to_dict()
                rows.append({"variant": label, "kind": kind, "period": pname, **r})
            for y, s0, a, b in windows:
                cg = np.median([stats(simulate(Oa, Cf, od, a, b, capital, costs, exit_rank=ex, no_loss=nl,
                                               rng=None if sd is None else np.random.default_rng(sd),
                                               hold_dist=hold_dist), capital)["cagr"] for sd in seeds])
                wrows.append({"variant": label, "kind": kind, "years": y, "start": s0.date(), "cagr": cg})
    for pname, (a, b) in periods.items():
        for k in ("nifty500", "niftybees"):
            rows.append({"variant": f"BENCH {k}", "kind": "bench", "period": pname,
                         "cagr": bench_cagr(k, a, b),
                         "max_dd": float((bench[k].iloc[a:b] / bench[k].iloc[a:b].cummax() - 1).min())})
    for y, s0, a, b in windows:
        for k in BENCH:
            wrows.append({"variant": f"BENCH {k}", "kind": "bench", "years": y, "start": s0.date(),
                          "cagr": bench_cagr(k, a, b)})
    P = pd.DataFrame(rows)
    W = pd.DataFrame(wrows)
    P.to_csv(out_dir / "momentum_portfolio.csv", index=False)
    W.to_csv(out_dir / "momentum_windows.csv", index=False)

    pct = ["cagr", "max_dd", "win_rate", "stuck_pct", "charges_pct_yr"]
    for pname in periods:
        t = P[P["period"] == pname].drop(columns="period").set_index(["variant", "kind"])
        for c in pct:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        for c in ["trades_yr", "median_hold"]:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        print(f"\n--- {pname} (single path, fresh start) ---")
        print(t.to_string())

    # rolling windows: rule vs its random control vs index
    print("\n--- Rolling windows (new start every quarter) ---")
    piv = W.pivot_table(index=["years", "start"], columns=["variant", "kind"], values="cagr")
    summ = []
    for name, ex, nl in variants:
        label = f"{name} exit>{ex}" + (" NO-LOSS" if nl else "")
        for y in (1, 3):
            p = piv.loc[y]
            r, rnd = p[(label, "rule")], p[(label, "random")]
            n5 = p[("BENCH nifty500", "bench")]
            summ.append({"variant": label, "years": y, "windows": len(r),
                         "median": r.median(), "worst": r.min(), "losing": (r < 0).mean(),
                         "beats_random": (r > rnd).mean(), "median_vs_random": (r - rnd).median(),
                         "beats_nifty500": (r > n5).mean(), "median_vs_nifty500": (r - n5).median()})
    for k in BENCH:
        for y in (1, 3):
            v = piv.loc[y][(f"BENCH {k}", "bench")].dropna()
            summ.append({"variant": f"BENCH {k}", "years": y, "windows": len(v), "median": v.median(),
                         "worst": v.min(), "losing": (v < 0).mean()})
    S = pd.DataFrame(summ)
    S.to_csv(out_dir / "momentum_rolling_summary.csv", index=False)
    show = S.copy()
    for c in show.columns:
        if c not in ("variant", "years", "windows"):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    print(show.to_string(index=False))

    # vs the Momentum 30 ETF over its own history
    if "mom30_etf" in bench:
        e = bench["mom30_etf"].dropna()
        a, b = cal.searchsorted(e.index[0]), i_end
        print(f"\n--- vs Nifty 200 Momentum 30 ETF (MOMENTUM.NS) since {e.index[0].date()} ---")
        print(f"  ETF buy-and-hold CAGR {bench_cagr('mom30_etf', a, b):+.1%} | "
              f"Nifty 500 {bench_cagr('nifty500', a, b):+.1%}")
        for name, ex, nl in variants:
            r = stats(simulate(Oa, Cf, orders[name], a, b, capital, costs, exit_rank=ex, no_loss=nl), capital)
            print(f"  {name} exit>{ex}{' NO-LOSS' if nl else '':9s} CAGR {r['cagr']:+.1%}  maxDD {r['max_dd']:.1%}")
    print(f"\nCSV files in {out_dir}/momentum_*.csv")
    print("Survivorship bias: universe = today's Nifty 500; flatters momentum most. Returns before 20% STCG.")
