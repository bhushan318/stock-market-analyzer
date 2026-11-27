"""Popular swing-trading strategies (books, Reddit, Indian setups) on one engine, one cost model.

    python -m trader strategies

Stage 1 of "best of all worlds" (owner, 2026-11-27): study each strategy alone, same universe
(point-in-time top 500 by turnover, price >= Rs 10, 20-day turnover >= Rs 2 cr), 5 slots, Rs 25k,
all charges, decisions at the close, fills at the next open. Splits: learn 2006-16, tune 2017-19,
test 2020-26 (2006-19 = history.db, 2020+ = market.db). Rules as commonly published:

  minervini  trend template (C > SMA50 > SMA150 > SMA200, SMA200 rising, >= 30% above the 52-week
             low, within 25% of the 52-week high, 12-month return in the top 30%) + close above the
             prior 20-day high on volume >= 1.5x its 50-day average. Exit: close < SMA50 or -8%.
  breakout52 close at a new 52-week closing high on volume >= 2x its 50-day average (Darvas-style).
             Exit: close below the lowest close of the prior 20 sessions (Turtle/Donchian).
  rsi2       Connors: C > SMA200 and RSI(2) < 10. Exit: close > SMA5, or after 10 sessions; no stop.
  ema20      EMA20 > EMA50 > SMA200, the day's low touches EMA20, closes above it and above its open.
             Stop = that day's low; target = 2x the risk; exit also on close < EMA50.
  supertrend Supertrend(10, 3) turns up while C > SMA200. Exit: Supertrend turns down.
  golden     SMA50 crosses above SMA200. Exit: SMA50 crosses below SMA200.
  nr7        day with the narrowest range of 7 while C > SMA50; next day closes above that day's high.
             Stop = the NR7 day's low; exit after 10 sessions.
Exits are checked at the close and sold at the next open (same for every strategy). Among several
signals on one day, the strongest 12-month momentum is bought first (one common tie-break).
"""

import math

import numpy as np
import pandas as pd

from . import ROOT
from .costs import buy_charges, sell_charges
from .data import download_prices
from .momentum_pit import pit_signals
from .pit import liquid_universe, load_panel

SPLITS = [("learn 2006-16", "2006-07-01", "2016-12-31"), ("tune 2017-19", "2017-01-01", "2019-12-31"),
          ("TEST 2020-26", "2020-01-01", "2026-12-31")]
N_SLOTS, CAPITAL = 5, 25_000


# ---------------------------------------------------------------- indicators (no look-ahead)

def sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)  # no down moves -> up/0 = inf -> RSI 100


def atr(H, L, C, n=10):
    pc = C.shift(1)
    tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
    return pd.DataFrame(tr, index=C.index, columns=C.columns).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend_up(H, L, C, n=10, mult=3.0):
    """True where the Supertrend is in an up-trend (standard final-band recursion)."""
    a = atr(H, L, C, n).to_numpy()
    mid = ((H + L) / 2).to_numpy()
    c = C.to_numpy()
    ub, lb = mid + mult * a, mid - mult * a
    T, N = c.shape
    fu, fl = np.full(N, np.nan), np.full(N, np.nan)
    up = np.zeros((T, N), bool)
    cur = np.ones(N, bool)
    for t in range(T):
        u, l_, ct = ub[t], lb[t], c[t]
        prev_c = c[t - 1] if t else np.full(N, np.nan)
        fu = np.where(np.isnan(fu) | (u < fu) | (prev_c > fu), u, fu)
        fl = np.where(np.isnan(fl) | (l_ > fl) | (prev_c < fl), l_, fl)
        cur = np.where(cur & (ct < fl), False, np.where(~cur & (ct > fu), True, cur))
        up[t] = cur & ~np.isnan(ct) & ~np.isnan(a[t])
    return pd.DataFrame(up, index=C.index, columns=C.columns)


# ---------------------------------------------------------------- signals

def build_signals(p, elig):
    """{name: dict(entry, exit, stop, target_r, max_hold)}; all DataFrames date x symbol, known at the close."""
    C = p["C"].ffill(limit=5)
    H, L, O, V = p["H"], p["L"], p["O"], p["V"]
    s50, s150, s200, s5 = sma(C, 50), sma(C, 150), sma(C, 200), sma(C, 5)
    e20, e50 = ema(C, 20), ema(C, 50)
    hi52, lo52 = C.rolling(252, min_periods=240).max(), C.rolling(252, min_periods=240).min()
    v50 = V.rolling(50, min_periods=40).mean()
    mom12 = C.shift(21) / C.shift(252) - 1
    rs = mom12.where(elig).rank(axis=1, pct=True)
    live = elig & C.notna() & H.notna()
    S = {}
    tt = ((C > s50) & (s50 > s150) & (s150 > s200) & (s200 > s200.shift(21)) & (C >= 1.3 * lo52)
          & (C >= 0.75 * hi52) & (rs >= 0.7))
    S["minervini"] = dict(entry=live & tt & (C > C.shift(1).rolling(20).max()) & (V >= 1.5 * v50),
                          exit=C < s50, stop_pct=0.08)
    S["breakout52"] = dict(entry=live & (C >= C.shift(1).rolling(251, min_periods=240).max()) & (V >= 2 * v50),
                           exit=C < C.shift(1).rolling(20).min())
    r2 = rsi(C, 2)
    S["rsi2"] = dict(entry=live & (C > s200) & (r2 < 10), exit=C > s5, max_hold=10)
    S["ema20"] = dict(entry=live & (e20 > e50) & (e50 > s200) & (L <= e20) & (C > e20) & (C > O),
                      exit=C < e50, stop_level=L, target_r=2.0)
    st = supertrend_up(H, L, C)
    S["supertrend"] = dict(entry=live & st & ~st.shift(1, fill_value=True) & (C > s200), exit=~st)
    gc = s50 > s200
    S["golden"] = dict(entry=live & gc & ~gc.shift(1, fill_value=True), exit=~gc)
    rng = H - L
    nr7 = rng <= rng.rolling(7, min_periods=7).min()
    nr7_hi, nr7_lo = H.where(nr7).shift(1), L.where(nr7).shift(1)
    S["nr7"] = dict(entry=live & nr7.shift(1, fill_value=False) & (C > nr7_hi) & (C > s50),
                    exit=pd.DataFrame(False, index=C.index, columns=C.columns), stop_level=nr7_lo, max_hold=10)
    return S, mom12


# ---------------------------------------------------------------- engine

def run_book(O, C, entry, priority, exit_sig, i0, i1, costs, capital=CAPITAL, n=N_SLOTS, stop_pct=None,
             stop_level=None, target_r=None, max_hold=None, regime=None, min_val=1500):
    """Daily engine. Numpy T x N arrays; entry/exit known at close i -> orders fill at open i+1.
    Exits (checked at each close): exit signal, close below stop, close at/above target, max_hold."""
    slip = costs["slippage"]
    cash, pos, eq, trades, charges = float(capital), {}, [], [], 0.0
    sells, buys = set(), []
    for i in range(i0, i1):
        for j in list(sells):
            o = O[i, j]
            if np.isnan(o):
                continue
            sh, basis, bi, stp, tgt = pos.pop(j)
            v = sh * o * (1 - slip)
            ch = sell_charges(v, costs)
            cash += v - ch
            charges += ch
            trades.append((i - bi, (v - ch) / basis - 1, j, bi, i, v - ch - basis))
            sells.discard(j)
        prev_eq = eq[-1] if eq else cash
        left = len(buys)
        for j, sig_stop in buys:
            o = O[i, j]
            left -= 1
            if np.isnan(o) or j in pos:
                continue
            px = o * (1 + slip)
            budget = min(prev_eq / n, cash / (left + 1))
            sh = math.floor(budget / (px * 1.0013))
            while sh > 0 and sh * px + buy_charges(sh * px, costs) > cash:
                sh -= 1
            if sh < 1 or sh * px < min_val * 0.5:
                continue
            ch = buy_charges(sh * px, costs)
            cash -= sh * px + ch
            charges += ch
            stp = px * (1 - stop_pct) if stop_pct else (sig_stop if sig_stop == sig_stop else None)
            if stp is not None and stp >= px:  # gap opened below the planned stop: exit at once
                stp = px * 0.999
            tgt = px + target_r * (px - stp) if (target_r and stp) else None
            pos[j] = [sh, sh * px + ch, i, stp, tgt]
        buys = []
        equity = cash + sum(p_[0] * C[i, j] for j, p_ in pos.items())
        eq.append(equity)
        if i + 1 >= i1:
            continue
        for j, (sh, basis, bi, stp, tgt) in pos.items():
            c = C[i, j]
            if (exit_sig[i, j] or (stp is not None and c < stp) or (tgt is not None and c >= tgt)
                    or (max_hold and i - bi >= max_hold)):
                sells.add(j)
        free = n - (len(pos) - len(sells))
        if free <= 0 or equity / n < min_val or (regime is not None and not regime[i]):
            continue
        cand = np.where(entry[i])[0]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(-np.nan_to_num(priority[i, cand], nan=-9), kind="stable")]
        for j in cand:
            if free == 0:
                break
            if j in pos or C[i, j] * 1.01 > equity / n:
                continue
            buys.append((j, stop_level[i, j] if stop_level is not None else np.nan))
            free -= 1
    return {"equity": np.array(eq), "trades": trades, "charges": charges}


def book_stats(res, capital=CAPITAL):
    eq, tr = res["equity"], res["trades"]
    yrs = len(eq) / 248
    r = np.array([t[1] for t in tr]) if tr else np.array([np.nan])
    return {"cagr": (eq[-1] / capital) ** (1 / yrs) - 1, "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "trades_yr": len(tr) / yrs, "win_rate": float(np.mean(r > 0)), "avg_trade": float(np.nanmean(r)),
            "avg_win": float(np.nanmean(r[r > 0])) if (r > 0).any() else np.nan,
            "avg_loss": float(np.nanmean(r[r <= 0])) if (r <= 0).any() else np.nan,
            "median_hold": float(np.median([t[0] for t in tr])) if tr else np.nan}


def run_strategies(cfg):
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, cal = p["C"], p["cal"]
    _, elig = pit_signals(C, p["TO"], liquid_universe(C, p["TO"]))
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    regime = ((n5 > n5.rolling(200, min_periods=150).mean()) | n5.isna()).to_numpy()
    print("Building signals ...", flush=True)
    S, mom12 = build_signals(p, elig)
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    pr = mom12.to_numpy()
    rows, curves = [], {}
    for name, s in S.items():
        ent, ex = s["entry"].fillna(False).to_numpy(), s["exit"].fillna(False).to_numpy()
        sl = s["stop_level"].to_numpy() if "stop_level" in s else None
        print(f"  {name}: {int(ent.sum()):,} signals", flush=True)
        for filt in (False, True):
            for sp, a, b in SPLITS:
                i0, i1 = cal.searchsorted(pd.Timestamp(a)), min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
                res = run_book(Oa, Cf, ent, pr, ex, i0, i1, cfg["costs"], stop_pct=s.get("stop_pct"), stop_level=sl,
                               target_r=s.get("target_r"), max_hold=s.get("max_hold"), regime=regime if filt else None)
                st = book_stats(res)
                bench = (n5.iloc[i1 - 1] / n5.iloc[i0]) ** (248 / (i1 - i0)) - 1
                rows.append({"strategy": name, "market_filter": filt, "split": sp, **st, "nifty500": bench})
                if not filt:
                    curves[(name, sp)] = pd.Series(res["equity"], index=cal[i0:i1])
    R = pd.DataFrame(rows)
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "strategies_stage1.csv", index=False)
    # daily-return correlation between strategies (learn period) - do they win at different times?
    lr = pd.DataFrame({k[0]: v.pct_change() for k, v in curves.items() if k[1] == SPLITS[0][0]})
    lr.corr().round(2).to_csv(out / "strategies_corr_learn.csv")
    for filt in (False, True):
        sub = R[R.market_filter == filt]
        print(f"\n=== Per year after charges (Rs 25k, 5 slots){' + market filter (no buys below 200d avg)' if filt else ''} ===")
        piv = sub.pivot(index="strategy", columns="split", values="cagr")[[s for s, _, _ in SPLITS]]
        dd = sub.pivot(index="strategy", columns="split", values="max_dd")[[s for s, _, _ in SPLITS]]
        show = piv.map(lambda v: f"{v:+.0%}") + " (" + dd.map(lambda v: f"{v:.0%}") + ")"
        show.loc["Nifty 500"] = [f"{sub[sub.split == s].nifty500.iloc[0]:+.0%}" for s, _, _ in SPLITS]
        print(show.to_string())
    full = R[(R.market_filter == False) & (R.split == SPLITS[0][0])]  # noqa: E712
    print("\n=== Trade profile, learn 2006-16 (no filter) ===")
    show = full[["strategy", "trades_yr", "win_rate", "avg_trade", "avg_win", "avg_loss", "median_hold"]].copy()
    for c in ("win_rate", "avg_trade", "avg_win", "avg_loss"):
        show[c] = show[c].map(lambda v: f"{v:+.1%}")
    show["trades_yr"] = show["trades_yr"].map("{:.0f}".format)
    print(show.to_string(index=False))
    print("\n=== Do they win at different times? daily-return correlation, learn 2006-16 ===")
    print(lr.corr().round(2).to_string())
    print(f"\nCSV: {out}/strategies_stage1.csv, strategies_corr_learn.csv")


# ---------------------------------------------------------------- stage 2: "best of all worlds"
# Chosen from the LEARN period only (Results 17): the market filter helped nearly every strategy;
# trend strategies with few, long trades worked (minervini, supertrend, golden, breakout52) while
# the short-hold ones (rsi2, ema20, nr7: 110-130 trades/yr) lost to charges. Two combinations:
#   confluence  one rule that needs several strategies to agree: Minervini trend template AND
#               Supertrend up AND momentum-combo rank in the top 20%, entered on a 20-day-high
#               close with volume >= 1.5x; exit on close < SMA50, Supertrend down, or -8%;
#               priority = momentum combo score.
#   multi-book  run the learn-period survivors side by side, equal money each (diversification):
#               momentum combo (weekly, Gate 1 rule), minervini, supertrend, golden, breakout52.
# All with the market filter. Judged on tune 2017-19 and TEST 2020-26, then forward paper.

def run_stage2(cfg):
    from .momentum_pit import _orders
    from .momentum_study import simulate, weekly_dates
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, cal = p["C"], p["cal"]
    sig, elig = pit_signals(C, p["TO"], liquid_universe(C, p["TO"]))
    combo = sig["combo"]
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    regime = ((n5 > n5.rolling(200, min_periods=150).mean()) | n5.isna()).to_numpy()
    S, mom12 = build_signals(p, elig)
    Cc = C.ffill(limit=5)
    s50, s150, s200 = sma(Cc, 50), sma(Cc, 150), sma(Cc, 200)
    hi52, lo52 = Cc.rolling(252, min_periods=240).max(), Cc.rolling(252, min_periods=240).min()
    v50 = p["V"].rolling(50, min_periods=40).mean()
    st = supertrend_up(p["H"], p["L"], Cc)
    tt = ((Cc > s50) & (s50 > s150) & (s150 > s200) & (s200 > s200.shift(21)) & (Cc >= 1.3 * lo52) & (Cc >= 0.75 * hi52))
    conf_entry = (elig & tt & st & (combo >= 0.8) & (Cc > Cc.shift(1).rolling(20).max()) & (p["V"] >= 1.5 * v50))
    conf_exit = (Cc < s50) | ~st
    S["confluence"] = dict(entry=conf_entry, exit=conf_exit, stop_pct=0.08)
    Oa, Cf = p["O"].to_numpy(), C.ffill().to_numpy()
    pri = {"default": mom12.to_numpy(), "confluence": combo.to_numpy()}
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(SPLITS[0][1])) & (rebal < cal[-1])]
    od = _orders(combo, [cal.get_loc(d) for d in rebal])
    books = ["minervini", "supertrend", "golden", "breakout52", "confluence"]
    rows = []
    for sp, a, b in SPLITS:
        i0, i1 = cal.searchsorted(pd.Timestamp(a)), min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
        curves = {}
        for name in books:
            s = S[name]
            res = run_book(Oa, Cf, s["entry"].fillna(False).to_numpy(), pri.get(name, pri["default"]),
                           s["exit"].fillna(False).to_numpy(), i0, i1, cfg["costs"], stop_pct=s.get("stop_pct"),
                           stop_level=s["stop_level"].to_numpy() if "stop_level" in s else None,
                           target_r=s.get("target_r"), max_hold=s.get("max_hold"), regime=regime)
            curves[name] = res["equity"] / CAPITAL
            rows.append({"split": sp, "portfolio": name, **book_stats(res)})
        mres = simulate(Oa, Cf, od, i0, i1, CAPITAL, cfg["costs"], exit_rank=50, regime=regime)
        curves["momentum"] = mres["equity"] / CAPITAL
        rows.append({"split": sp, "portfolio": "momentum (Gate 1)", **book_stats(mres)})
        multi = np.mean([curves[k] for k in ("momentum", "minervini", "supertrend", "golden", "breakout52")], axis=0)
        yrs = len(multi) / 248
        rows.append({"split": sp, "portfolio": "MULTI-BOOK (5 equal)", "cagr": multi[-1] ** (1 / yrs) - 1,
                     "max_dd": float((multi / np.maximum.accumulate(multi) - 1).min())})
        rows.append({"split": sp, "portfolio": "Nifty 500", "cagr": (n5.iloc[i1 - 1] / n5.iloc[i0]) ** (248 / (i1 - i0)) - 1,
                     "max_dd": float((n5.iloc[i0:i1] / n5.iloc[i0:i1].cummax() - 1).min())})
    R = pd.DataFrame(rows)
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "strategies_stage2.csv", index=False)
    piv = R.pivot(index="portfolio", columns="split", values="cagr")[[s for s, _, _ in SPLITS]]
    dd = R.pivot(index="portfolio", columns="split", values="max_dd")[[s for s, _, _ in SPLITS]]
    show = piv.map(lambda v: f"{v:+.0%}") + " (" + dd.map(lambda v: f"{v:.0%}") + ")"
    print("\n=== Stage 2, all with the market filter: per year after charges (worst fall) ===")
    print(show.to_string())
    t = R[(R.split == SPLITS[0][0]) & R.trades_yr.notna()][["portfolio", "trades_yr", "win_rate", "avg_win", "avg_loss", "median_hold"]]
    print("\nTrade profile, learn 2006-16:")
    print(t.round(3).to_string(index=False))
    print(f"\nCSV: {out}/strategies_stage2.csv")
