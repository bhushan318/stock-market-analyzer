"""How much money, and how to add it: Rs 1 lakh lump sum vs Rs 50k/month SIP (owner, 2026-11-28).

    python -m trader capital

Same panel, signals, charges and market filter as Results 17 (strategies.py). New engine
`run_pool` runs several strategies inside ONE account:
  * each position remembers which strategy bought it and exits by that strategy's rules;
  * slots: shared (any strategy may fill any of the 5 slots, strongest 12-month momentum first)
    or fixed per strategy (e.g. one slot each);
  * adds: cash paid in on given days (SIP); returns are reported two ways -
      TWR  time-weighted (unit NAV, like a mutual fund's CAGR; ignores when money came in)
      XIRR money-weighted (your actual rupees in -> rupees out, like an SIP statement);
  * liquidity cap: a buy is at most 2% of the stock's 20-day median turnover (big SIP accounts).
Momentum inside the pool = the Gate 1 rule as signals: on the weekly decision day buy the top 5
by combo score, sell when it drops below rank 50 (checked weekly).

The owner's limit of 5 open positions is kept, except in the "as tested" multi-book row
(5 separate accounts x 5 stocks = up to 25 positions), shown for comparison only.
"""

import math

import numpy as np
import pandas as pd

from . import ROOT
from .costs import buy_charges, sell_charges
from .data import download_prices
from .momentum_pit import pit_signals
from .momentum_study import weekly_dates
from .pit import liquid_universe, load_panel
from .strategies import SPLITS, build_signals

MULTI = ["momentum", "minervini", "supertrend", "golden", "breakout52"]
LIQ_SHARE = 0.02


def run_pool(O, C, books, priority, i0, i1, costs, capital, n=5, slots=None, regime=None, adds=None,
             liq=None, min_val=1500):
    """books: list of dicts {entry, exit (T x N bool), stop_pct, stop_level (T x N), target_r, max_hold}.
    slots: None = n shared slots, else per-book caps. adds: T array of rupees paid in at that day's open.
    liq: T x N max rupees per buy (None = no cap). Decide at close i, fill at open i+1."""
    slip = costs["slippage"]
    cash, pos, eq, trades, charges = float(capital), {}, [], [], 0.0
    units, nav, navs, invested = capital / 100.0, 100.0, [], float(capital)
    sells, buys = set(), []
    flows = [(i0, -float(capital))]
    for i in range(i0, i1):
        if adds is not None and i > i0 and adds[i] > 0:
            cash += adds[i]
            invested += adds[i]
            units += adds[i] / nav
            flows.append((i, -float(adds[i])))
        for j in list(sells):
            o = O[i, j]
            if np.isnan(o):
                continue
            sh, basis, bi, stp, tgt, k = pos.pop(j)
            v = sh * o * (1 - slip)
            ch = sell_charges(v, costs)
            cash += v - ch
            charges += ch
            trades.append((i - bi, (v - ch) / basis - 1, j, bi, i, v - ch - basis, k))
            sells.discard(j)
        prev_eq = cash + sum(p_[0] * C[i - 1, j] for j, p_ in pos.items()) if i > i0 else cash
        left = len(buys)
        for j, sig_stop, k in buys:
            o = O[i, j]
            left -= 1
            if np.isnan(o) or j in pos:
                continue
            px = o * (1 + slip)
            budget = min(prev_eq / n, cash / (left + 1))
            if liq is not None and not np.isnan(liq[i - 1, j]):
                budget = min(budget, liq[i - 1, j])
            sh = math.floor(budget / (px * 1.0013))
            while sh > 0 and sh * px + buy_charges(sh * px, costs) > cash:
                sh -= 1
            if sh < 1 or sh * px < min_val * 0.5:
                continue
            ch = buy_charges(sh * px, costs)
            cash -= sh * px + ch
            charges += ch
            b = books[k]
            stp = px * (1 - b["stop_pct"]) if b.get("stop_pct") else (sig_stop if sig_stop == sig_stop else None)
            if stp is not None and stp >= px:
                stp = px * 0.999
            tgt = px + b["target_r"] * (px - stp) if (b.get("target_r") and stp) else None
            pos[j] = [sh, sh * px + ch, i, stp, tgt, k]
        buys = []
        equity = cash + sum(p_[0] * C[i, j] for j, p_ in pos.items())
        nav = equity / units
        eq.append(equity)
        navs.append(nav)
        if i + 1 >= i1:
            continue
        for j, (sh, basis, bi, stp, tgt, k) in pos.items():
            b, c = books[k], C[i, j]
            if (b["exit"][i, j] or (stp is not None and c < stp) or (tgt is not None and c >= tgt)
                    or (b.get("max_hold") and i - bi >= b["max_hold"])):
                sells.add(j)
        held = {}
        for j, p_ in pos.items():
            if j not in sells:
                held[p_[5]] = held.get(p_[5], 0) + 1
        free = n - sum(held.values())
        if free <= 0 or equity / n < min_val or (regime is not None and not regime[i]):
            continue
        cand = []
        for k, b in enumerate(books):
            if slots is not None and held.get(k, 0) >= slots[k]:
                continue
            for j in np.where(b["entry"][i])[0]:
                cand.append((-np.nan_to_num(priority[i, j], nan=-9), k, j))
        cand.sort()
        taken = {}
        chosen = set()
        for _, k, j in cand:
            if free == 0:
                break
            if j in pos or j in chosen or C[i, j] * 1.01 > equity / n:
                continue
            if slots is not None and held.get(k, 0) + taken.get(k, 0) >= slots[k]:
                continue
            sl = books[k].get("stop_level")
            buys.append((j, sl[i, j] if sl is not None else np.nan, k))
            chosen.add(j)
            taken[k] = taken.get(k, 0) + 1
            free -= 1
    flows.append((i1 - 1, float(eq[-1])))
    return {"equity": np.array(eq), "nav": np.array(navs), "trades": trades, "charges": charges,
            "invested": invested, "flows": flows}


def xirr(flows, cal):
    """flows: [(day index, rupees)] (paid in < 0, final value > 0) -> annual money-weighted return."""
    t = np.array([(cal[i] - cal[flows[0][0]]).days / 365.25 for i, _ in flows])
    v = np.array([f for _, f in flows])
    f = lambda r: (v / (1 + r) ** t).sum()  # noqa: E731
    lo, hi = -0.99, 5.0
    if f(lo) * f(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def sip_days(cal, i0, i1):
    """First trading day of each month after the start."""
    s = pd.Series(np.arange(len(cal)), index=cal)[i0:i1]
    first = s.groupby([s.index.year, s.index.month]).min().values
    out = np.zeros(len(cal))
    out[first[first > i0]] = 1.0
    return out


def momentum_book(sig, elig, cal):
    """Gate 1 rule as entry/exit signals on weekly decision days."""
    combo = sig["combo"]
    rank = combo.rank(axis=1, ascending=False)
    wk = pd.Series(False, index=cal)
    wk[weekly_dates(cal)] = True
    w = wk.to_numpy()[:, None]
    entry = ((rank <= 5) & elig).to_numpy() & w
    exit_ = (rank.isna() | (rank > 50)).to_numpy() & w
    return {"entry": entry, "exit": exit_}


def stats(res, cal, i0, i1):
    navs, yrs = res["nav"], (i1 - i0) / 248
    return {"twr": (navs[-1] / 100.0) ** (1 / yrs) - 1, "xirr": xirr(res["flows"], cal),
            "max_dd": float((navs / np.maximum.accumulate(navs) - 1).min()),
            "invested": res["invested"], "final": float(res["equity"][-1]),
            "trades_yr": len(res["trades"]) / yrs, "charges": res["charges"]}


def combine(results):
    """Several separate accounts -> one: sum equity, cash flows; NAV from the summed account."""
    eq = sum(r["equity"] for r in results)
    flows = {}
    for r in results:
        for i, f in r["flows"][:-1]:
            flows[i] = flows.get(i, 0.0) + f
    last = results[0]["flows"][-1][0]
    fl = sorted(flows.items()) + [(last, float(eq[-1]))]
    # unit NAV of the combined account
    i0 = fl[0][0]
    units, nav, navs, add_at = 0.0, 100.0, [], dict(fl[:-1])
    for t, e in enumerate(eq):
        a = -add_at.get(i0 + t, 0.0)
        if a:
            units += a / nav
        nav = e / units
        navs.append(nav)
    return {"equity": eq, "nav": np.array(navs), "trades": sum((r["trades"] for r in results), []),
            "charges": sum(r["charges"] for r in results), "invested": sum(r["invested"] for r in results), "flows": fl}


def index_account(n5, cal, i0, i1, capital, adds, fee=0.002):
    """Nifty 500 index fund with the same cash flows (fee ~0.2%/yr)."""
    px = n5.to_numpy()
    units, eq, navs, flows, inv = capital / px[i0], [], [], [(i0, -capital)], capital
    drag = (1 - fee) ** (1 / 248)
    nav_units = capital / 100.0
    for i in range(i0, i1):
        if adds is not None and i > i0 and adds[i] > 0:
            nav_now = units * px[i] / nav_units
            units += adds[i] / px[i]
            nav_units += adds[i] / nav_now
            inv += adds[i]
            flows.append((i, -float(adds[i])))
        if i > i0:
            units *= drag
        e = units * px[i]
        eq.append(e)
        navs.append(e / nav_units)
    flows.append((i1 - 1, float(eq[-1])))
    return {"equity": np.array(eq), "nav": np.array(navs), "trades": [], "charges": 0.0, "invested": inv, "flows": flows}


def run_capital(cfg):
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    C, cal = p["C"], p["cal"]
    sig, elig = pit_signals(C, p["TO"], liquid_universe(C, p["TO"]))
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
    regime = ((n5 > n5.rolling(200, min_periods=150).mean()) | n5.isna()).to_numpy()
    print("Building signals ...", flush=True)
    S, mom12 = build_signals(p, elig)
    books = {"momentum": momentum_book(sig, elig, cal)}
    for name in MULTI[1:]:
        s = S[name]
        books[name] = {"entry": s["entry"].fillna(False).to_numpy(), "exit": s["exit"].fillna(False).to_numpy(),
                       "stop_pct": s.get("stop_pct"), "target_r": s.get("target_r"), "max_hold": s.get("max_hold"),
                       "stop_level": s["stop_level"].to_numpy() if "stop_level" in s else None}
    Oa, Cf, pr = p["O"].to_numpy(), C.ffill().to_numpy(), mom12.to_numpy()
    liq = (p["TO"].rolling(20, min_periods=15).median() * 1e7 * LIQ_SHARE).to_numpy()
    costs = cfg["costs"]
    allb = [books[k] for k in MULTI]
    SIP = 50_000

    def acct(kind, i0, i1, capital, adds):
        run = lambda bl, cap, ad, **kw: run_pool(Oa, Cf, bl, pr, i0, i1, costs, cap, regime=regime,  # noqa: E731
                                                  adds=ad, liq=liq, **kw)
        if kind == "minervini":
            return run([books["minervini"]], capital, adds)
        if kind == "momentum":
            return run([books["momentum"]], capital, adds)
        if kind == "pool shared":
            return run(allb, capital, adds)
        if kind == "pool 1 each":
            return run(allb, capital, adds, slots=[1] * 5)
        if kind == "pool mom+minervini":
            return run([books["momentum"], books["minervini"]], capital, adds)
        if kind == "multi-book 5x5 (25 stocks)":
            part = None if adds is None else adds / 5
            return combine([run([books[k]], capital / 5, part) for k in MULTI])
        if kind == "Nifty 500 fund":
            return index_account(n5, cal, i0, i1, capital, adds)
        raise KeyError(kind)

    kinds = ["minervini", "momentum", "pool mom+minervini", "pool shared", "pool 1 each",
             "multi-book 5x5 (25 stocks)", "Nifty 500 fund"]
    plans = [("Rs 25k lump", 25_000, False), ("Rs 1L lump", 100_000, False), ("Rs 50k/month SIP", SIP, True)]
    rows = []
    for sp, a, b in SPLITS:
        i0, i1 = cal.searchsorted(pd.Timestamp(a)), min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
        months = sip_days(cal, i0, i1) * SIP
        for plan, cap, is_sip in plans:
            for kind in kinds:
                if plan == "Rs 25k lump" and kind.startswith("multi"):
                    continue
                res = acct(kind, i0, i1, cap, months if is_sip else None)
                rows.append({"split": sp, "plan": plan, "portfolio": kind, **stats(res, cal, i0, i1)})
                print(f"  {sp} | {plan} | {kind}: {rows[-1]['twr']:+.1%}", flush=True)
    R = pd.DataFrame(rows)
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "capital_study.csv", index=False)
    order = [s for s, _, _ in SPLITS]
    for plan, _, is_sip in plans:
        sub = R[R.plan == plan]
        col = "xirr" if is_sip else "twr"
        piv = sub.pivot(index="portfolio", columns="split", values=col).reindex(kinds).dropna(how="all")[order]
        dd = sub.pivot(index="portfolio", columns="split", values="max_dd").reindex(piv.index)[order]
        show = piv.map(lambda v: f"{v:+.0%}") + " (" + dd.map(lambda v: f"{v:.0%}") + ")"
        print(f"\n=== {plan}: per year after charges ({'XIRR on your rupees' if is_sip else 'CAGR'}), worst fall ===")
        print(show.to_string())
        if is_sip:
            t = sub[sub.split == order[-1]].set_index("portfolio").reindex(kinds)
            print(f"\n  {order[-1]}: paid in Rs {t.invested.iloc[0] / 1e5:.1f} L ->")
            for k, r in t.iterrows():
                print(f"    {k:28s} Rs {r.final / 1e5:6.1f} L   ({r.trades_yr:.0f} trades/yr, charges Rs {r.charges / 1e3:.0f}k)")
    print(f"\nCSV: {out}/capital_study.csv")
