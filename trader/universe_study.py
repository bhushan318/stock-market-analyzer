"""Nifty 50 vs Nifty 500 for the stock slots - with random-pick controls.

    python -m trader universe

Survivorship bias: both stock lists are TODAY's constituents. For Nifty 500 that bias is large
(today's members include many small caps that multiplied to get in; the ones that collapsed are
missing). Random picks from the same list carry the same bias, so judge each rule by its gap
over the matching random control, not by its raw return.
"""

import copy

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices, wide
from .etf_study import BROAD
from .mixed_study import _by_date, run_one

START = "2016-01-01"
SEEDS = 5


def load_universe(cfg):
    n500 = pd.read_csv(ROOT / "universes" / "nifty500_2026-11.csv")["Symbol"].str.strip()
    n500 = [s + ".NS" for s in n500]
    dc = cfg["data"]
    sp = download_prices(n500, dc["start"], ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    ep = download_prices(BROAD, "2009-01-01", ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    prices = {**{t: df[df.index >= dc["start"]] for t, df in ep.items()}, **sp}
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in prices.values()])))
    frames = tuple(wide(prices, f, cal) for f in ["Open", "High", "Low", "Close"])
    volume = wide(prices, "Volume", cal)
    n50 = [t for t in cfg["universe"]["tickers"] if t in sp]
    return frames, volume, n50, [t for t in n500 if t in sp]


class Scores:
    """Dip z-scores; stocks only when above their 200-day average and liquid enough."""

    def __init__(self, frames, etfs, min_turnover_cr=2.0):
        C = frames[3]
        self.C = C
        mean20, std20 = C.rolling(20, min_periods=20).mean(), C.rolling(20, min_periods=20).std()
        self.z = -(C - mean20) / std20
        self.up = C > C.rolling(200, min_periods=200).mean()
        self.etfs = etfs
        self.etf_raw = -(C[etfs] / mean20[etfs] - 1)
        self.min_turnover_cr = min_turnover_cr

    def liquid(self, vol):
        to = (vol * self.C).rolling(20, min_periods=20).median() / 1e7
        return to >= self.min_turnover_cr

    def table(self, stocks, liquid, with_etfs, etf_raw=False):
        st = self.z[stocks].where(self.up[stocks] & liquid[stocks])
        if not with_etfs:
            return _by_date(st)
        e = self.etf_raw if etf_raw else self.z[self.etfs]
        return _by_date(pd.concat([e, st], axis=1))


def scorer(table):
    return lambda i, d: table.get(d)


def random_stock_scorer(table, stocks, seed):
    """Same eligible set and ETF ranking as the rule, but stocks in random order."""
    rng = np.random.default_rng(seed)
    stocks = set(stocks)

    def f(i, d):
        s = table.get(d)
        if s is None:
            return None
        s = s.copy()
        m = s.index.isin(list(stocks))
        s[m] = rng.random(m.sum()) * 10 - 5  # random scores in the same range as z-scores
        return s
    return f


def evaluate(cfg, frames, make_scorer, extra, stt, bench, seeds=(None,)):
    C = frames[3]
    full, r1, r3 = [], [], []
    for seed in seeds:
        sc = make_scorer(seed)
        full.append(run_one(cfg, frames, sc, START, None, extra, stt, bench))
        for y, bucket in ((1, r1), (3, r3)):
            for st in pd.date_range(START, C.index[-1] - pd.DateOffset(years=y), freq="3MS"):
                sc = make_scorer(seed)
                bucket.append(run_one(cfg, frames, sc, st, st + pd.DateOffset(years=y), extra, stt, bench))
    c1 = pd.Series([x["cagr"] for x in r1])
    c3 = pd.Series([x["cagr"] for x in r3])
    return {
        "10y_cagr": np.median([x["cagr"] for x in full]),
        "max_dd": np.median([x["max_drawdown"] for x in full]),
        "worst_pos": np.median([x["worst_unrealised_position"] for x in full]),
        "stuck": np.median([x["avg_stuck_pct"] for x in full]),
        "1y_median": c1.median(), "1y_losing": (c1 < 0).mean(), "1y_worst": c1.min(),
        "3y_median": c3.median(), "3y_worst": c3.min(),
    }


def run_universe(cfg):
    cfg = copy.deepcopy(cfg)
    frames, volume, n50, n500 = load_universe(cfg)
    C = frames[3]
    etfs = [t for t in BROAD if t in C]
    bench = C["NIFTYBEES.NS"].dropna()
    S = Scores(frames, etfs)
    liquid = S.liquid(volume).fillna(False).astype(bool)
    stt = {t: 0.00001 for t in etfs}
    groups = {**{t: "etf" for t in etfs}, **{t: "stock" for t in n500}}
    print(f"Capital Rs {cfg['capital']:,} | max 5 positions | +5% net | never sell at a loss | "
          f"{len(etfs)} ETFs, Nifty 50 = {len(n50)}, Nifty 500 = {len(n500)} stocks | from {START}")
    print("Stocks eligible only above their 200-day average with >= Rs 2 cr median daily turnover.")
    print(f"Random controls: median of {SEEDS} seeds, same eligible stocks, random order.\n")

    seeds = list(range(SEEDS))
    tabs = {"N50": (n50, S.table(n50, liquid, False), S.table(n50, liquid, True)),
            "N500": (n500, S.table(n500, liquid, False), S.table(n500, liquid, True))}
    rows = {}
    rows["A  ETFs only"] = evaluate(cfg, frames, lambda s: scorer(_by_date(S.etf_raw)), {}, stt, bench)
    for u, (stocks, only, mixed) in tabs.items():
        for label, tab, extra in [
            ("stocks only", only, {}),
            ("4 ETF + 1 stock", mixed, {"groups": groups, "group_limits": {"etf": 4, "stock": 1}}),
            ("3 ETF + 2 stock", mixed, {"groups": groups, "group_limits": {"etf": 3, "stock": 2}}),
        ]:
            print(f"  running {u} {label} ...")
            rows[f"{u} {label} - dip rule"] = evaluate(cfg, frames, lambda s, t=tab: scorer(t),
                                                         extra, stt, bench)
            rows[f"{u} {label} - RANDOM"] = evaluate(
                cfg, frames, lambda s, t=tab, st=stocks: random_stock_scorer(t, st, s),
                extra, stt, bench, seeds)
    df = pd.DataFrame(rows).T
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: f"{v:.1%}")
    print()
    print(out.to_string())
    out_dir = ROOT / cfg["backtest"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "universe_study.csv")
