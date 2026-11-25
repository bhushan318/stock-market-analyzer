"""Stocks + ETFs together, at most 5 open positions, never sell at a loss.

    python -m trader mixed

Dip score = how unusual today's dip is for that instrument:
    z = (close - 20-day average) / 20-day standard deviation; most negative first.
Stocks are only eligible above their 200-day average (avoid buying a collapsing stock).
"""

import copy

import numpy as np
import pandas as pd

from . import ROOT
from .backtest import run_backtest, summarize
from .data import download_prices, wide
from .etf_study import BROAD

START = "2016-01-01"     # stock data starts 2015; 200-day average needs a year
MODEL_START = "2019-01-01"
TP = 0.05
MAX_POS = 5


def _by_date(df):
    return {d: row.dropna() for d, row in df.iterrows()}


def build(cfg, stock_probs=None):
    stocks = cfg["universe"]["tickers"]
    dc = cfg["data"]
    sp = download_prices(stocks + [cfg["universe"]["benchmark"]], dc["start"],
                         ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    sp.pop(cfg["universe"]["benchmark"])
    ep = download_prices(BROAD, "2009-01-01", ROOT / dc["cache_dir"], dc["max_cache_age_hours"])
    prices = {**{t: df[df.index >= dc["start"]] for t, df in ep.items()}, **sp}
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in prices.values()])))
    O, H, L, C = (wide(prices, f, cal) for f in ["Open", "High", "Low", "Close"])

    mean20, std20 = C.rolling(20, min_periods=20).mean(), C.rolling(20, min_periods=20).std()
    z = (C - mean20) / std20
    uptrend = C > C.rolling(200, min_periods=200).mean()
    etf_cols = [t for t in BROAD if t in C]
    stock_cols = [t for t in stocks if t in C]

    etf_z = -z[etf_cols]
    etf_raw = -(C[etf_cols] / mean20[etf_cols] - 1)
    stock_z = (-z[stock_cols]).where(uptrend[stock_cols])
    sc = {
        "etf_raw": _by_date(etf_raw),
        "etf_z": _by_date(etf_z),
        "stock_z": _by_date(stock_z),
        "mixed_z": _by_date(pd.concat([etf_z, stock_z], axis=1)),
    }
    if stock_probs is not None:
        # model probability for stocks (uptrend only), rescaled so it can sit beside ETF z-scores;
        # with group slot limits the two groups never compete for the same slot anyway
        p = stock_probs.unstack("ticker").reindex(index=cal, columns=stock_cols)
        p = p.where(uptrend[stock_cols])
        sc["mixed_model"] = _by_date(pd.concat([etf_z, p * 10], axis=1))
    groups = {**{t: "etf" for t in etf_cols}, **{t: "stock" for t in stock_cols}}
    return (O, H, L, C), sc, groups, etf_cols


def _scorer(d):
    return lambda i, day: d.get(day)


def variants(sc, groups, etf_cols, with_model):
    stt = {t: 0.00001 for t in etf_cols}
    v = {
        "A ETF only (5)": ("etf_raw", {}),
        "B stocks only (5)": ("stock_z", {}),
        "C mixed pool (5)": ("mixed_z", {}),
        "D 3 ETF + 2 stock": ("mixed_z", {"groups": groups, "group_limits": {"etf": 3, "stock": 2}}),
        "E 2 ETF + 3 stock": ("mixed_z", {"groups": groups, "group_limits": {"etf": 2, "stock": 3}}),
        "F 4 ETF + 1 stock": ("mixed_z", {"groups": groups, "group_limits": {"etf": 4, "stock": 1}}),
    }
    if with_model:
        v["G 3 ETF + 2 model-stock"] = ("mixed_model", {"groups": groups,
                                                         "group_limits": {"etf": 3, "stock": 2}})
    return v, stt


def run_one(cfg, frames, scorer, start, end, extra, stt, bench):
    strat = {**cfg["strategy"], "take_profit": TP, "max_positions": MAX_POS, "max_hold_days": None,
             "stop_loss": None, "target_decay": None, "regime_filter": False, **extra}
    costs = {**cfg["costs"], "stt_by_ticker": stt}
    res = run_backtest(*frames, scorer, start, end, cfg["capital"], strat, costs,
                       cfg["backtest"]["stuck_days"])
    return summarize(res, cfg["capital"], bench, cfg["costs"]["stcg_tax"])


def study(cfg, frames, sc, groups, etf_cols, bench, start, with_model):
    vs, stt = variants(sc, groups, etf_cols, with_model)
    C = frames[3]
    full = {}
    roll = {y: {} for y in (1, 3)}
    for name, (key, extra) in vs.items():
        s = run_one(cfg, frames, _scorer(sc[key]), start, None, extra, stt, bench)
        full[name] = {"cagr": s["cagr"], "final_Rs": s["final_equity"], "max_dd": s["max_drawdown"],
                      "trades/yr": s["trades_per_year"], "med_days": s["median_days_held"],
                      "stuck": s["avg_stuck_pct"], "worst_pos": s["worst_unrealised_position"]}
        for y in roll:
            starts = pd.date_range(start, C.index[-1] - pd.DateOffset(years=y), freq="3MS")
            r = [run_one(cfg, frames, _scorer(sc[key]), st, st + pd.DateOffset(years=y), extra, stt, bench)
                 for st in starts]
            cg = pd.Series([x["cagr"] for x in r])
            bc = pd.Series([x["nifty_cagr"] for x in r])
            roll[y][name] = {"windows": len(r), "median": cg.median(), "worst_10pct": cg.quantile(0.1),
                             "worst": cg.min(), "pct_losing": (cg < 0).mean(),
                             "pct_beat_niftybees": (cg > bc).mean()}
    return full, roll


def show(title, full, roll):
    pc = lambda v: f"{v:.1%}"  # noqa: E731
    f = pd.DataFrame(full).T
    for c in ["cagr", "max_dd", "stuck", "worst_pos"]:
        f[c] = f[c].map(pc)
    f["final_Rs"] = f["final_Rs"].map("{:,.0f}".format)
    f["trades/yr"] = f["trades/yr"].map("{:.1f}".format)
    print(f"\n### {title}: one run from the start date")
    print(f.to_string())
    for y, d in roll.items():
        r = pd.DataFrame(d).T
        for c in ["median", "worst_10pct", "worst", "pct_losing", "pct_beat_niftybees"]:
            r[c] = r[c].map(pc)
        print(f"\nRolling {y}-year windows (new start every quarter)")
        print(r.to_string())


def run_mixed(cfg, stock_probs):
    cfg = copy.deepcopy(cfg)
    frames, sc, groups, etf_cols = build(cfg, stock_probs)
    bench = frames[3]["NIFTYBEES.NS"].dropna()
    print(f"Capital Rs {cfg['capital']:,} | max {MAX_POS} positions | target +{TP:.0%} net | "
          f"never sell at a loss | {len(etf_cols)} ETFs + {len(groups) - len(etf_cols)} stocks")
    full, roll = study(cfg, frames, sc, groups, etf_cols, bench, START, with_model=False)
    show(f"Since {START} (10y, rule-based stock picks)", full, roll)
    full, roll = study(cfg, frames, sc, groups, etf_cols, bench, MODEL_START, with_model=True)
    show(f"Since {MODEL_START} (adds model-picked stocks; model predictions start 2019)", full, roll)
    print("\nNote: stocks = today's Nifty 50, so stock results carry survivorship bias (flattering).")
