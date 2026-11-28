"""Compare strategy ideas that keep "never sell at a loss" - on the DEV period only.

    python -m trader ideas                  # dev period (2019 -> holdout_start)
    python -m trader ideas --holdout        # run once, at the end, on the holdout
"""

import numpy as np
import pandas as pd

from . import ROOT
from .backtest import run_backtest, summarize
from .data import download_prices, wide
from .features import build_panel

ETFS = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "PSUBNKBEES.NS", "ITBEES.NS",
        "PHARMABEES.NS", "GOLDBEES.NS", "SILVERBEES.NS", "CPSEETF.NS", "MON100.NS",
        "MAFANG.NS", "AUTOBEES.NS", "FMCGIETF.NS", "MOM100.NS", "ALPHA.NS"]

SHOW = ["cagr", "max_drawdown", "trades_per_year", "median_days_held", "p90_days_held",
        "avg_cash_idle_pct", "avg_stuck_pct", "worst_unrealised_position",
        "open_underwater_end", "charges_paid"]

DECAY = [[60, 0.02], [120, 0.005]]
AVG = {"trigger": 0.10, "max_adds": 2, "size": 1.0}


def _by_date(series):
    return {d: g.droplevel("date") for d, g in series.dropna().groupby(level="date")}


def _scorer(by_date, panel=None, uptrend=False):
    up = None
    if uptrend:
        ok = panel["dist_sma_200"] > 0
        up = {d: set(g.index.get_level_values("ticker")[g.to_numpy()])
              for d, g in ok.groupby(level="date")}

    def f(i, d):
        s = by_date.get(d)
        if s is None:
            return None
        if up is not None:
            s = s[s.index.isin(up.get(d, ()))]
        return s
    return f


def _random(by_date, seed):
    rng = np.random.default_rng(seed)

    def f(i, d):
        s = by_date.get(d)
        return None if s is None else pd.Series(rng.random(len(s)), index=s.index)
    return f


def run_ideas(cfg, stock_prices, market, stock_panel, stock_probs, holdout=False):
    from .__main__ import fmt  # local import to avoid a cycle

    m, b = cfg["model"], cfg["backtest"]
    start, end = (m["holdout_start"], None) if holdout else (m["first_test_date"],
                                                              pd.Timestamp(m["holdout_start"]) - pd.Timedelta(days=1))
    base = {**cfg["strategy"], "max_hold_days": None, "stop_loss": None}
    cap, costs = cfg["capital"], cfg["costs"]
    mkt = market["Close"]

    # --- stocks
    cal = market.index
    SO, SH, SL, SC = (wide(stock_prices, f, cal) for f in ["Open", "High", "Low", "Close"])
    model_bd = _by_date(stock_probs)
    dip_bd = _by_date(-stock_panel["dist_sma_20"])  # furthest below 20-day average first

    # --- ETFs (lower STT: 0.001% on sell only)
    etf_prices = download_prices(ETFS, cfg["data"]["start"], ROOT / cfg["data"]["cache_dir"],
                                 cfg["data"]["max_cache_age_hours"])
    etf_panel = build_panel(etf_prices, market, 0.05, m["horizon_days"])
    EO, EH, EL, EC = (wide(etf_prices, f, cal) for f in ["Open", "High", "Low", "Close"])
    etf_costs = {**costs, "stt": 0.00001}
    etf_dip_bd = _by_date(-etf_panel["dist_sma_20"])

    S = (SO, SH, SL, SC)
    E = (EO, EH, EL, EC)
    filt = dict(regime_filter=True)
    combo = dict(regime_filter=True, target_decay=DECAY)
    combo_avg = dict(combo, reserve_pct=0.3, avg_down=AVG)
    runs = {
        # name: (prices, scorer, strategy overrides, costs)
        "S1 model (your rules)": (S, _scorer(model_bd), {}, costs),
        "S2 + market filter": (S, _scorer(model_bd), filt, costs),
        "S3 + uptrend only": (S, _scorer(model_bd, stock_panel, True), {}, costs),
        "S4 + target decay": (S, _scorer(model_bd), {"target_decay": DECAY}, costs),
        "S5 + avg down (30% reserve)": (S, _scorer(model_bd), {"reserve_pct": 0.3, "avg_down": AVG}, costs),
        "S6 filter+uptrend+decay": (S, _scorer(model_bd, stock_panel, True), combo, costs),
        "S7 S6 + avg down": (S, _scorer(model_bd, stock_panel, True), combo_avg, costs),
        "S8 dip-in-uptrend+decay": (S, _scorer(dip_bd, stock_panel, True), combo, costs),
        "E1 ETF dip (your rules)": (E, _scorer(etf_dip_bd), {}, etf_costs),
        "E2 ETF dip + decay": (E, _scorer(etf_dip_bd), {"target_decay": DECAY}, etf_costs),
        "E3 ETF dip + avg down": (E, _scorer(etf_dip_bd), {"reserve_pct": 0.3, "avg_down": AVG}, etf_costs),
        "E4 ETF dip + decay + avg": (E, _scorer(etf_dip_bd), {"target_decay": DECAY, "reserve_pct": 0.3, "avg_down": AVG}, etf_costs),
    }
    rows = {}
    for name, (px, scorer, over, c) in runs.items():
        res = run_backtest(*px, scorer, start, end, cap, {**base, **over}, c, b["stuck_days"], mkt)
        rows[name] = summarize(res, cap, mkt, c["stcg_tax"])
    for name, (px, bd, c) in {"S0 stocks random (median)": (S, model_bd, costs),
                              "E0 ETF random (median)": (E, etf_dip_bd, etf_costs)}.items():
        rs = [summarize(run_backtest(*px, _random(bd, k), start, end, cap, base, c,
                                     b["stuck_days"], mkt), cap, mkt) for k in range(b["random_baseline_runs"])]
        rows[name] = pd.DataFrame(rs).median(numeric_only=True).to_dict()

    period = rows["S1 model (your rules)"]["period"]
    table = pd.DataFrame(rows).T[SHOW + ["nifty_cagr"]]
    table = table.loc[sorted(table.index)]
    print(f"\n===== {'HOLDOUT' if holdout else 'DEV'} period {period} | capital Rs {cap:,} | "
          f"never sell at a loss in every row =====")
    print(fmt(table.T).T.to_string())
