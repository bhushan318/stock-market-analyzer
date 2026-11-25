"""Command line entry point.

    python -m trader evaluate            # walk-forward model quality (dev vs holdout)
    python -m trader backtest            # simulate the strategy + random baseline + rule variants
    python -m trader picks --cash 20000  # today's ranked candidates
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, load_config
from .backtest import run_backtest, summarize
from .data import download_prices, wide
from .features import build_panel
from .model import evaluate, fit_latest, walk_forward

PCT = {"total_return", "cagr", "max_drawdown", "avg_cash_idle_pct", "avg_stuck_pct",
       "max_stuck_pct", "worst_unrealised_position", "nifty_cagr", "nifty_max_drawdown"}


def label_threshold(cfg):
    """Gross move needed so the trade nets `take_profit` after charges (approx.)."""
    s = cfg["strategy"]
    return s["take_profit"] + (0.0075 if s.get("take_profit_is_net", True) else 0.0)


def load(cfg, refresh=False):
    u, d = cfg["universe"], cfg["data"]
    tickers = u["tickers"] + [u["benchmark"]]
    print(f"Loading prices for {len(tickers)} tickers since {d['start']} ...")
    prices = download_prices(tickers, d["start"], ROOT / d["cache_dir"],
                             d["max_cache_age_hours"], refresh)
    market = prices.pop(u["benchmark"])
    return prices, market


def get_panel_and_probs(cfg, prices, market, model_name=None):
    m = cfg["model"]
    model_name = model_name or m["name"]
    panel = build_panel(prices, market, label_threshold(cfg), m["horizon_days"])
    key = json.dumps([model_name, label_threshold(cfg), m["horizon_days"], m["first_test_date"],
                      m["retrain_every_days"], str(panel.index[-1]), len(panel)])
    cache = ROOT / cfg["data"]["cache_dir"] / f"probs_{hashlib.md5(key.encode()).hexdigest()[:10]}.pkl"
    if cache.exists():
        probs = pd.read_pickle(cache)
    else:
        print(f"Walk-forward training ({model_name}) ...")
        probs = walk_forward(panel, model_name, m["first_test_date"], m["retrain_every_days"],
                             m["horizon_days"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        probs.to_pickle(cache)
    return panel, probs


def fmt(df):
    df = df.copy().astype(object)
    for r in df.index:
        for c in df.columns:
            v = df.loc[r, c]
            if isinstance(v, (float, np.floating)):
                df.loc[r, c] = f"{v:.1%}" if r in PCT else f"{v:,.1f}"
    return df


def cmd_evaluate(cfg, args):
    prices, market = load(cfg, args.refresh)
    m = cfg["model"]
    for name in ["logreg", "hgb"]:
        panel, probs = get_panel_and_probs(cfg, prices, market, name)
        print(f"\n== {name}: target +{label_threshold(cfg):.2%} within {m['horizon_days']} days ==")
        print(evaluate(probs, panel, m["holdout_start"], m["top_k"]).round(3).to_string())
    print("\nhit_rate_topK above base_rate = the ranking adds value. AUC 0.5 = coin flip.")


def cmd_backtest(cfg, args):
    prices, market = load(cfg, args.refresh)
    panel, probs = get_panel_and_probs(cfg, prices, market)
    cal = market.index
    O, H, L, C = (wide(prices, f, cal) for f in ["Open", "High", "Low", "Close"])
    by_date = {d: g.droplevel("date") for d, g in probs.dropna().groupby(level="date")}

    def model_scores(i, d):
        return by_date.get(d)

    def random_scores(seed):
        rng = np.random.default_rng(seed)

        def f(i, d):
            s = by_date.get(d)  # same tradable set as the model, random order
            return None if s is None else pd.Series(rng.random(len(s)), index=s.index)
        return f

    s, c, b, m = cfg["strategy"], cfg["costs"], cfg["backtest"], cfg["model"]
    out_dir = ROOT / b["output_dir"]
    out_dir.mkdir(exist_ok=True)

    periods = {"full": m["first_test_date"], "holdout": m["holdout_start"]}
    variants = {
        "your_rules": {},
        "max_pos_5": {"max_positions": 5},
        "max_pos_8": {"max_positions": 8},
        "time_exit_60d": {"max_hold_days": 60},
        "stop_loss_15": {"stop_loss": 0.15},
        "stop_loss_10": {"stop_loss": 0.10},
        "stop10_time60": {"stop_loss": 0.10, "max_hold_days": 60},
    }
    for pname, start in periods.items():
        rows = {}
        for vname, over in variants.items():
            strat = {**s, **over}
            res = run_backtest(O, H, L, C, model_scores, start, None, cfg["capital"], strat, c,
                               b["stuck_days"])
            rows[vname] = summarize(res, cfg["capital"], market["Close"], c["stcg_tax"])
            if vname == "your_rules":
                res["trades"].to_csv(out_dir / f"trades_{pname}.csv", index=False)
                res["equity"].to_csv(out_dir / f"equity_{pname}.csv")
                res["open"].to_csv(out_dir / f"open_positions_{pname}.csv", index=False)
                open_end = res["open"]
        rand = [summarize(run_backtest(O, H, L, C, random_scores(k), start, None, cfg["capital"],
                                       s, c, b["stuck_days"]), cfg["capital"], market["Close"])
                for k in range(b["random_baseline_runs"])]
        rd = pd.DataFrame(rand).drop(columns=["period"])
        rows["random_median"] = rd.median(numeric_only=True).to_dict()
        rows["random_p90"] = rd.quantile(0.9, numeric_only=True).to_dict()
        table = pd.DataFrame(rows)
        print(f"\n===== {pname.upper()} period: {rows['your_rules']['period']} =====")
        print(fmt(table.drop(index="period")).to_string())
        if len(open_end):
            print("\nPositions still open at the end (your rules):")
            print(open_end.assign(unrealised=open_end["unrealised"].map("{:.1%}".format)).to_string(index=False))
    print(f"\nTrade logs and equity curves written to {out_dir}")


def cmd_picks(cfg, args):
    prices, market = load(cfg, args.refresh)
    m, s = cfg["model"], cfg["strategy"]
    panel = build_panel(prices, market, label_threshold(cfg), m["horizon_days"])
    date, probs = fit_latest(panel, m["name"], m["horizon_days"])
    held = set(args.exclude or [])
    slots = args.slots or s["max_positions"]
    budget = args.cash / slots
    rows = []
    for t, p in probs.items():
        if t in held:
            continue
        close = prices[t]["Close"].iloc[-1]
        rows.append({"ticker": t, "prob": p, "close": close,
                     "shares_for_slot": int(budget // (close * 1.003)),
                     "gross_target": close * (1 + label_threshold(cfg))})
    df = pd.DataFrame(rows).head(args.top)
    print(f"\nScores as of close {date.date()} | slot budget Rs {budget:,.0f} ({slots} slots)")
    print(df.to_string(index=False, formatters={"prob": "{:.1%}".format, "close": "{:,.1f}".format,
                                                 "gross_target": "{:,.1f}".format}))
    print("\nshares_for_slot = 0 means the stock is too expensive for one slot.")


def cmd_ideas(cfg, args):
    from .experiments import run_ideas
    prices, market = load(cfg, args.refresh)
    panel, probs = get_panel_and_probs(cfg, prices, market)
    run_ideas(cfg, prices, market, panel, probs, holdout=args.holdout)


def cmd_etf(cfg, args):
    from .etf_study import run_study
    run_study(cfg, broad=args.broad)


def main():
    ap = argparse.ArgumentParser(prog="python -m trader")
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("backtest")
    sub.add_parser("ideas").add_argument("--holdout", action="store_true")
    sub.add_parser("etf").add_argument("--broad", action="store_true",
                                       help="broad index/asset-class ETFs only")
    p = sub.add_parser("picks")
    p.add_argument("--cash", type=float, default=None)
    p.add_argument("--slots", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--exclude", nargs="*", help="tickers already held")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "picks" and args.cash is None:
        args.cash = cfg["capital"]
    {"evaluate": cmd_evaluate, "backtest": cmd_backtest, "picks": cmd_picks,
     "ideas": cmd_ideas, "etf": cmd_etf}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
