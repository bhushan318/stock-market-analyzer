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


def cmd_mixed(cfg, args):
    from .mixed_study import run_mixed
    prices, market = load(cfg, args.refresh)
    _, probs = get_panel_and_probs(cfg, prices, market)
    run_mixed(cfg, probs)


def cmd_universe(cfg, args):
    from .universe_study import run_universe
    run_universe(cfg)


def cmd_momentum(cfg, args):
    from .momentum_study import run_momentum
    run_momentum(cfg)


def cmd_momentum_pit(cfg, args):
    from .momentum_pit import run_momentum_pit
    run_momentum_pit(cfg)


def cmd_combo(cfg, args):
    from .combo_study import run_combo
    run_combo(cfg)


def cmd_robust(cfg, args):
    from .robust_study import run_robust
    run_robust(cfg)


def cmd_delivery(cfg, args):
    from .delivery_study import run_delivery
    run_delivery(cfg)


def cmd_levers(cfg, args):
    from .levers import run_levers
    run_levers(cfg)


def cmd_exits(cfg, args):
    from .exits_study import run_exits
    run_exits(cfg)


def cmd_longrun(cfg, args):
    from .longrun_study import run_longrun
    run_longrun(cfg)


def cmd_history(cfg, args):
    from .history import main as history_main
    history_main(cfg, args)


def cmd_paper(cfg, args):
    from .paper import main as paper_main
    paper_main(cfg, args)


def cmd_collect(cfg, args):
    from .collect import main as collect_main
    collect_main(cfg, args)


def main():
    ap = argparse.ArgumentParser(prog="python -m trader")
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("backtest")
    sub.add_parser("ideas").add_argument("--holdout", action="store_true")
    sub.add_parser("mixed")
    sub.add_parser("universe")
    sub.add_parser("momentum", help="Gate 0 step 1: momentum / relative strength study")
    sub.add_parser("levers", help="Gate 0 step 4: lever sensitivities (crude, USD/INR, gold, rates)")
    sub.add_parser("combo", help="Gate 0: combo momentum - turnover, capital, market filter, exits")
    sub.add_parser("exits", help="Gate 0: target/stop as standing orders on daily high/low vs close-checked")
    sub.add_parser("robust", help="Gate 0: robustness of combo + market filter, core+satellite mixes")
    sub.add_parser("delivery", help="Gate 0 step 2: delivery %% signals (point-in-time universe)")
    sub.add_parser("momentum-pit", help="Gate 0 step 1 without survivorship bias (NSE bhavcopy universe)")
    c = sub.add_parser("collect", help="NSE data collectors (daily / news / backfill / status / export)")
    c.add_argument("job", choices=["daily", "news", "backfill", "hourly", "status", "export"])
    c.add_argument("--date", help="daily: collect as of this date (YYYY-MM-DD)")
    c.add_argument("--start", default="2020-01-01", help="backfill start date")
    c.add_argument("--end", help="backfill end date (default today)")
    c.add_argument("--weekends", action="store_true", help="backfill: try weekend special sessions only")
    sub.add_parser("longrun", help="Gate 1 rule on 2006-2019 (data/history.db), out of sample")
    h = sub.add_parser("history", help="old NSE bhavcopies 2005-2019 into data/history.db (PC, research)")
    h.add_argument("job", choices=["fetch", "status"])
    h.add_argument("--start", default="2005-01-01")
    h.add_argument("--end", default="2019-12-31")
    pp = sub.add_parser("paper", help="Gate 1 paper trading (no real orders): init / run / status / telegram (setup) / chatid")
    pp.add_argument("job", choices=["init", "run", "status", "chatid", "telegram"])
    pp.add_argument("--force", action="store_true", help="init: wipe and restart the paper books")
    pp.add_argument("--quiet", action="store_true", help="run: print only, no Telegram")
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
     "ideas": cmd_ideas, "etf": cmd_etf, "mixed": cmd_mixed, "universe": cmd_universe, "momentum": cmd_momentum, "momentum-pit": cmd_momentum_pit, "delivery": cmd_delivery, "combo": cmd_combo, "robust": cmd_robust, "levers": cmd_levers, "collect": cmd_collect, "paper": cmd_paper, "exits": cmd_exits, "history": cmd_history, "longrun": cmd_longrun}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
