"""Owner's follow-up to Results 13: if the model picks stocks likely to rise 5%, why sell at -5%?

    python -m trader winners-exits

Same model (gradient boosting, trained on 2006-16 only, label "+5% before -5%"), same picks, same
Monday-open buy and Tuesday-onwards checks; only the exit changes:
  +5% target / -5% stop (Results 13), +5% / -10%, +5% / no stop, no target / no stop (just hold
  Monday open -> Friday close). Compared with random picks under each exit and with Nifty 500.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .winners_study import FEATURES, SPLITS, build

EXITS = {"t5_s10": (0.05, 0.10), "t5_nostop": (0.05, 9.0), "hold": (9.0, 9.0)}
LABELS = {"": "+5% / -5% stop", "_t5_s10": "+5% / -10% stop", "_t5_nostop": "+5% / no stop",
          "_hold": "no target, no stop"}


def run_winners_exits(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    D, _ = build(cfg, exits=EXITS)
    D = D.dropna(subset=["mom12_1", "high52", "vol20"]).copy()
    X = D[FEATURES].replace([np.inf, -np.inf], np.nan)
    tr = (D.split == SPLITS[0][0]).to_numpy()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=200, l2_regularization=1.0)
    m.fit(X[tr], D.loc[tr, "win"])
    D["model"] = m.predict_proba(X)[:, 1]
    D["random"] = np.random.default_rng(0).random(len(D))
    rows = []
    for s, _, _ in SPLITS:
        g = D[D.split == s]
        bw = g.groupby("date")["bench"].first()
        for score in ("model", "combo", "random"):
            top = g.sort_values(score, ascending=False, kind="stable").groupby("date").head(5)
            for suf, lab in LABELS.items():
                col = "net" + suf
                wk = top.groupby("date")[col].mean()
                rows.append({"split": s, "picks": score, "exit": lab, "avg_trade": wk.mean(),
                             "worst_trade": top[col].min(), "weeks_up": (wk > 0).mean(),
                             "cagr": float(np.prod(1 + wk)) ** (52 / len(wk)) - 1})
        rows.append({"split": s, "picks": "Nifty 500", "exit": "-", "avg_trade": bw.mean(),
                     "weeks_up": (bw > 0).mean(), "cagr": float(np.prod(1 + bw)) ** (52 / len(bw)) - 1})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / cfg["backtest"]["output_dir"] / "winners_exits.csv", index=False)
    show = R.copy()
    for c in ("avg_trade", "worst_trade", "weeks_up", "cagr"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2%}" if c == "avg_trade" else f"{v:+.0%}")
    print("\n=== Same picks, different exits (buy Monday open, checks from Tuesday, sold by Friday close; "
          "Rs 5k each, after charges) ===")
    print(show.to_string(index=False))


def run_winners_hold(cfg):
    """Owner's actual rule: sell only when the day's high reaches +5% (after charges), never at a
    loss, no time limit; 5 slots; a free slot is filled on Friday with the best-scored stock
    (bought Monday open). Picks: model / momentum combo / random. Same simulator as everything
    else (momentum_study.simulate with intraday high/low, target checked from the day after buy)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from .momentum_study import simulate, stats
    from .winners_study import LAST
    D, cols = build(cfg)
    D = D.dropna(subset=["mom12_1", "high52", "vol20"]).copy()
    X = D[FEATURES].replace([np.inf, -np.inf], np.nan)
    tr = (D.split == SPLITS[0][0]).to_numpy()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=200, l2_regularization=1.0)
    m.fit(X[tr], D.loc[tr, "win"])
    D["model"] = m.predict_proba(X)[:, 1]
    p, n5 = LAST["panel"], LAST["n5"]
    cal = p["cal"]
    Oa, Cf = p["O"].to_numpy(), p["C"].ffill().to_numpy()
    HL = (p["H"].to_numpy(), p["L"].to_numpy())
    rng = np.random.default_rng(0)
    D["random"] = rng.random(len(D))
    orders = {}
    for score in ("model", "combo", "random"):
        od = {}
        for d, g in D.groupby("date"):
            od[cal.get_loc(d)] = g.sort_values(score, ascending=False, kind="stable")["j"].to_numpy()
        orders[score] = od
    rules = {"+5% target, hold losers (owner)": dict(target=0.05, no_loss=True, exit_rank=10**9, intraday=HL),
             "+10% target, hold losers": dict(target=0.10, no_loss=True, exit_rank=10**9, intraday=HL),
             "+5% target, else sell Friday (weekly)": None}
    rows = []
    for s, a, b in SPLITS:
        i0, i1 = cal.searchsorted(pd.Timestamp(a)), min(cal.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1)), len(cal))
        bench = (n5[i1 - 1] / n5[i0]) ** (248 / (i1 - i0)) - 1
        for rname, kw in rules.items():
            if kw is None:
                continue
            for score, od in orders.items():
                r = simulate(Oa, Cf, od, i0, i1, 25_000, cfg["costs"], **kw)
                st = stats(r, 25_000)
                rows.append({"split": s, "rule": rname, "picks": score, "cagr": st["cagr"], "max_dd": st["max_dd"],
                             "trades_yr": st["trades_yr"], "win_rate": st["win_rate"], "stuck": st["stuck_pct"],
                             "underwater_at_end": r["open_underwater"], "nifty500": bench})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / cfg["backtest"]["output_dir"] / "winners_hold.csv", index=False)
    show = R.copy()
    for c in ("cagr", "max_dd", "win_rate", "stuck", "nifty500"):
        show[c] = show[c].map(lambda v: f"{v:+.0%}")
    show["trades_yr"] = R["trades_yr"].map("{:.0f}".format)
    print("\n=== Hold losers until the target is hit (5 slots, Rs 25k, after charges; 'stuck' = share of money "
          "in stocks under water > 60 days) ===")
    print(show.to_string(index=False))
