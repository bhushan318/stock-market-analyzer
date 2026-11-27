""""Weekly winners": can we pick, on Friday, the stocks that gain ~5% in the next week?

    python -m trader winners

Owner's design (2026-11-27):
  - decide at Friday's close, buy at Monday's open (the buy price);
  - target / stop are checked from the NEXT session onwards (never the buy day's high/low);
  - label = the trade you could really take: target +TARGET hit before stop -STOP (if both are
    touched on one day the stop counts; a gap through either fills at the open), otherwise sold
    at the close of the HOLD-th session after the buy day. Charges on a Rs 5k position included.
  - time-series split, never random: train 2006-2016, tune 2017-2019, final test 2020-2026
    (looked at once). Features use only data up to the Friday close (tested for look-ahead).

Compared on the same weeks and the same exit rules: the model's top 5, the momentum "combo" top 5,
5 random eligible stocks (the base rate), and Nifty 500. Universe: point-in-time top 500 by
turnover, price >= Rs 10, 20-day median turnover >= Rs 2 cr. No delivery % (not in 2006-19 data).
"""

import numpy as np
import pandas as pd

from . import ROOT
from .costs import buy_charges, sell_charges
from .data import download_prices
from .momentum_pit import pit_signals
from .momentum_study import weekly_dates
from .pit import liquid_universe, load_panel

TARGET, STOP, HOLD = 0.05, 0.05, 4   # HOLD 4 = Tuesday..Friday: sold before the next Monday's buys
POS_RS = 5_000
SPLITS = [("train 2006-16", "2006-07-01", "2016-12-31"), ("tune 2017-19", "2017-01-01", "2019-12-31"),
          ("TEST 2020-26", "2020-01-01", "2026-12-31")]
FEATURES = ["mom12_1", "mom6_1", "mom1m", "ret5", "ret1", "high52", "vol20", "vol_ratio", "dist50", "dist200",
            "gap1", "range5", "turn_rank", "combo", "mkt_above200", "mkt_ret20"]


def trade_outcome(O, H, L, C, i, target=TARGET, stop=STOP, hold=HOLD, slip=0.0005):
    """Gross exit/entry ratio for every stock of a trade decided at close i (entry O[i+1]).
    Target/stop from session i+2 to i+1+hold; else close of i+1+hold. Returns (ratio, hit) with
    hit = +1 target, -1 stop, 0 time exit. NaN where no entry price."""
    entry = O[i + 1] * (1 + slip)
    tp, sp = entry * (1 + target), entry * (1 - stop)
    out = np.full(entry.shape, np.nan)
    hit = np.zeros(entry.shape)
    open_ = ~np.isnan(entry)
    for k in range(i + 2, i + 2 + hold):
        o, h, lo = O[k], H[k], L[k]
        o = np.where(np.isnan(o), C[k - 1], o)
        s_gap = open_ & (o <= sp)
        t_gap = open_ & ~s_gap & (o >= tp)
        s_in = open_ & ~s_gap & ~t_gap & (lo <= sp)
        t_in = open_ & ~s_gap & ~t_gap & ~s_in & (h >= tp)
        out[s_gap] = o[s_gap] * (1 - slip)
        out[t_gap] = o[t_gap]
        out[s_in] = sp[s_in] * (1 - slip)
        out[t_in] = tp[t_in]
        hit[s_gap | s_in], hit[t_gap | t_in] = -1, 1
        open_ &= ~(s_gap | t_gap | s_in | t_in)
    last = C[i + 1 + hold]
    out[open_] = last[open_] * (1 - slip)
    return out / entry, hit


def net_return(gross_ratio, costs, pos=POS_RS):
    """After all charges on a `pos` rupee position."""
    buy_v = pos
    sell_v = pos * gross_ratio
    return (sell_v - sell_charges(sell_v, costs)) / (buy_v + buy_charges(buy_v, costs)) - 1


def features(p, n5, combo):
    C = p["C"].ffill(limit=5)
    O, H, L, TO = p["O"], p["H"], p["L"], p["TO"]
    ret = C / C.shift(1) - 1
    f = {"mom12_1": C.shift(21) / C.shift(252) - 1, "mom6_1": C.shift(21) / C.shift(126) - 1,
         "mom1m": C / C.shift(21) - 1, "ret5": C / C.shift(5) - 1, "ret1": ret,
         "high52": C / C.rolling(252, min_periods=240).max(), "vol20": ret.rolling(20, min_periods=15).std(),
         "vol_ratio": p["V"].rolling(5).mean() / p["V"].rolling(60, min_periods=40).median(),
         "dist50": C / C.rolling(50).mean() - 1, "dist200": C / C.rolling(200, min_periods=150).mean() - 1,
         "gap1": O / C.shift(1) - 1, "range5": ((H - L) / C).rolling(5).mean() / ((H - L) / C).rolling(60, min_periods=40).mean(),
         "turn_rank": TO.rolling(20, min_periods=15).median().rank(axis=1, pct=True), "combo": combo}
    m = pd.Series(n5, index=C.index)
    mk = {"mkt_above200": (m > m.rolling(200, min_periods=150).mean()).astype(float), "mkt_ret20": m / m.shift(20) - 1}
    return f, mk


def build(cfg):
    print("Loading 2005-2026 panel ...", flush=True)
    p = load_panel(db=[ROOT / "data" / "history.db", ROOT / "data" / "market.db"], start="2005-01-01")
    cal = p["cal"]
    n5 = download_prices(["^CRSLDX"], "2005-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal).to_numpy()
    sig, elig = pit_signals(p["C"], p["TO"], liquid_universe(p["C"], p["TO"]))
    f, mk = features(p, n5, sig["combo"])
    rebal = weekly_dates(cal)
    rebal = rebal[(rebal >= pd.Timestamp(SPLITS[0][1])) & (rebal <= cal[-(HOLD + 3)])]
    O, H, L = (p[k].to_numpy() for k in "OHL")
    C = p["C"].ffill(limit=5).to_numpy()
    E = elig.to_numpy()
    F = {k: v.to_numpy() for k, v in f.items()}
    rows = []
    for d in rebal:
        i = cal.get_loc(d)
        ok = E[i] & ~np.isnan(O[i + 1])
        if not ok.any():
            continue
        g, hit = trade_outcome(O, H, L, C, i)
        cols = np.where(ok & ~np.isnan(g))[0]
        rec = {"date": np.repeat(d, len(cols)), "j": cols, "gross": g[cols], "hit": hit[cols]}
        for k, a in F.items():
            rec[k] = a[i, cols]
        for k, s in mk.items():
            rec[k] = np.repeat(s.iloc[i], len(cols))
        rec["bench"] = np.repeat(n5[i + 1 + HOLD] / n5[i] - 1, len(cols))
        rows.append(pd.DataFrame(rec))
    D = pd.concat(rows, ignore_index=True)
    D["net"] = net_return(D["gross"].to_numpy(), cfg["costs"])
    D["win"] = (D["hit"] == 1).astype(int)
    D["split"] = pd.cut(D["date"], [pd.Timestamp(a) for _, a, _ in SPLITS] + [pd.Timestamp("2100-01-01")],
                        labels=[s for s, _, _ in SPLITS], right=False)
    return D, p["C"].columns


def pick(D, score, k=5):
    """Top-k per week by `score` column -> mean net return per week (equal weight)."""
    top = D.sort_values(score, ascending=False, kind="stable").groupby("date").head(k)
    return top.groupby("date").agg(net=("net", "mean"), win=("win", "mean"), stop=("hit", lambda h: (h == -1).mean()))


def summarise(W, label, bench=None):
    n = len(W)
    r = W["net"]
    # 5 equal positions, the whole basket turned over weekly: compound weekly basket returns
    growth = float(np.prod(1 + r))
    yrs = n / 52
    return {"picks": label, "weeks": n, "avg_trade": r.mean(), "target_hit": W["win"].mean(),
            "stop_hit": W["stop"].mean(), "weeks_up": (r > 0).mean(),
            "cagr": growth ** (1 / yrs) - 1 if yrs > 0 and growth > 0 else -1.0}


def run_winners(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    D, cols = build(cfg)
    D = D.dropna(subset=["mom12_1", "high52", "vol20"]).copy()
    print(f"{len(D):,} stock-weeks, {D.date.nunique()} weeks. Base rate (target +{TARGET:.0%} hit before "
          f"-{STOP:.0%}, checked from the day after the buy): "
          + ", ".join(f"{s} {g.win.mean():.1%}" for s, g in D.groupby("split", observed=True)), flush=True)
    X = D[FEATURES].replace([np.inf, -np.inf], np.nan)
    tr = (D.split == SPLITS[0][0]).to_numpy()
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
              "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=200, l2_regularization=1.0)}
    rng = np.random.default_rng(0)
    D["random"] = rng.random(len(D))
    for name, m in models.items():
        Xm = X.fillna(X[tr].median()) if name == "logreg" else X
        m.fit(Xm[tr], D.loc[tr, "win"])
        D[name] = m.predict_proba(Xm)[:, 1]
        # also a model that predicts the net return directly would be the next step; keep one label
    from sklearn.metrics import roc_auc_score
    rows = []
    for s, _, _ in SPLITS:
        g = D[D.split == s]
        for score in ("logreg", "hgb", "combo", "random"):
            w = summarise(pick(g, score), score)
            w["split"] = s
            if score in ("logreg", "hgb"):
                w["auc"] = roc_auc_score(g["win"], g[score])
            rows.append(w)
        bw = g.groupby("date")["bench"].first()
        rows.append({"split": s, "picks": "Nifty 500 (same weeks)", "weeks": len(bw), "avg_trade": bw.mean(),
                     "weeks_up": (bw > 0).mean(), "cagr": float(np.prod(1 + bw)) ** (52 / len(bw)) - 1})
    R = pd.DataFrame(rows)[["split", "picks", "weeks", "avg_trade", "target_hit", "stop_hit", "weeks_up", "cagr", "auc"]]
    out = ROOT / cfg["backtest"]["output_dir"]
    R.to_csv(out / "winners_summary.csv", index=False)
    imp = pd.Series(models["logreg"][-1].coef_[0], index=FEATURES).sort_values()
    show = R.copy()
    for c in ("avg_trade", "target_hit", "stop_hit", "weeks_up", "cagr"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2%}" if c == "avg_trade" else f"{v:.0%}")
    show["auc"] = show["auc"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    print(f"\n=== Top 5 each Friday, buy Monday open, +{TARGET:.0%} target / -{STOP:.0%} stop from Tuesday, "
          f"else sell after {HOLD} sessions; Rs 5k each, after charges ===")
    print("(models trained on 2006-16 only; 'tune' is for choosing, 'TEST' is the one honest number)")
    print(show.to_string(index=False))
    print("\nLogistic regression weights (standardised; + = more likely to hit the target first):")
    print(imp.round(3).to_string())
    print(f"\nCSV: {out}/winners_summary.csv")
