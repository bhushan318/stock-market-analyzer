"""Walk-forward model: one pooled model across all stocks, retrained on a schedule.

Fixes versus the old StockPrediction.py:
- No look-ahead: training rows for a fold end `horizon + 1` days before the fold starts,
  so every training label was fully known at the time the model would have been trained.
- The scaler/imputer is fit inside the pipeline on training rows only.
- Predictions used for evaluation and backtesting are all out-of-sample.
- The model is never chosen using the holdout period.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns


def make_model(name, seed=0):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=300,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000))
    raise ValueError(f"unknown model {name!r}")


def fold_plan(n_dates, first_test_pos, retrain_every, horizon):
    """[(train_end_pos_inclusive, test_start_pos, test_end_pos_exclusive), ...]

    A model used from the close of day s-1 onward may only learn from rows whose label
    window (t+1 .. t+horizon) had closed by then: t + horizon <= s - 1.
    """
    folds = []
    for s in range(first_test_pos, n_dates, retrain_every):
        folds.append((s - horizon - 1, s, min(s + retrain_every, n_dates)))
    return folds


def _date_pos(panel):
    dates = panel.index.get_level_values("date")
    cal = pd.DatetimeIndex(dates.unique()).sort_values()
    return cal, cal.get_indexer(dates)


def walk_forward(panel, model_name, first_test_date, retrain_every, horizon, verbose=True):
    """Out-of-sample probability for every (date, ticker) row from `first_test_date` on."""
    feats = feature_columns(panel)
    X = panel[feats]
    y = panel["label"]
    cal, pos = _date_pos(panel)
    first = cal.searchsorted(pd.Timestamp(first_test_date))
    has_x = X.notna().sum(axis=1) >= len(feats) * 0.8  # allow a few NaNs (young stocks)

    out = pd.Series(np.nan, index=panel.index, name="prob")
    folds = fold_plan(len(cal), first, retrain_every, horizon)
    for k, (train_end, s, e) in enumerate(folds):
        tr = (pos <= train_end) & y.notna().to_numpy() & has_x.to_numpy()
        te = (pos >= s) & (pos < e) & has_x.to_numpy()
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        m = make_model(model_name)
        m.fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
        if verbose and (k % 8 == 0 or k == len(folds) - 1):
            print(f"  fold {k + 1}/{len(folds)}: train to {cal[max(train_end, 0)].date()}, "
                  f"predict {cal[s].date()}..{cal[e - 1].date()} ({tr.sum():,} train rows)")
    return out


def fit_latest(panel, model_name, horizon):
    """Train on everything whose label is known and score the most recent date (for live picks)."""
    feats = feature_columns(panel)
    X, y = panel[feats], panel["label"]
    cal, pos = _date_pos(panel)
    has_x = X.notna().sum(axis=1) >= len(feats) * 0.8
    tr = y.notna().to_numpy() & has_x.to_numpy()
    m = make_model(model_name).fit(X[tr], y[tr])
    last = (pos == len(cal) - 1) & has_x.to_numpy()
    probs = pd.Series(m.predict_proba(X[last])[:, 1], index=panel.index[last])
    return cal[-1], probs.droplevel("date").sort_values(ascending=False)


def evaluate(probs, panel, holdout_start, top_k=5):
    """AUC, base rate and precision@top_k, separately for the dev period and the holdout."""
    df = pd.DataFrame({"prob": probs, "label": panel["label"]}).dropna()
    dates = df.index.get_level_values("date")
    rows = []
    for name, mask in [("dev", dates < pd.Timestamp(holdout_start)),
                       ("holdout", dates >= pd.Timestamp(holdout_start))]:
        d = df[mask]
        if d.empty or d["label"].nunique() < 2:
            continue
        top = d.groupby(level="date", group_keys=False).apply(lambda g: g.nlargest(top_k, "prob"))
        rows.append({
            "period": name,
            "from": d.index.get_level_values("date").min().date(),
            "to": d.index.get_level_values("date").max().date(),
            "rows": len(d),
            "auc": roc_auc_score(d["label"], d["prob"]),
            "base_rate": d["label"].mean(),
            f"hit_rate_top{top_k}": top["label"].mean(),
        })
    return pd.DataFrame(rows).set_index("period")
