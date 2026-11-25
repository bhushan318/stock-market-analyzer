import numpy as np
import pandas as pd
import pytest

from trader import load_config
from trader.backtest import run_backtest
from trader.costs import buy_charges, sell_charges, target_price
from trader.features import build_panel, hit_target_label, market_features, stock_features
from trader.model import fold_plan

COSTS = load_config()["costs"]


def make_prices(n=400, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    v = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v}, index=idx)


def test_features_have_no_lookahead():
    df, mkt = make_prices(seed=1), make_prices(seed=2)
    full = stock_features(df, market_features(mkt))
    cut = 300
    part = stock_features(df.iloc[:cut], market_features(mkt.iloc[:cut]))
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-9)


def test_label_definition_and_tail_is_nan():
    idx = pd.bdate_range("2021-01-01", periods=8)
    df = pd.DataFrame({"Open": [100] * 8, "High": [100, 101, 102, 106, 100, 100, 100, 100],
                       "Low": [99] * 8, "Close": [100] * 8, "Volume": [1] * 8}, index=idx, dtype=float)
    lab = hit_target_label(df, 0.05, horizon=3)
    # row 0: entry Open[1]=100, highs[1..3] include 106 -> hit
    assert lab.iloc[0] == 1.0
    # row 3: highs[4..6] max 100 -> miss
    assert lab.iloc[3] == 0.0
    # last 3 rows: future window incomplete -> unknown, not 0
    assert lab.iloc[-3:].isna().all()


def test_panel_keeps_latest_rows_for_prediction():
    prices = {"A.NS": make_prices(seed=3), "B.NS": make_prices(seed=4)}
    panel = build_panel(prices, make_prices(seed=5), 0.05, 20)
    last = panel.index.get_level_values("date").max()
    assert last == prices["A.NS"].index[-1]
    assert panel.xs(last, level="date")["label"].isna().all()


def test_fold_plan_purges_label_window():
    for tr_end, s, e in fold_plan(1000, 500, 63, 20):
        assert tr_end + 20 < s          # last training label window closed before fold start
        assert s < e <= 1000


def test_costs_reasonable():
    b, s = buy_charges(3000, COSTS), sell_charges(3000, COSTS)
    assert 3 < b < 4.5                  # ~0.115% of 3000
    assert 18 < s < 21                  # STT 3 + DP ~15.9
    shares, px = 30, 100.0
    basis = shares * px + buy_charges(shares * px, COSTS)
    tgt = target_price(shares, px, basis, 0.05, COSTS, net=True)
    proceeds = shares * tgt - sell_charges(shares * tgt, COSTS)
    assert proceeds == pytest.approx(basis * 1.05, rel=1e-9)


def _frames(prices):
    cal = next(iter(prices.values())).index
    return [pd.DataFrame({t: d[f] for t, d in prices.items()}).reindex(cal)
            for f in ["Open", "High", "Low", "Close"]]


def test_backtest_never_sells_at_a_loss_and_conserves_cash():
    prices = {f"S{k}": make_prices(n=600, seed=10 + k, drift=0.0) for k in range(8)}
    O, H, L, C = _frames(prices)
    rng = np.random.default_rng(0)
    scores = lambda i, d: pd.Series(rng.random(8), index=C.columns)  # noqa: E731
    strat = {"take_profit": 0.05, "take_profit_is_net": True, "max_positions": 4,
             "min_position_value": 1000, "cooldown_days": 3, "min_prob": 0.0,
             "max_hold_days": None, "stop_loss": None}
    res = run_backtest(O, H, L, C, scores, C.index[50], None, 20000, strat, COSTS)
    tr = res["trades"]
    assert len(tr) > 10
    assert (tr["pnl"] > 0).all()
    assert (tr["sell_price"] >= tr["buy_price"]).all()
    assert (tr["reason"].isin(["target", "target_gap"])).all()
    assert (res["equity"]["cash"] >= -1e-6).all()
    assert (res["equity"]["n_positions"] <= 4).all()
