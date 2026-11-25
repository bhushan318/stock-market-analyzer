import numpy as np
import pandas as pd

from trader import load_config
from trader.momentum_study import signals, simulate, weekly_dates


def _panel(n_days=400, n=6, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-01", periods=n_days)
    C = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n)), axis=0)),
                     index=cal, columns=[f"S{k}" for k in range(n)])
    V = pd.DataFrame(1e6, index=cal, columns=C.columns)
    ind = pd.Series(["A", "A", "A", "B", "B", "B"], index=C.columns)
    return C, V, ind


def test_signals_have_no_lookahead():
    C, V, ind = _panel()
    sig, _ = signals(C, V, ind)
    d = C.index[300]
    C2 = C.copy()
    C2.loc[C.index > d] *= 3.0  # change the future only
    sig2, _ = signals(C2, V, ind)
    for k in sig:
        pd.testing.assert_series_equal(sig[k].loc[d], sig2[k].loc[d])


def test_weekly_dates_are_last_day_of_week():
    cal = pd.bdate_range("2024-01-01", "2024-01-31")
    w = weekly_dates(cal)
    assert all(d.day_of_week == 4 for d in w[:-1])
    assert w[-1] == pd.Timestamp("2024-01-31")


def _flat(n_days, prices):
    a = np.array(prices, dtype=float)
    return np.tile(a, (n_days, 1))


def test_rotation_sells_when_rank_drops_and_no_loss_holds():
    costs = load_config()["costs"]
    T = 12
    O = _flat(T, [100, 100, 100])
    C = O.copy()
    C[3:, 0] = O[3:, 0] = 90  # stock 0 falls after purchase
    order = {0: np.array([0, 1, 2]), 5: np.array([1, 2, 0])}
    r = simulate(O, C, order, 0, T, 20000, costs, n=1, exit_rank=1)
    assert len(r["trades"]) == 1 and r["trades"][0][1] < 0  # sold at a loss on the rotation
    r2 = simulate(O, C, order, 0, T, 20000, costs, n=1, exit_rank=1, no_loss=True)
    assert len(r2["trades"]) == 0 and r2["open_underwater"] == 1  # never sold below cost
    assert np.all(r["equity"] > 0)


def test_random_control_uses_hold_distribution():
    costs = load_config()["costs"]
    T = 30
    O = _flat(T, [100] * 5)
    order = {i: np.arange(5) for i in range(0, T, 5)}
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=2,
                 rng=np.random.default_rng(1), hold_dist=np.array([5]))
    holds = [t[0] for t in r["trades"]]
    assert holds and all(h == 5 for h in holds)
    # flat prices: every round trip loses exactly the charges
    assert all(t[1] < 0 for t in r["trades"])
