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


def test_pit_universe_and_signals_have_no_lookahead():
    from trader.momentum_pit import pit_signals
    from trader.pit import liquid_universe
    C, V, _ = _panel()
    TO = V * C / 1e7 * 5  # Rs cr
    d = C.index[300]
    C2, TO2 = C.copy(), TO.copy()
    C2.loc[C.index > d] *= 3.0
    TO2.loc[TO.index > d] *= 50.0
    u1, u2 = liquid_universe(C, TO, top=3), liquid_universe(C2, TO2, top=3)
    pd.testing.assert_series_equal(u1.loc[d], u2.loc[d])
    assert u1.loc[d].sum() == 3
    s1, _ = pit_signals(C, TO, u1)
    s2, _ = pit_signals(C2, TO2, u2)
    for k in s1:
        pd.testing.assert_series_equal(s1[k].loc[d], s2[k].loc[d])


def test_lever_fit_recovers_known_betas():
    from trader.levers import fit
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0, 0.02, (400, 3)), columns=["nifty", "crude", "usdinr"])
    y = 1.1 * X["nifty"] - 0.3 * X["crude"] + rng.normal(0, 0.005, 400)
    r = fit(y, X)
    assert abs(r["b_nifty"] - 1.1) < 0.05 and abs(r["b_crude"] + 0.3) < 0.05 and abs(r["b_usdinr"]) < 0.05
    assert r["t_crude"] < -10


def test_delivery_signals_have_no_lookahead():
    from trader.delivery_study import delivery_signals
    C, V, _ = _panel()
    rng = np.random.default_rng(2)
    D = pd.DataFrame(rng.uniform(20, 80, C.shape), index=C.index, columns=C.columns)
    TO = V * C / 1e7 * 5
    uni = C.notna()
    d = C.index[300]
    C2, V2, D2 = C.copy(), V.copy(), D.copy()
    after = C.index > d
    C2.loc[after] *= 2.0
    V2.loc[after] *= 10.0
    D2.loc[after] = 99.0
    s1, e1 = delivery_signals(C, V, TO, D, uni)
    s2, e2 = delivery_signals(C2, V2, V2 * C2 / 1e7 * 5, D2, uni)
    pd.testing.assert_series_equal(e1.loc[d], e2.loc[d])
    for k in s1:
        assert s1[k].loc[d].notna().all(), k
        pd.testing.assert_series_equal(s1[k].loc[d], s2[k].loc[d])
