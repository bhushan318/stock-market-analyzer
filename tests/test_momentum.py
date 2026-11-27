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


def test_target_stop_and_regime_rules():
    costs = load_config()["costs"]
    T = 15
    order = {0: np.array([0, 1]), 7: np.array([0, 1])}
    # target: stock 0 rises 30% on day 4 -> sold at day 5 open, no rank exit needed
    O = _flat(T, [100, 100]); O[4:, 0] = 130
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=1, exit_rank=10**9, target=0.2)
    assert len(r["trades"]) == 1 and r["trades"][0][4] == 5 and r["trades"][0][1] > 0.2
    # stop: 2% daily vol x 2 x sqrt(20) = 17.9% below the buy price; a fall to 80 triggers it
    O = _flat(T, [100, 100]); O[4:, 0] = 80
    vol = np.full(O.shape, 0.02)
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=1, exit_rank=10**9, stop_k=2.0, vol=vol)
    assert len(r["trades"]) == 1 and r["trades"][0][4] == 5 and r["trades"][0][1] < -0.18
    # a fall to 85 stays above the stop
    O[4:, 0] = 85
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=1, exit_rank=10**9, stop_k=2.0, vol=vol)
    assert len(r["trades"]) == 0
    # regime off on day 0: nothing bought until the day-7 rebalance (regime back on)
    regime = np.ones(T, bool); regime[:7] = False
    O = _flat(T, [100, 100])
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=1, regime=regime)
    assert r["equity"][5] == 20000 and r["equity"][-1] < 20000
    # regime_exit sells holdings when the filter turns off
    regime = np.ones(T, bool); regime[7] = False
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, n=1, regime=regime, regime_exit=True)
    assert len(r["trades"]) == 1 and r["trades"][0][4] == 8


def test_mix_equity_rebalances_to_weight():
    from trader.robust_study import mix_equity
    strat = [100, 200, 200]   # doubles on day 1
    etf = [10, 10, 10]        # flat
    e = mix_equity(strat, etf, 0.5, rebal_at=[1], fee=0.0)
    assert list(e) == [1.0, 1.5, 1.5]
    # all-in either side reproduces that side
    assert np.allclose(mix_equity(strat, etf, 1.0, fee=0.0), [1, 2, 2])
    assert np.allclose(mix_equity(strat, etf, 0.0, fee=0.0), [1, 1, 1])
    # after a rebalance the next move applies to the reset weights
    e = mix_equity([1, 2, 4], [1, 1, 1], 0.5, rebal_at=[1], fee=0.0)
    assert np.isclose(e[2], 1.5 + 0.75)


def test_intraday_target_and_stop_use_high_low():
    costs = load_config()["costs"]
    T = 10
    order = {0: np.array([0, 1])}
    O = _flat(T, [100, 100])
    kw = dict(n=1, exit_rank=10**9)
    # high touches +25% on day 4 but closes flat: intraday target sells that day at the target price
    H, L = O.copy(), O.copy()
    H[4, 0] = 125
    H[1, 0] = 200  # buy day (day 1): CNC shares cannot be sold the same day
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, target=0.2, intraday=(H, L), **kw)
    assert len(r["trades"]) == 1 and r["trades"][0][4] == 4 and abs(r["trades"][0][1] - 0.2) < 1e-6
    # checked at the close only, the same day never triggers
    assert len(simulate(O, O.copy(), order, 0, T, 20000, costs, target=0.2, **kw)["trades"]) == 0
    # gap above the target: filled at the (better) open
    O2 = O.copy(); O2[4:, 0] = 130
    H2 = np.maximum(H, O2)
    r = simulate(O2, O2.copy(), order, 0, T, 20000, costs, target=0.2, intraday=(H2, L), **kw)
    assert r["trades"][0][1] > 0.25
    # stop at 100 x (1 - 2 x 0.02 x sqrt(20)) = 82.1; low 80 triggers, filled at the stop - slippage
    vol = np.full(O.shape, 0.02)
    L3 = L.copy(); L3[4, 0] = 80
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, stop_k=2.0, vol=vol, intraday=(H, L3), **kw)
    assert r["trades"][0][4] == 4 and -0.19 < r["trades"][0][1] < -0.17
    # gap down through the stop: filled at the open (70), worse than the stop
    O4 = O.copy(); O4[4:, 0] = 70
    L4 = np.minimum(L, O4)
    r = simulate(O4, O4.copy(), order, 0, T, 20000, costs, stop_k=2.0, vol=vol, intraday=(H, L4), **kw)
    assert r["trades"][0][1] < -0.29
    # both touched on one day: the stop is assumed to come first
    r = simulate(O, O.copy(), order, 0, T, 20000, costs, target=0.2, stop_k=2.0, vol=vol, intraday=(H, L3), **kw)
    assert len(r["trades"]) == 1 and r["trades"][0][1] < 0


def test_timed_core_switches_with_one_day_lag():
    from trader.protect_study import timed_core
    idx = [100, 100, 50, 50, 50, 100]
    above = [True, False, False, False, True, True]
    check = [True] * 6
    v, sw = timed_core(idx, above, check, cash_rate=0.0, cost=0.0)
    # signal False at close 1 -> still in on day 2 (falls to 50), out from close 2;
    # signal True at close 4 -> back in at close 5, so day 5's jump is missed
    assert list(v) == [1.0, 1.0, 0.5, 0.5, 0.5, 0.5] and sw == 2
    # signals only read on check days
    v, sw = timed_core(idx, above, [True, False, False, False, False, False], cash_rate=0.0, cost=0.0)
    assert sw == 0 and v[-1] == 1.0


def test_weekly_winner_label_ignores_buy_day():
    from trader.winners_study import trade_outcome
    T = 9
    O = np.full((T, 4), 100.0); H = O + 1; L = O - 1; C = O.copy()
    # decided at close 0, bought at open 1. Stock 0: buy-day high +10% must NOT count
    H[1, 0] = 110
    # stock 1: Wednesday (session 3) high touches +5% -> target
    H[3, 1] = 105.5
    # stock 2: Tuesday low touches -5% and later high +5% -> stop first
    L[2, 2] = 94; H[4, 2] = 106
    # stock 3: same day both -> stop counts
    L[3, 3] = 94; H[3, 3] = 106
    g, hit = trade_outcome(O, H, L, C, 0, target=0.05, stop=0.05, hold=4, slip=0.0)
    assert list(hit) == [0, 1, -1, -1]
    assert np.allclose(g, [1.0, 1.05, 0.95, 0.95])   # stock 0: time exit at the close of session 5
