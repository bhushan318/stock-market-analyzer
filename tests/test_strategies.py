import numpy as np
import pandas as pd

from trader import load_config
from trader.strategies import rsi, run_book, supertrend_up

COSTS = load_config()["costs"]


def test_rsi_and_supertrend_basic():
    c = pd.DataFrame({"A": [10, 11, 12, 13, 12, 9, 8, 9, 10, 11]}, dtype=float)
    r = rsi(c, 2)["A"]
    assert r.iloc[3] > 90 and r.iloc[6] < 10          # three up days -> high; sharp drop -> low
    # steady rise then a crash: up-trend first, down-trend after the crash
    up = np.r_[np.linspace(100, 130, 40), np.linspace(128, 80, 20)]
    C = pd.DataFrame({"A": up})
    st = supertrend_up(C + 1, C - 1, C)["A"]
    assert st.iloc[35] and not st.iloc[-1]


def test_run_book_exits_and_next_open_fills():
    T = 12
    O = np.full((T, 2), 100.0); C = O.copy()
    entry = np.zeros((T, 2), bool); entry[1, 0] = True          # signal at close 1 -> buy open 2
    exit_sig = np.zeros((T, 2), bool); exit_sig[5, 0] = True    # exit at close 5 -> sell open 6
    pr = np.ones((T, 2))
    r = run_book(O, C, entry, pr, exit_sig, 0, T, COSTS)
    assert len(r["trades"]) == 1 and r["trades"][0][3] == 2 and r["trades"][0][4] == 6
    # stop: -8% close triggers, sold at the next open
    C2 = C.copy(); C2[4:, 0] = 90; O2 = O.copy(); O2[5:, 0] = 90
    r = run_book(O2, C2, entry, pr, np.zeros_like(exit_sig), 0, T, COSTS, stop_pct=0.08)
    assert r["trades"][0][4] == 5 and r["trades"][0][1] < -0.09
    # max_hold
    r = run_book(O, C, entry, pr, np.zeros_like(exit_sig), 0, T, COSTS, max_hold=3)
    assert r["trades"][0][4] == 6   # bought open 2, held 3 sessions at close 5, sold open 6
    # target at 2R with stop level 95 -> target 110 (fill ~100): close 111 triggers
    C3 = C.copy(); C3[7:, 0] = 111; O3 = O.copy(); O3[8:, 0] = 111
    sl = np.full((T, 2), 95.0)
    r = run_book(O3, C3, entry, pr, np.zeros_like(exit_sig), 0, T, COSTS, stop_level=sl, target_r=2.0)
    assert r["trades"][0][4] == 8 and r["trades"][0][1] > 0.09
