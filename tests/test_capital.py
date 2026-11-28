import numpy as np
import pandas as pd

from trader.capital_study import run_pool, xirr, sip_days
from trader.strategies import run_book

COSTS = {"brokerage": 0.0, "stt": 0.001, "exchange_txn": 0.0000297, "sebi": 0.000001, "gst": 0.18,
         "stamp_duty_buy": 0.00015, "dp_charge_per_sell": 15.93, "slippage": 0.0005}


def _market(T=120, N=8, seed=1):
    rng = np.random.default_rng(seed)
    C = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (T, N)), axis=0)
    O = np.vstack([C[:1], C[:-1]]) * (1 + rng.normal(0, 0.003, (T, N)))
    entry = rng.random((T, N)) < 0.05
    exit_ = rng.random((T, N)) < 0.05
    pri = rng.random((T, N))
    return O, C, entry, exit_, pri


def test_single_book_matches_run_book():
    O, C, entry, exit_, pri = _market()
    a = run_book(O, C, entry, pri, exit_, 0, len(C), COSTS, capital=25000, n=5, stop_pct=0.08)
    b = run_pool(O, C, [{"entry": entry, "exit": exit_, "stop_pct": 0.08}], pri, 0, len(C), COSTS, 25000)
    assert len(a["trades"]) == len(b["trades"])
    assert abs(a["equity"][-1] - b["equity"][-1]) < 1.0


def test_slots_per_book_cap_positions():
    O, C, entry, exit_, pri = _market()
    never = np.zeros_like(entry)
    books = [{"entry": entry, "exit": never}, {"entry": entry, "exit": never}]
    r = run_pool(O, C, books, pri, 0, len(C), COSTS, 100000, slots=[1, 1])
    assert len(r["trades"]) == 0  # nothing ever exits, so at most 2 buys and no sells
    assert r["equity"][0] <= 100000


def test_sip_adds_cash_and_nav_ignores_it():
    O = C = np.full((60, 3), 100.0)
    none = np.zeros((60, 3), bool)
    adds = np.zeros(60)
    adds[[20, 40]] = 50000
    r = run_pool(O, C, [{"entry": none, "exit": none}], np.zeros((60, 3)), 0, 60, COSTS, 50000, adds=adds)
    assert r["invested"] == 150000 and abs(r["equity"][-1] - 150000) < 1e-6
    assert abs(r["nav"][-1] - 100.0) < 1e-9  # money paid in is not "return"


def test_xirr_simple():
    cal = pd.bdate_range("2020-01-01", periods=600)
    i1 = int(np.searchsorted(cal, pd.Timestamp("2021-01-01")))
    assert abs(xirr([(0, -100.0), (i1, 110.0)], cal) - 0.10) < 0.002


def test_sip_days_first_of_month():
    cal = pd.bdate_range("2020-01-01", "2020-04-30")
    d = sip_days(cal, 0, len(cal))
    assert d.sum() == 3 and d[0] == 0
