from datetime import date

import numpy as np

from trader.paper import decide, fill_buys, is_decision_day, rebalance_amount, sell_fill

COSTS = {"brokerage": 0.0, "stt": 0.001, "exchange_txn": 0.0000297, "sebi": 0.000001, "gst": 0.18,
         "stamp_duty_buy": 0.00015, "dp_charge_per_sell": 15.93, "slippage": 0.0005}


def test_decide_matches_backtest_rules():
    ranked = [f"S{i}" for i in range(100)]
    close = {s: 100.0 for s in ranked}
    close["S1"] = 6000.0                      # too expensive for a Rs 5k slot -> skipped
    # held S3 (rank 4) stays, S70 (rank 71 > 50) is sold; 4 free slots -> S0, S2, S4, S5
    sells, buys = decide({"S3", "S70"}, ranked, close, 25_000, True)
    assert sells == ["S70"] and buys == ["S0", "S2", "S4", "S5"]
    # a held stock that dropped out of the eligible list is sold
    assert decide({"GONE"}, ranked, close, 25_000, True)[0] == ["GONE"]
    # market filter off: nothing at all (like simulate: no buys and no rank sells)
    assert decide({"S70"}, ranked, close, 25_000, False) == ([], [])
    # slot below Rs 1,500 -> no buys
    assert decide(set(), ranked, close, 5_000, True)[1] == []


def test_fill_buys_and_sell_charges():
    cash, fills, waiting = fill_buys(25_000, 25_000, {"A": 100.0, "B": np.nan, "C": 250.0}, ["A", "B", "C"], COSTS)
    assert waiting == ["B"]
    a, c = fills
    assert a[0] == "A" and a[1] == 49 and c[1] == 19      # floor(5000 / (price x 1.0005 x 1.0013))
    spent = sum(sh * px + ch for _, sh, px, ch in fills)
    assert abs(cash - (25_000 - spent)) < 1e-6 and cash > 0
    px, ch, net = sell_fill(49, 110.0, COSTS)
    assert abs(px - 110 * 0.9995) < 1e-9 and ch > 15.93 and abs(net - (49 * px - ch)) < 1e-9


def test_rebalance_and_decision_day():
    # 75/25: strategy grew to 40k, core 80k -> target 30k: move 10k of idle cash to the core
    assert rebalance_amount(40_000, 12_000, 80_000, 0.25) == -10_000
    assert rebalance_amount(40_000, 3_000, 80_000, 0.25) == -3_000        # only idle cash moves
    assert rebalance_amount(20_000, 0, 90_000, 0.25) == 7_500             # core tops up the strategy
    fri, thu = date(2026, 11, 27), date(2026, 11, 26)
    assert is_decision_day(fri, fri, set())
    assert not is_decision_day(fri, fri, {"2026-W48"})
    assert not is_decision_day(thu, thu, set())                          # Friday may still trade
    assert is_decision_day(thu, date(2026, 11, 28), set())               # holiday Friday: Saturday run
