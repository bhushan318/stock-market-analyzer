"""Day-by-day portfolio simulation of the take-profit strategy.

Timeline for each trading day d:
  09:15  buy orders decided at the previous close fill at today's Open (+ slippage)
  intraday  a standing sell order at each position's target fills if High >= target
            (at the Open if the stock gaps up above the target)
  15:30  mark-to-market at Close; rank stocks with today's scores; queue tomorrow's buys
Positions are never sold below the target unless `max_hold_days` / `stop_loss` are set.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import buy_charges, sell_charges, target_price


@dataclass
class Position:
    ticker: str
    shares: int
    buy_price: float
    cost_basis: float  # shares * buy_price + buy charges
    buy_pos: int
    target: float


def run_backtest(opens, highs, lows, closes, scores, start, end, capital, strat, costs,
                 stuck_days=60):
    """opens/highs/lows/closes: date x ticker frames on one calendar.

    scores: callable(pos, date) -> pd.Series(ticker -> score) available at that day's close,
            or None/empty if no score exists for that day.
    """
    cal = closes.index
    i0 = cal.searchsorted(pd.Timestamp(start))
    i1 = cal.searchsorted(pd.Timestamp(end), side="right") if end else len(cal)
    O, H, L, C = (x.to_numpy() for x in (opens, highs, lows, closes))
    tickers = list(closes.columns)
    col = {t: j for j, t in enumerate(tickers)}
    last_close = closes.ffill().to_numpy()

    tp = strat["take_profit"]
    net = strat.get("take_profit_is_net", True)
    max_pos = strat["max_positions"]
    min_val = strat["min_position_value"]
    cooldown = strat.get("cooldown_days", 0) or 0
    min_prob = strat.get("min_prob", 0.0) or 0.0
    max_hold = strat.get("max_hold_days")
    stop = strat.get("stop_loss")
    slip = costs["slippage"]

    cash = float(capital)
    positions: dict[str, Position] = {}
    pending: list[tuple[str, int]] = []
    last_exit: dict[str, int] = {}
    trades, equity_rows = [], []
    worst_unrealised = 0.0
    charges_paid = 0.0

    for i in range(i0, i1):
        d = cal[i]

        # 1) fill queued buys at the open
        for t, shares in pending:
            j = col[t]
            o = O[i, j]
            if np.isnan(o) or t in positions:
                continue
            px = o * (1 + slip)
            while shares > 0 and shares * px + buy_charges(shares * px, costs) > cash:
                shares -= 1
            if shares == 0:
                continue
            ch = buy_charges(shares * px, costs)
            cash -= shares * px + ch
            charges_paid += ch
            basis = shares * px + ch
            positions[t] = Position(t, shares, px, basis, i,
                                    target_price(shares, px, basis, tp, costs, net))
        pending = []

        # 2) exits during the day (not on the day of purchase)
        for t, p in list(positions.items()):
            if p.buy_pos == i:
                continue
            j = col[t]
            o, h, lo, c = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(o):
                continue
            exit_px, why = None, None
            if o >= p.target:
                exit_px, why = o, "target_gap"
            elif h >= p.target:
                exit_px, why = p.target, "target"
            elif stop and lo <= p.buy_price * (1 - stop):
                exit_px, why = min(o, p.buy_price * (1 - stop)), "stop_loss"
            elif max_hold and i - p.buy_pos >= max_hold:
                exit_px, why = c * (1 - slip), "time_exit"
            if exit_px is None:
                continue
            value = p.shares * exit_px
            ch = sell_charges(value, costs)
            cash += value - ch
            charges_paid += ch
            pnl = value - ch - p.cost_basis
            trades.append({
                "ticker": t, "buy_date": cal[p.buy_pos].date(), "sell_date": d.date(),
                "days_held": i - p.buy_pos, "shares": p.shares,
                "buy_price": round(p.buy_price, 2), "sell_price": round(exit_px, 2),
                "pnl": round(pnl, 2), "return_net": pnl / p.cost_basis, "reason": why,
            })
            last_exit[t] = i
            del positions[t]

        # 3) mark to market
        held_val, stuck_val = 0.0, 0.0
        for t, p in positions.items():
            px = last_close[i, col[t]]
            val = p.shares * px
            held_val += val
            ur = val / p.cost_basis - 1
            worst_unrealised = min(worst_unrealised, ur)
            if ur < 0 and i - p.buy_pos >= stuck_days:
                stuck_val += val
        equity = cash + held_val
        equity_rows.append({"date": d, "equity": equity, "cash": cash, "invested": held_val,
                            "n_positions": len(positions), "stuck_value": stuck_val})

        # 4) choose tomorrow's buys
        free = max_pos - len(positions)
        if i + 1 >= i1 or free <= 0 or cash < min_val:
            continue
        s = scores(i, d)
        if s is None or len(s) == 0:
            continue
        s = s[s >= min_prob].sort_values(ascending=False)
        budget_cash = cash
        for t in s.index:
            if free == 0:
                break
            if t in positions or (t in last_exit and i - last_exit[t] < cooldown):
                continue
            c = C[i, col[t]]
            if np.isnan(c):
                continue
            budget = budget_cash / free
            if budget < min_val:
                break
            est = c * (1 + slip) * 1.003  # small buffer for an overnight move + charges
            shares = math.floor(budget / est)
            if shares < 1:
                continue  # can't afford even one share - try the next candidate
            pending.append((t, shares))
            budget_cash -= shares * est
            free -= 1

    eq = pd.DataFrame(equity_rows).set_index("date")
    tr = pd.DataFrame(trades)
    open_pos = pd.DataFrame([{
        "ticker": p.ticker, "buy_date": cal[p.buy_pos].date(), "days_held": i1 - 1 - p.buy_pos,
        "shares": p.shares, "buy_price": round(p.buy_price, 2),
        "last_close": round(last_close[i1 - 1, col[p.ticker]], 2), "target": round(p.target, 2),
        "unrealised": p.shares * last_close[i1 - 1, col[p.ticker]] / p.cost_basis - 1,
    } for p in positions.values()])
    return {"equity": eq, "trades": tr, "open": open_pos,
            "worst_unrealised": worst_unrealised, "charges": charges_paid}


def summarize(res, capital, benchmark_close=None, stcg=0.20):
    eq = res["equity"]["equity"]
    tr = res["trades"]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = eq.iloc[-1]
    dd = (eq / eq.cummax() - 1).min()
    realised = tr["pnl"].sum() if len(tr) else 0.0
    out = {
        "period": f"{eq.index[0].date()} -> {eq.index[-1].date()} ({years:.1f}y)",
        "final_equity": final,
        "total_return": final / capital - 1,
        "cagr": (final / capital) ** (1 / years) - 1,
        "max_drawdown": dd,
        "trades_closed": len(tr),
        "trades_per_year": len(tr) / years,
        "median_days_held": tr["days_held"].median() if len(tr) else np.nan,
        "p90_days_held": tr["days_held"].quantile(0.9) if len(tr) else np.nan,
        "max_days_held_closed": tr["days_held"].max() if len(tr) else np.nan,
        "loss_trades": int((tr["pnl"] < 0).sum()) if len(tr) else 0,
        "charges_paid": res["charges"],
        "est_stcg_tax_on_realised": max(realised, 0) * stcg,
        "avg_cash_idle_pct": (res["equity"]["cash"] / eq).mean(),
        "avg_stuck_pct": (res["equity"]["stuck_value"] / eq).mean(),
        "max_stuck_pct": (res["equity"]["stuck_value"] / eq).max(),
        "worst_unrealised_position": res["worst_unrealised"],
        "open_positions_end": len(res["open"]),
        "open_underwater_end": int((res["open"]["unrealised"] < 0).sum()) if len(res["open"]) else 0,
    }
    if benchmark_close is not None:
        b = benchmark_close.reindex(eq.index).ffill().dropna()
        out["nifty_cagr"] = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
        out["nifty_max_drawdown"] = (b / b.cummax() - 1).min()
    return out
