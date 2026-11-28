"""Gate 0 step 2: does delivery % (shares taken home, not day-traded) predict the next 5-20 days?

    python -m trader delivery

NSE's bhavcopy gives, per stock and day, the % of traded shares that were delivered. A rise in
delivery above the stock's own normal level is often read as "real buying" (investors holding,
not intraday churn). Signals, all measured only from data up to the ranking date:

  deliv_rel  5-day average delivery % minus the stock's own 60-day median (percentage points)
  accum      10-day average of (delivery % minus its 60-day median) x sign of that day's return:
             high delivery on up days and low delivery on down days = accumulation
  dq_rel     delivered shares (volume x delivery %) over the last 5 days vs their 60-day median:
             a surge in shares changing hands for keeps
  accum_mom  average percentile rank of accum and 12-1 month momentum (does delivery add to
             the momentum signal?)

Same harness as momentum-pit: point-in-time top-500 universe (no survivorship bias), weekly
ranking, top-decile signal test, 5-stock portfolio after all charges vs random picks from the
same eligible stocks and vs Nifty 500, rolling 1y/3y windows, learn 2021-23 / check 2024+.
"""

import numpy as np

from .momentum_pit import MIN_TURNOVER_CR, load_context, run_pit_study

VARIANTS = [("deliv_rel", 20, False), ("accum", 20, False), ("dq_rel", 20, False),
            ("accum_mom", 20, False)]


def delivery_signals(C, V, TO, D, universe):
    """C close, V volume, TO turnover (Rs cr), D delivery % (EQ series only, else NaN)."""
    Cf = C.ffill(limit=5)
    ret1 = C / Cf.shift(1) - 1
    base = D.rolling(60, min_periods=40).median()
    deliv_rel = D.rolling(5, min_periods=4).mean() - base
    accum = ((D - base) * np.sign(ret1)).rolling(10, min_periods=8).mean()
    dq = V * D / 100.0
    dq_rel = dq.rolling(5, min_periods=4).mean() / dq.rolling(60, min_periods=40).median()
    dq_rel = dq_rel.replace([np.inf, -np.inf], np.nan)
    mom12 = Cf.shift(21) / Cf.shift(252) - 1
    turn20 = TO.rolling(20, min_periods=15).median()
    elig = universe & C.notna() & (turn20 >= MIN_TURNOVER_CR) & base.notna()
    sig = {"deliv_rel": deliv_rel, "accum": accum, "dq_rel": dq_rel}
    sig = {k: v.where(elig) for k, v in sig.items()}
    em = elig & mom12.notna() & accum.notna()
    sig["accum_mom"] = (accum.where(em).rank(axis=1, pct=True) + mom12.where(em).rank(axis=1, pct=True)) / 2
    return sig, elig


def _signals(p, uni):
    return delivery_signals(p["C"], p["V"], p["TO"], p["DELIV"], uni)


def run_delivery(cfg):
    ctx = load_context(cfg)
    D = ctx["p"]["DELIV"]
    print(f"  delivery % available on {D.notna().to_numpy().mean():.0%} of stock-days "
          f"(EQ series only; median {np.nanmedian(D.to_numpy()):.0f}%)")
    run_pit_study(cfg, ctx, _signals, VARIANTS, "deliv", universes=["point-in-time top500"])
