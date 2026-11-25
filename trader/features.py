"""Feature engineering and labels.

Rules that keep this honest:
- Every feature at date t uses only data up to and including the close of t.
- Features are scale-free (ratios, returns, oscillators) so one model can learn
  across all stocks at once - raw prices / SMAs / OBV are not comparable across stocks.
- The label looks forward and is NaN where the future is not yet known; rows are
  never dropped because of a missing label, so today's features are always available.
"""

import numpy as np
import pandas as pd


def _rsi(close, n):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 * gain / (gain + loss)


def market_features(mkt):
    c = mkt["Close"]
    return pd.DataFrame({
        "mkt_ret_5": c.pct_change(5),
        "mkt_ret_20": c.pct_change(20),
        "mkt_dist_sma_200": c / c.rolling(200).mean() - 1,
        "mkt_vol_20": c.pct_change().rolling(20).std(),
    })


def stock_features(df, mkt_feats):
    o, h, l, c, v = (df[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (20, 50, 200):
        f[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_vs_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1

    f["rsi_14"] = _rsi(c, 14)
    f["rsi_6"] = _rsi(c, 6)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["bb_width"] = 4 * sd / mid

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    f["vol_20"] = r.rolling(20).std()
    f["vol_60"] = r.rolling(60).std()

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = (c - ll) / (hh - ll)
    f["close_loc"] = (c - l) / (h - l)
    f["gap"] = o / c.shift() - 1

    f["vol_ratio"] = v / v.rolling(20).mean()
    f["dist_52w_high"] = c / h.rolling(252).max() - 1
    f["dist_52w_low"] = c / l.rolling(252).min() - 1

    up = r > 0
    f["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()

    f = f.join(mkt_feats.reindex(df.index))
    f["rel_ret_20"] = f["ret_20"] - f["mkt_ret_20"]
    return f.replace([np.inf, -np.inf], np.nan)


def hit_target_label(df, take_profit, horizon):
    """1 if, buying at the next day's open, the High reaches +take_profit within `horizon` days.

    Row t: entry = Open[t+1]; window = High[t+1 .. t+horizon]. NaN when the window is incomplete.
    """
    entry = df["Open"].shift(-1)
    fut_high = df["High"].rolling(horizon, min_periods=horizon).max().shift(-horizon)
    label = (fut_high >= entry * (1 + take_profit)).astype(float)
    return label.where(fut_high.notna() & entry.notna())


CS_RANK_COLS = ["ret_20", "ret_60", "vol_20", "rel_ret_20", "dist_52w_high"]


def build_panel(prices, market, take_profit, horizon):
    """Long panel indexed by (date, ticker) with feature columns and `label`."""
    mf = market_features(market)
    parts = {}
    for t, df in prices.items():
        f = stock_features(df, mf)
        f["label"] = hit_target_label(df, take_profit, horizon)
        parts[t] = f
    panel = pd.concat(parts, names=["ticker", "date"]).swaplevel().sort_index()
    # Cross-sectional ranks: where a stock stands versus the rest of the universe that day.
    g = panel.groupby(level="date")
    for col in CS_RANK_COLS:
        panel[f"cs_{col}"] = g[col].rank(pct=True)
    return panel


def feature_columns(panel):
    return [c for c in panel.columns if c != "label"]
