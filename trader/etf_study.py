"""ETF dip-buying study: profit targets x time windows x seasonality.

    python -m trader etf

Strategy under test (never sells at a loss):
  each evening rank ETFs by how far they closed below their 20-day average, buy the deepest
  dips next morning into free slots, sell each one at +target (net of all charges).
Variants: plain, and with averaging down (30% cash reserve, buy again at -10%, max 2 times).
"""

import numpy as np
import pandas as pd

from . import ROOT
from .backtest import run_backtest, summarize
from .data import download_prices, wide
from .features import hit_target_label

ETF_UNIVERSE = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "PSUBNKBEES.NS", "GOLDBEES.NS",
                "INFRABEES.NS", "MON100.NS", "MOM100.NS", "CPSEETF.NS", "ITBEES.NS",
                "PHARMABEES.NS", "SILVERBEES.NS", "MAFANG.NS", "AUTOBEES.NS", "FMCGIETF.NS",
                "ALPHA.NS"]
# Broad index / asset-class ETFs only. Sector & thematic ETFs can fall 55-78% and stay down
# for years (PSUBNKBEES 2010-2020, CPSEETF, SILVERBEES), which is fatal for "never sell at a loss".
BROAD = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "MON100.NS",
         "ITBEES.NS", "MAFANG.NS"]
BENCH = "NIFTYBEES.NS"
TARGETS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
AVG = {"trigger": 0.10, "max_adds": 2, "size": 1.0}
VARIANTS = {"plain": {}, "avg_down": {"reserve_pct": 0.3, "avg_down": AVG}}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def load(cfg):
    prices = download_prices(ETF_UNIVERSE, "2009-01-01", ROOT / cfg["data"]["cache_dir"],
                             cfg["data"]["max_cache_age_hours"])
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in prices.values()])))
    frames = tuple(wide(prices, f, cal) for f in ["Open", "High", "Low", "Close"])
    return prices, frames


def dip_scores(C):
    dist = C / C.rolling(20, min_periods=20).mean() - 1
    by_date = {d: (-row).dropna() for d, row in dist.iterrows()}
    return lambda i, d: by_date.get(d)


def _strat(cfg, tp, variant, **extra):
    return {**cfg["strategy"], "take_profit": tp, "max_hold_days": None, "stop_loss": None,
            "target_decay": None, "regime_filter": False, **VARIANTS[variant], **extra}


def _costs(cfg):
    return {**cfg["costs"], "stt": 0.00001}  # ETFs: 0.001% STT on sell only


def run_window(cfg, frames, scorer, bench, start, end, tp, variant, **extra):
    res = run_backtest(*frames, scorer, start, end, cfg["capital"], _strat(cfg, tp, variant, **extra),
                       _costs(cfg), cfg["backtest"]["stuck_days"])
    s = summarize(res, cfg["capital"], bench, cfg["costs"]["stcg_tax"])
    s["bench_cagr"] = s.pop("nifty_cagr")
    s["bench_max_drawdown"] = s.pop("nifty_max_drawdown")
    return s


def fixed_windows(cfg, frames, scorer, bench):
    C = frames[3]
    end = C.index[-1]
    first = C.index[25]
    windows = {"1y": end - pd.DateOffset(years=1), "3y": end - pd.DateOffset(years=3),
               "5y": end - pd.DateOffset(years=5), "10y": end - pd.DateOffset(years=10),
               f"all({(end - first).days / 365.25:.0f}y)": first}
    rows = []
    for wname, start in windows.items():
        for variant in VARIANTS:
            for tp in TARGETS:
                s = run_window(cfg, frames, scorer, bench, start, None, tp, variant)
                rows.append({"window": wname, "variant": variant, "target": f"{tp:.0%}",
                             "cagr": s["cagr"], "max_dd": s["max_drawdown"],
                             "trades/yr": s["trades_per_year"], "med_days": s["median_days_held"],
                             "stuck%": s["avg_stuck_pct"], "worst_pos": s["worst_unrealised_position"],
                             "bench_cagr": s["bench_cagr"], "bench_dd": s["bench_max_drawdown"]})
    return pd.DataFrame(rows)


def rolling_windows(cfg, frames, scorer, bench, years, step_months=3, start_from=None, **extra):
    C = frames[3]
    first = pd.Timestamp(start_from) if start_from else C.index[25]
    last_start = C.index[-1] - pd.DateOffset(years=years)
    starts = pd.date_range(first, last_start, freq=f"{step_months}MS")
    rows = []
    for st in starts:
        en = st + pd.DateOffset(years=years)
        for variant in VARIANTS:
            for tp in TARGETS:
                s = run_window(cfg, frames, scorer, bench, st, en, tp, variant, **extra)
                rows.append({"start": st.date(), "variant": variant, "target": tp,
                             "cagr": s["cagr"], "bench_cagr": s["bench_cagr"],
                             "max_dd": s["max_drawdown"], "stuck": s["avg_stuck_pct"]})
    df = pd.DataFrame(rows)
    g = df.groupby(["variant", "target"])
    out = pd.DataFrame({
        "windows": g.size(),
        "median_cagr": g["cagr"].median(),
        "worst_10pct_cagr": g["cagr"].quantile(0.1),
        "worst_cagr": g["cagr"].min(),
        "pct_losing": g["cagr"].apply(lambda x: (x < 0).mean()),
        "pct_beat_niftybees": g.apply(lambda x: (x["cagr"] > x["bench_cagr"]).mean(), include_groups=False),
        "median_max_dd": g["max_dd"].median(),
        "median_stuck": g["stuck"].median(),
    })
    out.index = out.index.set_levels([out.index.levels[0], [f"{t:.0%}" for t in out.index.levels[1]]])
    return out, df


# ---------------------------------------------------------------- seasonality

def entry_outcomes(prices, tp=0.04, horizon=20):
    """For every ETF-day that closed below its 20-day average (a dip we'd consider buying):
    did buying next open reach +tp within `horizon` trading days?"""
    parts = []
    for t, df in prices.items():
        c = df["Close"]
        dip = c / c.rolling(20).mean() - 1 < 0
        lab = hit_target_label(df, tp, horizon)
        fr = pd.DataFrame({"hit": lab, "dip": dip}).dropna()
        fr = fr[fr["dip"]].drop(columns="dip")
        fr["ticker"] = t
        parts.append(fr)
    out = pd.concat(parts)
    idx = out.index
    out["month"] = idx.month
    out["weekday"] = idx.dayofweek
    # trading-day position within the month, using the union calendar of these rows
    cal = pd.DatetimeIndex(sorted(idx.unique()))
    s = pd.Series(cal, index=cal)
    pos = s.groupby([cal.year, cal.month]).cumcount()
    rev = s.groupby([cal.year, cal.month]).cumcount(ascending=False)
    bucket = pd.Series(np.where(pos < 3, "first 3 days", np.where(rev < 3, "last 3 days", "mid-month")),
                       index=cal)
    out["month_part"] = bucket.reindex(idx).to_numpy()
    return out


def seasonal_table(outcomes, key, split, labels=None):
    first = outcomes[outcomes.index < pd.Timestamp(split)]
    second = outcomes[outcomes.index >= pd.Timestamp(split)]
    t = pd.DataFrame({
        "hit_2009_17": first.groupby(key)["hit"].mean() - first["hit"].mean(),
        "hit_2018_26": second.groupby(key)["hit"].mean() - second["hit"].mean(),
        "n_2018_26": second.groupby(key).size(),
    })
    if labels:
        t.index = [labels[i] for i in t.index]
    corr = t["hit_2009_17"].corr(t["hit_2018_26"], method="spearman")
    return t, corr


def nifty_month_returns(bench, split):
    m = bench.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({"ret": m, "month": m.index.month})
    a = df[df.index < pd.Timestamp(split)].groupby("month")["ret"].mean()
    b = df[df.index >= pd.Timestamp(split)].groupby("month")["ret"].mean()
    t = pd.DataFrame({"avg_ret_2009_17": a, "avg_ret_2018_26": b})
    t.index = MONTHS
    return t, a.corr(b, method="spearman")


# ---------------------------------------------------------------- report

def pct(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: f"{v:+.1%}" if "hit" in c or "ret" in c else f"{v:.1%}")
    return df


def run_study(cfg, broad=False):
    out_dir = ROOT / cfg["backtest"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    prices, frames = load(cfg)
    if broad:
        prices = {t: prices[t] for t in BROAD}
        frames = tuple(f[BROAD] for f in frames)
        out_dir = out_dir / "broad"
        out_dir.mkdir(exist_ok=True)
    C = frames[3]
    bench = C[BENCH].dropna()
    scorer = dip_scores(C)
    print(f"ETFs: {len(prices)} | data {C.index[0].date()} -> {C.index[-1].date()} | "
          f"capital Rs {cfg['capital']:,} | max {cfg['strategy']['max_positions']} positions | "
          f"never sell at a loss")

    print("\n### 1. Fixed windows ending today (benchmark = buy & hold NIFTYBEES)")
    fw = fixed_windows(cfg, frames, scorer, bench)
    fw.to_csv(out_dir / "etf_fixed_windows.csv", index=False)
    for variant in VARIANTS:
        t = fw[fw["variant"] == variant].pivot(index="target", columns="window", values="cagr")
        t = t[[c for c in fw["window"].unique()]]
        t.loc["NIFTYBEES B&H"] = fw[fw["variant"] == variant].groupby("window")["bench_cagr"].first()
        print(f"\nCAGR by window - {variant}")
        print(t.map(lambda v: f"{v:.1%}").to_string())
    allw = fw[fw["window"].str.startswith("all")]
    print("\nFull-period detail:")
    print(pct(allw.drop(columns=["window"]).set_index(["variant", "target"]),
              ["cagr", "max_dd", "stuck%", "worst_pos", "bench_cagr", "bench_dd"]).round(1).to_string())

    for years in (1, 3):
        print(f"\n### 2. Rolling {years}-year windows, a new Rs {cfg['capital']:,} start every quarter")
        agg, raw = rolling_windows(cfg, frames, scorer, bench, years)
        raw.to_csv(out_dir / f"etf_rolling_{years}y.csv", index=False)
        print(pct(agg.reset_index(), ["median_cagr", "worst_10pct_cagr", "worst_cagr", "pct_losing",
                                      "pct_beat_niftybees", "median_max_dd", "median_stuck"])
              .set_index(["variant", "target"]).to_string())

    split = "2018-01-01"
    print("\n### 3. Seasonality (learn on 2009-2017, check on 2018-2026)")
    print("Values = hit rate of a dip entry reaching +4% within 20 days, minus that period's average.")
    oc = entry_outcomes(prices)
    for key, labels in [("month", dict(enumerate(MONTHS, 1))),
                        ("month_part", None),
                        ("weekday", dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split())))]:
        t, corr = seasonal_table(oc, key, split, labels)
        print(f"\nBy {key}: rank correlation between the two periods = {corr:+.2f} "
              f"(near +1 = repeatable pattern, near 0 = noise)")
        print(pct(t, ["hit_2009_17", "hit_2018_26"]).to_string())
    nm, ncorr = nifty_month_returns(bench, split)
    print(f"\nNIFTYBEES average monthly return by calendar month (rank correlation {ncorr:+.2f}):")
    print(pct(nm, ["avg_ret_2009_17", "avg_ret_2018_26"]).T.to_string())

    # Out-of-sample seasonal rule: skip new buys in the 2 worst months learned from 2009-2017
    first = oc[oc.index < pd.Timestamp(split)]
    worst = first.groupby("month")["hit"].mean().nsmallest(2).index.tolist()
    print(f"\nSeasonal rule test: no new buys in {[MONTHS[m - 1] for m in worst]} "
          f"(worst months in 2009-2017), judged on 3-year windows starting 2018+")
    base, _ = rolling_windows(cfg, frames, scorer, bench, 3, start_from=split)
    seas, _ = rolling_windows(cfg, frames, scorer, bench, 3, start_from=split, blackout_months=worst)
    cmp_ = pd.DataFrame({"median_cagr_base": base["median_cagr"], "median_cagr_seasonal": seas["median_cagr"],
                         "worst10_base": base["worst_10pct_cagr"], "worst10_seasonal": seas["worst_10pct_cagr"]})
    print(cmp_.map(lambda v: f"{v:.1%}").to_string())
    print(f"\nCSV files written to {out_dir}")
