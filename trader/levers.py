"""Gate 0 step 4: measured lever sensitivities (betas) per stock and per industry.

    python -m trader levers

Weekly (Friday-to-Friday) returns of each of today's Nifty 500 stocks are regressed on
    Nifty 50 + Brent crude + USD/INR + gold + US 10y yield change
all at once, so each lever beta is the stock's reaction BEYOND the general market move:
  crude / usdinr / gold : % stock move per +1% move in the lever (USD/INR up = rupee weaker)
  us10y                 : % stock move per +0.10 pp (10 bp) rise in the US 10-year yield
(India 10y is not on Yahoo; US 10y is a global-rates stand-in until an NSE G-Sec series is added.)

Fitted separately on 2016-2020 and 2021-2026: a lever is only trusted where the sign holds in
both halves. These numbers are the check for the sector lever maps (drafted per sub-industry):
a map saying "crude up hurts paints" should show a negative crude beta for paint stocks.
Survivorship: today's constituents - fine for describing sensitivities, not for returns.
"""

import numpy as np
import pandas as pd

from . import ROOT
from .data import download_prices

LEVERS = {"nifty": "^NSEI", "crude": "BZ=F", "usdinr": "INR=X", "gold": "GC=F", "us10y": "^TNX"}
HALVES = {"2016-20": ("2016-01-01", "2020-12-31"), "2021-26": ("2021-01-01", "2100-01-01")}
MIN_WEEKS = 100


def weekly(close):
    """Last close of each week (Friday-ending), forward-filled across holidays."""
    return close.ffill(limit=5).resample("W-FRI").last()


def lever_returns(px):
    X = pd.DataFrame({k: weekly(px[t]).pct_change() for k, t in LEVERS.items() if k != "us10y"})
    X["us10y"] = weekly(px[LEVERS["us10y"]]).diff() / 0.10  # per 10 bp
    return X


def fit(y, X):
    """OLS with intercept. Returns betas, t-stats, n, r2 (rows with any NaN dropped)."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    n = len(d)
    if n < MIN_WEEKS:
        return None
    A = np.column_stack([np.ones(n), d[X.columns].to_numpy()])
    yv = d["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ coef
    dof = n - A.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    t = coef / np.sqrt(np.diag(cov))
    r2 = 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))
    names = list(X.columns)
    return {"n": n, "r2": r2, **{f"b_{k}": coef[i + 1] for i, k in enumerate(names)},
            **{f"t_{k}": t[i + 1] for i, k in enumerate(names)}}


def run_levers(cfg):
    dc = cfg["data"]
    uni = pd.read_csv(ROOT / "universes" / "nifty500_2026-11.csv")
    uni["t"] = uni["Symbol"].str.strip() + ".NS"
    cache = ROOT / dc["cache_dir"]
    lp = download_prices(list(LEVERS.values()), dc["start"], cache, dc["max_cache_age_hours"])
    missing = [t for t in LEVERS.values() if t not in lp]
    if missing:
        raise SystemExit(f"lever series missing from Yahoo: {missing}")
    sp = download_prices(list(uni["t"]), dc["start"], cache, dc["max_cache_age_hours"])
    X = lever_returns({t: lp[t]["Close"] for t in LEVERS.values()})
    lev = [k for k in X.columns if k != "nifty"]

    rows = []
    for r in uni.itertuples():
        if r.t not in sp:
            continue
        y = weekly(sp[r.t]["Close"]).pct_change()
        for half, (a, b) in {"all": ("2016-01-01", "2100-01-01"), **HALVES}.items():
            res = fit(y.loc[a:b], X.loc[a:b])
            if res:
                rows.append({"symbol": r.Symbol.strip(), "industry": r.Industry, "period": half, **res})
    B = pd.DataFrame(rows)
    out_dir = ROOT / cfg["backtest"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    B.to_csv(out_dir / "lever_betas.csv", index=False)

    # stability per stock: same sign in both halves
    w = B[B["period"] != "all"].pivot_table(index="symbol", columns="period", values=[f"b_{k}" for k in lev])
    print(f"Stocks with >= {MIN_WEEKS} weeks in both halves: {w.dropna().shape[0]}")
    print("\n=== Per-stock stability: share of stocks whose lever beta keeps its sign 2016-20 -> 2021-26 ===")
    print("(50% = coin flip; correlation = how well the 2016-20 ranking of stocks predicts 2021-26)")
    for k in lev:
        a, b = w[(f"b_{k}", "2016-20")], w[(f"b_{k}", "2021-26")]
        ok = a.notna() & b.notna()
        print(f"  {k:7s} same sign {np.mean(np.sign(a[ok]) == np.sign(b[ok])):.0%}   "
              f"corr {a[ok].corr(b[ok]):+.2f}   rank corr {a[ok].rank().corr(b[ok].rank()):+.2f}")

    # industry level: median beta in each half; trusted if sign agrees and most stocks agree
    def ind_table(period):
        g = B[B["period"] == period].groupby("industry")
        t = g[[f"b_{k}" for k in lev]].median()
        t["stocks"] = g.size()
        return t
    full, h1, h2 = ind_table("all"), ind_table("2016-20"), ind_table("2021-26")
    out = []
    for ind in full.index:
        row = {"industry": ind, "stocks": int(full.at[ind, "stocks"])}
        for k in lev:
            sub = B[(B["period"] == "all") & (B["industry"] == ind)]
            m = full.at[ind, f"b_{k}"]
            row[k] = m
            row[f"{k}_agree"] = float((np.sign(sub[f"b_{k}"]) == np.sign(m)).mean())
            row[f"{k}_sig"] = float((sub[f"t_{k}"].abs() > 2).mean())
            a, b = h1[f"b_{k}"].get(ind, np.nan), h2[f"b_{k}"].get(ind, np.nan)
            row[f"{k}_stable"] = bool(np.sign(a) == np.sign(b)) if pd.notna(a) and pd.notna(b) else False
        out.append(row)
    I = pd.DataFrame(out).sort_values("stocks", ascending=False)
    I.to_csv(out_dir / "lever_betas_by_industry.csv", index=False)

    print("\n=== Industry median lever betas (all years). * = same sign in both halves AND >= 70% of the")
    print("    industry's stocks agree in sign. Units: % per +1% lever move; us10y per +10 bp ===")
    show = pd.DataFrame({"industry": I["industry"].str[:34], "n": I["stocks"]})
    for k in lev:
        star = I[f"{k}_stable"] & (I[f"{k}_agree"] >= 0.7)
        show[k] = [f"{v:+.2f}{'*' if s else ' '}" for v, s in zip(I[k], star)]
    print(show.to_string(index=False))
    print(f"\nCSV: {out_dir}/lever_betas.csv (per stock, t-stats, r2), lever_betas_by_industry.csv")
