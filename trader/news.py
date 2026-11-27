"""NSE corporate announcements history for a shortlist of stocks, and their price impact.

    python -m trader news fetch [--top 50]     # shortlist = today's top N by the combo ranking
    python -m trader news impact               # event study (see news_impact)

Source: www.nseindia.com/api/corporate-announcements (free, history back to ~2019 per symbol).
NSE files each announcement under its own category ("desc", e.g. "Outcome of Board Meeting",
"Acquisition", "Credit Rating") - no LLM is used, so the test cannot leak hindsight.
Stored in data/news.db (research, PC). Shortlist caveat: picked from TODAY's ranking, so these
stocks did well recently - fine for "what moves a price after news", not for strategy returns.
"""

import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import ROOT, nse

DB = ROOT / "data" / "news.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS ann (seq_id TEXT PRIMARY KEY, symbol TEXT, ts TEXT, category TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS shortlist (symbol TEXT PRIMARY KEY, rank INTEGER, as_of TEXT);
CREATE TABLE IF NOT EXISTS fetched (symbol TEXT, year INTEGER, n INTEGER, PRIMARY KEY (symbol, year));
"""


def connect():
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def parse_announcements(items):
    """API rows -> dicts (ts ISO local time, category, text)."""
    out = []
    for x in items or []:
        try:
            ts = datetime.strptime(x["an_dt"], "%d-%b-%Y %H:%M:%S").isoformat()
        except (KeyError, ValueError, TypeError):
            continue
        out.append({"seq_id": str(x.get("seq_id") or f"{x.get('symbol')}|{ts}|{x.get('desc')}"),
                    "symbol": x.get("symbol"), "ts": ts, "category": (x.get("desc") or "").strip(),
                    "text": (x.get("attchmntText") or "").strip()[:2000]})
    return out


def shortlist(cfg, top=50):
    """Today's top `top` stocks by the combo ranking (point-in-time top-500 universe)."""
    from .momentum_pit import pit_signals
    from .pit import liquid_universe, load_panel
    p = load_panel(start=(date.today().replace(year=date.today().year - 2)).isoformat())
    s = pit_signals(p["C"], p["TO"], liquid_universe(p["C"], p["TO"]))[0]["combo"].iloc[-1].dropna()
    return list(s.sort_values(ascending=False).index[:top]), p["cal"][-1].date()


def fetch(cfg, top=50, first_year=2019):
    con, client = connect(), nse.NSEClient()
    syms, as_of = shortlist(cfg, top)
    con.execute("DELETE FROM shortlist")
    con.executemany("INSERT INTO shortlist VALUES (?,?,?)", [(s, k + 1, str(as_of)) for k, s in enumerate(syms)])
    con.commit()
    print(f"Shortlist ({len(syms)}, ranking as of {as_of}): {', '.join(syms)}", flush=True)
    done = {(s, y) for s, y in con.execute("SELECT symbol, year FROM fetched")}
    this_year = date.today().year
    for s in syms:
        for y in range(first_year, this_year + 1):
            if (s, y) in done and y < this_year:
                continue
            to = date(y, 12, 31) if y < this_year else date.today()
            url = (f"{nse.WWW}/api/corporate-announcements?index=equities&symbol={requests_quote(s)}"
                   f"&from_date=01-01-{y}&to_date={to:%d-%m-%Y}")
            try:
                d = client.get(url, api=True).json()
            except Exception as e:  # noqa: BLE001 - reported, retried next run
                print(f"  {s} {y}: {type(e).__name__}")
                continue
            rows = parse_announcements(d if isinstance(d, list) else d.get("data"))
            con.executemany("INSERT OR IGNORE INTO ann VALUES (:seq_id,:symbol,:ts,:category,:text)", rows)
            con.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?)", (s, y, len(rows)))
            con.commit()
        print(f"  {s}: {con.execute('SELECT COUNT(*) FROM ann WHERE symbol=?', (s,)).fetchone()[0]} announcements",
              flush=True)


def requests_quote(s):
    from urllib.parse import quote
    return quote(s, safe="")


def event_entry_index(ts, cal):
    """First session whose OPEN comes after the announcement: same day if announced before
    09:15, else the next session (during/after market hours you can only act at the next open)."""
    t = pd.Timestamp(ts)
    day = t.normalize()
    if t.time() < datetime.strptime("09:15", "%H:%M").time():
        return int(cal.searchsorted(day))
    return int(cal.searchsorted(day, side="right"))


def news_impact(cfg, min_events=15):
    """Per category: return from the entry open to the close h sessions later, minus Nifty 500
    over the same days ("excess"); also the overnight gap you could NOT trade. Learn 2019-2023,
    check 2024+. All in percent, before charges."""
    from .data import download_prices
    from .pit import load_panel
    con = connect()
    ann = pd.read_sql("SELECT symbol, ts, category FROM ann", con)
    p = load_panel(start="2019-01-01")
    O, C, cal = p["O"], p["C"].ffill(), p["cal"]
    n5 = download_prices(["^CRSLDX"], "2019-01-01", ROOT / cfg["data"]["cache_dir"], 12)["^CRSLDX"]["Close"]
    n5 = n5.reindex(cal.union(n5.index)).ffill().reindex(cal).to_numpy()
    rows = []
    for r in ann.itertuples(index=False):
        if r.symbol not in C.columns:
            continue
        i = event_entry_index(r.ts, cal)
        if i < 1 or i + 20 >= len(cal):
            continue
        j = C.columns.get_loc(r.symbol)
        o, prev = O.iat[i, j], C.iat[i - 1, j]
        if not (o == o and prev == prev and o > 0):
            continue
        rec = {"symbol": r.symbol, "date": cal[i], "category": r.category, "gap": o / prev - 1}
        for h in (1, 5, 20):
            c = C.iat[i + h - 1, j]
            bench = n5[i + h - 1] / n5[i - 1] - 1  # index from the previous close (approx. open)
            rec[f"x{h}"] = c / o - 1 - bench
        rows.append(rec)
    E = pd.DataFrame(rows)
    E["period"] = np.where(E["date"] < pd.Timestamp("2024-01-01"), "learn 2019-23", "check 2024+")
    out = ROOT / cfg["backtest"]["output_dir"]
    E.to_csv(out / "news_events.csv", index=False)

    def summ(g):
        return pd.Series({"events": len(g), "gap": g.gap.mean(), "x1": g.x1.mean(), "x5": g.x5.mean(),
                          "x20": g.x20.mean(), "x20_median": g.x20.median(), "x20_up": (g.x20 > 0).mean(),
                          "t20": g.x20.mean() / (g.x20.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan})
    S = E.groupby(["category", "period"]).apply(summ).unstack("period")
    keep = S[("events", "learn 2019-23")].fillna(0) >= min_events
    S = S[keep].sort_values(("events", "learn 2019-23"), ascending=False)
    S.to_csv(out / "news_impact.csv")
    allrow = E.groupby("period").apply(summ)
    print(f"\n{len(E)} events, {E.symbol.nunique()} stocks. Excess return vs Nifty 500 from the entry open "
          f"(next tradable open after the announcement), before charges.")
    print("\nAll announcements:")
    print(allrow.round(4).to_string())
    pd.set_option("display.width", 250)
    cols = ["events", "gap", "x5", "x20", "x20_up", "t20"]
    show = S[cols].copy()
    for c in show.columns:
        if c[0] in ("gap", "x5", "x20", "x20_up"):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
        elif c[0] == "t20":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}")
        else:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    print(f"\nBy NSE category (>= {min_events} events in 2019-23). t20 = t-statistic; |t| < 2 is noise:")
    print(show.to_string())
    print(f"\nCSV: {out}/news_events.csv, news_impact.csv")


def main(cfg, args):
    if args.job == "fetch":
        return fetch(cfg, top=args.top)
    if args.job == "baseline":
        from .news_baseline import run_news_baseline
        return run_news_baseline(cfg)
    return news_impact(cfg)
