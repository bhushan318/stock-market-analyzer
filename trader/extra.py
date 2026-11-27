"""Forward data for stock selection (collected daily on the Pi from 2026-11-27 on).

    python -m trader collect extra     # F&O summary, results + shareholding filings, levers
    (business news RSS runs inside `collect news`, hourly)

Owner's decision 2026-11-27: build selection ideas on data collected from now on and judge them
only on paper-trading results going forward. Sources (all free, no login):
  fno_daily     NSE F&O UDiFF bhavcopy -> per-underlying futures/options OI and volume (fno.py)
  filings       NSE latest quarterly results and shareholding-pattern filings (raw JSON kept)
  levers        Yahoo daily closes: S&P 500, Brent, USD/INR, gold, US 10y, India VIX, Nifty, Nifty 500
  news          ET Markets, Moneycontrol, Livemint, PIB press releases (headlines + summaries)
Not available: NSE insider-trading (PIT) API returns nothing after early May 2026 - insider and
SAST disclosures still arrive via the announcements RSS.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests

from . import fno, nse
from .collect import CATCHUP_DAYS, _weekdays
from .store import log, upsert

SCHEMA = fno.SCHEMA.split("CREATE TABLE IF NOT EXISTS fno_log")[0] + """
CREATE TABLE IF NOT EXISTS filings (kind TEXT, key TEXT, symbol TEXT, ts TEXT, json TEXT, fetched_at TEXT,
    PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS levers (date TEXT, ticker TEXT, close REAL, PRIMARY KEY (date, ticker));
CREATE TABLE IF NOT EXISTS news (link TEXT PRIMARY KEY, source TEXT, published TEXT, title TEXT, summary TEXT,
    fetched_at TEXT, tags TEXT);
"""
LEVERS = {"^GSPC": "S&P 500", "BZ=F": "Brent crude", "INR=X": "USD/INR", "GC=F": "Gold", "^TNX": "US 10y yield",
          "^INDIAVIX": "India VIX", "^NSEI": "Nifty 50", "^CRSLDX": "Nifty 500"}
NEWS_FEEDS = {"et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
              "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
              "livemint": "https://www.livemint.com/rss/markets",
              "pib": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"}
FNO_COLS = ["date", "symbol", "kind", "fut_oi", "fut_oi_chg", "fut_contracts", "fut_close", "ce_oi", "ce_oi_chg",
            "ce_contracts", "pe_oi", "pe_oi_chg", "pe_contracts"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_fno(con, client, today):
    have = {r[0] for r in con.execute("SELECT DISTINCT date FROM fno_daily")}
    report = {}
    for d in _weekdays(today - timedelta(days=CATCHUP_DAYS), today):
        if d.isoformat() in have:
            continue
        try:
            text = fno.unzip_text(client.get(fno.udiff_url(d)).content)
        except nse.NotPublished:
            continue
        rows = [r for r in fno.aggregate(fno.parse_contracts(text)) if r["date"] == d.isoformat()]
        con.executemany(f"INSERT OR REPLACE INTO fno_daily ({','.join(FNO_COLS)}) VALUES ({','.join('?' * len(FNO_COLS))})",
                        [tuple(r[c] for c in FNO_COLS) for r in rows])
        con.commit()
        log(con, "fno", d.isoformat(), "ok", len(rows))
        report[d.isoformat()] = len(rows)
    return report


def filing_rows(kind, items, symbol_key, ts_key):
    """Raw API rows -> filings rows; key = hash of the whole row (re-filings stay distinct)."""
    out = []
    for x in items or []:
        raw = json.dumps(x, sort_keys=True)
        out.append({"kind": kind, "key": hashlib.sha1(raw.encode()).hexdigest()[:20], "symbol": x.get(symbol_key),
                    "ts": x.get(ts_key), "json": raw, "fetched_at": _now()})
    return out


def collect_filings(con, client):
    rep = {}
    for kind, url, sk, tk in [("results", f"{nse.WWW}/api/corporates-financial-results?index=equities&period=Quarterly",
                               "symbol", "broadCastDate"),
                              ("shareholding", f"{nse.WWW}/api/corporate-share-holdings-master?index=equities",
                               "symbol", "broadcastDate")]:
        d = client.get(url, api=True).json()
        d = d.get("data", d) if isinstance(d, dict) else d
        rep[kind] = upsert(con, "filings", filing_rows(kind, d, sk, tk))
        log(con, kind, "", "ok", rep[kind])
    return rep


def collect_levers(con):
    import yfinance as yf
    d = yf.download(list(LEVERS), period="10d", progress=False, auto_adjust=True)["Close"]
    rows = [{"date": idx.date().isoformat(), "ticker": t, "close": float(v)}
            for idx, r in d.iterrows() for t, v in r.items() if v == v]
    n = upsert(con, "levers", rows, replace=True)
    log(con, "levers", "", "ok", n)
    return n


def parse_news_rss(xml_text, source, fetched_at):
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()  # noqa: E731
        if not g("link"):
            continue
        out.append({"link": g("link"), "source": source, "published": g("pubDate"), "title": g("title"),
                    "summary": g("description")[:1000], "fetched_at": fetched_at, "tags": None})
    return out


def collect_news_feeds(con):
    rep = {}
    for src, url in NEWS_FEEDS.items():
        try:
            r = requests.get(url, headers={"User-Agent": nse.UA}, timeout=30)
            r.raise_for_status()
            rep[src] = upsert(con, "news", parse_news_rss(r.content, src, _now()))  # bytes: honours BOM + declared encoding
            log(con, f"news_{src}", "", "ok", rep[src])
        except Exception as e:  # noqa: BLE001 - one feed failing must not stop the others
            log(con, f"news_{src}", "", "error", 0, f"{type(e).__name__}: {str(e)[:200]}")
    return rep


def run_extra(con, today=None):
    con.executescript(SCHEMA)
    client, today, rep = nse.NSEClient(), today or date.today(), {}
    for name, job in [("fno", lambda: collect_fno(con, client, today)), ("filings", lambda: collect_filings(con, client)),
                      ("levers", lambda: collect_levers(con))]:
        try:
            rep[name] = job()
        except Exception as e:  # noqa: BLE001 - logged; the other sources still run
            log(con, name, "", "error", 0, f"{type(e).__name__}: {str(e)[:200]}")
            rep[name] = f"error {type(e).__name__}"
    return rep
