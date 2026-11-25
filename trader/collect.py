"""Daily data collection jobs (run by systemd timers on the Pi, or by hand).

    python -m trader collect daily              # bhavcopy (+ catch-up), deals, FII/DII
    python -m trader collect news               # announcements + results RSS
    python -m trader collect backfill --start 2020-01-01
    python -m trader collect status
"""

import csv
import gzip
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone

from . import ROOT
from . import nse
from .store import connect, log, upsert

CATCHUP_DAYS = 10  # re-check this many recent weekdays so Pi downtime never leaves gaps


def db_path(cfg):
    return ROOT / cfg.get("collect", {}).get("db_path", "data/market.db")


def tracked_symbols(cfg):
    """Nifty 500 + every NSE ETF + any extras; None means store everything."""
    c = cfg.get("collect", {})
    if c.get("store_all"):
        return None
    syms = set(c.get("extra_symbols") or [])
    uni = ROOT / "universes"
    n500 = uni / "nifty500_2026-11.csv"
    if n500.exists():
        syms |= {r["Symbol"].strip() for r in csv.DictReader(open(n500, encoding="utf-8"))}
    etfs = uni / "nse_etfs.csv"
    if etfs.exists():
        syms |= {r["Symbol"].strip() for r in csv.DictReader(open(etfs, encoding="latin1"))}
    return syms


def refresh_reference_lists(client):
    """ETF list, symbol changes and company-name -> symbol map; refreshed at most weekly."""
    uni = ROOT / "universes"
    uni.mkdir(exist_ok=True)
    for name, url in [("nse_etfs.csv", f"{nse.ARCHIVE}/content/equities/eq_etfseclist.csv"),
                      ("nse_equity_list.csv", f"{nse.ARCHIVE}/content/equities/EQUITY_L.csv"),
                      ("nse_symbol_changes.csv", f"{nse.ARCHIVE}/content/equities/symbolchange.csv")]:
        p = uni / name
        if not p.exists() or (datetime.now().timestamp() - p.stat().st_mtime) > 7 * 86400:
            p.write_bytes(client.get(url).content)
    names = nse.parse_equity_list((uni / "nse_equity_list.csv").read_text(encoding="latin1"))
    n500 = uni / "nifty500_2026-11.csv"
    if n500.exists():  # index file spells some names differently; add those spellings too
        for r in csv.DictReader(open(n500, encoding="utf-8")):
            names.setdefault(nse.norm_name(r["Company Name"]), r["Symbol"].strip())
    return names


FULL_DAY_MIN_SYMBOLS = 1000  # a store_all day has ~2,000+ EQ/BE symbols; filtered days ~860


def _have_dates(con, min_symbols=0):
    """Days already stored, plus past days confirmed as holidays / never published.

    min_symbols > 0 treats days stored with fewer symbols (e.g. before store_all was switched
    on) as missing, so they are fetched again; upsert overwrites the old rows in place.
    """
    have = {r[0] for r in con.execute(
        "SELECT date FROM eq_daily GROUP BY date HAVING COUNT(*) >= ?", (min_symbols,))}
    have |= {r[0] for r in con.execute(
        "SELECT DISTINCT ref FROM collect_log WHERE source='bhavcopy' AND status='holiday'")}
    # 'not_published' only counts once the day is well past (files appear the same evening)
    have |= {r[0] for r in con.execute(
        "SELECT DISTINCT ref FROM collect_log WHERE source='bhavcopy' AND status='not_published' "
        "AND ref < date('now', '-3 days')")}
    return have


def collect_bhavcopy(con, client, day, symbols):
    try:
        text = client.get(nse.bhavcopy_url(day)).text
    except nse.NotPublished:
        log(con, "bhavcopy", day.isoformat(), "not_published")
        return None
    rows = nse.parse_bhavcopy(text, symbols)
    file_dates = {r["date"] for r in rows}
    if file_dates and day.isoformat() not in file_dates:
        # NSE re-publishes the previous session's file on exchange holidays
        log(con, "bhavcopy", day.isoformat(), "holiday", 0, f"file holds {sorted(file_dates)}")
        return None
    n = upsert(con, "eq_daily", rows)
    log(con, "bhavcopy", day.isoformat(), "ok", n)
    return n


def _weekdays(start, end, weekends=False):
    d = start
    while d <= end:
        if (d.weekday() >= 5) == weekends:
            yield d
        d += timedelta(days=1)


def run_daily(cfg, day=None):
    con, client = connect(db_path(cfg)), nse.NSEClient()
    symbols = tracked_symbols(cfg)
    try:
        refresh_reference_lists(client)
        symbols = tracked_symbols(cfg)
    except Exception as e:  # reference lists are a convenience; never block the price data
        log(con, "reference", "", "error", 0, str(e))
    today = day or date.today()
    have = _have_dates(con, FULL_DAY_MIN_SYMBOLS if symbols is None else 0)
    report = {}
    for d in _weekdays(today - timedelta(days=CATCHUP_DAYS), today):
        if d.isoformat() not in have:
            n = collect_bhavcopy(con, client, d, symbols)
            if n is not None:
                report[d.isoformat()] = n
    for kind in ("bulk", "block"):
        try:
            rows = nse.parse_deals(client.get(f"{nse.ARCHIVE}/content/equities/{kind}.csv").text, kind)
            report[kind] = upsert(con, "deals", rows)
            log(con, kind, "", "ok", report[kind])
        except Exception as e:
            log(con, kind, "", "error", 0, str(e))
            report[kind] = f"error: {e}"
    try:
        rows = nse.parse_fii_dii(client.get(f"{nse.WWW}/api/fiidiiTradeReact", api=True).json())
        report["fii_dii"] = upsert(con, "fii_dii", rows, replace=True)
        log(con, "fii_dii", "", "ok", report["fii_dii"])
    except Exception as e:
        log(con, "fii_dii", "", "error", 0, str(e))
        report["fii_dii"] = f"error: {e}"
    return report


def run_news(cfg):
    con, client = connect(db_path(cfg)), nse.NSEClient()
    try:
        names = refresh_reference_lists(client)
    except Exception:
        names = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {}
    for feed, url in nse.FEEDS.items():
        try:
            rows = nse.parse_rss(client.get(url).text, feed, names, now)
            report[feed] = upsert(con, "announcements", rows)
            log(con, feed, "", "ok", report[feed])
        except Exception as e:
            log(con, feed, "", "error", 0, str(e))
            report[feed] = f"error: {e}"
    return report


def run_backfill(cfg, start, end=None, weekends=False):
    """weekends=True tries Saturdays/Sundays only: NSE trades on a few (budget day, muhurat,
    DR drills); the rest are logged as not_published/holiday and skipped next time."""
    con, client = connect(db_path(cfg)), nse.NSEClient(pause=1.0)
    try:
        refresh_reference_lists(client)
    except Exception:
        pass
    symbols = tracked_symbols(cfg)
    have = _have_dates(con, FULL_DAY_MIN_SYMBOLS if symbols is None else 0)
    end = end or date.today()
    days = [d for d in _weekdays(start, end, weekends) if d.isoformat() not in have]
    print(f"Backfilling {len(days)} weekdays ({start} -> {end}); ~1 request/second ...")
    done = 0
    for k, d in enumerate(days):
        try:
            if collect_bhavcopy(con, client, d, symbols) is not None:
                done += 1
        except Exception as e:
            log(con, "bhavcopy", d.isoformat(), "error", 0, str(e))
            print(f"  ! {d}: {e}")
        if k % 50 == 0:
            print(f"  {k}/{len(days)} checked, {done} trading days stored")
    print(f"Done: {done} trading days stored.")


def export_snapshot(src, dest):
    """Consistent copy of the live DB (SQLite online backup: safe while a collector is writing),
    gzipped for copying to the research PC. Returns the .gz path."""
    dest = dest.with_suffix(".db") if dest.suffix != ".db" else dest
    if dest.exists():
        dest.unlink()
    s, d = sqlite3.connect(src), sqlite3.connect(dest)
    with d:
        s.backup(d)
    s.close()
    d.close()
    gz = dest.with_name(dest.name + ".gz")
    with open(dest, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    dest.unlink()
    return gz


def status(cfg):
    con = connect(db_path(cfg))
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    print(f"Database: {db_path(cfg)}")
    print("eq_daily     : %s rows, %s days, %s -> %s, %s symbols" %
          (*q("SELECT COUNT(*), COUNT(DISTINCT date), MIN(date), MAX(date) FROM eq_daily"),
           q("SELECT COUNT(DISTINCT symbol) FROM eq_daily")[0]))
    print("deals        : %s rows, latest %s" % q("SELECT COUNT(*), MAX(date) FROM deals"))
    print("fii_dii      : %s rows, latest %s" % q("SELECT COUNT(*), MAX(date) FROM fii_dii"))
    print("announcements: %s rows, latest %s, matched to a symbol %s" %
          (*q("SELECT COUNT(*), MAX(published) FROM announcements"),
           q("SELECT COUNT(*) FROM announcements WHERE symbol IS NOT NULL")[0]))
    print("\nLast 8 runs:")
    for r in con.execute("SELECT * FROM collect_log ORDER BY rowid DESC LIMIT 8"):
        print("  ", " | ".join(str(x) for x in r))
    errs = q("SELECT COUNT(*) FROM collect_log WHERE status='error' AND run_at > datetime('now','-2 days')")[0]
    if errs:
        print(f"\n!! {errs} errors in the last 2 days - check collect_log")


def main(cfg, args):
    if args.job == "daily":
        day = date.fromisoformat(args.date) if args.date else None
        print(run_daily(cfg, day))
    elif args.job == "news":
        print(run_news(cfg))
    elif args.job == "backfill":
        run_backfill(cfg, date.fromisoformat(args.start),
                     date.fromisoformat(args.end) if args.end else None, weekends=args.weekends)
    elif args.job == "status":
        status(cfg)
    elif args.job == "export":
        src = db_path(cfg)
        gz = export_snapshot(src, src.with_name("market_snapshot.db"))
        print(f"Snapshot: {gz} ({gz.stat().st_size / 1e6:.0f} MB). On the PC:\n"
              f"  scp {__import__('getpass').getuser()}@<pi-ip>:{gz} data/\n"
              f"  then unzip to data/market.db (python -c \"import gzip,shutil;"
              f"shutil.copyfileobj(gzip.open('data/market_snapshot.db.gz'),open('data/market.db','wb'))\")")
