"""Old NSE equity bhavcopies (2005-2019) for long backtests - research only, runs on the PC.

    python -m trader history fetch [--start 2005-01-01] [--end 2019-12-31]
    python -m trader history status

Source: nsearchives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
(free, no login). Columns: open, high, low, close, prev close, volume, traded value - no delivery %.
Stored in data/history.db, table eq_daily with the same layout as market.db (delivery columns
empty), so pit.load_panel(db=...) works on it unchanged. Resumable: days already logged are
skipped; ~1 request/second.
"""

import csv
import io
import zipfile
from datetime import date, datetime

from . import ROOT, nse
from .collect import _weekdays
from .store import connect, log, upsert

DB = ROOT / "data" / "history.db"


def old_bhavcopy_url(day):
    mon = day.strftime("%b").upper()
    return f"{nse.ARCHIVE}/content/historical/EQUITIES/{day:%Y}/{mon}/cm{day:%d}{mon}{day:%Y}bhav.csv.zip"


def parse_old_bhavcopy(text, series=("EQ", "BE")):
    """Rows for eq_daily from the pre-2024 'cm...bhav.csv' format (TOTTRDVAL in rupees)."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        if r.get("SERIES") not in series:
            continue
        try:
            d = datetime.strptime(r["TIMESTAMP"], "%d-%b-%Y").date().isoformat()
        except (KeyError, ValueError):
            continue
        val = nse._num(r.get("TOTTRDVAL"))
        out.append({"date": d, "symbol": r["SYMBOL"], "series": r["SERIES"],
                    "prev_close": nse._num(r.get("PREVCLOSE")), "open": nse._num(r.get("OPEN")),
                    "high": nse._num(r.get("HIGH")), "low": nse._num(r.get("LOW")),
                    "last": nse._num(r.get("LAST")), "close": nse._num(r.get("CLOSE")),
                    "volume": nse._num(r.get("TOTTRDQTY"), int),
                    "turnover_lacs": None if val is None else val / 1e5})
    return out


def unzip_text(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    return z.read(z.namelist()[0]).decode("latin1")


def fetch(start, end):
    con, client = connect(DB), nse.NSEClient()
    done = {r[0] for r in con.execute("SELECT ref FROM collect_log WHERE source='old_bhavcopy' "
                                      "AND status IN ('ok','not_published','holiday')")}
    days = [d for d in _weekdays(start, end) if d.isoformat() not in done]
    print(f"{len(days)} weekdays to fetch ({start} -> {end}), ~{len(days) * 1.3 / 3600:.1f} h", flush=True)
    for k, d in enumerate(days):
        try:
            rows = parse_old_bhavcopy(unzip_text(client.get(old_bhavcopy_url(d)).content))
        except nse.NotPublished:
            log(con, "old_bhavcopy", d.isoformat(), "not_published")
            continue
        except Exception as e:  # noqa: BLE001 - logged, retried on the next run
            log(con, "old_bhavcopy", d.isoformat(), "error", 0, f"{type(e).__name__}: {str(e)[:200]}")
            continue
        rows = [r for r in rows if r["date"] == d.isoformat()]
        if not rows:
            log(con, "old_bhavcopy", d.isoformat(), "holiday")
            continue
        n = upsert(con, "eq_daily", rows)
        log(con, "old_bhavcopy", d.isoformat(), "ok", n)
        if k % 50 == 0:
            print(f"  {d}: {n} rows ({k + 1}/{len(days)})", flush=True)
    status()


def status():
    con = connect(DB)
    for st, n in con.execute("SELECT status, COUNT(*) FROM collect_log WHERE source='old_bhavcopy' GROUP BY status"):
        print(f"  {st}: {n} days")
    r = con.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(*) FROM eq_daily").fetchone()
    print(f"  eq_daily: {r[0]} -> {r[1]}, {r[2]} sessions, {r[3]:,} rows")


def main(cfg, args):
    if args.job == "status":
        return status()
    fetch(date.fromisoformat(args.start), date.fromisoformat(args.end))
