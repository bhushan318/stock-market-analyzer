"""NSE F&O daily bhavcopy (futures + options) -> one summary row per underlying per day.

    python -m trader fno fetch [--start 2008-01-01] [--end today]
    python -m trader fno status

Two formats: the old "foDDMONYYYYbhav.csv.zip" (2000s -> Jul 2024) and UDiFF
"BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip" (Jul 2024 ->). Raw files have 30-40k contract rows
per day, so only per-underlying totals are stored (data/fno.db, PC research):
  fut_oi, fut_oi_chg, fut_contracts, fut_close (nearest expiry)       futures
  ce_oi, ce_oi_chg, ce_contracts / pe_...                              calls / puts
Open interest is in shares (both formats), volume in contracts (UDiFF shares / lot size).
Ratios built from these (put/call, OI build-up) are unit-free. Resumable, ~1 request/second.
"""

import csv
import io
import sqlite3
import zipfile
from collections import defaultdict
from datetime import date, datetime

from . import ROOT, nse
from .collect import _weekdays

DB = ROOT / "data" / "fno.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS fno_daily (
    date TEXT, symbol TEXT, kind TEXT,            -- kind: stock / index
    fut_oi REAL, fut_oi_chg REAL, fut_contracts REAL, fut_close REAL,
    ce_oi REAL, ce_oi_chg REAL, ce_contracts REAL,
    pe_oi REAL, pe_oi_chg REAL, pe_contracts REAL,
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS fno_log (date TEXT PRIMARY KEY, status TEXT, rows INTEGER, detail TEXT);
"""
UDIFF_FROM = date(2024, 7, 8)
KIND = {"FUTSTK": ("stock", "F"), "OPTSTK": ("stock", "O"), "FUTIDX": ("index", "F"), "OPTIDX": ("index", "O"),
        "STF": ("stock", "F"), "STO": ("stock", "O"), "IDF": ("index", "F"), "IDO": ("index", "O")}


def old_url(d):
    mon = d.strftime("%b").upper()
    return f"{nse.ARCHIVE}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"


def udiff_url(d):
    return f"{nse.ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def _f(x):
    try:
        return float((x or "").strip() or 0)
    except ValueError:
        return 0.0


def parse_contracts(text):
    """Either format -> list of (date, kind, symbol, F/CE/PE, expiry, close, oi, oi_chg, contracts)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    if rows and "INSTRUMENT" in rows[0]:
        for r in rows:
            k = KIND.get((r.get("INSTRUMENT") or "").strip())
            if not k:
                continue
            leg = "F" if k[1] == "F" else (r.get("OPTION_TYP") or "").strip()
            if leg not in ("F", "CE", "PE"):
                continue
            d = datetime.strptime(r["TIMESTAMP"].strip(), "%d-%b-%Y").date().isoformat()
            exp = datetime.strptime(r["EXPIRY_DT"].strip(), "%d-%b-%Y").date().isoformat()
            out.append((d, k[0], r["SYMBOL"].strip(), leg, exp, _f(r["CLOSE"]), _f(r["OPEN_INT"]),
                        _f(r["CHG_IN_OI"]), _f(r["CONTRACTS"])))
    elif rows and "FinInstrmTp" in rows[0]:
        for r in rows:
            k = KIND.get((r.get("FinInstrmTp") or "").strip())
            if not k:
                continue
            leg = "F" if k[1] == "F" else (r.get("OptnTp") or "").strip()
            if leg not in ("F", "CE", "PE"):
                continue
            lot = _f(r.get("NewBrdLotQty")) or 1.0
            out.append((r["TradDt"].strip(), k[0], r["TckrSymb"].strip(), leg, r["XpryDt"].strip(),
                        _f(r["ClsPric"]), _f(r["OpnIntrst"]), _f(r["ChngInOpnIntrst"]), _f(r["TtlTradgVol"]) / lot))
    return out


def aggregate(contracts):
    """Per (date, symbol) totals; fut_close = close of the nearest-expiry future."""
    acc = defaultdict(lambda: {"fut_oi": 0.0, "fut_oi_chg": 0.0, "fut_contracts": 0.0, "fut_close": None, "_exp": None,
                               "ce_oi": 0.0, "ce_oi_chg": 0.0, "ce_contracts": 0.0,
                               "pe_oi": 0.0, "pe_oi_chg": 0.0, "pe_contracts": 0.0, "kind": None})
    for d, kind, sym, leg, exp, close, oi, chg, n in contracts:
        a = acc[(d, sym)]
        a["kind"] = kind
        p = {"F": "fut", "CE": "ce", "PE": "pe"}[leg]
        a[f"{p}_oi"] += oi
        a[f"{p}_oi_chg"] += chg
        a[f"{p}_contracts"] += n
        if leg == "F" and (a["_exp"] is None or exp < a["_exp"]):
            a["_exp"], a["fut_close"] = exp, close
    out = []
    for (d, sym), a in acc.items():
        a.pop("_exp")
        out.append({"date": d, "symbol": sym, **a})
    return out


def unzip_text(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    return z.read(z.namelist()[0]).decode("latin1")


def connect():
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def fetch(start, end):
    con, client = connect(), nse.NSEClient()
    done = {r[0] for r in con.execute("SELECT date FROM fno_log WHERE status IN ('ok','not_published')")}
    days = [d for d in _weekdays(start, end) if d.isoformat() not in done]
    print(f"{len(days)} weekdays to fetch ({start} -> {end}), ~{len(days) * 1.6 / 3600:.1f} h", flush=True)
    cols = ["date", "symbol", "kind", "fut_oi", "fut_oi_chg", "fut_contracts", "fut_close", "ce_oi", "ce_oi_chg",
            "ce_contracts", "pe_oi", "pe_oi_chg", "pe_contracts"]
    for k, d in enumerate(days):
        urls = [udiff_url(d), old_url(d)] if d >= UDIFF_FROM else [old_url(d), udiff_url(d)]
        text, status = None, "not_published"
        for u in urls:
            try:
                text = unzip_text(client.get(u).content)
                break
            except nse.NotPublished:
                continue
            except Exception as e:  # noqa: BLE001 - logged, retried next run
                status = f"error {type(e).__name__}"
                break
        if text is None:
            con.execute("INSERT OR REPLACE INTO fno_log VALUES (?,?,?,?)", (d.isoformat(), status.split()[0], 0, status))
            con.commit()
            continue
        rows = [r for r in aggregate(parse_contracts(text)) if r["date"] == d.isoformat()]
        con.executemany(f"INSERT OR REPLACE INTO fno_daily ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [tuple(r[c] for c in cols) for r in rows])
        con.execute("INSERT OR REPLACE INTO fno_log VALUES (?,?,?,?)",
                    (d.isoformat(), "ok" if rows else "holiday", len(rows), ""))
        con.commit()
        if k % 100 == 0:
            print(f"  {d}: {len(rows)} underlyings ({k + 1}/{len(days)})", flush=True)
    status()


def status():
    con = connect()
    for st, n in con.execute("SELECT status, COUNT(*) FROM fno_log GROUP BY status"):
        print(f"  {st}: {n} days")
    r = con.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(*), "
                    "COUNT(DISTINCT CASE WHEN kind='stock' THEN symbol END) FROM fno_daily").fetchone()
    print(f"  fno_daily: {r[0]} -> {r[1]}, {r[2]} sessions, {r[3]:,} rows, {r[4]} stock underlyings ever")


def main(cfg, args):
    if args.job == "status":
        return status()
    fetch(date.fromisoformat(args.start), date.fromisoformat(args.end) if args.end else date.today())
