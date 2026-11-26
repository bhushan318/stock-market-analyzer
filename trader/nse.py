"""Fetch and parse NSE public files. Parsers are pure functions (tested offline)."""

import csv
import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests

ARCHIVE = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0 Safari/537.36")


class NotPublished(Exception):
    """File does not exist (holiday, weekend, or not yet published)."""


class NSEClient:
    def __init__(self, timeout=30, retries=3, pause=1.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
        self.timeout, self.retries, self.pause = timeout, retries, pause
        self._warmed = False

    def _warm(self):
        # www.nseindia.com API calls need the cookies set by a normal page visit
        if not self._warmed:
            try:
                self.s.get(WWW, timeout=self.timeout)
            except requests.RequestException:
                pass
            self._warmed = True

    def get(self, url, api=False):
        if api:
            self._warm()
        last = None
        for attempt in range(self.retries):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    raise NotPublished(url)
                if r.status_code in (401, 403) and api:
                    self._warmed = False
                    self._warm()
                r.raise_for_status()
                time.sleep(self.pause)  # be polite
                return r
            except NotPublished:
                raise
            except requests.RequestException as e:
                last = e
                time.sleep(self.pause * 2 ** attempt * 2)
        raise RuntimeError(f"failed after {self.retries} tries: {url}: {last}")


def _num(x, cast=float):
    x = (x or "").strip().replace(",", "")
    if x in ("", "-", "NA"):
        return None
    try:
        return cast(float(x)) if cast is int else cast(x)
    except ValueError:
        return None


# ------------------------------------------------------------------ bhavcopy

def bhavcopy_url(day):
    return f"{ARCHIVE}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv"


def xlsx_to_csv(content):
    """NSE occasionally publishes a bhavcopy as .xlsx under the .csv name (e.g. 2022-08-08).
    Minimal reader (first sheet, shared strings + numbers) -> CSV text; stdlib only."""
    import zipfile
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(io.BytesIO(content))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    out = io.StringIO()
    w = csv.writer(out)
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", ns):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                col = col * 26 + ord(ch) - 64
            v = c.find("m:v", ns)
            if c.get("t") == "inlineStr":
                txt = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
            elif v is None:
                txt = ""
            elif c.get("t") == "s":
                txt = shared[int(v.text)]
            else:
                txt = v.text
            vals[col - 1] = txt
        if vals:
            w.writerow([vals.get(i, "") for i in range(max(vals) + 1)])
    return out.getvalue()


def bhavcopy_text(content):
    """Response bytes -> CSV text, whatever format NSE served."""
    if content[:2] == b"PK":
        return xlsx_to_csv(content)
    return content.decode("utf-8", errors="replace")


def parse_bhavcopy(text, symbols=None, series=("EQ", "BE")):
    rows = []
    reader = csv.reader(io.StringIO(text))
    header = [h.strip() for h in next(reader)]
    for raw in reader:
        if not raw:
            continue
        r = dict(zip(header, (v.strip() for v in raw)))
        if r.get("SERIES") not in series:
            continue
        if symbols is not None and r["SYMBOL"] not in symbols:
            continue
        rows.append({
            "date": datetime.strptime(r["DATE1"], "%d-%b-%Y").date().isoformat(),
            "symbol": r["SYMBOL"], "series": r["SERIES"],
            "prev_close": _num(r["PREV_CLOSE"]), "open": _num(r["OPEN_PRICE"]),
            "high": _num(r["HIGH_PRICE"]), "low": _num(r["LOW_PRICE"]),
            "last": _num(r["LAST_PRICE"]), "close": _num(r["CLOSE_PRICE"]),
            "avg_price": _num(r["AVG_PRICE"]), "volume": _num(r["TTL_TRD_QNTY"], int),
            "turnover_lacs": _num(r["TURNOVER_LACS"]), "trades": _num(r["NO_OF_TRADES"], int),
            "deliv_qty": _num(r["DELIV_QTY"], int), "deliv_pct": _num(r["DELIV_PER"]),
        })
    return rows


# ------------------------------------------------------------------ deals

def parse_deals(text, kind):
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("Symbol"):
            continue
        rows.append({
            "date": datetime.strptime(r["Date"], "%d-%b-%Y").date().isoformat(),
            "kind": kind, "symbol": r["Symbol"], "security": r.get("Security Name", ""),
            "client": r.get("Client Name", ""), "side": r.get("Buy/Sell", ""),
            "qty": _num(r.get("Quantity Traded"), int),
            "price": _num(r.get("Trade Price / Wght. Avg. Price")),
            "remarks": r.get("Remarks", ""),
        })
    return rows


# ------------------------------------------------------------------ FII / DII

def parse_fii_dii(data):
    return [{
        "date": datetime.strptime(d["date"], "%d-%b-%Y").date().isoformat(),
        "category": d["category"], "buy": _num(d.get("buyValue")),
        "sell": _num(d.get("sellValue")), "net": _num(d.get("netValue")),
    } for d in data]


# ------------------------------------------------------------------ announcements

FEEDS = {
    "announcements": f"{ARCHIVE}/content/RSS/Online_announcements.xml",
    "results": f"{ARCHIVE}/content/RSS/Financial_Results.xml",
}

_SUFFIXES = r"\b(limited|ltd|private|pvt|india|the|co|company|corporation|corp|and|&)\b"


def norm_name(name):
    n = re.sub(r"[^a-z0-9& ]", " ", (name or "").lower())
    n = re.sub(_SUFFIXES, " ", n)
    return re.sub(r"\s+", " ", n).strip()


def parse_equity_list(text):
    """NSE EQUITY_L.csv -> {normalised company name: symbol}"""
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("SYMBOL") and r.get("NAME OF COMPANY"):
            out[norm_name(r["NAME OF COMPANY"])] = r["SYMBOL"]
    return out


def parse_pubdate(text):
    """NSE RSS pubDate -> ISO; seen as '25-Nov-2026 21:25:48' and '26-NOV-2026 19:38'."""
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            pass
    return text


def parse_rss(xml_text, feed, name_to_symbol=None, fetched_at=""):
    root = ET.fromstring(xml_text)
    rows = []
    for it in root.findall("./channel/item"):
        get = lambda tag: (it.findtext(tag) or "").strip()  # noqa: E731
        desc = get("description")
        subject = desc.split("|SUBJECT:", 1)[1].strip() if "|SUBJECT:" in desc else ""
        published = parse_pubdate(get("pubDate"))
        company = get("title")
        rows.append({
            "link": get("link"), "feed": feed, "published": published, "company": company,
            "symbol": (name_to_symbol or {}).get(norm_name(company)),
            "subject": subject, "description": desc.split("|SUBJECT:", 1)[0].strip(),
            "fetched_at": fetched_at,
        })
    return [r for r in rows if r["link"]]
