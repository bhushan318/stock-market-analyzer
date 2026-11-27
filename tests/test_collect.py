import json

from trader import nse
from trader.store import connect, upsert

BHAV = """SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
20MICRONS, EQ, 25-Nov-2026, 180.44, 180.61, 181.21, 177.33, 178.00, 178.10, 179.02, 39466, 71.66, 2063, 21631, 54.81
TCS, EQ, 25-Nov-2026, 2500.00, 2505.00, 2530.00, 2490.00, 2517.00, 2517.30, 2512.10, 1200000, 30145.20, 80000, 700000, 58.33
TCS, BL, 25-Nov-2026, 2500.00, 2505.00, 2505.00, 2505.00, 2505.00, 2505.00, 2505.00, 100, 2.5, 1, -, -
NIFTYBEES, EQ, 25-Nov-2026, 293.00, 293.50, 295.00, 292.80, 294.07, 294.07, 294.00, 5000000, 14700.00, 30000, 4000000, 80.00
"""

DEALS = """Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks
25-NOV-2026,AGIIL,Agi Infra Limited,THAKKAR NILESHKUMAR FARSHURAM HUF,BUY,1600000,303.32,-
"""

RSS = """<rss version="2.0"><channel><title>x</title>
<item><title>Tata Consultancy Services Limited</title><link>https://x/a.pdf</link>
<description>TCS has informed the Exchange about results |SUBJECT: Outcome of Board Meeting</description>
<pubDate>25-Nov-2026 21:25:48</pubDate></item>
<item><title>Unknown Co Ltd</title><link>https://x/b.pdf</link><description>no subject</description>
<pubDate>25-Nov-2026 20:00:00</pubDate></item>
</channel></rss>"""

EQUITY_L = """SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING
TCS,Tata Consultancy Services Limited,EQ,25-AUG-2004
"""


def test_parse_bhavcopy_filters_series_and_symbols():
    rows = nse.parse_bhavcopy(BHAV)
    assert {(r["symbol"], r["series"]) for r in rows} == {("20MICRONS", "EQ"), ("TCS", "EQ"), ("NIFTYBEES", "EQ")}
    tcs = next(r for r in rows if r["symbol"] == "TCS")
    assert tcs["date"] == "2026-11-25" and tcs["close"] == 2517.30 and tcs["deliv_pct"] == 58.33
    assert tcs["volume"] == 1200000
    only = nse.parse_bhavcopy(BHAV, symbols={"TCS"})
    assert [r["symbol"] for r in only] == ["TCS"]


def test_parse_deals_and_fii():
    d = nse.parse_deals(DEALS, "bulk")[0]
    assert d["date"] == "2026-11-25" and d["qty"] == 1600000 and d["price"] == 303.32
    f = nse.parse_fii_dii(json.loads('[{"buyValue":"22379.82","category":"DII","date":"25-Nov-2026",'
                                     '"netValue":"4018.72","sellValue":"18361.1"}]'))[0]
    assert f == {"date": "2026-11-25", "category": "DII", "buy": 22379.82, "sell": 18361.1, "net": 4018.72}


def test_parse_rss_maps_company_to_symbol():
    names = nse.parse_equity_list(EQUITY_L)
    rows = nse.parse_rss(RSS, "announcements", names, "now")
    assert rows[0]["symbol"] == "TCS" and rows[0]["subject"] == "Outcome of Board Meeting"
    assert rows[0]["published"] == "2026-11-25T21:25:48"
    assert rows[1]["symbol"] is None and rows[1]["subject"] == ""
    # NSE sometimes drops the seconds and upper-cases the month
    assert nse.parse_pubdate("26-NOV-2026 19:38") == "2026-11-26T19:38:00"
    assert nse.parse_pubdate("garbage") == "garbage"


def test_upsert_is_idempotent(tmp_path):
    con = connect(tmp_path / "t.db")
    rows = nse.parse_bhavcopy(BHAV)
    assert upsert(con, "eq_daily", rows) == 3
    assert upsert(con, "eq_daily", rows) == 0
    assert con.execute("SELECT COUNT(*) FROM eq_daily").fetchone()[0] == 3


def test_have_dates_refetches_partial_days(tmp_path):
    from trader.collect import _have_dates
    con = connect(tmp_path / "m.db")
    rows = [{"date": "2022-01-03", "symbol": f"S{i}", "series": "EQ"} for i in range(5)]
    rows += [{"date": "2022-01-04", "symbol": "S0", "series": "EQ"}]
    upsert(con, "eq_daily", rows)
    assert _have_dates(con) >= {"2022-01-03", "2022-01-04"}
    have = _have_dates(con, min_symbols=3)
    assert "2022-01-03" in have and "2022-01-04" not in have


def test_pit_symbol_changes_and_split_adjustment():
    import pandas as pd
    from trader.pit import adjustment_multiplier, apply_symbol_changes, split_events
    d = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"])
    # 1:5 split effective 2021-01-06; NSE's prev_close is NOT adjusted (as in the real files),
    # the stock opens at 20.6 (gap 0.206 -> snaps to 1/5) and rallies to 23 that day
    df = pd.DataFrame({"date": d, "symbol": ["OLD", "OLD", "NEW", "NEW"],
                       "prev_close": [99.0, 100.0, 100.0, 23.0], "open": [99.0, 100.0, 20.6, 23.0],
                       "close": [100.0, 100.0, 23.0, 22.0]})
    df = apply_symbol_changes(df, [("OLD", "NEW", pd.Timestamp("2021-01-06"))])
    assert set(df["symbol"]) == {"NEW"}
    ev = split_events(df)
    assert len(ev) == 1 and abs(ev.factor[0] - 0.2) < 1e-9 and ev.date[0] == d[2]
    M = adjustment_multiplier(ev, pd.DatetimeIndex(d), ["NEW"])
    assert list(M["NEW"]) == [0.2, 0.2, 1.0, 1.0]
    # a 10% opening gap (missed Saturday session, news) is not a split
    df.loc[2:, ["open", "close"]] = [[90.0, 92.0], [92.0, 93.0]]
    assert split_events(df).empty
    # an intraday crash (opens flat, closes -30%) is not a split either
    df.loc[2:, ["open", "close"]] = [[99.0, 70.0], [70.0, 68.0]]
    assert split_events(df).empty
    # a demerger-like gap with no clean ratio keeps its own factor
    from trader.pit import snap_ratio
    assert abs(snap_ratio([0.605])[0] - 0.6) < 1e-9      # 2:3 bonus / demerger near 3/5
    assert abs(snap_ratio([0.45])[0] - 0.45) < 1e-9      # no clean ratio nearby: keep the gap
    assert abs(snap_ratio([0.198])[0] - 0.2) < 1e-9      # IRCTC 1:5 split opened at 0.198


def test_export_snapshot_roundtrip(tmp_path):
    import gzip
    import shutil
    import sqlite3
    from trader.collect import export_snapshot
    src = tmp_path / "market.db"
    con = connect(src)
    upsert(con, "eq_daily", [{"date": "2022-01-03", "symbol": "ABC", "series": "EQ", "close": 10.0}])
    gz = export_snapshot(src, tmp_path / "snap.db")
    assert gz.name == "snap.db.gz" and not (tmp_path / "snap.db").exists()
    out = tmp_path / "restored.db"
    shutil.copyfileobj(gzip.open(gz), open(out, "wb"))
    assert sqlite3.connect(out).execute("SELECT close FROM eq_daily").fetchone() == (10.0,)


def test_bhavcopy_served_as_xlsx_is_parsed():
    import io
    import zipfile
    m = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    head = ["SYMBOL", " SERIES", " DATE1", " PREV_CLOSE", " OPEN_PRICE", " HIGH_PRICE", " LOW_PRICE",
            " LAST_PRICE", " CLOSE_PRICE", " AVG_PRICE", " TTL_TRD_QNTY", " TURNOVER_LACS",
            " NO_OF_TRADES", " DELIV_QTY", " DELIV_PER"]
    strings = head + ["ABC", " EQ", " 08-Aug-2022"]
    sst = f"<sst {m}>" + "".join(f"<si><t>{x}</t></si>" for x in strings) + "</sst>"
    cols = "ABCDEFGHIJKLMNO"
    r1 = "".join(f'<c r="{cols[i]}1" t="s"><v>{i}</v></c>' for i in range(15))
    nums = [100, 101, 105, 99, 104, 104.5, 103, 1000, 10.3, 50, 600, 60]
    r2 = "".join(f'<c r="{cols[i]}2" t="s"><v>{15 + i}</v></c>' for i in range(3))
    r2 += "".join(f'<c r="{cols[3 + i]}2"><v>{v}</v></c>' for i, v in enumerate(nums))
    sheet = f'<worksheet {m}><sheetData><row r="1">{r1}</row><row r="2">{r2}</row></sheetData></worksheet>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", sst)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    rows = nse.parse_bhavcopy(nse.bhavcopy_text(buf.getvalue()))
    assert len(rows) == 1
    r = rows[0]
    assert (r["date"], r["symbol"], r["series"], r["close"], r["volume"], r["deliv_pct"]) == \
        ("2022-08-08", "ABC", "EQ", 104.5, 1000, 60.0)


def test_old_bhavcopy_format():
    from datetime import date
    from trader.history import old_bhavcopy_url, parse_old_bhavcopy
    assert old_bhavcopy_url(date(2008, 1, 21)).endswith("/EQUITIES/2008/JAN/cm21JAN2008bhav.csv.zip")
    text = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
            "3IINFOTECH,EQ,132,132,111,117.75,117.75,131.5,1006260,118590976.85,21-JAN-2008,\n"
            "3MINDIA,EQ,584,585,580,581.1,581,575.55,1387,808306.35,3-JAN-2005,\n"
            "XYZ,N1,100,100,100,100,100,100,1,100,21-JAN-2008,\n")
    rows = parse_old_bhavcopy(text)
    assert [r["symbol"] for r in rows] == ["3IINFOTECH", "3MINDIA"]
    r = rows[0]
    assert (r["date"], r["open"], r["low"], r["close"], r["volume"]) == ("2008-01-21", 132.0, 111.0, 117.75, 1006260)
    assert abs(r["turnover_lacs"] - 1185.9097685) < 1e-6 and rows[1]["date"] == "2005-01-03"


def test_hourly_frame_to_rows():
    import numpy as np
    import pandas as pd
    from trader.hourly import frame_to_rows
    idx = pd.DatetimeIndex(["2026-11-26 09:15", "2026-11-26 10:15", "2026-11-26 11:15"], tz="Asia/Kolkata")
    df = pd.DataFrame({"Open": [100, np.nan, 101], "High": [102, 1, 103], "Low": [99, 1, 0],
                       "Close": [101, 1, 102], "Volume": [1000, 5, np.nan]}, index=idx)
    rows = frame_to_rows(df, "TCS")   # NaN row and zero-low row dropped
    assert rows == [{"ts": "2026-11-26T09:15", "symbol": "TCS", "open": 100.0, "high": 102.0,
                     "low": 99.0, "close": 101.0, "volume": 1000}]
    utc = df.iloc[:1].copy()
    utc.index = pd.DatetimeIndex(["2026-11-26 03:45"])  # naive = UTC -> 09:15 IST
    assert frame_to_rows(utc, "TCS")[0]["ts"] == "2026-11-26T09:15"


def test_news_parse_and_entry_day():
    import pandas as pd
    from trader.news import event_entry_index, parse_announcements
    rows = parse_announcements([{"an_dt": "26-Nov-2026 16:02:31", "desc": " Acquisition ", "symbol": "ABC",
                                 "seq_id": 7, "attchmntText": "buys XYZ"}, {"an_dt": "bad"}])
    assert rows == [{"seq_id": "7", "symbol": "ABC", "ts": "2026-11-26T16:02:31", "category": "Acquisition",
                     "text": "buys XYZ"}]
    cal = pd.DatetimeIndex(["2026-11-25", "2026-11-26", "2026-11-27", "2026-11-30"])
    assert event_entry_index("2026-11-26T08:30:00", cal) == 1   # before the open: same day's open
    assert event_entry_index("2026-11-26T11:00:00", cal) == 2   # during market: next session
    assert event_entry_index("2026-11-27T19:00:00", cal) == 3   # Friday evening: Monday
    assert event_entry_index("2026-11-28T10:00:00", cal) == 3   # Saturday: Monday
