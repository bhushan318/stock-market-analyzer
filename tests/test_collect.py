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


def test_upsert_is_idempotent(tmp_path):
    con = connect(tmp_path / "t.db")
    rows = nse.parse_bhavcopy(BHAV)
    assert upsert(con, "eq_daily", rows) == 3
    assert upsert(con, "eq_daily", rows) == 0
    assert con.execute("SELECT COUNT(*) FROM eq_daily").fetchone()[0] == 3
