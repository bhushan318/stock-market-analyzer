"""Gate 1: paper trading of the chosen rule (Results 9), run on the Pi. NO real orders.

    python -m trader paper init      # start the paper books (once)
    python -m trader paper run       # daily job (systemd timer, after the 19:30 collector)
    python -m trader paper status    # print holdings and value
    python -m trader paper chatid    # find your Telegram chat id after messaging the bot

Rule (identical to the backtest in momentum_study.simulate / robust_study "base"):
  every week, on its last session's close: rank the point-in-time top-500 by "combo" momentum;
  if Nifty 500 is above its 200-day average: sell holdings that fell out of the top 50, buy the
  best-ranked stocks not held until 5 are held (slot = strategy value / 5); if below: do nothing.
  Orders fill at the next session's open (+0.05% slippage, all Zerodha charges).
Money: `paper.capital` split into `strategy_share` for the stocks and the rest in a Nifty 500
index fund ("core"), reset to that split on the first session of each January.
Books: "main" (top 500) and "large" (top 200 only, tracked for comparison).

Data: data/market.db (collectors), Nifty 500 index from Yahoo (^CRSLDX). State: data/paper.db.
Secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) come from .env and are never printed or logged.
"""

import math
import os
import sqlite3
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from . import ROOT
from .costs import buy_charges, sell_charges

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (name TEXT PRIMARY KEY, universe TEXT, started TEXT, capital REAL,
    share REAL, cash REAL, core_units REAL);
CREATE TABLE IF NOT EXISTS positions (book TEXT, symbol TEXT, shares INTEGER, basis REAL, buy_date TEXT,
    adj_through TEXT, PRIMARY KEY (book, symbol));
CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, book TEXT, decided TEXT, side TEXT, symbol TEXT,
    rank INTEGER, status TEXT, fill_date TEXT, price REAL, shares INTEGER, charges REAL, note TEXT);
CREATE TABLE IF NOT EXISTS trades (book TEXT, symbol TEXT, buy_date TEXT, sell_date TEXT, shares INTEGER,
    basis REAL, proceeds REAL, pnl REAL);
CREATE TABLE IF NOT EXISTS equity (book TEXT, date TEXT, strategy REAL, core REAL, total REAL, nifty500 REAL,
    regime INTEGER, PRIMARY KEY (book, date));
CREATE TABLE IF NOT EXISTS decisions (book TEXT, week TEXT, date TEXT, regime INTEGER, PRIMARY KEY (book, week));
"""


# ---------------------------------------------------------------- pure rule functions (tested)

def decide(held, ranked, close, equity, regime_ok, n=5, exit_rank=50, min_slot=1500):
    """Weekly decision, same as simulate(): returns (sells, buys).
    held: symbols held; ranked: eligible symbols best first; close: {symbol: last close}."""
    if not regime_ok:
        return [], []
    rank = {s: i for i, s in enumerate(ranked)}
    sells = sorted(s for s in held if rank.get(s, 10**9) >= exit_rank)
    free = n - (len(held) - len(sells))
    slot = equity / n
    buys = []
    if free > 0 and slot >= min_slot:
        for s in ranked:
            if free == 0:
                break
            px = close.get(s, np.nan)
            if s in held or not px == px or px * 1.01 > slot:
                continue
            buys.append(s)
            free -= 1
    return sells, buys


def fill_buys(cash, prev_equity, opens, buys, costs, n=5, min_val=1500):
    """Fill buy orders at the open like simulate(). opens: {symbol: open or nan}.
    Returns (cash, [(symbol, shares, price, charges)], [symbols not filled: no open])."""
    slip = costs["slippage"]
    slot = prev_equity / n
    todo = [s for s in buys if opens.get(s, np.nan) == opens.get(s, np.nan)]
    waiting = [s for s in buys if s not in todo]
    left, fills = len(todo), []
    for s in todo:
        px = opens[s] * (1 + slip)
        budget = min(slot, cash / left)
        left -= 1
        sh = math.floor(budget / (px * 1.0013))
        while sh > 0 and sh * px + buy_charges(sh * px, costs) > cash:
            sh -= 1
        if sh < 1 or sh * px < min_val * 0.5:
            fills.append((s, 0, px, 0.0))
            continue
        ch = buy_charges(sh * px, costs)
        cash -= sh * px + ch
        fills.append((s, sh, px, ch))
    return cash, fills, waiting


def sell_fill(shares, open_px, costs):
    """(price, charges, net proceeds) of selling at the open."""
    px = open_px * (1 - costs["slippage"])
    v = shares * px
    ch = sell_charges(v, costs)
    return px, ch, v - ch


def rebalance_amount(strategy_value, strategy_cash, core_value, share):
    """Rupees to move core -> strategy cash (negative = strategy cash -> core) to get back to `share`.
    Only idle cash can go to the core; stocks are never sold for it."""
    diff = share * (strategy_value + core_value) - strategy_value
    return diff if diff >= 0 else -min(-diff, strategy_cash)


def is_decision_day(last_session, today, decided_weeks):
    """Decide on the last session of each ISO week: Friday's close, or (holiday Friday) the
    Saturday catch-up run after a Thursday close - once per week."""
    wk = "%d-W%02d" % last_session.isocalendar()[:2]
    if wk in decided_weeks:
        return False
    friday = last_session - timedelta(days=last_session.weekday() - 4)
    return last_session.weekday() == 4 or today > friday


# ---------------------------------------------------------------- Telegram

def load_env(path=None):
    env = {}
    path = path or ROOT / ".env"
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("TELEGRAM_")}}


def telegram(text, env=None, tries=3):
    """Send a plain-text message; returns True on success. Never logs the token."""
    env = env or load_env()
    tok, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("(Telegram not configured - message printed only)\n" + text)
        return False
    for k in range(tries):
        try:
            for part in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
                r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                                  json={"chat_id": chat, "text": part}, timeout=20)
                r.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"telegram attempt {k + 1} failed: {type(e).__name__} {getattr(e.response, 'status_code', '')}")
            time.sleep(5 * 2 ** k)
    return False


def show_chat_ids():
    env = load_env()
    tok = env.get("TELEGRAM_BOT_TOKEN")
    if not tok:
        print("Put TELEGRAM_BOT_TOKEN=... in .env first.")
        return
    r = requests.get(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=20)
    r.raise_for_status()
    chats = {(u.get("message") or {}).get("chat", {}).get("id"): (u.get("message") or {}).get("chat", {})
             for u in r.json().get("result", [])}
    chats.pop(None, None)
    if not chats:
        print("No messages yet: open your bot in Telegram, press Start / send 'hi', then run this again.")
    for cid, c in chats.items():
        print(f"chat id {cid}  ({c.get('type')}, {c.get('first_name', '') or c.get('title', '')})")
    print("Add TELEGRAM_CHAT_ID=<id> to .env")


# ---------------------------------------------------------------- market data

def market(cfg):
    """Panel (recent history only), signals per book, regime series, Nifty 500 series."""
    from .data import download_prices
    from .momentum_pit import pit_signals
    from .momentum_study import BENCH
    from .pit import liquid_universe, load_panel
    pc = cfg["paper"]
    start = (date.today() - timedelta(days=pc["history_days"])).isoformat()
    p = load_panel(start=start)
    C, cal = p["C"], p["cal"]
    unis = {"top500": liquid_universe(C, p["TO"], top=500), "top200": liquid_universe(C, p["TO"], top=200)}
    combo = {k: pit_signals(C, p["TO"], u)[0]["combo"] for k, u in unis.items()}
    dc = cfg["data"]
    n5 = None
    try:
        bp = download_prices([BENCH["nifty500"]], start, ROOT / dc["cache_dir"], 6)
        n5 = bp[BENCH["nifty500"]]["Close"]
    except Exception as e:  # noqa: BLE001 - reported, rule falls back to "no new buys"
        print(f"Nifty 500 download failed: {type(e).__name__}")
    if n5 is not None:
        stale = n5.index[-1] < cal[-1] - pd.Timedelta(days=4)
        n5c = n5.reindex(cal.union(n5.index)).ffill().reindex(cal)
        ma = n5c.rolling(pc["regime_ma"], min_periods=int(pc["regime_ma"] * 0.75)).mean()
        regime = (n5c > ma) & ~stale
    else:
        n5c, regime = pd.Series(np.nan, index=cal), pd.Series(False, index=cal)
    return {"p": p, "cal": cal, "combo": combo, "n5": n5c, "regime": regime,
            "regime_known": n5 is not None and not (n5.index[-1] < cal[-1] - pd.Timedelta(days=4))}


# ---------------------------------------------------------------- state

def connect(cfg):
    con = sqlite3.connect(ROOT / cfg["paper"]["db"])
    con.executescript(SCHEMA)
    return con


def init(cfg, force=False):
    pc = cfg["paper"]
    con = connect(cfg)
    if con.execute("SELECT COUNT(*) FROM books").fetchone()[0] and not force:
        print("Paper books already exist (status: python -m trader paper status).")
        return
    m = market(cfg)
    d = m["cal"][-1]
    n5 = m["n5"].iloc[-1]
    if not n5 == n5:
        raise SystemExit("No Nifty 500 price - cannot value the index-fund part. Try again later.")
    cap, share = pc["capital"], pc["strategy_share"]
    for t in ("books", "positions", "orders", "trades", "equity", "decisions"):
        con.execute(f"DELETE FROM {t}")
    for name, uni in pc["books"].items():
        con.execute("INSERT INTO books VALUES (?,?,?,?,?,?,?)",
                    (name, uni, d.date().isoformat(), cap, share, cap * share, cap * (1 - share) / n5))
    con.commit()
    print(f"Paper books started {d.date()}: Rs {cap:,.0f} each, {share:.0%} strategy / "
          f"{1 - share:.0%} Nifty 500 index fund. Books: {', '.join(pc['books'])}.")


def _prices(m, d):
    p = m["p"]
    i = m["cal"].get_loc(d)
    return p["O"].iloc[i], p["C"].ffill().iloc[i]


def run(cfg, today=None, notify=True):
    pc, costs = cfg["paper"], cfg["costs"]
    con = connect(cfg)
    books = con.execute("SELECT name, universe, started, capital, share, cash, core_units FROM books").fetchall()
    if not books:
        raise SystemExit("No paper books - run: python -m trader paper init")
    m = market(cfg)
    cal, p = m["cal"], m["p"]
    d = cal[-1]
    ds = d.date().isoformat()
    today = today or date.today()
    n_today = int(p["C"].iloc[-1].notna().sum())
    if n_today < 1000:
        raise RuntimeError(f"bhavcopy for {ds} looks incomplete ({n_today} stocks) - nothing done")
    O, Cl = _prices(m, d)
    n5 = float(m["n5"].iloc[-1])
    decay = (1 - pc["etf_fee"]) ** (1 / 248)
    msgs, alerts = [], []
    if not m["regime_known"]:
        alerts.append("Nifty 500 index data missing or stale - no new buys this week until it is back.")

    for name, uni, started, capital, share, cash, core_units in books:
        last = con.execute("SELECT MAX(date) FROM equity WHERE book=?", (name,)).fetchone()[0]
        new_day = last is None or ds > last
        if new_day and ds >= started:
            # 1) splits/bonuses on held stocks since we last looked
            ev = p["events"]
            for sym, sh, adj in con.execute("SELECT symbol, shares, adj_through FROM positions WHERE book=?",
                                            (name,)).fetchall():
                e = ev[(ev.symbol == sym) & (ev.date > pd.Timestamp(adj)) & (ev.date <= d)]
                for r in e.itertuples():
                    sh = int(round(sh / r.factor))
                    alerts.append(f"[{name}] {sym}: split/bonus on {r.date.date()} (x{1 / r.factor:.2f} shares)")
                con.execute("UPDATE positions SET shares=?, adj_through=? WHERE book=? AND symbol=?",
                            (sh, ds, name, sym))
                if sym not in Cl.index or not Cl[sym] == Cl[sym]:
                    alerts.append(f"[{name}] {sym}: no price today (renamed / suspended?) - check manually")
            # 2) fill pending orders at today's open (decided on an earlier session)
            prev_eq = con.execute("SELECT strategy FROM equity WHERE book=? ORDER BY date DESC LIMIT 1",
                                  (name,)).fetchone()
            pend = con.execute("SELECT id, side, symbol, decided FROM orders WHERE book=? AND status='pending' "
                               "AND decided < ? ORDER BY id", (name, ds)).fetchall()
            fills = []
            for oid, side, sym, dec in [o for o in pend if o[1] == "sell"]:
                o = O.get(sym, np.nan)
                pos = con.execute("SELECT shares, basis, buy_date FROM positions WHERE book=? AND symbol=?",
                                  (name, sym)).fetchone()
                if pos is None:
                    con.execute("UPDATE orders SET status='cancelled', note='not held' WHERE id=?", (oid,))
                    continue
                if not o == o:
                    alerts.append(f"[{name}] sell {sym}: no open price today, will retry")
                    continue
                px, ch, net = sell_fill(pos[0], o, costs)
                cash += net
                con.execute("DELETE FROM positions WHERE book=? AND symbol=?", (name, sym))
                con.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                            (name, sym, pos[2], ds, pos[0], pos[1], net, net - pos[1]))
                con.execute("UPDATE orders SET status='filled', fill_date=?, price=?, shares=?, charges=? WHERE id=?",
                            (ds, px, pos[0], ch, oid))
                fills.append(f"SOLD {sym} {pos[0]} @ {px:,.2f}  P&L Rs {net - pos[1]:+,.0f} ({net / pos[1] - 1:+.1%})")
            buys = [o for o in pend if o[1] == "buy"]
            if buys:
                base_eq = prev_eq[0] if prev_eq else cash
                cash, bf, waiting = fill_buys(cash, base_eq, dict(O.items()), [b[2] for b in buys], costs,
                                              n=pc["positions"])
                ids = {b[2]: (b[0], b[3]) for b in buys}
                for sym, sh, px, ch in bf:
                    oid = ids[sym][0]
                    if sh == 0:
                        con.execute("UPDATE orders SET status='cancelled', note='slot too small' WHERE id=?", (oid,))
                        continue
                    con.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?)",
                                (name, sym, sh, sh * px + ch, ds, ds))
                    con.execute("UPDATE orders SET status='filled', fill_date=?, price=?, shares=?, charges=? "
                                "WHERE id=?", (ds, px, sh, ch, oid))
                    fills.append(f"BOUGHT {sym} {sh} @ {px:,.2f} = Rs {sh * px:,.0f}")
                for sym in waiting:
                    oid, dec = ids[sym]
                    if (d - pd.Timestamp(dec)).days > 7:
                        con.execute("UPDATE orders SET status='cancelled', note='no open for a week' WHERE id=?",
                                    (oid,))
                        alerts.append(f"[{name}] buy {sym} cancelled: no trading for a week")
            # 3) index-fund part: fee, January reset to the target split
            if last is not None:
                core_units *= decay
            pos = con.execute("SELECT symbol, shares FROM positions WHERE book=?", (name,)).fetchall()
            held_val = sum(sh * float(Cl.get(s, np.nan)) for s, sh in pos if Cl.get(s, np.nan) == Cl.get(s, np.nan))
            if last is not None and d.year != pd.Timestamp(last).year:
                move = rebalance_amount(cash + held_val, cash, core_units * n5, share)
                if abs(move) >= 500:
                    ch = sell_charges(abs(move), costs) if move > 0 else buy_charges(abs(move), costs)
                    cash += move - (ch if move > 0 else 0)
                    core_units -= (move + (ch if move < 0 else 0)) / n5
                    fills.append(f"January reset: Rs {move:+,.0f} moved {'from' if move > 0 else 'to'} the index fund")
            strat, core = cash + held_val, core_units * n5
            con.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?,?)",
                        (name, ds, strat, core, strat + core, n5, int(bool(m["regime"].iloc[-1]))))
            con.execute("UPDATE books SET cash=?, core_units=? WHERE name=?", (cash, core_units, name))
            if fills:
                msgs.append(f"[{name}] filled at today's open ({ds}):\n  " + "\n  ".join(fills))

        # 4) weekly decision
        decided = {r[0] for r in con.execute("SELECT week FROM decisions WHERE book=?", (name,))}
        if ds >= started and is_decision_day(d.date(), today, decided):
            s = m["combo"][uni].iloc[-1].dropna().sort_values(ascending=False, kind="stable")
            ranked = list(s.index)
            held = {r[0] for r in con.execute("SELECT symbol FROM positions WHERE book=?", (name,))}
            held |= {r[0] for r in con.execute("SELECT symbol FROM orders WHERE book=? AND side='buy' "
                                               "AND status='pending'", (name,))}
            eq_row = con.execute("SELECT strategy FROM equity WHERE book=? AND date=?", (name, ds)).fetchone()
            equity = eq_row[0] if eq_row else cash
            reg = bool(m["regime"].iloc[-1])
            sells, buys = decide(held, ranked, dict(Cl.items()), equity, reg, pc["positions"], pc["exit_rank"])
            rk = {sy: i + 1 for i, sy in enumerate(ranked)}
            for side, lst in (("sell", sells), ("buy", buys)):
                for sy in lst:
                    con.execute("INSERT INTO orders (book, decided, side, symbol, rank, status) VALUES (?,?,?,?,?,?)",
                                (name, ds, side, sy, rk.get(sy), "pending"))
            con.execute("INSERT INTO decisions VALUES (?,?,?,?)",
                        (name, "%d-W%02d" % d.date().isocalendar()[:2], ds, int(reg)))
            msgs.append(weekly_report(con, name, ds, reg, sells, buys, ranked, rk, Cl, n5, pc))
        con.commit()

    text = "\n\n".join(alerts + msgs)
    if text:
        text = f"PAPER TRADING (no real money) - data to {ds}\n\n" + text
        print(text)
        if notify:
            telegram(text)
    else:
        print(f"{ds}: nothing to report")


def weekly_report(con, name, ds, reg, sells, buys, ranked, rk, Cl, n5, pc):
    b = con.execute("SELECT started, capital FROM books WHERE name=?", (name,)).fetchone()
    first = con.execute("SELECT nifty500 FROM equity WHERE book=? ORDER BY date LIMIT 1", (name,)).fetchone()
    row = con.execute("SELECT strategy, core, total FROM equity WHERE book=? AND date=?", (name, ds)).fetchone()
    lines = [f"== Book '{name}' ({pc['books'][name]}) - week ending {ds} =="]
    if row:
        idx = n5 / first[0] - 1 if first and first[0] else float("nan")
        lines.append(f"Total Rs {row[2]:,.0f} ({row[2] / b[1] - 1:+.1%} since {b[0]}); "
                     f"Nifty 500 {idx:+.1%}. Stocks Rs {row[0]:,.0f}, index fund Rs {row[1]:,.0f}")
    lines.append("Market filter: " + ("ON - Nifty 500 above its 200-day average" if reg else
                                      "OFF - Nifty 500 below its 200-day average: no buys or sells this week"))
    pos = con.execute("SELECT symbol, shares, basis, buy_date FROM positions WHERE book=?", (name,)).fetchall()
    for s, sh, basis, bd in pos:
        v = sh * float(Cl.get(s, np.nan))
        lines.append(f"  holding {s}: {sh} sh, Rs {v:,.0f} ({v / basis - 1:+.1%}), rank {rk.get(s, '-')}, since {bd}")
    if sells:
        lines.append("SELL at Monday's open: " + ", ".join(f"{s} (rank {rk.get(s, 'out')})" for s in sells))
    if buys:
        lines.append("BUY at Monday's open: " + ", ".join(f"{s} (rank {rk[s]}, ~Rs {Cl[s]:,.1f})" for s in buys))
    if not sells and not buys:
        lines.append("No changes this week.")
    if name == next(iter(pc["books"])):
        lines.append("Top 20 now: " + ", ".join(ranked[:20]))
    return "\n".join(lines)


def status(cfg):
    con = connect(cfg)
    for name, started, cap, cash in con.execute("SELECT name, started, capital, cash FROM books"):
        row = con.execute("SELECT date, strategy, core, total, nifty500 FROM equity WHERE book=? "
                          "ORDER BY date DESC LIMIT 1", (name,)).fetchone()
        print(f"\n{name}: started {started}, Rs {cap:,.0f}; cash Rs {cash:,.0f}")
        if row:
            print(f"  {row[0]}: total Rs {row[3]:,.0f} ({row[3] / cap - 1:+.1%}), stocks Rs {row[1]:,.0f}, "
                  f"index fund Rs {row[2]:,.0f}")
        for r in con.execute("SELECT symbol, shares, basis, buy_date FROM positions WHERE book=?", (name,)):
            print(f"  holding {r[0]} {r[1]} sh, cost Rs {r[2]:,.0f}, since {r[3]}")
        for r in con.execute("SELECT side, symbol, decided FROM orders WHERE book=? AND status='pending'", (name,)):
            print(f"  pending {r[0]} {r[1]} (decided {r[2]})")
        t = con.execute("SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM trades WHERE book=?", (name,)).fetchone()
        print(f"  closed trades: {t[0]}, realised P&L Rs {t[1]:,.0f}")


def main(cfg, args):
    if args.job == "chatid":
        return show_chat_ids()
    if args.job == "init":
        return init(cfg, force=args.force)
    if args.job == "status":
        return status(cfg)
    try:
        run(cfg, notify=not args.quiet)
    except Exception as e:
        telegram(f"PAPER job failed {datetime.now():%Y-%m-%d %H:%M}: {type(e).__name__}: {str(e)[:300]}")
        raise
