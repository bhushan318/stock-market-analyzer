# CLAUDE.md — Systematic swing-trading research (Indian equities)

Read `PROGRESS.md` first: it holds every decision, result and the current next step.

## What this project is

A personal, data-backed **swing-trading system for NSE stocks and ETFs**, built and tested
step by step before any real money is used. The owner trades via Zerodha. Branch `swing-engine`.

**Two machines (decided 2026-11-25):**
- **Raspberry Pi** (64-bit Raspberry Pi OS, arm64, repo `~/stock-market-analyzer`): always on.
  Owns the master `data/market.db` (NSE collectors via systemd timers) and later runs the live
  weekly scoring / paper trading / orders. No heavy studies here (slow, ~3 GB free disk).
- **Windows PC**: research - backtests, studies, writing code, tests. Gets a read-only copy of the
  DB: on the Pi `python -m trader collect export` -> `data/market_snapshot.db.gz`; copy it with scp
  and unzip to `data/market.db` (command is printed by export). Data flows Pi -> PC only; code
  flows both ways via GitHub (Pi pushes with deploy key alias `github-sma`).

The legacy files `app.py`, `StockPrediction.py`, `nifty_50_analyzer.py` are kept for reference
only — do not build on them.

## The owner's goals and constraints (never violate)

1. **No real money until the system passes the gates** in PROGRESS.md (backtest → paper → small
   live ₹20k → scale in steps). Never place real orders or connect a broker without an explicit go-ahead.
2. **At most 5 open positions.** Delivery (CNC) only — no F&O, no margin/MTF, no leverage.
3. **Zero or near-zero running cost.** Free data only (NSE public files, Yahoo Finance).
   LLM = Google Gemini API **free tier** (key from AI Studio, in `.env`, never committed).
   Zerodha Kite Connect *Personal* (free, orders only) later; the paid ₹500/month data plan is
   not needed. Ask before adding anything that needs a sign-up or costs money.
4. The owner prefers the rule **"never sell below the buy price"**. Recommended alternative under
   test: target + thesis-break + volatility stop. Keep both variants and let data decide
   (track the "stuck rate").
5. Capital: start ₹20k, may scale to ₹1 lakh+. Percent returns are size-independent; the flat
   DP charge makes larger positions slightly cheaper.
6. Secrets (`.env`, API keys, broker tokens) are never logged, committed, or sent to an LLM.
   Only public market data/news may be sent to the LLM — never holdings or account details.

## Research discipline (the reason this project exists)

- Every strategy result must be compared with **random picks from the same universe** and with
  **buy-and-hold of the index** (NIFTYBEES / a Nifty 500 ETF). A rule that doesn't beat random has no edge.
- Use **rolling windows** (new start every quarter, 1y and 3y) — single-path results on a small
  account are very noisy (±10%/yr differences are often luck).
- Learn on an early period, check on a later one. The stock-model holdout is 2024-01-01 onward
  and has already been looked at several times — treat it as partly used.
- Survivorship bias: universes are *today's* constituents. Always say so when reporting stock results.
- All returns are **after charges** (STT, stamp duty, exchange/SEBI fees + GST, ₹15.93 DP charge per sell,
  slippage). Report the 20% STCG tax separately.
- Yahoo data has bad prints (mis-scaled rows, zero/phantom opens); `trader/data.py:clean_bad_ticks`
  repairs them. If a result looks too good or a holding shows −99%, suspect data first.
- LLM output never supplies numbers; it only classifies/explains data we give it.

## Layout and commands

```
trader/            engine package (python -m trader <command>)
  data.py          Yahoo download + cache + bad-tick cleaning
  features.py      scale-free features, hit-target label (no look-ahead; tested)
  model.py         walk-forward pooled model with purge gap
  costs.py         Zerodha delivery charges, net take-profit price
  backtest.py      day-by-day portfolio simulator (all rule options documented at top)
  experiments.py   stock rule variants              -> python -m trader ideas [--holdout]
  etf_study.py     ETF targets/windows/seasonality  -> python -m trader etf [--broad]
  mixed_study.py   ETF + stock slot splits          -> python -m trader mixed
  universe_study.py Nifty 50 vs 500 + random ctrl   -> python -m trader universe
  momentum_study.py Gate 0.1 momentum/RS signals    -> python -m trader momentum   (biased: today's N500)
  pit.py           point-in-time panel from market.db: renames, split/bonus adjust, liquid top-500
  momentum_pit.py  Gate 0.1 redone point-in-time    -> python -m trader momentum-pit
  levers.py        Gate 0.4 lever betas             -> python -m trader levers
  nse.py, collect.py, store.py  NSE collectors      -> python -m trader collect daily|news|backfill|status|export
config.yaml        every strategy rule, costs, collector settings
universes/         saved constituent lists (nifty500_2026-11.csv)
deploy/            systemd user timers + install.sh for the Pi
tests/             pytest suite — must stay green
outputs/, data/    generated, gitignored (data/market.db = collected NSE data)
```

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements-trader.txt`.
Tests: `.venv/bin/python -m pytest -q tests` (run before every commit).

## Working conventions

- Small modules, pure parsing functions, tests for anything that touches money logic or parsing.
- Every network call: timeout, retries with backoff, logged to `collect_log`.
- Be polite to NSE: ~1 request/second, archive files (nsearchives.nseindia.com) preferred over APIs.
- Heavy studies run on the PC. On the Pi only collectors/live jobs; long Pi jobs via `nohup`/`tmux`.
- Commit to `swing-engine` with clear messages; push when a step is done; update PROGRESS.md
  (results + next step) at the end of every work session.
- Explain results to the owner in plain language and rupee terms, with honest caveats.
