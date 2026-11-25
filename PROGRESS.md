# PROGRESS.md — decisions, results, next steps

Last updated: 2026-11-25 (session moved from the Windows PC to the Raspberry Pi after this).

## Where we are

**Pi set up (2026-11-25), backfill running, timers live. Gate 0 step 1 (momentum) run once:
result is dominated by survivorship bias -> needs a point-in-time universe before it means
anything (see Results 6). Sector lever maps: Claude drafts them (decision below).**

## The plan (agreed with the owner)

Systematic, data-backed **swing trading** (holding days to weeks) on Nifty 500 stocks (+ ETFs),
max 5 positions, real money only after passing gates:

| Gate | What | Money | Pass criteria |
|---|---|---|---|
| 0 Backtest | Everything testable on history: momentum/relative strength, delivery %, results reactions, lever sensitivities | ₹0 | Beats random picks **and** Nifty 500 buy-and-hold on rolling windows |
| 1 Paper (3 months) | Full system live: 5 positions + shadow list of top 20-30 scored stocks | ₹0 | Beats Nifty 500 ETF after charges+tax; max drawdown < 10% |
| 2 Small live (2-3 months) | Real Zerodha orders | ₹20k | Live ≈ paper within ~1%; no operational errors |
| 3 Scale | 25% → 50% → 100% of target capital, ~1 month each | partial | Risk limits held |

Hard risk rules at every stage: ≤5 stocks, ≤25% of capital per stock, ≤2 stocks per sector,
CNC only, account kill-switch at −10% from peak (no new buys until reviewed).

Selection = fundamentals quality filter + sector "lever" outlook + momentum/trend + catalysts
(results, announcements, deals, delivery %) — not price patterns alone.

Phases: (1) universe + data collectors ✅ → (2) sector lever maps (~20 sectors; LLM drafts, owner
reviews) + measured lever sensitivities → (3) news archive tagged by LLM → (4) weekly scoring →
(5) paper portfolio + shadow list.

## Open decisions

- **Exit rule:** owner leans "never sell at a loss" and believes good selection keeps stuck cases
  rare. Agreed to test it: paper-track both (a) never-sell-at-loss and (b) target + thesis-break +
  volatility-sized stop; compare returns and stuck rate (held >60 days underwater).
- **LLM:** owner has Google AI Pro (app subscription, no API). Plan: free Gemini API key from
  Google AI Studio, stored in `.env` (`GEMINI_API_KEY=...`), model name in config. Not created yet.
  Only public news/filings go to it.
- **Sector lever maps (decided 2026-11-25):** one-time job, drafted by Claude in a work session
  (no API cost), not Gemini. Gemini is only for the recurring news tagging. Method: NSE's official
  4-level industry classification (fetched from NSE, not LLM-labelled) -> Claude drafts levers per
  sub-industry (direction, mechanism, lag, confidence, data series; no magnitudes) -> checked
  against measured betas (Gate 0 step 4), disagreements flagged -> owner reviews. Refresh at each
  March/September index rebalance or after a major policy change.
- Universe: use official Nifty 500 list (+ liquidity filter), *not* an LLM-generated list
  (LLM may only suggest extras that are verified against NSE's equity list).

## Results so far (all after charges, before 20% STCG tax)

### 1. Legacy code review
`StockPrediction.py` results were unreliable: latest signal 1-5 days stale (dropna removed the
newest rows), misaligned accuracy/F1, best model chosen on the test set, "expected 5-day return"
= annual/50, no costs, 1-month test window. Replaced by `trader/`.

### 2. Stock model: "hit +5.75% within 20 days" (Nifty 50, walk-forward)
- AUC ≈ 0.60 (dev 2019-23 and holdout 2024-26); logreg ≥ gradient boosting on dev → `logreg`.
- Top-5 hit rate 57% vs 41% base in holdout — modest edge, but the portfolio built on it
  **did worse than random picks in 2024-26** (it favours volatile stocks that crash and get stuck).

### 3. "Never sell at a loss" + 5% target, Nifty 50 stocks, ₹20k
- Holdout 2024-26: ~3-11%/yr (path-sensitive), ~50% of capital stuck underwater; worst holding −74%
  (TMPV), JIOFIN/ONGC/ADANIENT stuck 1.5+ years.
- 10% stop-loss made it worse (holdout −0.5%/yr): 5% wins vs 10% losses + volatile picks + charges.
- Market filter, uptrend filter, target decay, averaging down: none fixed the structural stuck-capital problem.

### 4. ETF dip strategy (owner found it convincing, but feels it is "investing", not trading)
Rule: rank ETFs by how far the close is below its 20-day average, buy the deepest dips into free
slots, sell at +5% net, never at a loss. Data 2009-2026 (Yahoo ETF history starts 2009).
- Targets 3-6% all similar; 4-5% is fine.
- **Broad ETFs only** (NIFTYBEES, JUNIORBEES, BANKBEES, GOLDBEES, MON100, ITBEES, MAFANG):
  ~16%/yr over 18y vs NIFTYBEES 13.8%; rolling 3-year windows: **0% losing**, worst +7.1%/yr,
  beats NIFTYBEES 78%; rolling 1-year: 6% losing, worst −5%.
- Sector ETFs are dangerous under no-loss: PSUBNKBEES −78% (2010-20), CPSE −56%, SILVER −61%.
- Averaging down (30% reserve, add at −10%, max 2): ~2-3%/yr lower return, smoother.
- ₹20k → ~₹3k/yr typical (~₹250/month) before tax; ₹1 lakh → ~₹16.7k/yr. Same % at any size;
  6 slots max (8 slots > 7 ETFs leaves cash idle).
- With 7 ETFs and 5-6 slots it re-buys the same ETF often (75% within 30 days), usually 2-4%
  below the last sell price; 27-40% of re-buys are above the last sell (sold too early).
- Untested ideas: only buy if below the 20-day average; never re-buy above last sell price;
  expand to 12-15 broad ETFs (midcap 150, smallcap 250, momentum 30, value 20, global).
- Seasonality: month-of-year pattern did not repeat (rank corr +0.11); last-3-trading-days-of-month
  entries slightly better in both halves (+3-5% hit rate) — small, maybe an entry-timing rule later.
- NSE lists 367 ETFs / 114 underlying indices; most are duplicates, debt/liquid, illiquid or sector.

### 5. Stocks + ETFs, max 5 positions (2016-2026, rule-based stock dips in uptrend)
| Setup | 10y CAGR | 1y losing | worst 1y | worst 3y |
|---|---|---|---|---|
| ETF only | 19.3% | 2.5% | −8.8% | +7.0% |
| 4 ETF + 1 stock (N50) | 21.8% | 10% | −17.9% | +3.9% |
| 3 ETF + 2 stock (N50) | 23.5% | 17.5% | −24.6% | +1.5% |
| stocks only (N50 / N500) | 14.3% / 12.0% | 27.5% / 37.5% | | +0.4% / −6.5% |

**Key finding: the price-only dip rule did not beat random stock picks (N50 or N500) in any setup.**
Nifty 500 was worse than Nifty 50 for this rule. → motivates the fundamentals + catalysts approach.

### 6. Gate 0 step 1: momentum on today's Nifty 500 (2026-11-25) - NOT TRUSTWORTHY YET
`python -m trader momentum` (trader/momentum_study.py). Signals: 12-1 and 6-1 month return,
distance to 52w high, 6m return vs sector median, combo. (Return vs Nifty = same ranking as raw
return, so not separate.) Weekly ranking, 499 stocks, ~400 eligible (>= Rs 2 cr turnover).
- Signal test (top decile minus average eligible, 20d forward, gross): 12-1 momentum +1.25%
  (learn) / +0.91% (check 2024+), beats in 64-67% of weeks; 6-1 and sector RS faded in 2024+
  (+0.3%); 52w-high ~0.
- Portfolio (5 stocks, Rs 20k, sell when out of top 20, all charges): 12-1 momentum 44%/yr
  2016-26 vs random 18%, Nifty 500 12.5%; 3y rolling windows beat random 91%.
- **But** the whole gain comes from 10 of 128 trades (ADANIGREEN +828%, TTML, DIXON, CGPOWER,
  APARINDS, TITAGARH, MAZDOCK ...); the other 118 trades net to a loss. These are exactly the
  stocks that are in today's Nifty 500 *because* they multiplied -> survivorship bias. No data
  glitches found. The real point-in-time Nifty 200 Momentum 30 ETF made +8.8%/yr since Aug 2022
  vs Nifty 500 +11.4%, while our version showed +40%.
- "Never sell at a loss" variant: lower return (28% vs 45% on the same biased data), stuck 27%.
- 52w-high variant churns (5-day holds, 150 trades/yr) because many stocks tie at the high; ignore.
- Note: `charges_pct_yr` is charges / *starting* capital per year, so it balloons as equity grows.
- Fix needed: point-in-time universe (NSE full bhavcopy incl. later-collapsed stocks, liquidity
  ranked each month, split/bonus adjusted) and/or the official Nifty 200 Momentum 30 index history
  (niftyindices.com, point-in-time by construction) as the honest benchmark.

## Phase 1: data collectors (done, tested on live NSE data 2026-11-25)

`python -m trader collect daily|news|backfill|status` → `data/market.db` (SQLite):
- `eq_daily`: full bhavcopy incl. **delivery %** (Nifty 500 + all NSE ETFs, ~860 symbols/day).
  Archive available back to at least 2020 → backfill with `collect backfill --start 2020-01-01` (~1 h).
- `deals`: bulk + block deals (current file only → collect daily from now).
- `fii_dii`: FII/DII cash flows (API gives only today → collect daily from now).
- `announcements`: NSE announcements + results RSS hourly, company → symbol match ~78%
  (unmatched are mostly SME/debt issuers).
- Holidays detected (NSE republishes the previous file); catch-up of last 10 weekdays each run.
- systemd user timers: daily Mon-Fri 19:30 IST + Sat 10:00 catch-up; news hourly 07:00-23:00 IST.

## Pi setup checklist (do these first on the Pi)

```bash
cd ~ && git clone -b swing-engine https://github.com/bhushan318/stock-market-analyzer.git   # or git pull
cd ~/stock-market-analyzer
python3 -m venv .venv && .venv/bin/pip install -r requirements-trader.txt
.venv/bin/python -m pytest -q tests                      # expect 12 passed
.venv/bin/python -m trader collect daily && .venv/bin/python -m trader collect news
nohup .venv/bin/python -m trader collect backfill --start 2020-01-01 > backfill.log 2>&1 &
bash deploy/install.sh                                   # timers; needs sudo for enable-linger
.venv/bin/python -m trader collect status
```
- [x] Pi set up, tests pass (16 passed, Python 3.13, 2026-11-25)
- [ ] Backfill finished. First run (Nifty 500 + ETFs only) died at 2022-01-20 when the Pi rebooted.
  Restarted 22:45 IST 2026-11-25 with `store_all: true` (every EQ/BE stock, ~1,640/day in 2020,
  est. ~500-700 MB) so a point-in-time universe is possible; days stored with <1,000 symbols are
  re-fetched automatically. ~1,774 weekdays at ~10/min => ~3 h. If it dies, just rerun the same
  command (it resumes): `nohup .venv/bin/python -u -m trader collect backfill --start 2020-01-01 > backfill.log 2>&1 &`
- [x] Timers installed and confirmed firing (news fired 22:05 IST). Linger was already on.
- [x] Git push from the Pi: deploy key `~/.ssh/github_sma` (write access, this repo only), remote `git@github-sma:bhushan318/stock-market-analyzer.git` (host alias in `~/.ssh/config`).
- Disk: only ~1.8 GB free on the Pi SD card (14 GB, 87% used) - matters for an all-stocks bhavcopy store.
- [ ] Gemini API key created and put in `.env` (owner)

## NEXT: Gate 0 backtests (no LLM needed)

Work on the Pi (studies take minutes; use nohup/tmux). For each signal: rank Nifty 500 stocks
weekly, form a 5-stock portfolio + measure forward 5/10/20-day returns of the top decile; compare
with random picks from the same eligible set and with Nifty 500 buy-and-hold; rolling 1y/3y windows;
learn on 2020-2023, check on 2024-2026. Note survivorship bias in every report.

1. **Momentum / relative strength** — first run done, biased (Results 6); redo on a point-in-time universe.
   Original spec: (prices from Yahoo, 2015+): 6- and 12-month return skipping the
   last month, distance to 52-week high, return vs sector and vs Nifty. Compare with the
   Nifty 200 Momentum 30 ETF (if the ETF matches, prefer the ETF).
2. **Delivery %** (from `eq_daily`, 2020+): does rising delivery % on up-days / high delivery vs the
   stock's own 20-day average predict the next 5-20 days?
3. **Results reactions**: around result dates (from announcements going forward; for history use
   big-volume days / NSE results calendar if obtainable) — drift after strong vs weak reactions.
4. **Lever sensitivities**: regress each stock's weekly returns on crude (BZ=F), USD/INR (INR=X),
   gold, India 10y yield proxy, Nifty, sector index; store betas per stock for the lever maps.
5. Exit-rule comparison on whatever signal survives: never-sell-at-loss vs target + vol stop
   (track stuck rate).

Then: Phase 2 lever maps (Gemini drafts per sector, owner reviews) → Phase 3 news tagging →
weekly scoring → Gate 1 paper trading with Telegram alerts (owner OK'd Telegram in principle;
bot not created yet).

## Known issues / notes

- Yahoo ETF data 2009 has zero opens; NSE ETF early data rough — cleaner handles it.
- `data/cache` probs cache key = model + label + last date + panel length; delete `data/cache`
  after changing features or cleaning logic.
- Universe file `universes/nifty500_2026-11.csv` = constituents as of Nov 2026 (refresh every
  March/September when NSE rebalances; keep old files for history).
- The repo is also edited on Windows: `.gitattributes` forces LF for `*.sh`, `*.service`, `*.timer`.
