# PROGRESS.md — decisions, results, next steps

Last updated: 2026-11-26 (Results 8: combo variants).

## Where we are

**2026-11-26: Pi rebuilt, backfill complete (1,714 days 2020-26, every NSE stock), DB copied to
the PC. Gate 0 run on point-in-time data (Results 7): only the momentum "combo" ranking beats
random picks and Nifty 500 in every 3-year window, but with a −42% drawdown, a smaller edge in
2024+, and heavy charges at Rs 20k. Levers and delivery % alone: no stock-picking edge.
Results 8 (18 combo variants): the market filter "no new buys while Nifty 500 < 200-day avg"
is the one change that helped consistently; slower rotation, Rs 1 lakh, stops and targets gave
mixed, noisy results; never-sell-at-loss again weak. Every variant still falls 32-56% at worst.
Next: robustness checks of combo + filter, then decide on Gate 1 paper trading (see NEXT).**

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

### 7. Gate 0 on point-in-time data (2026-11-26, PC, NSE bhavcopy 2020-01 -> 2026-11-26)
Universe each week = top 500 stocks by 6-month median turnover at that date (incl. stocks that
later collapsed/delisted: 366 stopped trading). 5 stocks, Rs 20k, weekly ranking, sell when out of
the top 20, all charges, no dividends, before 20% STCG. Learn 2021-23 / check 2024+ (the 2024+
period was already seen in Results 6, so it is only partly fresh).

**Data bug found and fixed first:** NSE's PREV_CLOSE is *not* split-adjusted, so the first run
detected only 14 splits in 7 years and every split looked like a crash. Now opening-gap detection
(752 events; MAZDOCK, IRCTC, RELIANCE, CDSL, TMPV demerger checked). Rights entitlements (-RE)
dropped. Remaining liquid moves >25%/day: 16 in 5M stock-days, all real (Adani, Zee, IndusInd...).

Momentum (`momentum-pit`), CAGR full 2021-26 / check 2024+:
| variant | rule | random | Nifty 500 | 3y windows beat random / Nifty 500 | max DD | trades/yr | charges/yr* |
|---|---|---|---|---|---|---|---|
| mom12_1 | 15.9% / 24.1% | 11.3% / 0.4% | 13.3% / 7.7% | 33% / 33% | −50% | 14 | 1.8% |
| mom6_1 | 6.2% / 13.2% | 11.6% / 1.9% | | 50% / 50% | −62% | 23 | 2.7% |
| **combo** (12-1, 6-1, 52w-high ranks) | **31.5% / 12.7%** | 9.0% / −4.5% | | **100% / 100%** (median +24% / +20%/yr) | −42% | 59 | 11.5% |
| mom12_1 never-sell-at-loss | 11.5% / 11.3% | 10.1% / 4.0% | | 50% / 33% | −45% | 4 | 0.6% |
\* charges as % of the *starting* Rs 20k per year.
- Survivorship bias measured: random picks from *today's* Nifty 500 make 26%/yr vs 11% from the
  point-in-time universe; mom12_1 52% vs 16%. Results 6 was mostly this bias.
- combo is the only candidate: beats random and Nifty 500 in every 3-year window, both halves.
  Caveats: 2024+ edge much smaller (+5%/yr vs Nifty 500); −42% drawdown (Nifty 500 −19%);
  profit concentrated in ~10 big winners (CUPID, COCHINSHIP, MTARTECH, IRFC, MIRZAINT); 59
  trades/yr cost 11.5% of Rs 20k per year (DP Rs 15.93 per sell on ~Rs 4k positions).
- Nifty 200 Momentum 30 ETF since 2022-08: +9.0%/yr vs Nifty 500 +11.3% (the real index lagged).
- Never-sell-at-loss again ~= random, 30-43% of holdings stuck.

Delivery % (`delivery`, same harness): delivery on its own has no edge (deliv_rel +7%,
accum −7%, dq_rel −17% full period; random −2%... all crushed by 145-160 trades/yr = 17-22%
charges). **accum_mom** (delivery accumulation + 12-1 momentum): 21.0% vs random 3.7%, check
12.6% vs −10.5%, beats random in 92% of 3y windows but Nifty 500 only 33%; 114 trades/yr.

Levers (`levers`, Yahoo 2016-26, 361 stocks): per-stock betas to crude/USDINR/gold/US10y keep
their sign between halves only 50-58% of the time (coin flip) -> useless for stock picking.
Industry level, stable: pharma/healthcare +0.41% per +1% USDINR, FMCG +0.23, IT +0.20;
consumer durables and realty −0.23; metals +0.10 crude / +0.12 gold; US yields ~0. Small effects
-> lever maps are context for sector tilts, not a signal.

**Key practical finding: at Rs 20k, charges dominate any weekly-rotation strategy** (DP charge is
0.4% of a Rs 4k sell). Test slower rotation (monthly, exit when out of top 50) and Rs 1 lakh.

### 8. combo variants: turnover, capital, market filter, exits (2026-11-26, `python -m trader combo`)
Point-in-time top 500, 5 stocks, 2021-01 -> 2026-11, after charges, no dividends, before STCG.
Random control = random picks under the *same* rules (rebalance days, filter, stops). Nifty 500:
13.3%/yr full, 19.1% learn, 7.7% check, max DD −19%; Rs 20k -> ~Rs 42k.
W/M = weekly/monthly ranking, 20/50 = sell when out of the top 20/50.

| variant | Rs 20k -> | full | random | check 2024+ | max DD | trades/yr | charges/yr | 3y beat random / N500 | worst 3y |
|---|---|---|---|---|---|---|---|---|---|
| W20 (Results 7) | 1.01 L | 32% | 9% | 13% | −42% | 59 | 12% | 100% / 100% | +17% |
| W50 | 1.35 L | 38% | 11% | 1% | −44% | 27 | 6% | 75% / 92% | +7% |
| M20 | 0.77 L | 26% | 8% | −6% | −51% | 29 | 5% | 50% / 58% | +4% |
| M50 | 0.90 L | 29% | 19% | 8% | −54% | 16 | 3% | 75% / 83% | +10% |
| **W50 + filter (block buys)** | **1.74 L** | **44%** | 19% | **11%** | **−36%** | 25 | 6% | **100% / 100%** | +15% |
| M50 + filter (block buys) | 1.70 L | 44% | 12% | 30% | −49% | 14 | 3% | 92% / 100% | +15% |
| W50 + filter (sell all) | 0.66 L | 23% | 10% | −3% | −43% | 30 | 6% | 92% / 92% | +9% |
| W50 never-sell-at-loss | 0.53 L | 18% | 6% | 2% | −30% | 5 | 1% | 75% / 67% | +1% |
| W50 + vol stop | 1.57 L | 42% | 14% | 1% | −43% | 27 | 6% | 83% / 92% | +5% |
| W50 +15% target + stop | 0.32 L | 8% | 15% | −1% | −50% | 24 | 3% | 50% / 33% | −5% |
| M50 +15% target + stop | 0.84 L | 28% | 17% | 12% | −32% | 21 | 4% | 92% / 100% | +19% |
Rs 1 lakh (same rules): W20 34% (charges 9%/yr), W50 39%, M20 25%, M50 30% - a little better
than Rs 20k, within path noise (W20 check 2024+: 13% at 20k vs 8% at 1 lakh just from rounding
of share counts -> treat ±5%/yr differences between variants as noise).

Reading (honest):
- 18 variants on one dataset: the best one is flattered by selection. Only conclusions that hold
  across several variants are trusted:
  - **Market filter "block new buys"** helps in both W50 and M50 (higher return, check 2024+
    better, 1y losing windows 30% -> 20% for W50). "Sell everything" on the filter hurts (whipsaw).
  - Slower rotation cuts charges (12% -> 3-6%/yr of Rs 20k) but does not reliably raise returns.
  - Never-sell-at-loss: low returns, 34-40% of capital stuck. Rejected again by data.
  - Stops/targets: inconsistent between weekly and monthly -> no conclusion.
- Profit still depends on a few big winners (top 10 trades > 100% of total profit in most
  variants; the rest roughly break even after charges).
- Drawdowns of 32-56% in every variant vs 19% for the index. The owner says fear/greed is their
  weak spot -> sizing / ETF core must be decided before Gate 1.

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

## Rebuild the Pi (after a reset) — 2026-11-26

The Pi was reset on 2026-11-26: its `data/market.db` (backfill in progress + ~1 day of FII/DII,
deals, news) was lost. Code/config/notes are all in GitHub. Prices + delivery are re-downloadable;
FII/DII, deals and news are not (only from the day they are collected), so get the timers back fast.

1. **OS:** Raspberry Pi OS **64-bit** (Lite is fine), SSH enabled, same username as before if possible.
2. **Deploy key** (lets the Pi push to this repo only). On the Pi:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/github_sma -N "" -C "pi-stock-market-analyzer"
   cat >> ~/.ssh/config <<'EOF'
   Host github-sma
       HostName github.com
       User git
       IdentityFile ~/.ssh/github_sma
       IdentitiesOnly yes
   EOF
   chmod 600 ~/.ssh/config
   cat ~/.ssh/github_sma.pub
   ```
   On GitHub: repo → Settings → Deploy keys → **delete the old Pi key** (its private half is gone
   with the old SD card/stick) → Add deploy key → paste, tick **Allow write access**.
   Test: `ssh -T github-sma` → "successfully authenticated".
3. **Everything else, one command** (idempotent; clones if needed, venv, tests, first collection, timers):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/bhushan318/stock-market-analyzer/swing-engine/deploy/setup_pi.sh -o setup_pi.sh
   less setup_pi.sh        # read it before running
   bash setup_pi.sh
   ```
4. **Data on the external disk + PC merge + backfill, one command.** The Pi now has a 1.4 TB
   external HDD (exFAT, label "Bhushan", mounted at `/mnt/ehd`, shared with other files). SQLite
   is safest on ext4, so `deploy/move_data_to_ehd.sh` makes a 10 GB ext4 image
   `/mnt/ehd/pi-data/sma.img`, loop-mounts it at `/mnt/sma` (fstab, nofail), moves `data/` to
   `/mnt/sma/sma-data` and leaves `data` as a symlink (disk missing → collectors fail loudly,
   never fill the flash). If `~/pc_market.db` exists it merges its news/deals/FII-DII
   (INSERT OR IGNORE — never scp the PC DB over the Pi's: that would wipe Pi data).
   Then it starts the backfill in tmux (log: `data/backfill.log`).
   ```bash
   # on the PC (optional, older news 2026-11-03..25): scp data/market.db bhushan@<pi-ip>:~/pc_market.db
   cd ~/stock-market-analyzer && git pull && bash deploy/move_data_to_ehd.sh
   ```
5. **Backfill** (~3 h, resumes if interrupted; weekend special sessions afterwards) — started by step 4.
6. `.env` with `GEMINI_API_KEY` when created (`chmod 600`).
7. Claude Code on the Pi: reinstall per Anthropic's docs, log in, start it in `~/stock-market-analyzer`.

Checklist after the reset:
- [ ] 64-bit OS, SSH
- [ ] deploy key + `github-sma` alias, old key removed on GitHub
- [ ] `setup_pi.sh` ran: tests pass (22 on the PC), timers listed
- [x] `setup_pi.sh` ran on the Pi 2026-11-26: 22 tests pass, 8 days x ~2,950 symbols, timers on
- [x] PC news/deals/FII-DII merged into the Pi DB (announcements now from 2026-11-03)
- [x] backfill done 2026-11-26 22:29 IST: 1,714 trading days 2020-01-01 -> 2026-11-26,
      3,751 symbols, 533 MB, no partial days. DB still on the flash disk (4.3 GB free).
- [x] hourly news job now also fetches deals + FII/DII (both are "latest day only" and appear
      after the 19:30 run; 2026-11-26's deals were missed because of this)
- [ ] external disk: first attempt hung — Seagate Expansion (0bc2:2036) under the UAS driver never
      completed a cache flush. Fixed with `usb-storage.quirks=0bc2:2036:u` in
      /boot/firmware/cmdline.txt (backup `cmdline.txt.bak-uas`). `fsck.exfat -n` then found real
      exFAT damage (cross-linked clusters, a corrupted folder) - likely from the old flush failures.
      Owner is repairing it on Windows (chkdsk /f). After that: delete `pi-data/sma.img.tmp`,
      run `deploy/move_data_to_ehd.sh` (needs sudo password) - it moves data/ into the ext4 image.
      Only ever touch `pi-data/` on that disk; it holds the owner's personal files.
- [ ] reboot test: `/mnt/sma` mounts by itself
- [ ] backfill + `--weekends` finished; `collect status` shows ~1,700+ days
- [ ] Claude Code reinstalled
- [ ] Gemini key in `.env` (owner, still not created)

Notes kept from the first setup: Python 3.13 on the Pi worked; linger must be on (install.sh does it);
system disk is a USB flash stick with little free space — keep no extra index on eq_daily (see below);
expect ~0.7 GB for the all-stocks DB.

## Workflow: PC for research, Pi for collection + live (decided 2026-11-25)

Backtests are ~2,000 full simulations per study - too slow on the Pi. The daily/weekly live run
(rank today's stocks, 0-4 orders a week) is seconds and stays on the Pi. Data flows Pi -> PC only:
on the Pi `python -m trader collect export` makes `data/market_snapshot.db.gz` (SQLite online
backup, safe while collectors write; 69 MB DB -> 28 MB gz); scp it to the PC and unzip to
`data/market.db`. Live-system risks on the Pi are reliability (reboots, SD card, network), not
speed: build catch-up checks, retries and failure alerts into Gate 1.

## Point-in-time data (built 2026-11-25, `trader/pit.py`)

- `collect.store_all: true`: every EQ/BE stock (~1,640/day in 2020), not just today's N500.
- Symbol renames from NSE `symbolchange.csv` (effective-dated, chains resolved).
- Split/bonus factor = NSE PREV_CLOSE(t) / CLOSE(t-1); only < 0.85 or > 1.18 counts (smaller gaps
  are missed Saturday sessions, penny-stock ticks, dividends) and only if the previous row is
  <= 10 days earlier. Dividends are not added back (~1-1.5%/yr lower for rule and random alike).
- 2026-11-26: backfill crawled (250 -> 25 days/hour, 19 GB written in 11 h) because the Pi's
  system disk is a USB flash stick and the eq_daily(symbol, date) index made every insert hit
  ~2,000 scattered pages. Index dropped (nothing reads by symbol; all reads are by date = PK):
  6 days/min again, and far less flash wear. Keep new indexes off eq_daily unless really needed.
- NSE served 2022-08-08 as an .xlsx under the .csv name; `nse.bhavcopy_text` now detects that
  and converts it (stdlib only).
- Saturday special sessions (budget day 2020-02-01 etc.) caused false 10-12% gaps: backfill
  `--weekends` fetches them (queued to run automatically after the main backfill).
- Universe each week = top 500 by 6-month median turnover at that date, price >= Rs 10, ETFs out.
- Stocks that stop trading > 10 sessions before the end can be exited at their last close
  (optimistic for collapses; counted in the output).
- `momentum-pit` also runs the same rules on "today's Nifty 500" over the same data: the gap is
  the survivorship bias of Results 6. Learn 2021-23, check 2024+ (signals need 12 months).
- Smoke run on partial data completed end to end (numbers meaningless).

## NEXT: Gate 0 backtests (no LLM needed)

Immediate (after Results 8) - robustness of "combo + block-buys filter" without re-tuning:
- neighbours: exit rank 30/40, filter on 150/250-day average (results should change smoothly);
- other universes: top 200 by turnover (large caps only) and top 300-500 only (mid/small);
- core + satellite: e.g. 50% NIFTYBEES + 50% combo, to see the drawdown the owner would live with;
- Gate 1 plan: paper-trade the chosen rule weekly from the Pi (Telegram), shadow list of top 20.
- Results reactions (step 3) once announcements history is long enough (only Nov 2026 so far).
- Done: turnover/capital/filter/exit comparison (Results 8), split detection fixed (Results 7).

Work on the Pi (studies take minutes; use nohup/tmux). For each signal: rank Nifty 500 stocks
weekly, form a 5-stock portfolio + measure forward 5/10/20-day returns of the top decile; compare
with random picks from the same eligible set and with Nifty 500 buy-and-hold; rolling 1y/3y windows;
learn on 2020-2023, check on 2024-2026. Note survivorship bias in every report.

1. **Momentum / relative strength** — first run done, biased (Results 6); point-in-time redo = `momentum-pit` (written, to run).
   Original spec: (prices from Yahoo, 2015+): 6- and 12-month return skipping the
   last month, distance to 52-week high, return vs sector and vs Nifty. Compare with the
   Nifty 200 Momentum 30 ETF (if the ETF matches, prefer the ETF).
2. **Delivery %** (from `eq_daily`, 2020+): does rising delivery % on up-days / high delivery vs the
   stock's own 20-day average predict the next 5-20 days?
3. **Results reactions**: around result dates (from announcements going forward; for history use
   big-volume days / NSE results calendar if obtainable) — drift after strong vs weak reactions.
4. **Lever sensitivities** (`levers`, written, to run): regress each stock's weekly returns on crude (BZ=F), USD/INR (INR=X),
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
