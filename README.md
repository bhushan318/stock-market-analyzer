# Stock Market Analyzer

Two things live in this repo:

| Part | Files | Status |
|---|---|---|
| **`trader/` – swing-trading engine** | `trader/`, `config.yaml`, `tests/` | Active. Model + backtest for a small-capital take-profit strategy on Nifty 50. |
| Legacy Streamlit app | `app.py`, `StockPrediction.py`, `nifty_50_analyzer.py` | Old version, kept for reference. Its backtest numbers are not reliable (see "Why the rewrite"). |

## The strategy being tested

- Start with ₹20,000 (`capital` in `config.yaml`).
- Hold at most 5–8 Nifty 50 stocks (`strategy.max_positions`).
- Sell a stock when it is up 5% **after all charges** (`strategy.take_profit`).
- Default rule: never sell below the buy price. Optional `max_hold_days` and `stop_loss`
  exist so the backtest can show what that rule costs.
- Cash from a sale is spread across free slots and reinvested the next morning.
- A model ranks stocks by the probability of reaching the target within 20 trading days.

## Setup on a Raspberry Pi (Raspberry Pi OS 64-bit / Debian) or any Linux/macOS machine

```bash
sudo apt update && sudo apt install -y git python3-venv     # Pi / Debian only
git clone -b swing-engine https://github.com/bhushan318/stock-market-analyzer.git
cd stock-market-analyzer
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-trader.txt
.venv/bin/python -m pytest -q tests          # should print "6 passed"
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

All packages have 64-bit ARM wheels, so nothing compiles on the Pi. Use the 64-bit OS image;
32-bit Raspberry Pi OS is not supported by recent scikit-learn/pandas wheels.

## Commands

```bash
.venv/bin/python -m trader evaluate                 # model quality: dev period vs holdout
.venv/bin/python -m trader backtest                 # strategy, rule variants and random baseline
.venv/bin/python -m trader picks --cash 20000       # today's ranked candidates
.venv/bin/python -m trader picks --cash 8000 --slots 2 --exclude TCS.NS INFY.NS
.venv/bin/python -m trader --refresh backtest       # force a fresh price download
```

### ETF study (current preferred direction)

```bash
.venv/bin/python -m trader etf --broad   # broad ETFs: targets 3-8%, windows 1/3/5/10/18y, rolling starts, seasonality
.venv/bin/python -m trader etf           # same with sector/thematic ETFs included
.venv/bin/python -m trader ideas         # stock-strategy variants (market filter, uptrend, averaging down...)
```

Findings so far (broad ETFs, buy the deepest dip below the 20-day average, sell at +5% net,
never at a loss, Rs 20k, 6 slots): ~16% a year over 2009-2026 before tax; no 3-year rolling
window lost money; sector ETFs (PSU Bank -78%, Silver -61%) are excluded because a decade-long
fall freezes money under the no-loss rule; seasonality did not repeat out of sample.

- Prices come from Yahoo Finance (free) and are cached in `data/cache/` for 12 hours.
  Today's bar is ignored until 15:45 IST so a half-finished day never feeds the model.
- `evaluate` takes ~1–2 minutes on a laptop (expect several minutes on a Pi 4).
  Walk-forward predictions are cached, so `backtest` is fast afterwards.
- Outputs (trade log, equity curve, open positions) go to `outputs/`.

## Reading the backtest

Each column is one variant of the rules; rows include:

- `cagr` / `max_drawdown` – yearly growth and worst peak-to-trough fall of the whole account.
- `avg_stuck_pct` – average share of money sitting in positions held 60+ days and still underwater.
- `worst_unrealised_position` – the deepest paper loss any single holding reached.
- `random_median` / `random_p90` – the same rules with random stock picks (20 runs).
  If the model doesn't beat `random_median`, its ranking isn't adding value.
- **FULL** (2019→today) vs **HOLDOUT** (2024→today). Trust the holdout more.
  Both are flattered by survivorship bias: the universe is *today's* Nifty 50.

## How the model avoids fooling itself

- Features at day *t* use only data up to the close of *t* (a test checks this).
- The label is "does the High reach the target within the next 20 days, buying at the next open".
  The last 20 days have no label but still get predictions, so signals are current.
- One model is trained across all stocks and retrained every ~3 months on past data only,
  with a 21-day gap so no training label overlaps the period being predicted.
- The model type is chosen on 2019–2023 only. 2024 onward is the holdout.

## Charges modelled (Zerodha delivery, approx.)

STT 0.1% each side, stamp duty 0.015% on buy, NSE + SEBI fees with 18% GST,
₹15.93 DP charge per stock sold, 0.05% slippage on buys at the open.
Short-term capital gains tax (20%) is estimated in the summary, not deducted daily.
Update the `costs` section of `config.yaml` if your broker's rates differ.

## Why the rewrite

The legacy `StockPrediction.py` had problems that made its results look better than reality:
the "latest" signal was 1–5 days old, accuracy/F1 compared misaligned rows, the "best" model
was picked using the test period, "expected 5-day return" was annual return ÷ 50, there were
no transaction costs, and the default test window was one month.

## Roadmap

1. ✅ Engine: data, features, walk-forward model, cost-aware backtest, tests.
2. ⬜ Model tuned for the chosen exit rules (e.g. prefer steady low-volatility stocks).
3. ⬜ Paper-trading pipeline on the Pi: SQLite state, paper broker, 07:30 / 09:20 / 15:45
   systemd timers, Telegram alerts, several strategy variants run side by side.
4. ⬜ News veto (log-only during paper trading).
5. ⬜ Zerodha Kite broker (CNC only, GTT target/stop orders, hard capital and order limits),
   tested with ₹3–5k before the full amount.

Not investment advice. Backtests are estimates; past results don't predict future returns.
