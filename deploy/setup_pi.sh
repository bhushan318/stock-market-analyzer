#!/usr/bin/env bash
# Rebuild the Pi from scratch (safe to re-run: every step checks before acting).
#
# Before running (one-time, needs the GitHub website - see PROGRESS.md "Rebuild the Pi"):
#   - deploy key ~/.ssh/github_sma added to the repo with write access
#   - ~/.ssh/config has the "github-sma" host alias
#
# Run:   bash setup_pi.sh            (from anywhere; clones the repo if missing)
# Then:  start the backfill it prints at the end.
set -euo pipefail

REPO_DIR="$HOME/stock-market-analyzer"
REMOTE="git@github-sma:bhushan318/stock-market-analyzer.git"
BRANCH="swing-engine"

step() { echo; echo "==> $*"; }

step "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip sqlite3 tmux

step "Timezone (timers name Asia/Kolkata explicitly; this just makes logs readable)"
if [ "$(timedatectl show -p Timezone --value)" != "Asia/Kolkata" ]; then
  sudo timedatectl set-timezone Asia/Kolkata
fi
timedatectl show -p Timezone --value

step "GitHub access via deploy key"
# (ssh -T always exits non-zero on GitHub, so check its message, not its exit code)
gh_auth=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T github-sma 2>&1 || true)
if ! grep -q "successfully authenticated" <<<"$gh_auth"; then
  echo "!! 'ssh -T github-sma' failed. Set up the deploy key first (PROGRESS.md: Rebuild the Pi)."
  echo "   Falling back to read-only HTTPS clone; switch the remote later with:"
  echo "   git -C $REPO_DIR remote set-url origin $REMOTE"
  REMOTE="https://github.com/bhushan318/stock-market-analyzer.git"
fi

step "Repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q origin
  git -C "$REPO_DIR" checkout -q "$BRANCH"
  git -C "$REPO_DIR" pull -q --ff-only
else
  git clone -q -b "$BRANCH" "$REMOTE" "$REPO_DIR"
fi
cd "$REPO_DIR"
git log --oneline -1

step "Python environment"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-trader.txt
.venv/bin/python --version

step "Tests"
.venv/bin/python -m pytest -q tests | tail -1

step "First collection (creates data/market.db if missing)"
mkdir -p data
.venv/bin/python -m trader collect daily || echo "!! daily collect failed - check network, rerun later"
.venv/bin/python -m trader collect news || echo "!! news collect failed - check network, rerun later"

step "Scheduled jobs (systemd user timers)"
bash deploy/install.sh

step "Disk space (the all-stocks DB needs ~0.7 GB)"
df -h "$HOME" | tail -1

cat <<EOF

Done. Remaining manual steps:
  1. (optional) Restore news/flows collected earlier: copy a market.db from the PC into
     $REPO_DIR/data/ BEFORE the backfill (days with too few symbols are re-fetched automatically).
  2. Start the price/delivery backfill (~3 h, resumes if interrupted):
       cd $REPO_DIR && tmux new -d -s backfill '.venv/bin/python -u -m trader collect backfill --start 2020-01-01 > backfill.log 2>&1 && .venv/bin/python -u -m trader collect backfill --start 2020-01-01 --weekends >> backfill.log 2>&1'
     Watch:  tail -f $REPO_DIR/backfill.log
  3. Gemini key (when created):  echo 'GEMINI_API_KEY=...' > $REPO_DIR/.env && chmod 600 $REPO_DIR/.env
  4. Check any time:  .venv/bin/python -m trader collect status
EOF
