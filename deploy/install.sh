#!/usr/bin/env bash
# Install the collectors as systemd *user* timers on the Pi.
# Run from the repo root:  bash deploy/install.sh
# Assumes the repo is at ~/stock-market-analyzer with its .venv already set up.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(pwd)" = "$HOME/stock-market-analyzer" ] || { echo "Repo must live at ~/stock-market-analyzer (it is at $(pwd))"; exit 1; }
[ -x .venv/bin/python ] || { echo "Create the venv first (see README)"; exit 1; }

mkdir -p ~/.config/systemd/user
cp deploy/*.service deploy/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now trader-collect-daily.timer trader-collect-news.timer trader-paper.timer trader-collect-hourly.timer trader-collect-extra.timer
# keep user timers running when you are not logged in
sudo loginctl enable-linger "$USER"

echo; systemctl --user list-timers 'trader-*'
echo; echo "Logs:   journalctl --user -u trader-collect-daily -u trader-collect-news -u trader-paper -n 50"
echo "Status: .venv/bin/python -m trader collect status"
