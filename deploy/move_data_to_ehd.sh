#!/usr/bin/env bash
# Put data/ (market.db) on the external disk instead of the USB flash system disk.
#
# The external disk is exFAT (shared with other files), and SQLite is safest on ext4, so this
# creates one ext4 image file on it (default 10 GB), loop-mounts it at /mnt/sma via /etc/fstab
# (nofail: the Pi still boots without the disk), moves data/ there and leaves a symlink.
# If the disk is ever missing the symlink dangles and the collectors fail loudly instead of
# silently filling the flash disk.
#
# Also, if ~/pc_market.db exists (copied from the PC), merges its announcements / deals /
# FII-DII rows in (existing rows win), and starts the backfill in tmux unless --no-backfill.
#
# Safe to re-run: every step checks before acting.
#   bash deploy/move_data_to_ehd.sh [--no-backfill]
set -euo pipefail

EHD="${EHD:-/mnt/ehd}"           # where the external disk is mounted
SIZE_GB="${SIZE_GB:-10}"
IMG="$EHD/pi-data/sma.img"
MNT=/mnt/sma
DATA="$MNT/sma-data"
REPO="$HOME/stock-market-analyzer"
PC_DB="$HOME/pc_market.db"
BACKFILL=1
[ "${1:-}" = "--no-backfill" ] && BACKFILL=0

step() { echo; echo "==> $*"; }

step "External disk"
mountpoint -q "$EHD" || { echo "!! $EHD is not mounted - plug in / mount the disk first"; exit 1; }
if ! grep -Eqs "[[:space:]]$EHD[[:space:]]" /etc/fstab; then
  echo "!! $EHD is not in /etc/fstab, so it may not mount at boot (then /mnt/sma won't either)."
elif ! grep -Es "[[:space:]]$EHD[[:space:]]" /etc/fstab | grep -q nofail; then
  echo "!! $EHD's /etc/fstab line has no 'nofail': the Pi won't boot cleanly if the disk is unplugged."
fi
df -h "$EHD" | tail -1

step "Stop collectors + paper trading while data moves"
systemctl --user stop trader-collect-daily.timer trader-collect-news.timer trader-paper.timer \
  trader-collect-daily.service trader-collect-news.service trader-paper.service 2>/dev/null || true

step "ext4 image $IMG (${SIZE_GB} GB)"
if [ ! -f "$IMG" ]; then
  sudo mkdir -p "$(dirname "$IMG")"
  # exFAT has no sparse files, so this really writes SIZE_GB (a few minutes)
  sudo dd if=/dev/zero of="$IMG.tmp" bs=1M count=$((SIZE_GB * 1024)) status=progress
  sudo mkfs.ext4 -q -F -L sma-data "$IMG.tmp"
  sudo mv "$IMG.tmp" "$IMG"
else
  echo "exists"
fi

step "Mount at $MNT (fstab, nofail)"
sudo mkdir -p "$MNT"
if ! grep -Fqs "$IMG " /etc/fstab; then
  sudo cp /etc/fstab /etc/fstab.bak-sma
  echo "$IMG $MNT ext4 loop,noatime,nofail,x-systemd.requires-mounts-for=$EHD 0 0" | sudo tee -a /etc/fstab
  sudo systemctl daemon-reload
fi
mountpoint -q "$MNT" || sudo mount "$MNT"
sudo mkdir -p "$DATA"
sudo chown "$(id -u):$(id -g)" "$DATA"
df -h "$MNT" | tail -1

step "Move data/ -> $DATA"
if [ -L "$REPO/data" ]; then
  echo "already a symlink -> $(readlink "$REPO/data")"
else
  if [ -d "$REPO/data" ]; then
    shopt -s dotglob nullglob
    files=("$REPO/data"/*)
    for f in "${files[@]}"; do
      if [ -e "$DATA/$(basename "$f")" ]; then
        echo "!! $DATA/$(basename "$f") already exists - not overwriting; sort it out by hand"; exit 1
      fi
    done
    [ ${#files[@]} -gt 0 ] && mv "${files[@]}" "$DATA"/
    rmdir "$REPO/data"   # fails (and stops) if anything could not be moved
  fi
  ln -s "$DATA" "$REPO/data"
  echo "data -> $DATA"
fi

cd "$REPO"
if [ -f "$PC_DB" ]; then
  step "Merge news/deals/flows from $PC_DB"
  sqlite3 data/market.db <<EOF
ATTACH '$PC_DB' AS pc;
INSERT OR IGNORE INTO announcements (link, feed, published, company, symbol, subject, description, fetched_at, tags)
  SELECT link, feed, published, company, symbol, subject, description, fetched_at, tags FROM pc.announcements;
INSERT OR IGNORE INTO deals (date, kind, symbol, security, client, side, qty, price, remarks)
  SELECT date, kind, symbol, security, client, side, qty, price, remarks FROM pc.deals;
INSERT OR IGNORE INTO fii_dii (date, category, buy, sell, net)
  SELECT date, category, buy, sell, net FROM pc.fii_dii;
DETACH pc;
EOF
  mv "$PC_DB" "$PC_DB.merged"
  echo "merged (source kept as $PC_DB.merged; delete it when happy)"
fi

step "Database check"
.venv/bin/python -m trader collect status

step "Restart collectors"
systemctl --user start trader-collect-daily.timer trader-collect-news.timer trader-paper.timer
systemctl --user list-timers 'trader-*' --no-pager

if [ "$BACKFILL" = 1 ]; then
  step "Backfill 2020 -> today in tmux session 'backfill' (~3 h, resumes if interrupted)"
  if tmux has-session -t backfill 2>/dev/null; then
    echo "already running"
  else
    tmux new -d -s backfill "cd $REPO && .venv/bin/python -u -m trader collect backfill --start 2020-01-01 > data/backfill.log 2>&1 && .venv/bin/python -u -m trader collect backfill --start 2020-01-01 --weekends >> data/backfill.log 2>&1"
    echo "started"
  fi
  echo "Watch:  tail -f $REPO/data/backfill.log"
fi

cat <<EOF

Done. Once the backfill has finished, reboot once to confirm the disk comes back by itself:
  sudo reboot      then:  ls -l $REPO/data/ && df -h $MNT
(Rebooting earlier is fine too: re-run this script afterwards and the backfill resumes.)
EOF
