#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/asr-gcp-uploader"
PID_FILE="$APP_DIR/run/asr_upload.pid"
LOG_DIR="$APP_DIR/logs"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found: $PID_FILE"
  exit 1
fi

PID="$(cat "$PID_FILE")"

if ps -p "$PID" > /dev/null 2>&1; then
  echo "ASR upload is running."
  echo "PID: $PID"
else
  echo "ASR upload is not running."
  echo "Last known PID: $PID"
fi

echo
echo "Latest log:"

LATEST_LOG="$(
  find "$LOG_DIR" -maxdepth 1 -type f -name 'asr_upload_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"

if [ -z "$LATEST_LOG" ]; then
  echo "No logs found."
  exit 0
fi

echo "$LATEST_LOG"
echo
tail -100 "$LATEST_LOG"