#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/asr-gcp-uploader"
PID_FILE="$APP_DIR/run/asr_upload.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"

if ps -p "$PID" > /dev/null 2>&1; then
  echo "Stopping ASR upload PID=$PID"
  kill "$PID"

  sleep 5

  if ps -p "$PID" > /dev/null 2>&1; then
    echo "Process still running. Sending SIGKILL."
    kill -9 "$PID"
  fi

  echo "Stopped."
else
  echo "Process is not running. Last known PID=$PID"
fi