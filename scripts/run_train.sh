#!/usr/bin/env bash
# Wrapper around scripts/train.sh: run it in the background and/or tail its
# log live.
#
# Usage:
#   scripts/run_train.sh                # run train.sh in the foreground
#   scripts/run_train.sh --bg           # launch in the background, print PID + log path, return immediately
#   scripts/run_train.sh --watch        # tail the log of the currently running background job (if any)
#   scripts/run_train.sh --bg --watch   # launch in the background, then immediately tail its log
#                                        # (Ctrl-C only stops watching -- the training keeps running)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
PID_FILE="$LOG_DIR/train.pid"
LATEST_LINK="$LOG_DIR/train.latest.log"

BG=0
WATCH=0
for arg in "$@"; do
  case "$arg" in
    --bg) BG=1 ;;
    --watch) WATCH=1 ;;
    *) echo "unknown flag: $arg (expected --bg and/or --watch)" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

run_watch() {
  local log="$1"
  echo "Watching $log (Ctrl-C stops watching only -- training keeps running in the background)"
  tail -n +1 -f "$log"
}

if [[ "$BG" -eq 1 ]]; then
  if is_running; then
    echo "Training already running (PID $(cat "$PID_FILE")). Not starting a new run." >&2
    [[ "$WATCH" -eq 1 ]] && run_watch "$LATEST_LINK"
    exit 0
  fi

  LOG_FILE="$LOG_DIR/train.$(date +%Y%m%d_%H%M%S).log"
  ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

  cd "$REPO_ROOT"
  nohup bash scripts/train.sh > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Training started in background: PID $(cat "$PID_FILE"), log: $LOG_FILE"

  [[ "$WATCH" -eq 1 ]] && run_watch "$LOG_FILE"
  exit 0
fi

if [[ "$WATCH" -eq 1 ]]; then
  if is_running; then
    run_watch "$LATEST_LINK"
  else
    echo "No background training run is currently active -- nothing to watch. Pass --bg (optionally with --watch) to start one." >&2
    exit 1
  fi
  exit 0
fi

cd "$REPO_ROOT"
bash scripts/train.sh
