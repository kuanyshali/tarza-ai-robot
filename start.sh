#!/bin/bash
# Start Aisana (realtime mode) on the Jetson:  ./start.sh
# Any setting can be changed for one run:      AISANA_HUMOR=70 ./start.sh
# Stop: Ctrl+C here, or ./stop.sh from another terminal.
cd "$(dirname "$0")" || exit 1

export AISANA_LEGS_URL="${AISANA_LEGS_URL:-http://10.42.0.148:5000}"   # Raspberry Pi (legs)
export AISANA_ECHO_TAIL_SECONDS="${AISANA_ECHO_TAIL_SECONDS:-0.4}"
export AISANA_PLAYBACK_LEAD_MS="${AISANA_PLAYBACK_LEAD_MS:-150}"

PATTERN='python[0-9.]* -u aisana_realtime[.]py'
if [ ! -x .venv/bin/python ]; then
    echo "No .venv here: create it first (see README)." >&2
    exit 1
fi
if pgrep -f "$PATTERN" >/dev/null; then
    echo "Aisana is already running. Stop it first: ./stop.sh" >&2
    exit 1
fi

.venv/bin/python -u aisana_realtime.py 2>&1 | tee realtime.log
