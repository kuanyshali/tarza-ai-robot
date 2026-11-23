#!/bin/bash
# Start Aisana (realtime mode) on the Jetson:  ./start.sh
# Any setting can be changed for one run:      AISANA_HUMOR=70 ./start.sh
# Stop: Ctrl+C here, or ./stop.sh from another terminal.
cd "$(dirname "$0")" || exit 1

# The OpenAI key lives in ~/.bashrc, which is skipped when started over SSH,
# by nohup or by a service (Nov 23: "OPENAI_API_KEY is missing"). Read just that line.
if [ -z "$OPENAI_API_KEY" ] && [ -f "$HOME/.bashrc" ]; then
    eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENAI_API_KEY=' "$HOME/.bashrc" | tail -1)"
    export OPENAI_API_KEY
fi

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

# One command for the day: if the wheels need calibrating (ODrive was just
# powered on), offer it first; it still asks to lift the robot.
legs="$(curl -s --max-time 3 "$AISANA_LEGS_URL/motor/status")"
case "$legs" in
    *alibrat*|*DISCONNECTED*)
        # After the ODrive is powered on it needs a calibration (that also connects it).
        echo "The wheels need calibrating (ODrive just switched on)."
        # Automatic, no questions; one retry (a failed try is often fixed by the second).
        if ./calibrate_legs.sh --yes || ./calibrate_legs.sh --yes; then
            echo "Wheels ready."
        else
            echo "Starting without wheels; run ./calibrate_legs.sh --yes later."
        fi ;;
    "")
        echo "Legs Pi at $AISANA_LEGS_URL does not answer; starting without wheels." ;;
esac

.venv/bin/python -u aisana_realtime.py 2>&1 | tee realtime.log
