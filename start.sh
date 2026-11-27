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
restart_camera_service() {
    # Fresh camera service on every start: after an unclean exit nvargus-daemon is
    # broken and the camera says "Failed to create CameraProvider" (Nov 23).
    # Needs the sudo rule from ./install_panel_service.sh (no password asked).
    if sudo -n systemctl restart nvargus-daemon 2>/dev/null; then
        sleep 2
    else
        echo "Camera service not restarted (run ./install_panel_service.sh once)." >&2
    fi
}
if [ ! -x .venv/bin/python ]; then
    echo "No .venv here: create it first (see README)." >&2
    exit 1
fi
if pgrep -f "$PATTERN" >/dev/null; then
    echo "Aisana is already running. Stop it first: ./stop.sh" >&2
    exit 1
fi
restart_camera_service

# Wheels: Aisana switches the motors on and keeps them on by herself (keep-alive
# in aisana_legs.py), calibrating only if the ODrive needs it. Nothing to wait for here.
if [ -n "$(curl -s --max-time 3 "$AISANA_LEGS_URL/motor/status")" ]; then
    echo "Legs Pi answers; Aisana will switch the motors on by herself." | tee realtime.log
else
    echo "Legs Pi at $AISANA_LEGS_URL does not answer; Aisana keeps trying." | tee realtime.log
fi

.venv/bin/python -u aisana_realtime.py 2>&1 | tee -a realtime.log   # after the wheel lines
