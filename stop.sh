#!/bin/bash
# Stop Aisana from any terminal:  ./stop.sh
# The Pi stops the wheels by itself 0.35 s after the last heartbeat.
PATTERN='python[0-9.]* -u aisana_realtime[.]py'   # the running robot, not an editor with the file open
if ! pgrep -f "$PATTERN" >/dev/null; then
    echo "Aisana is not running."
    exit 0
fi
pkill -TERM -f "$PATTERN"
for _ in 1 2 3 4 5 6; do
    sleep 0.5
    pgrep -f "$PATTERN" >/dev/null || { echo "Aisana stopped."; exit 0; }
done
pkill -KILL -f "$PATTERN"
echo "Aisana stopped (forced)."
