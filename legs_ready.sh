#!/bin/bash
# Make the wheels ready with no questions (used by start.sh and the phone panel):
#   1. wait while the Pi is still connecting to the ODrive (up to 30 s);
#   2. enable WITHOUT calibration (enough if the ODrive was not power-cycled);
#   3. only if that fails: calibrate (the wheels spin a little), one retry.
# Nov 23: start.sh took "CALIBRATING / ENABLING" (the Pi just connecting) for
# "needs calibration" and started a second calibration on top of it.
cd "$(dirname "$0")" || exit 1
URL="${AISANA_LEGS_URL:-http://10.42.0.148:5000}"
status() { curl -s --max-time 3 "$URL/motor/status"; }
enabled() { case "$1" in *'"enabled": true'*|*'"enabled":true'*) return 0 ;; esac; return 1; }
connecting() { case "$1" in *'"connecting": true'*|*'"connecting":true'*) return 0 ;; esac; return 1; }

now="$(status)"
if [ -z "$now" ]; then
    echo "Legs Pi at $URL does not answer; no wheels." >&2
    exit 2
fi
for _ in $(seq 1 30); do
    connecting "$now" || break
    sleep 1
    now="$(status)"
done
if enabled "$now"; then
    echo "Wheels ready."
    exit 0
fi
echo "Enabling the wheels (no calibration)..."
curl -s --max-time 5 -X POST -H "Content-Type: application/json" -d '{"calibrate": false}' "$URL/motor/enable" >/dev/null
for _ in $(seq 1 15); do
    sleep 1
    now="$(status)"
    if enabled "$now"; then
        echo "Wheels ready."
        exit 0
    fi
    connecting "$now" || case "$now" in *'"error": null'*|*'"error":null'*) ;; *) break ;; esac
done
echo "Not enabled ($now); calibrating."
if ./calibrate_legs.sh --yes || ./calibrate_legs.sh --yes; then
    exit 0
fi
echo "Wheels NOT ready: $(status)" >&2
exit 1
