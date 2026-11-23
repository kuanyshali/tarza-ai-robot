#!/bin/bash
# Calibrate the wheel motors (ODrive) after the ODrive was powered on:
#   ./calibrate_legs.sh          (asks first)
#   ./calibrate_legs.sh --yes    (no question, 3 s warning)
# The wheels spin during calibration, so the robot MUST be lifted: wheels off
# the floor. Aisana herself never calibrates (not by voice, not on her own).
URL="${AISANA_LEGS_URL:-http://10.42.0.148:5000}"

status() { curl -s --max-time 3 "$URL/motor/status"; }

if ! status >/dev/null; then
    echo "The legs Pi at $URL does not answer. Is it on and connected by LAN?" >&2
    exit 1
fi
echo "Legs now: $(status)"
echo
if [ "$1" = "--yes" ]; then
    # No question (Nov 23: the owner calibrates on the floor). The wheels still
    # spin a little: 3 s to step away.
    echo "!!! Calibrating the wheels in 3 s: they will spin a little. Step away from the robot."
    sleep 3
else
    echo "!!! The wheels WILL SPIN. Lift the robot so both wheels are off the floor."
    read -r -p "Wheels are off the floor? Type yes: " answer
    if [ "$answer" != "yes" ] && [ "$answer" != "да" ]; then
        echo "Cancelled."
        exit 1
    fi
fi

curl -s --max-time 5 -X POST -H "Content-Type: application/json" -d '{"calibrate": true}' "$URL/motor/enable" >/dev/null
echo "Calibrating (up to 60 s)..."
for _ in $(seq 1 60); do
    sleep 1
    now="$(status)"
    case "$now" in
        *'"enabled": true'*|*'"enabled":true'*)
            echo "Done: $now"
            echo "Done. Aisana can drive now."
            exit 0 ;;
        *'"connecting": false'*|*'"connecting":false'*)
            case "$now" in *'"error": null'*|*'"error":null'*) ;; *)
                echo "Calibration FAILED: $now" >&2
                exit 1 ;;
            esac ;;
    esac
done
echo "No result after 60 s: $(status)" >&2
exit 1
