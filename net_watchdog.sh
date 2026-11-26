#!/bin/bash
# Internet watchdog for the Jetson (owner, Nov 26): the 4G Altel USB router (eth1)
# is the only main channel; the Wi-Fi "hommedumonde" is the backup and is used
# only while the USB router gives no internet.
#
# Every CHECK_S seconds it pings through eth1. After 3 failures in a row:
#   - eth1's default route gets metric 500 (the Wi-Fi backup, metric 300, wins);
#   - the USB connection is re-applied (Nov 26: connected, but no default route).
# When eth1 answers again its metric goes back to 50.
# The Wi-Fi profile itself has autoconnect with endless retries (NetworkManager
# keeps looking for it); the password lives only in NetworkManager, never here.
#
#   sudo ./net_watchdog.sh --install    # systemd service aisana-net (as root)
#   sudo ./net_watchdog.sh --remove
set -u
USB_DEV="${AISANA_USB_DEV:-eth1}"
USB_CONN="${AISANA_USB_CONN:-Beeline-router-USB}"
WIFI_CONN="${AISANA_WIFI_CONN:-hommedumonde}"
CHECK_S="${AISANA_NET_CHECK_S:-20}"
TARGETS="1.1.1.1 8.8.8.8"

if [ "${1:-}" = "--install" ] || [ "${1:-}" = "--remove" ]; then
    [ "$(id -u)" = 0 ] || { echo "Run with sudo." >&2; exit 1; }
    unit=/etc/systemd/system/aisana-net.service
    if [ "$1" = "--remove" ]; then
        systemctl disable --now aisana-net 2>/dev/null; rm -f "$unit"; systemctl daemon-reload
        echo "aisana-net removed."; exit 0
    fi
    cat > "$unit" <<EOF
[Unit]
Description=Aisana internet watchdog (Altel USB main, Wi-Fi backup)
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
ExecStart=$(readlink -f "$0")
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload && systemctl enable --now aisana-net && echo "aisana-net installed and running."
    exit 0
fi

usb_ok() {
    local t
    for t in $TARGETS; do
        ping -I "$USB_DEV" -c 1 -W 3 "$t" >/dev/null 2>&1 && return 0
    done
    return 1
}

fails=0
backup=0
while true; do
    if ! ip link show "$USB_DEV" >/dev/null 2>&1; then
        fails=0                                  # router unplugged: Wi-Fi is the only route anyway
    elif usb_ok; then
        if [ "$backup" = 1 ]; then
            nmcli device modify "$USB_DEV" ipv4.route-metric 50 ipv6.route-metric 50 >/dev/null 2>&1
            echo "USB internet is back: main channel again."
            backup=0
        fi
        fails=0
    else
        fails=$((fails + 1))
        if [ "$fails" = 3 ]; then
            echo "No internet through $USB_DEV: Wi-Fi backup takes over, re-applying the USB connection."
            nmcli device modify "$USB_DEV" ipv4.route-metric 500 ipv6.route-metric 500 >/dev/null 2>&1 \
                || nmcli connection up "$USB_CONN" >/dev/null 2>&1
            backup=1
        elif [ "$fails" -gt 3 ] && [ $((fails % 15)) = 0 ]; then
            nmcli connection up "$USB_CONN" >/dev/null 2>&1      # every ~5 min: try the router again
            [ "$backup" = 1 ] && nmcli device modify "$USB_DEV" ipv4.route-metric 500 ipv6.route-metric 500 >/dev/null 2>&1
        fi
    fi
    # The backup Wi-Fi stays connected (unused while the USB works) so a switch is instant.
    if nmcli radio wifi | grep -q disabled; then
        nmcli radio wifi on >/dev/null 2>&1
    fi
    sleep "$CHECK_S"
done
