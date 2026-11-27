#!/bin/bash
# Install autostart of the phone panel only (not Aisana).
#   ./install_panel_service.sh            install and start
#   ./install_panel_service.sh --remove   remove autostart
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
UNIT=/etc/systemd/system/aisana-panel.service
if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now aisana-panel.service || true
  sudo rm -f "$UNIT" /etc/sudoers.d/aisana-camera
  sudo systemctl daemon-reload
  echo "Panel autostart removed."
  exit 0
fi
# start.sh restarts the camera service on every start without a password:
# a sudo rule for exactly this one command, nothing else.
RULE=/etc/sudoers.d/aisana-camera
echo "$(id -un) ALL=(root) NOPASSWD: /bin/systemctl restart nvargus-daemon" > /tmp/aisana-camera.sudoers
if sudo visudo -cf /tmp/aisana-camera.sudoers >/dev/null; then
  sudo install -m 0440 -o root -g root /tmp/aisana-camera.sudoers "$RULE"
fi
rm -f /tmp/aisana-camera.sudoers
sed -e "s|__USER__|$(id -un)|" -e "s|__DIR__|$DIR|" -e "s|__HOME__|$HOME|" \
  deploy/aisana-panel.service | sudo tee "$UNIT" >/dev/null
# stop a panel started by hand, so the port is free
pkill -f 'python[0-9.]* -u aisana_panel[.]py' 2>/dev/null || true
pkill -f 'python[0-9.]* aisana_panel[.]py' 2>/dev/null || true
sleep 1
sudo systemctl daemon-reload
sudo systemctl enable --now aisana-panel.service
sleep 2
systemctl --no-pager --lines=3 status aisana-panel.service || true
echo "Panel: http://$(hostname -I | awk '{print $1}'):${AISANA_PANEL_PORT:-8080}"
