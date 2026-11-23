#!/bin/bash
# Connect the Jetson to NU_Campus_WiFi (WPA2-Enterprise: PEAP + MSCHAPv2 with the
# university e-mail login). Run it yourself on the Jetson and type the login and
# password: they are stored only in NetworkManager's root-only file, not in git.
#   ./connect_nu_campus.sh            connect, 4G router (Altel) first, campus Wi-Fi as backup
#   ./connect_nu_campus.sh --first    connect, campus Wi-Fi first, 4G router as backup
#   ./connect_nu_campus.sh --remove   forget the campus Wi-Fi (and the stored password)
set -u
NAME=NU-Campus
SSID=NU_Campus_WiFi
if [ "${1:-}" = "--remove" ]; then
    sudo nmcli connection delete "$NAME" && echo "NU_Campus removed."
    exit 0
fi
# Route metric: router over USB has 50; lower = preferred.
METRIC=250
[ "${1:-}" = "--first" ] && METRIC=40

read -r -p "Логин университетской почты (например name.surname или name.surname@nu.edu.kz): " LOGIN
read -r -s -p "Пароль: " PASS
echo
[ -n "$LOGIN" ] && [ -n "$PASS" ] || { echo "Нужны логин и пароль." >&2; exit 1; }

add() {   # $1 = check the server certificate (yes/no)
    sudo nmcli connection delete "$NAME" >/dev/null 2>&1
    local extra=()
    if [ "$1" = yes ]; then
        # Only talk to NU's real login server: protects the password from a fake hotspot.
        extra=(802-1x.ca-cert /etc/ssl/certs/ca-certificates.crt 802-1x.domain-suffix-match nu.edu.kz)
    fi
    sudo nmcli connection add type wifi con-name "$NAME" ifname wlan0 ssid "$SSID" \
        wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.phase2-auth mschapv2 \
        802-1x.identity "$LOGIN" 802-1x.password "$PASS" 802-1x.password-flags 0 \
        ${extra[@]+"${extra[@]}"} \
        connection.autoconnect yes connection.autoconnect-priority 60 \
        ipv4.route-metric "$METRIC" ipv6.route-metric "$METRIC" >/dev/null
    sudo nmcli --wait 40 connection up "$NAME"
}

echo "Подключаюсь с проверкой сертификата NU..."
if ! add yes; then
    echo
    echo "С проверкой сертификата не вышло (у NU может быть свой сертификат)."
    read -r -p "Подключиться без проверки сертификата? Это слабее защищает пароль от поддельной точки. [y/N] " OK
    if [ "$OK" = y ] || [ "$OK" = Y ]; then
        add no || { echo "Не подключилось: проверь логин/пароль (попробуй с @nu.edu.kz и без)." >&2;
                    sudo nmcli connection delete "$NAME" >/dev/null 2>&1; unset PASS; exit 1; }
    else
        sudo nmcli connection delete "$NAME" >/dev/null 2>&1
        unset PASS
        exit 1
    fi
fi
unset PASS
sleep 3
echo
ip route get 8.8.8.8 | head -1
if curl -s -m 10 -o /dev/null -w '%{http_code}' https://api.openai.com/v1/models | grep -q 401; then
    echo "NU_Campus подключён, OpenAI доступен."
else
    echo "NU_Campus подключён, но OpenAI не отвечает: сеть может его блокировать." >&2
fi
echo "Порядок: $( [ "$METRIC" -lt 50 ] && echo 'NU_Campus, потом роутер 4G' || echo 'роутер 4G, потом NU_Campus' )."
echo "Айсана держит старое соединение: перезапустите её с пульта (Выключить -> Включить)."
