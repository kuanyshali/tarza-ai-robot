"""Local control port of a running Aisana (127.0.0.1 only), used by aisana_panel.py.

One JSON object per line in, one JSON object per line out:
    {"action": "status"}
    {"action": "stop"}                 wheels + arms stop at once
    {"action": "wave"}                 arm up, hand waves, arm down
    {"action": "dance"}
    {"action": "say", "text": "..."}   she says exactly this
    {"action": "ask", "text": "..."}   as if someone asked her this

Python 3.6, stdlib only. Never listens on other interfaces: the panel is the
only door from the network, and it can only send these actions.
"""
import json
import socket
import socketserver
import threading

CONTROL_PORT = 8091
ACTIONS = ('status', 'stop', 'wave', 'dance', 'say', 'ask')
MAX_TEXT = 300


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline(4096)
        try:
            request = json.loads(line.decode('utf-8'))
            reply = self.server.dispatch(request)
        except Exception as exc:
            reply = {'ok': False, 'error': str(exc)}
        self.wfile.write((json.dumps(reply, ensure_ascii=False) + '\n').encode('utf-8'))


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class ControlServer(object):
    """Runs in Aisana's process; `get_session` returns the current RealtimeAisana."""

    def __init__(self, get_session, port=CONTROL_PORT):
        self.get_session = get_session
        self.server = _Server(('127.0.0.1', port), _Handler)
        self.server.dispatch = self.dispatch
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def dispatch(self, request):
        action = request.get('action')
        if action not in ACTIONS:
            return {'ok': False, 'error': 'unknown action'}
        session = self.get_session()
        if session is None:
            return {'ok': False, 'error': 'Aisana is starting'}
        if action == 'status':
            return dict(ok=True, **session.panel_status())
        text = str(request.get('text') or '').strip()[:MAX_TEXT]
        if action in ('say', 'ask') and not text:
            return {'ok': False, 'error': 'empty text'}
        return session.panel_action(action, text)


def send(request, port=CONTROL_PORT, timeout=4.0):
    """Client side (the panel): one request, one reply; None if Aisana is not running."""
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout) as sock:
            sock.sendall((json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8'))
            data = b''
            while not data.endswith(b'\n'):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        return json.loads(data.decode('utf-8')) if data else None
    except (OSError, ValueError):
        return None
