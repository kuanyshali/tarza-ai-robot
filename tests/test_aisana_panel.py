"""Phone panel and Aisana's local control port (no hardware, no network)."""
import socket
import threading
import time
import unittest

import aisana_control
import aisana_panel


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class FakeSession(object):
    def __init__(self):
        self.actions = []

    def panel_status(self):
        return {'arms': 'ready', 'legs': 'not connected'}

    def panel_action(self, action, text=''):
        self.actions.append((action, text))
        return {'ok': True}


class ControlPortTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.session = FakeSession()
        self.server = aisana_control.ControlServer(lambda: self.session, port=self.port).start()

    def tearDown(self):
        self.server.close()

    def test_status_and_actions(self):
        status = aisana_control.send({'action': 'status'}, port=self.port)
        self.assertEqual(status, {'ok': True, 'arms': 'ready', 'legs': 'not connected'})
        self.assertEqual(aisana_control.send({'action': 'say', 'text': 'Hello, guests!'}, port=self.port), {'ok': True})
        self.assertEqual(self.session.actions, [('say', 'Hello, guests!')])

    def test_only_known_actions_and_text(self):
        self.assertFalse(aisana_control.send({'action': 'rm -rf'}, port=self.port)['ok'])
        self.assertFalse(aisana_control.send({'action': 'say', 'text': '  '}, port=self.port)['ok'])
        aisana_control.send({'action': 'ask', 'text': 'x' * 1000}, port=self.port)
        self.assertEqual(len(self.session.actions[-1][1]), aisana_control.MAX_TEXT)

    def test_listens_on_localhost_only(self):
        self.assertEqual(self.server.server.server_address[0], '127.0.0.1')

    def test_not_running(self):
        self.assertIsNone(aisana_control.send({'action': 'status'}, port=free_port()))


class PanelTests(unittest.TestCase):
    def test_log_tail_hides_camera_noise(self):
        original = aisana_panel.LOG
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as handle:
            handle.write('GST_ARGUS: Running\nYou: Aisana, dance\n\nAisana: Dancing!\n')
        aisana_panel.LOG = handle.name
        try:
            self.assertEqual(aisana_panel.log_tail(), ['You: Aisana, dance', 'Aisana: Dancing!'])
        finally:
            aisana_panel.LOG = original

    def test_page_has_the_buttons(self):
        for label in ('СТОП', 'Включить', 'Выключить', 'Помахать', 'Танцевать', 'Сказать это'):
            self.assertIn(label, aisana_panel.PAGE)


if __name__ == '__main__':
    unittest.main()
