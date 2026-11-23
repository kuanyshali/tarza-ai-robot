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

    def phone_audio(self, mode, pressed, pcm):
        self.actions.append(('mic', mode, pressed, pcm))
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

    def test_iphone_audio_packet(self):
        import base64
        pcm = b'\x10\x00' * 24000                      # 1 s of audio fits in one line
        reply = aisana_control.send({'action': 'mic', 'mode': 'ptt', 'pressed': True,
                                     'pcm': base64.b64encode(pcm).decode('ascii')}, port=self.port)
        self.assertEqual(reply, {'ok': True})
        self.assertEqual(self.session.actions[-1], ('mic', 'ptt', True, pcm))
        self.assertFalse(aisana_control.send({'action': 'mic', 'mode': 'ptt', 'pcm': '%%%'}, port=self.port)['ok'])

    def test_listens_on_localhost_only(self):
        self.assertEqual(self.server.server.server_address[0], '127.0.0.1')

    def test_not_running(self):
        self.assertIsNone(aisana_control.send({'action': 'status'}, port=free_port()))


class PanelTests(unittest.TestCase):
    def test_speaker_found_by_label_only(self):
        cards = (' 0 [tegrahda       ]: tegra-hda - tegra-hda\n'
                 '                      tegra-hda at 0x70038000 irq 115\n'
                 ' 4 [Device         ]: USB-Audio - USB Composite Device\n'
                 '                      GeneralPlus USB Composite Device at usb-70090000, full speed\n'
                 ' 3 [Audio          ]: USB-Audio - KT USB Audio\n')
        self.assertEqual(aisana_panel.speaker_card(cards), 4)      # the number moves after a replug
        self.assertIsNone(aisana_panel.speaker_card(' 3 [Audio ]: USB-Audio - KT USB Audio\n'))   # no fallback
        self.assertEqual(aisana_panel.parse_volume('  Front Left: Playback 127 [86%] [-4.68dB] [on]'), 86)

    def test_volume_without_speaker_fails_closed(self):
        original = aisana_panel.speaker_card
        aisana_panel.speaker_card = lambda cards_text=None: None
        try:
            self.assertFalse(aisana_panel.set_volume(50)['ok'])
            self.assertIsNone(aisana_panel.speaker_volume())
        finally:
            aisana_panel.speaker_card = original

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
