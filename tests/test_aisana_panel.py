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

    def panel_frame(self):
        import base64
        return {'ok': True, 'jpeg': base64.b64encode(b'\xff\xd8fake\xff\xd9').decode('ascii')}


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

    def test_camera_frame_is_read_only(self):
        reply = aisana_control.send({'action': 'frame'}, port=self.port)
        self.assertTrue(reply['ok'])
        self.assertEqual(self.session.actions, [])                 # nothing was done to the robot
        self.assertEqual(aisana_panel.camera_frame(port=self.port), b'\xff\xd8fake\xff\xd9')

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

    def test_no_camera_frame_when_aisana_is_off(self):
        self.assertIsNone(aisana_panel.camera_frame(port=free_port()))

    def test_camera_not_on_the_public_link(self):
        class H(object):
            headers = {}
        h = H()
        for address, public in (('127.0.0.1', True), ('10.42.0.148', False), ('192.168.8.5', False),
                                ('100.101.2.3', False)):
            h.client_address = (address, 1234)
            self.assertEqual(aisana_panel.Handler.via_public_link(h), public, address)

    def test_battery_percent_8s_lifepo4(self):
        percent = aisana_panel.battery_percent
        self.assertEqual(percent(27.2, cells=8), 100)        # 3.40 V/cell, rested full
        self.assertEqual(percent(29.2, cells=8), 100)        # on the charger: never above 100
        self.assertEqual(percent(26.56, cells=8), 80)        # Nov 25 reading on the robot
        self.assertEqual(percent(26.08, cells=8), 50)
        self.assertEqual(percent(24.0, cells=8), 10)
        self.assertEqual(percent(20.0, cells=8), 0)
        self.assertEqual(percent(18.0, cells=8), 0)
        self.assertIsNone(percent(None))

    def test_battery_from_the_legs_pi(self):
        reading = aisana_panel.battery(fetch=lambda: {'state': 'ENABLED', 'vbus_voltage': 26.56})
        self.assertEqual(reading, {'ok': True, 'voltage': 26.6, 'percent': 80})
        self.assertFalse(aisana_panel.battery(fetch=lambda: {'vbus_voltage': None})['ok'])

        def down():
            raise OSError('no route to host')
        self.assertFalse(aisana_panel.battery(fetch=down)['ok'])

    def test_photo_button_saves_the_newest_picture(self):
        import os
        import tempfile
        folder = os.path.join(tempfile.mkdtemp(), 'robot_photos')      # created on first photo
        grab = lambda: b'\xff\xd8photo\xff\xd9'
        first = aisana_panel.take_photo(grab=grab, folder=folder, now=0)
        second = aisana_panel.take_photo(grab=grab, folder=folder, now=0)  # same second
        self.assertTrue(first['ok'] and second['ok'])
        self.assertNotEqual(first['file'], second['file'])
        self.assertTrue(first['file'].startswith('photo-') and first['file'].endswith('.jpg'))
        with open(os.path.join(folder, first['file']), 'rb') as handle:
            self.assertEqual(handle.read(), b'\xff\xd8photo\xff\xd9')

    def test_no_photo_without_a_picture(self):
        import tempfile
        result = aisana_panel.take_photo(grab=lambda: None, folder=tempfile.mkdtemp())
        self.assertFalse(result['ok'])

    def test_internet_speed_from_the_counters(self):
        counters = {'rx': 1000000, 'tx': 200000}
        read = lambda name, kind: counters[kind]
        first = aisana_panel.traffic('ethX', now=100.0, read=read)
        self.assertEqual(first['total_mb'], 1.2)
        self.assertNotIn('down_kbs', first)                     # needs two samples
        counters['rx'] += 1024 * 30                              # 30 KB in 2 s down
        counters['tx'] += 1024 * 128                             # 128 KB up (her microphone)
        second = aisana_panel.traffic('ethX', now=102.0, read=read)
        self.assertEqual((second['down_kbs'], second['up_kbs']), (15.0, 64.0))

    def test_microphone_has_its_own_page(self):
        # Owner, Nov 26: a separate page with a big microphone button, none on the panel.
        self.assertIn('id="big"', aisana_panel.MIC_PAGE)
        self.assertIn('/api/mic', aisana_panel.MIC_PAGE)
        self.assertNotIn('/api/mic', aisana_panel.PAGE)
        self.assertIn('href="/mic"', aisana_panel.PAGE)

    def test_page_has_the_buttons(self):
        for label in ('СТОП', 'Включить', 'Выключить', 'Помахать', 'Танцевать', 'Сказать это'):
            self.assertIn(label, aisana_panel.PAGE)


if __name__ == '__main__':
    unittest.main()
