"""Offline tests for the Arduino arm link; no serial port or pyserial needed."""
import threading
import time
import unittest
from types import SimpleNamespace

import aisana_arm as arm


class FakeSerial(object):
    """Behaves like the Arm team firmware: boots, then answers commands."""

    def __init__(self, boot_lines=(b'Starting both arms...\n', b'Both arms ready.\n')):
        self.written = []
        self.incoming = list(boot_lines)
        self.lock = threading.Lock()
        self.closed = False

    def read(self, size):
        time.sleep(0.005)
        with self.lock:
            if self.incoming:
                return self.incoming.pop(0)
        return b''

    def write(self, data):
        self.written.append(data)
        command = data.decode('ascii').strip()
        with self.lock:
            if command == 'p':
                self.incoming.append(b'RIGHT stepper: 107.2  target 107.0  [at position 2]\n')
            elif command == 'r 9':
                self.incoming.append(b'RIGHT stepper: position 9 is not set.\n')

    def flush(self):
        pass

    def close(self):
        self.closed = True


def open_controller(fake):
    logs = []
    controller = arm.ArmController('/dev/ttyACM0', serial_factory=lambda port: fake,
                                   ready_timeout=2, echo=False, log=logs.append)
    controller.open()
    return controller, logs


class BuildCommandTests(unittest.TestCase):
    def test_whitelisted_firmware_commands(self):
        self.assertEqual(arm.build_command('position', 'both', 1), 'b 1')
        self.assertEqual(arm.build_command('position', 'left', 0), 'l 0')
        self.assertEqual(arm.build_command('position', 'right', 3, 'shoulder'), 'r m 3')
        self.assertEqual(arm.build_command('position', 'left', 4, 'elbow'), 'l s 4')
        self.assertEqual(arm.build_command('gripper_open', 'left'), 'l g o')
        self.assertEqual(arm.build_command('gripper_close', 'both'), 'b g c')
        self.assertEqual(arm.build_command('stop'), 'x')
        self.assertEqual(arm.build_command('status'), 'p')

    def test_rejects_anything_outside_the_whitelist(self):
        for args in (('position', 'both', 5), ('position', 'both', -1),
                     ('position', 'both', '2; x'), ('position', 'both', 1.5),
                     ('position', 'middle', 1), ('position', 'both', 1, 'wrist'),
                     ('dance', 'both'), ('gripper_open', 'b g a 180')):
            with self.subTest(args=args):
                with self.assertRaises(arm.ArmError):
                    arm.build_command(*args)


class PortDetectionTests(unittest.TestCase):
    def test_single_arduino_found(self):
        ports = [SimpleNamespace(device='/dev/ttyTHS1', vid=None),
                 SimpleNamespace(device='/dev/ttyACM0', vid=0x2341)]
        self.assertEqual(arm.find_arduino_port(ports, environ={}), '/dev/ttyACM0')

    def test_ch340_clone_found(self):
        ports = [SimpleNamespace(device='/dev/ttyUSB0', vid=0x1A86)]
        self.assertEqual(arm.find_arduino_port(ports, environ={}), '/dev/ttyUSB0')

    def test_genuine_arduino_preferred_over_other_acm_device(self):
        ports = [SimpleNamespace(device='/dev/ttyACM0', vid=0x1546, description='u-blox GPS'),
                 SimpleNamespace(device='/dev/ttyACM1', vid=0x2341, description='Arduino Mega 2560')]
        self.assertEqual(arm.find_arduino_port(ports, environ={}), '/dev/ttyACM1')

    def test_two_unknown_acm_devices_listed_in_error(self):
        ports = [SimpleNamespace(device='/dev/ttyACM0', vid=None, description='n/a'),
                 SimpleNamespace(device='/dev/ttyACM1', vid=None, description='n/a')]
        with self.assertRaisesRegex(arm.ArmError, 'by-id'):
            arm.find_arduino_port(ports, environ={})

    def test_several_boards_refused(self):
        ports = [SimpleNamespace(device='/dev/ttyACM0', vid=0x2341),
                 SimpleNamespace(device='/dev/ttyACM1', vid=0x2341)]
        with self.assertRaisesRegex(arm.ArmError, 'found 2'):
            arm.find_arduino_port(ports, environ={})

    def test_no_board_and_override(self):
        with self.assertRaisesRegex(arm.ArmError, 'found 0'):
            arm.find_arduino_port([], environ={})
        self.assertEqual(arm.find_arduino_port([], environ={'AISANA_ARM_PORT': '/dev/ttyACM3'}),
                         '/dev/ttyACM3')

    def test_disabled_link_never_opens_port(self):
        logs = []
        self.assertIsNone(arm.connect_from_environment(log=logs.append, environ={'AISANA_ARM': '0'}))
        self.assertIn('disabled', logs[0])


class ControllerTests(unittest.TestCase):
    def test_waits_for_ready_then_sends_newline_command(self):
        fake = FakeSerial()
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('position', 'both', 2)
            self.assertEqual(fake.written, [b'b 2\n'])
            self.assertIn('"b 2"', summary)
        finally:
            controller.close()

    def test_status_reply_is_collected(self):
        fake = FakeSerial()
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('status')
            self.assertIn('at position 2', summary)
        finally:
            controller.close()

    def test_not_ready_arduino_refuses_moves_but_allows_emergency_stop(self):
        fake = FakeSerial(boot_lines=(b'Starting both arms...\n',))
        controller = arm.ArmController('/dev/ttyACM0', serial_factory=lambda port: fake,
                                       ready_timeout=0.1, echo=False, log=lambda *a: None)
        controller.open()
        try:
            with self.assertRaisesRegex(arm.ArmError, 'did not report'):
                controller.execute('gripper_close', 'right')
            controller.emergency_stop()
            self.assertEqual(fake.written, [b'x\n'])
        finally:
            controller.close()


class VoiceParserTests(unittest.TestCase):
    def check(self, text, action, side='both', position=None, joint='whole'):
        self.assertEqual(arm.parse_arm_command(text),
                         {'action': action, 'arm': side, 'position': position, 'joint': joint},
                         text)

    def test_positions(self):
        self.check('подними руки в позицию 2', 'position', 'both', 2)
        self.check('правую руку во вторую позицию', 'position', 'right', 2)
        self.check('левая рука позиция четыре', 'position', 'left', 4)
        self.check('руки в исходное положение', 'position', 'both', 0)
        self.check('move the left arm to position 3', 'position', 'left', 3)
        self.check('right elbow to position 1', 'position', 'right', 1, 'elbow')
        self.check('левое плечо руки в позицию 1', 'position', 'left', 1, 'shoulder')

    def test_grippers_and_stop(self):
        self.check('открой правую руку', 'gripper_open', 'right')
        self.check('сожми захват на левой руке', 'gripper_close', 'left')
        self.check('close both grippers', 'gripper_close', 'both')
        self.check('останови руки', 'stop')
        self.check('stop the arms', 'stop')

    def test_not_arm_commands(self):
        for text in ('что у меня в руке?', 'what am I holding in my hand?',
                     'открой окно', 'какая погода в Астане?', 'қолымда не бар?',
                     'расскажи про руки человека', ''):
            with self.subTest(text=text):
                self.assertIsNone(arm.parse_arm_command(text))

    def test_spoken_replies_in_all_languages(self):
        intent = {'action': 'position', 'arm': 'both', 'position': 2, 'joint': 'whole'}
        self.assertEqual(arm.arm_reply(intent, 'Russian'), 'Двигаю обе руки в позицию 2.')
        self.assertIn('position 2', arm.arm_reply(intent, 'English'))
        for language in ('Kazakh', 'Egyptian Arabic'):
            self.assertTrue(arm.arm_reply(intent, language))
        grip = {'action': 'gripper_open', 'arm': 'left', 'position': None, 'joint': 'whole'}
        self.assertEqual(arm.arm_reply(grip, 'Russian'), 'Открываю захват на левой руке.')
        self.assertIn('не отвечает', arm.arm_reply(grip, 'Russian', error='x'))


if __name__ == '__main__':
    unittest.main()
