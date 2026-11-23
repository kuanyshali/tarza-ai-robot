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


class StatusSerial(FakeSerial):
    """Answers "p" with a right-stepper position: first before, then after."""

    def __init__(self, before, after, target):
        FakeSerial.__init__(self)
        self.before, self.after, self.target, self.status_calls = before, after, target, 0

    def write(self, data):
        self.written.append(data)
        if data.strip() == b'p':
            self.status_calls += 1
            value = self.before if self.status_calls % 2 == 1 else self.after
            with self.lock:
                self.incoming.append('RIGHT stepper: {:.2f}  target {:.2f}  [between positions]\n'.format(
                    value, self.target if self.status_calls % 2 == 0 else value).encode())


def moves(fake):
    """Commands sent, without the status ("p") reads used for verification."""
    return [data for data in fake.written if data != b'p\n']


def open_controller(fake):
    logs = []
    controller = arm.ArmController('/dev/ttyACM0', serial_factory=lambda port: fake,
                                   ready_timeout=2, echo=0, log=logs.append)
    controller.open()
    return controller, logs


def open_controller_with_echo(fake, echo):
    logs = []
    controller = arm.ArmController('/dev/ttyACM0', serial_factory=lambda port: fake,
                                   ready_timeout=2, echo=echo, log=logs.append)
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


class NudgeCommandTests(unittest.TestCase):
    def test_up_follows_each_joints_position_direction(self):
        # Shoulders count up toward position 4; the right elbow counts DOWN.
        self.assertEqual(arm.nudge_commands('right', 'shoulder', 'up', 5), ['r m r 5'])
        self.assertEqual(arm.nudge_commands('left', 'shoulder', 'down', 10), ['l m r -10'])
        # checked5: the right elbow has no pot, its positions are servo angles 80..184 (up = bigger).
        self.assertEqual(arm.nudge_commands('right', 'elbow', 'up', 5), ['r s r 5'])
        self.assertEqual(arm.nudge_commands('left', 'elbow', 'up', 5), ['l s r 5'])
        self.assertEqual(arm.nudge_commands('both', 'elbow', 'up', 10), ['r s r 10', 'l s r 10'])
        self.assertEqual(arm.nudge_commands('both', 'whole', 'up', 10), ['r m r 10', 'l m r 10'])

    def test_direction_can_be_flipped(self):
        self.assertEqual(arm.nudge_commands('right', 'shoulder', 'up', 5, up_toward_4=False), ['r m r -5'])

    def test_nudge_limits(self):
        for args in (('right', 'shoulder', 'up', 0), ('right', 'shoulder', 'up', 46),
                     ('right', 'shoulder', 'sideways', 5), ('right', 'wrist', 'up', 5),
                     ('right', 'shoulder', 'up', '5; x'), ('middle', 'shoulder', 'up', 5)):
            with self.subTest(args=args):
                with self.assertRaises(arm.ArmError):
                    arm.nudge_commands(*args)


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

    def test_real_jetson_ports_arduino_next_to_xiaomi(self):
        # From the Jetson: ls -l /dev/serial/by-id/ on 2026-11-19
        xiaomi = SimpleNamespace(device='/dev/ttyACM0', vid=0x2717,
                                 description='USB Composite Device', manufacturer='Xiaomi Communications Co., Ltd.')
        arduino = SimpleNamespace(device='/dev/ttyACM1', vid=0x2341,
                                  description='Arduino Mega 2560', manufacturer='Arduino (www.arduino.cc)')
        self.assertEqual(arm.find_arduino_port([xiaomi, arduino], environ={}), '/dev/ttyACM1')
        # Arduino unplugged: never fall back to the speaker's serial port.
        with self.assertRaisesRegex(arm.ArmError, 'found 0'):
            arm.find_arduino_port([xiaomi], environ={})
        no_ids = SimpleNamespace(device='/dev/ttyACM0', vid=None, description='USB Composite Device')
        with self.assertRaisesRegex(arm.ArmError, 'found 0'):
            arm.find_arduino_port([no_ids], environ={})

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
    def setUp(self):
        self.verify_seconds = arm.VERIFY_SECONDS
        arm.VERIFY_SECONDS = 0.05

    def tearDown(self):
        arm.VERIFY_SECONDS = self.verify_seconds

    def test_waits_for_ready_then_sends_newline_command(self):
        fake = FakeSerial()
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('position', 'both', 2)
            self.assertEqual(moves(fake), [b'b 2\n'])
            self.assertIn('"b 2"', summary)
        finally:
            controller.close()

    def test_nudge_sends_one_command_per_arm(self):
        fake = FakeSerial()
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('nudge', 'both', None, 'elbow', 'up', 5)
            self.assertEqual(moves(fake), [b'r s r 5\n', b'l s r 5\n'])
            self.assertIn('"r s r 5"', summary)
        finally:
            controller.close()

    def test_positions_table_from_arduino_replaces_default(self):
        fake = FakeSerial(boot_lines=(
            b'Both arms ready.\n',
            b'RIGHT stepper           \t167\t137\t107\t77\t47\tlimits 10-178\n'))
        controller, _ = open_controller(fake)
        try:
            time.sleep(0.1)
            self.assertEqual(controller.position_table[('right', 'shoulder')][0], 167.0)
            controller.execute('nudge', 'right', None, 'shoulder', 'up', 5)
            self.assertEqual(moves(fake), [b'r m r -5\n'])
        finally:
            controller.close()

    def test_console_shows_only_important_arduino_lines(self):
        fake = FakeSerial(boot_lines=(
            b'Both arms ready.\n', b'RIGHT stepper reached 77.48\n',
            b'LEFT STEPPER STOPPED: joint outside limits.\n', b'RIGHT elbow -> 129.00\n',
            b'Other:     p (status) | t (positions table) | x (EMERGENCY STOP) | h\n'))
        controller, logs = open_controller_with_echo(fake, 1)
        try:
            time.sleep(0.2)
            arduino = [line for line in logs if line.startswith('Arduino:')]
            self.assertEqual(arduino, ['Arduino: LEFT STEPPER STOPPED: joint outside limits.'])
        finally:
            controller.close()

    def test_verification_reports_real_motion(self):
        fake = StatusSerial(before=47.0, after=47.2, target=77.0)
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('position', 'right', 1, 'shoulder')
            self.assertIn('right shoulder: 47 -> 47, target 77, DID NOT MOVE', summary)
            self.assertIn('never claim it moved', summary)
            fake.after = 63.0
            summary = controller.execute('position', 'right', 1, 'shoulder')
            self.assertIn('still moving', summary)
        finally:
            controller.close()

    def test_watch_reports_only_real_problems(self):
        fake = StatusSerial(before=47.0, after=47.2, target=77.0)
        controller, _ = open_controller(fake)
        try:
            reports = []
            intents = arm.parse_arm_commands('правое плечо в позицию 1')
            controller.watch(intents, reports.append, first=0.05, second=0.4).join(2)
            self.assertIn('right shoulder is not moving', reports[0][0])
            reports[:] = []
            fake.after = 63.0                        # moving: stay silent
            controller.watch(intents, reports.append, first=0.05, second=0.4).join(2)
            self.assertEqual(reports, [])
            fake.after = 47.2                        # user said stop meanwhile: silent
            thread = controller.watch(intents, reports.append, first=0.05, second=0.4)
            controller.emergency_stop()
            thread.join(2)
            self.assertEqual(reports, [])
        finally:
            controller.close()

    def test_status_reply_is_collected(self):
        fake = FakeSerial()
        controller, _ = open_controller(fake)
        try:
            summary = controller.execute('status')
            self.assertIn('right shoulder: 107 (target 107) [at position 2]', summary)
        finally:
            controller.close()

    def test_not_ready_arduino_refuses_moves_but_allows_emergency_stop(self):
        fake = FakeSerial(boot_lines=(b'Starting both arms...\n',))
        controller = arm.ArmController('/dev/ttyACM0', serial_factory=lambda port: fake,
                                       ready_timeout=0.1, echo=False, log=lambda *a: None)
        controller.open()
        try:
            with self.assertRaisesRegex(arm.ArmError, 'did not report'):
                controller.execute('gripper_close', 'left')
            controller.emergency_stop()
            self.assertEqual(fake.written, [b'x\n'])
        finally:
            controller.close()


class VoiceParserTests(unittest.TestCase):
    def check(self, text, action, side='both', position=None, joint='whole',
              direction=None, degrees=None):
        self.assertEqual(arm.parse_arm_command(text),
                         {'action': action, 'arm': side, 'position': position, 'joint': joint,
                          'direction': direction, 'degrees': degrees},
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

    def test_relative_moves(self):
        self.check('айсана рука чуть-чуть наверх', 'nudge', 'both', None, 'shoulder', 'up', 5)
        self.check('подними правую руку немного', 'nudge', 'right', None, 'shoulder', 'up', 10)
        self.check('опусти левую руку', 'nudge', 'left', None, 'shoulder', 'down', 15)
        self.check('левый локоть чуть ниже', 'nudge', 'left', None, 'elbow', 'down', 5)
        self.check('правую руку выше на 20 градусов', 'nudge', 'right', None, 'shoulder', 'up', 20)
        self.check('raise the right arm a lot', 'nudge', 'right', None, 'shoulder', 'up', 30)
        self.check('руки вверх на 90 градусов', 'nudge', 'both', None, 'shoulder', 'up', 45)
        # A position number wins over "подними".
        self.check('подними правую руку, позиция один', 'position', 'right', 1)

    def test_several_commands_in_one_utterance(self):
        intents = arm.parse_arm_commands('Айсана, подвигай плечом. Плечо правое в позицию 1. '
                                         'Позиция 4. Айсана, правое плечо в позицию 2.')
        self.assertEqual([(i['arm'], i['joint'], i['position']) for i in intents],
                         [('right', 'shoulder', 1), ('right', 'shoulder', 4), ('right', 'shoulder', 2)])
        intents = arm.parse_arm_commands('правую руку в позицию 1, потом левую в позицию тройку')
        self.assertEqual([(i['arm'], i['position']) for i in intents], [('right', 1), ('left', 3)])
        self.assertEqual(arm.parse_arm_commands('правую в позицию 2. Стоп!'),
                         [arm.make_intent('stop')])
        self.assertEqual(arm.parse_arm_commands('какая погода?'), [])
        intents = arm.parse_arm_commands('правая рука позиция один, левая рука позиция два')
        self.assertEqual([(i['arm'], i['position']) for i in intents], [('right', 1), ('left', 2)])
        intents = arm.parse_arm_commands('подними правую руку, позиция один')
        self.assertEqual([(i['action'], i['position']) for i in intents], [('position', 1)])
        intents = arm.parse_arm_commands('открой правый захват и закрой левый захват')
        self.assertEqual([(i['action'], i['arm']) for i in intents],
                         [('gripper_open', 'right'), ('gripper_close', 'left')])

    def test_not_arm_commands(self):
        for text in ('что у меня в руке?', 'what am I holding in my hand?',
                     'открой окно', 'какая погода в Астане?', 'қолымда не бар?',
                     'расскажи про руки человека', ''):
            with self.subTest(text=text):
                self.assertIsNone(arm.parse_arm_command(text))

    def test_nudge_reply(self):
        intent = arm.parse_arm_command('подними правую руку чуть-чуть')
        self.assertEqual(arm.arm_reply(intent, 'Russian'), 'Поднимаю правую руку на 5°.')
        self.assertIn('Lowering', arm.arm_reply(dict(intent, direction='down'), 'English'))

    def test_spoken_replies_in_all_languages(self):
        intent = {'action': 'position', 'arm': 'both', 'position': 2, 'joint': 'whole'}
        self.assertEqual(arm.arm_reply(intent, 'Russian'), 'Двигаю обе руки в позицию 2.')
        self.assertIn('position 2', arm.arm_reply(intent, 'English'))
        for language in ('Kazakh', 'Egyptian Arabic'):
            self.assertTrue(arm.arm_reply(intent, language))
        grip = {'action': 'gripper_open', 'arm': 'left', 'position': None, 'joint': 'whole'}
        self.assertEqual(arm.arm_reply(grip, 'Russian'), 'Открываю захват на левой руке.')
        self.assertIn('не отвечает', arm.arm_reply(grip, 'Russian', error='x'))



class Checked5FirmwareTests(unittest.TestCase):
    """robotic_arms_checked5.ino (Nov 23): right elbow without pot, right HAND instead of gripper."""

    BOOT = (b'Starting both arms...\n',
            b'RIGHT elbow (no pot): starting at REST (down) = 80.00 deg, holding.\n',
            b'Both arms ready.\n',
            b'POSITIONS (pot readings)  zero\t1\t2\t3\t4\n',
            b'RIGHT stepper           \t47\t77\t107\t137\t167\tlimits 10-178\n',
            b'RIGHT elbow (servo deg) \t80\t106\t132\t158\t184\tlimits 0-180\n',
            b'LEFT  stepper           \t52\t82\t112\t142\t172\tlimits 13-184\n',
            b'LEFT  elbow             \t144\t174\t204\t234\t264\tlimits 91-280\n',
            b'RIGHT hand:    not started yet (send g s / g n)\n',
            b'LEFT gripper: not started yet (send g o / g c)\n')

    def open(self):
        fake = FakeSerial(boot_lines=self.BOOT)
        controller, _ = open_controller(fake)
        time.sleep(0.3)                      # boot lines read
        return fake, controller

    def test_status_of_elbow_without_pot(self):
        status = arm.parse_status(['RIGHT elbow:   servo 106.00 deg (no pot)  [at position 1]'])
        self.assertEqual(status[('right', 'elbow')][:2], (106.0, None))

    def test_table_and_hand_read_from_the_boot_text(self):
        fake, controller = self.open()
        try:
            self.assertEqual(controller.position_table[('right', 'elbow')], (80, 106, 132, 158, 184))
            self.assertEqual(controller.hands, {'right': True, 'left': False})
        finally:
            controller.close()

    def test_hand_shakes_and_right_gripper_is_not_sent(self):
        fake, controller = self.open()
        try:
            controller.execute_many([arm.make_intent('hand_shake', 'right')], verify=False)
            self.assertEqual(moves(fake), [b'r g s\n'])
            summary = controller.execute_many([arm.make_intent('gripper_open', 'right')], verify=False)
            self.assertIn('has a hand', summary)
            self.assertEqual(moves(fake), [b'r g s\n'])                # nothing new sent
            controller.execute_many([arm.make_intent('gripper_close', 'both')], verify=False)
            self.assertEqual(moves(fake)[-1], b'l g c\n')             # only the arm with a gripper
        finally:
            controller.close()

    def test_voice_commands_for_the_hand(self):
        self.assertEqual(arm.parse_arm_commands('Айсана, помаши рукой')[0]['action'], 'hand_shake')
        self.assertEqual(arm.parse_arm_commands('Айсана, пожми руку')[0]['action'], 'hand_shake')
        self.assertEqual(arm.parse_arm_commands('Айсана, хватит махать рукой')[0]['action'], 'hand_still')
        self.assertEqual(arm.parse_arm_commands('Стоп, руки!')[0]['action'], 'stop')
        self.assertEqual(arm.build_command('hand_shake'), 'r g s')
        # Jetson log, Nov 23: a repeated phrase sent "r g s" twice.
        self.assertEqual(len(arm.parse_arm_commands('Aisana, wave the hand, ah, wave the hand on the right arm.')), 1)
        self.assertEqual(arm.build_command('hand_still'), 'r g n')

if __name__ == '__main__':
    unittest.main()
