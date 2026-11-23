"""USB serial link from Aisana (Jetson) to the Arm team's Arduino Mega.

The Arduino runs the Arm team's firmware (robotic_arms_*.ino, 115200 baud,
newline-terminated text commands such as "b 1", "l g o", "x", "p").
This module only ever sends commands built from a fixed whitelist, never
free text from speech recognition or from the language model.

Python 3.6 compatible. Needs pyserial on the Jetson:
    python3 -m pip install --user "pyserial==3.5"
and the user must be in the dialout group (sudo usermod -aG dialout $USER).

Environment:
    AISANA_ARM=0             disable the arm link completely (default 1)
    AISANA_ARM_PORT=/dev/... use this serial port instead of auto-detection
    AISANA_ARM_ECHO=1        print only important Arduino lines (default);
                             0 = nothing, 2 = everything (debugging)
    AISANA_ARM_UP=position4  "up" means toward position 4 (default);
                             set position0 if the arms move the wrong way
"""
import collections
import os
import re
import threading
import time

BAUD = 115200
READY_TEXT = 'Both arms ready'
ARM_LETTERS = {'right': 'r', 'left': 'l', 'both': 'b'}
JOINT_LETTERS = {'whole': '', 'shoulder': 'm', 'elbow': 's'}
ACTIONS = ('position', 'nudge', 'gripper_open', 'gripper_close', 'hand_shake', 'hand_still', 'stop', 'status')
MAX_NUDGE_DEGREES = 45
# "Помаши рукой": the right arm is raised first (shoulder 3, elbow 4), then the hand waves.
# AISANA_WAVE_POSE="3,4" (shoulder,elbow); "-" = do not raise.
def _wave_pose(text):
    try:
        shoulder, elbow = [int(v) for v in text.split(',')]
        if 0 <= shoulder <= 4 and 0 <= elbow <= 4:
            return shoulder, elbow
    except ValueError:
        pass
    return None


WAVE_POSE = _wave_pose(os.environ.get('AISANA_WAVE_POSE', '3,4'))
VERIFY_SECONDS = float(os.environ.get('AISANA_ARM_VERIFY_SECONDS', '1.0'))
# Positions 0..4 from robotic_arms_checked5.ino (Nov 23). Replaced at runtime by
# the table the Arduino prints at start-up ("POSITIONS (pot readings)").
# The right elbow has no pot now: its positions are servo angles (rest 80 = down,
# +26 per position, 184 is clamped to 180 by the firmware).
DEFAULT_POSITION_TABLE = {
    ('right', 'shoulder'): (47, 77, 107, 137, 167),
    ('right', 'elbow'): (80, 106, 132, 158, 184),
    ('left', 'shoulder'): (52, 82, 112, 142, 172),
    ('left', 'elbow'): (144, 174, 204, 234, 264),
}
TABLE_LINE = re.compile(
    r'^(RIGHT|LEFT)\s+(stepper|elbow)\s+(?:\(servo deg\)\s+)?((?:\S+\s+){5})limits', re.I)
# Arduino lines worth showing at AISANA_ARM_ECHO=1 (errors and safety only).
IMPORTANT_LINE = re.compile(
    r'STOPPED|EMERGENCY|limit is|nearest limit|limit reached|not set|feedback is OFF|'
    r'can\'t|cannot|calibrat|wrong way|outside|!!|doesn\'t match', re.I)
# Firmware lines that mean a joint really failed (EMERGENCY STOP is user-requested).
SERIOUS_LINE = re.compile(r'STEPPER STOPPED|wrong way|outside limits|feedback is OFF|is not set', re.I)
STOPPED_AT = re.compile(r'stopped at (-?[\d.]+) \(target (-?[\d.]+)\)', re.I)
# Lines of the firmware's help text ("h"), printed at every start.
HELP_LINE = re.compile(r'^(?:-{5,}|Arm:|Whole arm:|Stepper:|Jog:|Elbow:|Gripper:|Hand:|Right elbow:|Other:)', re.I)
# "p" status lines, e.g. "RIGHT stepper: 42.01  target 42.01  [between positions]"
STATUS_LINE = re.compile(
    r'^(RIGHT|LEFT)\s+(stepper|elbow):\s+(-?[\d.]+)\s+target\s+(-?[\d.]+)\s*(.*)$', re.I)
# Elbow without a pot (right arm, checked5): "RIGHT elbow:   servo 80.00 deg (no pot)  [at position 0]"
NO_POT_LINE = re.compile(r'^(RIGHT|LEFT)\s+elbow:\s+servo\s+(-?[\d.]+)\s+deg\s+\(no pot\)\s*(.*)$', re.I)
# "RIGHT hand:    SHAKING" = a shaking hand instead of a gripper on that arm.
HAND_LINE = re.compile(r'^(RIGHT|LEFT)\s+(hand|gripper):', re.I)
JOINT_TITLES = {('right', 'shoulder'): 'right shoulder', ('right', 'elbow'): 'right elbow',
                ('left', 'shoulder'): 'left shoulder', ('left', 'elbow'): 'left elbow'}


def parse_status(lines):
    """{(side, joint): (now, target, note)} from the firmware's "p" reply."""
    result = {}
    for line in lines:
        match = STATUS_LINE.match(line.strip())
        if match:
            joint = 'shoulder' if match.group(2).lower() == 'stepper' else 'elbow'
            result[(match.group(1).lower(), joint)] = (
                float(match.group(3)), float(match.group(4)), match.group(5).strip())
            continue
        match = NO_POT_LINE.match(line.strip())
        if match:                      # target None: moves by timing, no feedback
            result[(match.group(1).lower(), 'elbow')] = (float(match.group(2)), None,
                                                          (match.group(3).strip() + ' no pot').strip())
    return result


def affected_joints(action, arm='both', joint='whole'):
    if action == 'hand_shake' and WAVE_POSE:
        return [('right', 'shoulder')]   # raised for waving; the elbow has no sensor
    if action not in ('position', 'nudge'):
        return []
    sides = ('right', 'left') if arm == 'both' else (arm,)
    if action == 'nudge' and joint in (None, '', 'whole'):
        joint = 'shoulder'
    joints = ('shoulder', 'elbow') if joint in (None, '', 'whole') else (joint,)
    return [(side, name) for side in sides for name in joints]
# USB vendor IDs: genuine Arduino boards, then common clone USB-serial chips.
GENUINE_ARDUINO_VIDS = (0x2341, 0x2A03)
CLONE_SERIAL_VIDS = (0x1A86, 0x0403, 0x10C4)
ARDUINO_VIDS = GENUINE_ARDUINO_VIDS + CLONE_SERIAL_VIDS


class ArmError(RuntimeError):
    """The arm link is missing, not ready, or a command is not allowed."""


def build_command(action, arm='both', position=None, joint='whole'):
    """Translate a validated intent into one firmware command line."""
    if action == 'stop':
        return 'x'                       # firmware EMERGENCY STOP for everything
    if action == 'status':
        return 'p'
    letter = ARM_LETTERS.get(arm)
    if letter is None:
        raise ArmError('Unknown arm: {!r}'.format(arm))
    if action == 'position':
        try:
            number = int(position)
        except (TypeError, ValueError):
            raise ArmError('Position must be 0..4, got {!r}'.format(position))
        if number < 0 or number > 4 or str(number) != str(position).strip():
            raise ArmError('Position must be 0..4, got {!r}'.format(position))
        joint_letter = JOINT_LETTERS.get(joint or 'whole')
        if joint_letter is None:
            raise ArmError('Unknown joint: {!r}'.format(joint))
        return ' '.join(part for part in (letter, joint_letter, str(number)) if part)
    if action in ('gripper_open', 'gripper_close'):
        return letter + (' g o' if action == 'gripper_open' else ' g c')
    if action in ('hand_shake', 'hand_still'):
        return 'r g ' + ('s' if action == 'hand_shake' else 'n')   # only the right arm has the hand
    raise ArmError('Unknown arm action: {!r}'.format(action))


def _is_known_non_arduino(port):
    text = ' '.join(str(getattr(port, name, '') or '') for name in
                    ('description', 'manufacturer', 'product', 'hwid'))
    return bool(re.search(r'xiaomi|composite device|KT USB Audio', text, re.I))


def _describe(port):
    vid = getattr(port, 'vid', None)
    return '{} ({}{})'.format(
        port.device, getattr(port, 'description', None) or 'unknown device',
        ', VID {:04x}'.format(vid) if vid is not None else '')


def nudge_commands(arm, joint, direction, degrees, table=None, up_toward_4=True):
    """Relative move of one joint: "up" = toward position 4 of that joint.

    The firmware's "m r <deg>" / "s r <deg>" move a joint relative to its
    target in pot degrees and clamp to its limits. Elbows are mirrored
    (right counts down, left up), so each arm gets its own sign.
    """
    table = table or DEFAULT_POSITION_TABLE
    if direction not in ('up', 'down'):
        raise ArmError('Direction must be up or down, got {!r}'.format(direction))
    joint = 'shoulder' if joint in (None, '', 'whole') else joint
    if joint not in ('shoulder', 'elbow'):
        raise ArmError('Unknown joint: {!r}'.format(joint))
    try:
        amount = int(degrees)
    except (TypeError, ValueError):
        raise ArmError('Degrees must be 1..{}, got {!r}'.format(MAX_NUDGE_DEGREES, degrees))
    if amount < 1 or amount > MAX_NUDGE_DEGREES or str(amount) != str(degrees).strip():
        raise ArmError('Degrees must be 1..{}, got {!r}'.format(MAX_NUDGE_DEGREES, degrees))
    if arm == 'both':
        sides = ('right', 'left')
    elif arm in ('right', 'left'):
        sides = (arm,)
    else:
        raise ArmError('Unknown arm: {!r}'.format(arm))
    commands = []
    for side in sides:
        positions = table[(side, joint)]
        sign = 1 if positions[4] > positions[0] else -1
        if not up_toward_4:
            sign = -sign
        if direction == 'down':
            sign = -sign
        commands.append('{} {} r {}'.format(ARM_LETTERS[side], 'm' if joint == 'shoulder' else 's',
                                            sign * amount))
    return commands


def find_arduino_port(ports=None, environ=None):
    """Return exactly one Arduino serial port; never guess between equals.

    Preference: genuine Arduino USB IDs, then common clone USB-serial chips,
    then any /dev/ttyACM*. Several ports in the best tier is an error.
    """
    environ = os.environ if environ is None else environ
    override = environ.get('AISANA_ARM_PORT', '').strip()
    if override:
        return override
    if ports is None:
        from serial.tools import list_ports
        ports = list_ports.comports()
    ports = list(ports)
    tiers = (
        [p for p in ports if getattr(p, 'vid', None) in GENUINE_ARDUINO_VIDS],
        [p for p in ports if getattr(p, 'vid', None) in CLONE_SERIAL_VIDS],
        # Last resort: a ttyACM port that reports no USB IDs at all. Ports of
        # other known devices (the Xiaomi speaker exposes /dev/ttyACM0!) never count.
        [p for p in ports if 'ttyACM' in (getattr(p, 'device', '') or '')
         and getattr(p, 'vid', None) is None and not _is_known_non_arduino(p)],
    )
    for tier in tiers:
        devices = sorted(set(p.device for p in tier))
        if len(devices) == 1:
            return devices[0]
        if len(devices) > 1:
            raise ArmError(
                'Expected one Arduino serial port, found {}: {}. '
                'Set AISANA_ARM_PORT to the right one (see ls -l /dev/serial/by-id/).'.format(
                    len(devices), '; '.join(_describe(p) for p in tier)))
    raise ArmError('Expected one Arduino serial port, found 0. Plug in the Arduino Mega '
                   'or set AISANA_ARM_PORT=/dev/ttyACM0.')


class ArmController(object):
    """Owns the serial port; a background thread reads Arduino output."""

    def __init__(self, port, serial_factory=None, ready_timeout=15.0,
                 echo=1, log=print, up_toward_4=True):
        self.port = port
        self._factory = serial_factory
        self._serial = None
        self._write_lock = threading.Lock()
        self._lines = collections.deque(maxlen=300)
        self._lines_lock = threading.Lock()
        self._sequence = 0
        self._stops = 0
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._reader = None
        self.ready_timeout = ready_timeout
        self.echo = int(echo)
        self.up_toward_4 = up_toward_4
        self.position_table = dict(DEFAULT_POSITION_TABLE)
        # checked5: a shaking hand on the right arm instead of a gripper. Updated from
        # the firmware's status ("RIGHT hand:" / "RIGHT gripper:").
        self.hands = {'right': True, 'left': False}
        self.log = log

    def open(self):
        factory = self._factory
        if factory is None:
            import serial

            def factory(port):
                return serial.Serial(port, BAUD, timeout=0.2, write_timeout=1)
        self._serial = factory(self.port)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Opening the USB port restarts an Arduino Mega (DTR). The firmware is
        # designed for that: elbows start from saved calibration, steppers hold.
        self.log('Arm: connected to {}; waiting for "{}"...'.format(self.port, READY_TEXT))
        return self

    def close(self):
        self._stop.set()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

    def _read_loop(self):
        pending = b''
        while not self._stop.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception as error:
                if not self._stop.is_set():
                    self.log('Arm: serial read failed: {}'.format(error))
                self.ready.clear()
                return
            if not chunk:
                continue
            pending += chunk
            while b'\n' in pending:
                raw, pending = pending.split(b'\n', 1)
                self._remember(raw.decode('utf-8', 'replace').strip())

    def _remember(self, line):
        if not line:
            return
        with self._lines_lock:
            self._sequence += 1
            self._lines.append((self._sequence, line))
        table = TABLE_LINE.match(line)
        if table:
            values = table.group(3).split()
            joint = 'shoulder' if table.group(2).lower() == 'stepper' else 'elbow'
            try:
                if '-' not in (values[0], values[4]):
                    self.position_table[(table.group(1).lower(), joint)] = tuple(float(v) for v in values)
            except ValueError:
                pass
        if self.echo >= 2 or (self.echo == 1 and IMPORTANT_LINE.search(line)
                               and not HELP_LINE.match(line)):
            self.log('Arduino: ' + line)
        hand = HAND_LINE.match(line)
        if hand:
            self.hands[hand.group(1).lower()] = hand.group(2).lower() == 'hand'
        if READY_TEXT in line and not self.ready.is_set():
            self.ready.set()
            self.log('Arm: ready.')

    def _lines_after(self, sequence):
        with self._lines_lock:
            return [line for number, line in self._lines if number > sequence]

    def send(self, command, collect_seconds=0.0):
        """Write one whitelisted command; optionally collect the reply lines."""
        if self._serial is None:
            raise ArmError('Arm is not connected')
        if command != 'x' and not self.ready.wait(self.ready_timeout):
            raise ArmError('Arduino did not report "{}"'.format(READY_TEXT))
        with self._lines_lock:
            start = self._sequence
        if command == 'x':
            self._stops += 1
        with self._write_lock:
            self._serial.write((command + '\n').encode('ascii'))
            self._serial.flush()
        if self.echo >= 2:
            self.log('Arm command: ' + command)
        if collect_seconds > 0:
            time.sleep(collect_seconds)
        return self._lines_after(start)

    def execute(self, action, arm='both', position=None, joint='whole',
                direction=None, degrees=None, verify=True):
        """Run one intent; see execute_many."""
        return self.execute_many([_intent(action, arm, position, joint, direction, degrees)], verify)

    def execute_many(self, intents, verify=True):
        """Run intents in order and report what the joints REALLY did.

        With verify, the firmware status ("p") is read once before and once
        ~1 s after the moves, so the summary says "47 -> 61, target 77, still
        moving" or "DID NOT MOVE" instead of assuming success.
        """
        commands, joints, notes = [], [], []
        for intent in intents:
            if intent['action'] in ('gripper_open', 'gripper_close') and self.hands.get(intent['arm']):
                notes.append('the {} arm has a hand, not a gripper: it cannot open or close '
                             '(it can shake: "помаши рукой")'.format(intent['arm']))
                continue
            if (intent['action'] in ('gripper_open', 'gripper_close') and intent['arm'] == 'both'
                    and any(self.hands.values())):
                side = 'left' if self.hands.get('right') else 'right'
                commands.append(build_command(intent['action'], side))   # only the arm with a gripper
                continue
            if intent['action'] == 'hand_shake' and WAVE_POSE and self.hands.get('right'):
                commands += ['r m {}'.format(WAVE_POSE[0]), 'r s {}'.format(WAVE_POSE[1])]   # raise, then wave
            if intent['action'] == 'nudge':
                commands += nudge_commands(intent['arm'], intent['joint'], intent['direction'],
                                           intent['degrees'], self.position_table, self.up_toward_4)
            else:
                commands.append(build_command(intent['action'], intent['arm'],
                                              intent['position'], intent['joint']))
            for key in affected_joints(intent['action'], intent['arm'], intent['joint']):
                if key not in joints:
                    joints.append(key)
        if not verify or 'x' in commands:
            joints = []
        if not commands:
            return 'Nothing sent: ' + '; '.join(notes) + '. Tell the user in a few words.'
        status_only = commands == ['p']
        before = parse_status(self.send('p', collect_seconds=0.3)) if joints else {}
        lines = []
        for command in commands:
            lines.extend(self.send(command, collect_seconds=0.6 if command == 'p' else 0.15))
        problems = [line for line in lines if IMPORTANT_LINE.search(line)]
        summary = 'Sent {} to the Arduino.'.format(', '.join('"{}"'.format(c) for c in commands))
        if status_only:
            status = parse_status(lines)
            summary += (' ' + '; '.join(
                '{}: {:.0f} (target {:.0f}){}'.format(JOINT_TITLES[key], now, target,
                                                      ' ' + note if note else '')
                for key, (now, target, note) in sorted(status.items()))) if status else ' No status reply yet.'
        if joints:
            time.sleep(VERIFY_SECONDS)
            reply = self.send('p', collect_seconds=0.4)
            after = parse_status(reply)
            problems += [line for line in reply if IMPORTANT_LINE.search(line)]
            report = []
            for key in joints:
                if key not in after:
                    report.append('{}: no feedback'.format(JOINT_TITLES[key]))
                    continue
                now, target, _note = after[key]
                start_value = before.get(key, (None,))[0]
                moved = start_value is not None and abs(now - start_value) >= 1.5
                if target is None:          # no pot: the servo angle it is sent to, no feedback
                    report.append('{}: servo at {:.0f} deg (no sensor, moves by timing)'.format(
                        JOINT_TITLES[key], now))
                    continue
                if abs(target - now) <= 3:
                    state = 'reached' if moved or start_value is None else 'already there'
                elif moved:
                    state = 'still moving'
                else:
                    state = 'DID NOT MOVE'
                report.append('{}: {} -> {:.0f}, target {:.0f}, {}'.format(
                    JOINT_TITLES[key], '?' if start_value is None else '{:.0f}'.format(start_value),
                    now, target, state))
            summary += ' Feedback: ' + '; '.join(report) + '.'
            if any('DID NOT MOVE' in item for item in report):
                summary += (' Tell the user honestly which joint did not move '
                            '(check motor power, limit or load); never claim it moved.')
        if notes:
            summary += ' Not sent: ' + '; '.join(notes) + '.'
        if problems:
            summary += ' Arduino reported: ' + ' | '.join(sorted(set(problems))[-3:])
        return summary

    def watch(self, intents, report, first=0.8, second=2.5):
        """Check the sensors in the background after a move; call report(problems)
        only if something really went wrong. Nothing is said when all is fine."""
        joints = []
        for intent in intents:
            for key in affected_joints(intent['action'], intent['arm'], intent['joint']):
                if key not in joints:
                    joints.append(key)
        if not joints:
            return None
        with self._lines_lock:
            start = self._sequence
        stops = self._stops

        def worker():
            try:
                time.sleep(first)
                early = parse_status(self.send('p', collect_seconds=0.3))
                time.sleep(max(0.0, second - first - 0.3))
                late = parse_status(self.send('p', collect_seconds=0.3))
            except Exception as error:
                self.log('Arm: check failed: {}'.format(error))
                return
            if self._stops != stops:
                return                      # user said "стоп": holding is expected
            problems = []
            for line in self._lines_after(start):
                stopped = STOPPED_AT.search(line)
                if SERIOUS_LINE.search(line) or (
                        stopped and abs(float(stopped.group(1)) - float(stopped.group(2))) > 8):
                    if line not in problems:
                        problems.append(line)
            for key in joints:
                if key in early and key in late and late[key][1] is not None:
                    now, target, _note = late[key]
                    if abs(target - now) > 4 and abs(now - early[key][0]) < 1.0:
                        problems.append('{} is not moving: at {:.0f}, target {:.0f}'.format(
                            JOINT_TITLES[key], now, target))
            if problems:
                report(problems[:3])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def emergency_stop(self):
        try:
            self.send('x')
        except Exception as error:
            self.log('Arm: emergency stop could not be sent: {}'.format(error))


def connect_from_environment(log=print, environ=None):
    """Open the arm link if enabled and present; never stop the robot if not."""
    environ = os.environ if environ is None else environ
    if environ.get('AISANA_ARM', '1') != '1':
        log('Arm: disabled (AISANA_ARM=0).')
        return None
    try:
        port = find_arduino_port(environ=environ)
        try:
            echo = int(environ.get('AISANA_ARM_ECHO', '1'))
        except ValueError:
            echo = 1
        controller = ArmController(port, echo=echo, log=log,
                                   up_toward_4=environ.get('AISANA_ARM_UP', 'position4') != 'position0')
        return controller.open()
    except ImportError:
        log('Arm: pyserial is missing; run python3 -m pip install --user "pyserial==3.5".')
    except Exception as error:
        hint = ''
        if 'busy' in str(error).lower() or 'Errno 16' in str(error):
            hint = (' The port is used by another program: close the Arduino IDE Serial Monitor '
                    'or the other running Aisana (see: fuser -v {}).'.format(environ.get('AISANA_ARM_PORT') or '/dev/ttyACM*'))
        log('Arm: not connected ({}).{} Voice and camera keep working.'.format(error, hint))
    return None


# ---------------------------------------------------------------------------
# Local (no API call) parsing of explicit spoken arm commands, used by robot.py.
# A command needs an arm word AND an action, so "что у меня в руке?" is not one.
ARM_WORDS = re.compile(
    r'(?:рук[аиуеой]\w*|клешн\w*|захват\w*|кист[ьи]\w*|манипулятор\w*|'
    r'\barms?\b|\bgrippers?\b|\bhands?\b|\bclaws?\b|\bқол\w*|ذراع\w*|إيد\w*|ايد\w*|'
    r'локт\w*|локоть|плеч[оиа]\w*|\belbows?\b|\bshoulders?\b)', re.I)
NUMBER_WORDS = (
    (0, r'0|ноль|нуль|нулев\w*|zero|нөл\w*|صفر'),
    (1, r'1|один|одн[уа]|перв\w*|единиц\w*|one|first|бір\w*|واحد'),
    (2, r'2|два|две|втор\w*|двойк\w*|two|second|екі\w*|اتنين|اثنين'),
    (3, r'3|три|трет\w*|тройк\w*|three|third|үш\w*|تلاتة|ثلاثة'),
    (4, r'4|четыре|четв[её]рт\w*|четв[её]рк\w*|four|fourth|төрт\w*|اربعة|أربعة'),
)
POSITION_WORD = r'(?:позици\w*|положени\w*|positions?|pos|позиция\w*|қалып\w*|وضع\w*)'
HOME_WORDS = re.compile(
    r'(?:исходн\w*|в ноль|на ноль|домой|\bhome\b|\breset\b|бастапқы\w*)', re.I)
STOP_WORDS = re.compile(r'(?:\bстоп\b|останов\w*|замри|\bstop\b|\bhalt\b|freeze|тоқта\w*|وقف)', re.I)
OPEN_WORDS = re.compile(
    r'(?:открой|открыть|раскрой|разожми|разжать|отпусти|\bopen\b|\brelease\b|\bаш\b|افتح)', re.I)
CLOSE_WORDS = re.compile(
    r'(?:закрой|закрыть|сожми|сжать|схвати|возьми|\bclose\b|\bgrab\b|\bgrip\b|\bжап\b|اقفل|امسك)', re.I)
RIGHT_WORDS = re.compile(r'(?:прав\w*|\bright\b|\bоң\w*|يمين|اليمين)', re.I)
LEFT_WORDS = re.compile(r'(?:лев\w*|\bleft\b|\bсол\b|شمال|يسار|اليسار)', re.I)
SHOULDER_WORDS = re.compile(r'(?:плеч\w*|shoulder|stepper|шаговик\w*|иық\w*)', re.I)
ELBOW_WORDS = re.compile(r'(?:локт\w*|локоть|elbow|шынтақ\w*|كوع)', re.I)
UP_WORDS = re.compile(
    r'(?:наверх|вверх|выше|подними|подыми|приподними|подняй|\bup\b|\braise\b|\bhigher\b|'
    r'\blift\b|жоғары|көтер\w*|فوق|ارفع)', re.I)
DOWN_WORDS = re.compile(
    r'(?:вниз|ниже|опусти|\bdown\b|\blower\b|төмен|түсір\w*|تحت|نزل)', re.I)
DEGREES = re.compile(r'(\d{1,2})\s*(?:градус\w*|°|degrees?|deg\b|درج\w*)', re.I)
TINY = re.compile(r'(?:чуть|слегка|капельку|чуточку|slightly|a little|a tiny|tiny|кішкене|сәл|شوية صغيرين)', re.I)
SMALL = re.compile(r'(?:немного|немножко|a bit|bit\b|аздап|شوية)', re.I)
LARGE = re.compile(r'(?:сильно|намного|гораздо|максимально|a lot|much|far|қатты|كتير)', re.I)


def nudge_amount(text):
    """чуть-чуть = 5, немного = 10, no amount = 15, сильно = 30; explicit degrees win."""
    explicit = DEGREES.search(text)
    if explicit:
        return max(1, min(MAX_NUDGE_DEGREES, int(explicit.group(1))))
    if TINY.search(text):
        return 5
    if SMALL.search(text):
        return 10
    if LARGE.search(text):
        return 30
    return 15


def _position_number(text):
    for number, pattern in NUMBER_WORDS:
        if re.search(POSITION_WORD + r'\s*(?:номер\s*|number\s*|№\s*)?(?:' + pattern + r')\b', text, re.I):
            return number
        if re.search(r'\b(?:' + pattern + r')\s*' + POSITION_WORD, text, re.I):
            return number
    if HOME_WORDS.search(text):
        return 0
    return None


def _intent(action, arm='both', position=None, joint='whole', direction=None, degrees=None):
    return {'action': action, 'arm': arm, 'position': position, 'joint': joint,
            'direction': direction, 'degrees': degrees}


CLAUSE_SPLIT = re.compile(r'[.!?;]+|\b(?:потом|затем|после этого|and then|then|кейін)\b', re.I)
ACTION_CUE = re.compile(POSITION_WORD + r'|исходн|в ноль|\bhome\b', re.I)
_CONTEXT_WORDS = {'right': 'правая', 'left': 'левая', 'both': 'обе'}
_JOINT_WORDS = {'shoulder': 'плечо', 'elbow': 'локоть', 'whole': 'рука'}


PART_SPLIT = re.compile(r',|\s(?:и|а|and|және)\s', re.I)


def _parse_clauses(clauses, previous):
    """Parse clauses in order; context (arm, joint) carries over from the last one."""
    intents = list(previous)
    for clause in clauses:
        clause = (clause or '').strip(' ,')
        if not clause:
            continue
        intent = parse_arm_command(clause)
        names_side = RIGHT_WORDS.search(clause) or LEFT_WORDS.search(clause)
        if intent is None and names_side and ACTION_CUE.search(clause):
            intent = parse_arm_command('рука ' + clause)      # "правую в позицию 2"
        if intent is None and intents and STOP_WORDS.search(clause) and len(clause.split()) <= 3:
            intent = _intent('stop')                         # "... Стоп!" after a move
        if intent is None and intents and (ACTION_CUE.search(clause) or UP_WORDS.search(clause)
                                           or DOWN_WORDS.search(clause)):
            last = intents[-1]
            context = '{} {} '.format('' if names_side else _CONTEXT_WORDS[last['arm']],
                                      _JOINT_WORDS[last['joint']])
            intent = parse_arm_command(context + clause)
        if intent is None:
            continue
        # "подними правую руку, позиция один": the position replaces the vague lift.
        if (intents[len(previous):] and intent['action'] == 'position'
                and intents[-1]['action'] == 'nudge' and intents[-1]['arm'] == intent['arm']):
            intents.pop()
        intents.append(intent)
    return intents[len(previous):]


def parse_arm_commands(text, limit=4):
    """All arm commands in an utterance, in order.

    "Плечо правое в позицию 1. Позиция 4. Правое плечо в позицию 2" gives three
    position commands for the right shoulder; executed in order, the last wins.
    A clause without an arm word ("Позиция 4.") inherits arm and joint from the
    previous command.
    """
    intents = []
    for sentence in CLAUSE_SPLIT.split(text or ''):
        sentence = (sentence or '').strip(' ,')
        if not sentence:
            continue
        # "правая рука позиция один, левая рука позиция два": commas and "и" can
        # separate commands too; use the split only if it finds several of them.
        parts = [part for part in PART_SPLIT.split(sentence) if part and part.strip(' ,')]
        found = _parse_clauses(parts, intents[-1:]) if len(parts) > 1 else []
        if len(found) < 2:
            found = _parse_clauses([sentence], intents[-1:])
        intents.extend(found)
    if any(intent['action'] == 'stop' for intent in intents):
        return [_intent('stop')]           # stop always wins, nothing else is sent
    # "wave the hand, ah, wave the hand on the right arm": one command, not two.
    unique = []
    for intent in intents:
        if not unique or intent != unique[-1]:
            unique.append(intent)
    return unique[-limit:]


make_intent = _intent
NUMBER_ANY = re.compile(r'\d|' + '|'.join(pattern for _number, pattern in NUMBER_WORDS), re.I)
MOVE_WORDS = re.compile(r'(?:подвига\w*|подвинь|двига\w*|помаши|пошевели|\bmove\b|\bwave\b|қозға\w*)', re.I)


def describe(intents):
    """Short English description of what was sent, for the model."""
    side = {'right': 'right arm', 'left': 'left arm', 'both': 'both arms'}
    parts = []
    for intent in intents:
        who = side[intent['arm']] + ('' if intent['joint'] in (None, 'whole') else ' ' + intent['joint'])
        if intent['action'] == 'position':
            parts.append('{} -> position {}'.format(who, intent['position']))
        elif intent['action'] == 'nudge':
            parts.append('{} {} {} degrees'.format(who, intent['direction'], intent['degrees']))
        elif intent['action'] in ('gripper_open', 'gripper_close'):
            parts.append('{} gripper {}'.format(side[intent['arm']], intent['action'].split('_')[1]))
        elif intent['action'] == 'hand_shake':
            parts.append('right hand shaking (until told to stop)')
        elif intent['action'] == 'hand_still':
            parts.append('right hand stops shaking')
        else:
            parts.append(intent['action'])
    return '; '.join(parts)
# The right hand (checked5) shakes: "помаши рукой", "потряси рукой", "дай пять"...
SHAKE_WORDS = re.compile(
    r'(?:помаши|помахай|маши|махни|махать|маха\w*|потряс\w*|тряс\w*|покачай|пожми|рукопожат\w*|дай пять|'
    r'\bwave\b|\bshake\b|\bhigh five\b|бұлға\w*|сілке\w*)', re.I)
STILL_WORDS = re.compile(
    r'(?:перестань\w*|хватит|прекрати\w*|не маши|не тряси|успокой\w*|stop (?:waving|shaking)|'
    r'enough|жетер\w*)', re.I)
CONTINUE_WORDS = re.compile(
    r'(?:ещ[её]|снова|опять|повтори|\bagain\b|\bmore\b|тағы|كمان)', re.I)


def parse_arm_command(text):
    """Return an intent dict (action, arm, position, joint, direction, degrees) or None."""
    q = ' '.join((text or '').casefold().split())
    if not q or not ARM_WORDS.search(q):
        return None
    if RIGHT_WORDS.search(q) and not LEFT_WORDS.search(q):
        arm = 'right'
    elif LEFT_WORDS.search(q) and not RIGHT_WORDS.search(q):
        arm = 'left'
    else:
        arm = 'both'                    # "руки", "обе руки", or both sides named
    if SHAKE_WORDS.search(q) and STILL_WORDS.search(q):
        return _intent('hand_still', 'right')     # "хватит махать рукой"
    if STOP_WORDS.search(q):
        return _intent('stop')
    if SHAKE_WORDS.search(q) and not LEFT_WORDS.search(q):
        return _intent('hand_shake', 'right')     # "помаши рукой", "пожми руку"
    number = _position_number(q)
    if number is not None:
        joint = 'whole'
        if SHOULDER_WORDS.search(q) and not ELBOW_WORDS.search(q):
            joint = 'shoulder'
        elif ELBOW_WORDS.search(q) and not SHOULDER_WORDS.search(q):
            joint = 'elbow'
        return _intent('position', arm, number, joint)
    if OPEN_WORDS.search(q):
        return _intent('gripper_open', arm)
    if CLOSE_WORDS.search(q):
        return _intent('gripper_close', arm)
    up, down = UP_WORDS.search(q), DOWN_WORDS.search(q)
    if bool(up) != bool(down):
        joint = 'elbow' if ELBOW_WORDS.search(q) and not SHOULDER_WORDS.search(q) else 'shoulder'
        return _intent('nudge', arm, None, joint, 'up' if up else 'down', nudge_amount(q))
    return None


_SIDE = {
    'Russian': {'right': 'правую руку', 'left': 'левую руку', 'both': 'обе руки'},
    'Kazakh': {'right': 'оң қолды', 'left': 'сол қолды', 'both': 'екі қолды'},
    'Egyptian Arabic': {'right': 'الإيد اليمين', 'left': 'الإيد الشمال', 'both': 'الإيدين'},
    'English': {'right': 'the right arm', 'left': 'the left arm', 'both': 'both arms'},
}


def arm_reply(intent, language='Russian', error=None):
    """Short spoken confirmation without an extra model call."""
    language = language if language in _SIDE else 'English'
    if error is not None:
        return {
            'Russian': 'Не могу управлять руками: контроллер рук не отвечает.',
            'Kazakh': 'Қолдарды басқара алмаймын: контроллер жауап бермей тұр.',
            'Egyptian Arabic': 'مش قادرة أحرك الإيدين، الكنترولر مش بيرد.',
            'English': 'I cannot move my arms right now: the arm controller is not responding.',
        }[language]
    side = _SIDE[language][intent.get('arm', 'both')]
    action = intent['action']
    if action == 'stop':
        return {'Russian': 'Остановила руки.', 'Kazakh': 'Қолдарды тоқтаттым.',
                'Egyptian Arabic': 'وقفت الإيدين.', 'English': 'Arms stopped.'}[language]
    if action == 'status':
        return {'Russian': 'Статус рук выведен в консоль.', 'Kazakh': 'Қолдардың күйі консольде.',
                'Egyptian Arabic': 'حالة الإيدين في الكونسول.', 'English': 'Arm status is in the console.'}[language]
    if action == 'nudge':
        up = intent.get('direction') == 'up'
        degrees = intent.get('degrees')
        return {
            'Russian': '{} {} на {}°.'.format('Поднимаю' if up else 'Опускаю', side, degrees),
            'Kazakh': '{} {}° {}.'.format(side.capitalize(), degrees, 'көтеремін' if up else 'түсіремін'),
            'Egyptian Arabic': '{} {} {} درجات.'.format('بعلّي' if up else 'بنزّل', side, degrees),
            'English': '{} {} by {} degrees.'.format('Raising' if up else 'Lowering', side, degrees),
        }[language]
    if action in ('hand_shake', 'hand_still'):
        shake = action == 'hand_shake'
        return {'Russian': 'Машу!' if shake else 'Всё, не машу.',
                'Kazakh': 'Бұлғап тұрмын!' if shake else 'Болды, тоқтаттым.',
                'Egyptian Arabic': 'بسلّم!' if shake else 'خلاص وقفت.',
                'English': 'Waving!' if shake else 'Stopped waving.'}[language]
    if action == 'position':
        number = intent['position']
        return {
            'Russian': 'Двигаю {} в позицию {}.'.format(side, number),
            'Kazakh': '{} {}-қалыпқа апарамын.'.format(side.capitalize(), number),
            'Egyptian Arabic': 'بحرك {} للوضع {}.'.format(side, number),
            'English': 'Moving {} to position {}.'.format(side, number),
        }[language]
    opening = action == 'gripper_open'
    return {
        'Russian': ('Открываю захват' if opening else 'Закрываю захват') + (' на обеих руках.' if intent.get('arm') == 'both' else ' на ' + {'right': 'правой', 'left': 'левой'}[intent['arm']] + ' руке.'),
        'Kazakh': side.capitalize() + (' ашамын.' if opening else ' жабамын.'),
        'Egyptian Arabic': ('بفتح ' if opening else 'بقفل ') + side + '.',
        'English': ('Opening' if opening else 'Closing') + ' the gripper on ' + side + '.',
    }[language]
