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
import math
import os
import re
import threading
import time

BAUD = 115200
READY_TEXT = 'Both arms ready'
ARM_LETTERS = {'right': 'r', 'left': 'l', 'both': 'b'}
JOINT_LETTERS = {'whole': '', 'shoulder': 'm', 'elbow': 's'}
ACTIONS = ('position', 'nudge', 'gripper_open', 'gripper_close', 'stop', 'status')
MAX_NUDGE_DEGREES = 45
# Reference only: positions in the supplied 4.0 sketch. Actual motion requires
# the live position table printed by the connected Arduino; None means unset.
DEFAULT_POSITION_TABLE = {
    ('right', 'shoulder'): (101.8, 131.8, 161.8, 171, None),
    ('right', 'elbow'): (159, 129, 99, 69, 39),
    ('left', 'shoulder'): (114, 144, 174, 180, None),
    ('left', 'elbow'): (144, 174, 204, 234, 264),
}
TABLE_LINE = re.compile(
    r'^(RIGHT|LEFT)\s+(stepper|elbow)\s+((?:\S+\s+){5})limits', re.I)
# Arduino lines worth showing at AISANA_ARM_ECHO=1 (errors and safety only).
IMPORTANT_LINE = re.compile(
    r'STOPPED|EMERGENCY|limit|not set|feedback is OFF|can\'t|cannot|calibrat|'
    r'wrong way|outside|!!|doesn\'t match', re.I)
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
    if action == 'gripper_open':
        return letter + ' g o'
    if action == 'gripper_close':
        return letter + ' g c'
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
        valid = [value for value in positions if value is not None]
        if len(valid) < 2 or positions[0] is None or valid[-1] == valid[0]:
            raise ArmError('No calibrated movement direction for {} {}'.format(side, joint))
        sign = 1 if valid[-1] > valid[0] else -1
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
        self._command_lock = threading.Lock()
        self._stop_generation = 0
        self._table_rows = set()
        self._lines = collections.deque(maxlen=300)
        self._lines_lock = threading.Lock()
        self._sequence = 0
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._reader = None
        self.ready_timeout = ready_timeout
        self.echo = int(echo)
        self.up_toward_4 = up_toward_4
        self.position_table = dict(DEFAULT_POSITION_TABLE)
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
                key = (table.group(1).lower(), joint)
                parsed = tuple(None if v == '-' else float(v) for v in values)
                if len(parsed) == 5 and all(v is None or math.isfinite(v) for v in parsed):
                    with self._lines_lock:
                        self.position_table[key] = parsed
                        self._table_rows.add(key)
            except ValueError:
                pass
        if self.echo >= 2 or (self.echo == 1 and IMPORTANT_LINE.search(line)):
            self.log('Arduino: ' + line)
        if line.startswith('Starting both arms'):
            self.ready.clear()
            with self._lines_lock:
                self._table_rows.clear()
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
        with self._write_lock:
            self._serial.write((command + '\n').encode('ascii'))
            self._serial.flush()
        if self.echo >= 2:
            self.log('Arm command: ' + command)
        if collect_seconds > 0:
            time.sleep(collect_seconds)
        return self._lines_after(start)

    def execute(self, action, arm='both', position=None, joint='whole',
                direction=None, degrees=None):
        """Validate live presets, send commands and report Arduino feedback, not guesses."""
        if action == 'stop':
            lines = self.emergency_stop()
            if any('EMERGENCY STOP' in line.upper() for line in lines):
                return 'Arduino confirmed the stop command; joints are commanded to hold.'
            return 'Stop command sent; Arduino did not confirm it.'
        with self._command_lock:
            if not self.ready.wait(self.ready_timeout):
                raise ArmError('Arduino did not report "{}"'.format(READY_TEXT))
            generation = self._stop_generation
            if action == 'status':
                lines = self.send('p', collect_seconds=0.6)
                status = [line for line in lines if re.match(r'^(RIGHT|LEFT)\s+(stepper|elbow|gripper):', line, re.I)]
                return 'Arduino status: ' + ' | '.join(status) if status else 'No Arduino status reply.'
            if arm not in ARM_LETTERS:
                raise ArmError('Unknown arm: {!r}'.format(arm))
            sides = ('right', 'left') if arm == 'both' else (arm,)
            joint = joint or 'whole'
            if action == 'nudge' and joint == 'whole':
                joint = 'shoulder'
            if action in ('position', 'nudge'):
                if joint not in JOINT_LETTERS:
                    raise ArmError('Unknown joint: {!r}'.format(joint))
                joints = ('shoulder', 'elbow') if joint == 'whole' else (joint,)
                keys = [(side, j) for side in sides for j in joints]
                with self._lines_lock:
                    missing = [key for key in keys if key not in self._table_rows]
                if missing:
                    self.send('t', collect_seconds=0.6)
                with self._lines_lock:
                    if any(key not in self._table_rows for key in keys):
                        raise ArmError('Live Arduino position table unavailable; motion not sent.')
                    table = dict(self.position_table)
                if action == 'position':
                    commands = [build_command(action, arm, position, joint)]
                    if any(table[key][int(position)] is None for key in keys):
                        unset = ['{} {}'.format(*key) for key in keys if table[key][int(position)] is None]
                        raise ArmError('Position {} is not set for {}; no movement command sent.'.format(
                            position, ', '.join(unset)))
                else:
                    commands = nudge_commands(arm, joint, direction, degrees, table, self.up_toward_4)
            else:
                joints = ('gripper',)
                commands = [build_command(action, arm, position, joint)]
            lines = []
            for command in commands:
                if generation != self._stop_generation:
                    return 'Movement cancelled by stop; do not claim success.'
                lines.extend(self.send(command, collect_seconds=0.6))
            sent = ', '.join('"{}"'.format(c) for c in commands)
            print('Arm feedback:', ' | '.join(lines) if lines else '(no Arduino reply)')
            problems = [line for line in lines if IMPORTANT_LINE.search(line)]
            expected = [(side, j) for side in sides for j in joints]
            acknowledged = []
            for side, j in expected:
                motor = {'shoulder': 'stepper', 'elbow': 'elbow', 'gripper': 'gripper'}[j]
                suffix = r'\s+(?:OPEN|CLOSE)\b' if j == 'gripper' else r'\s*->\s*[-+\d.]'
                if any(re.match(r'^' + side + r'\s+' + motor + suffix, line, re.I) for line in lines):
                    acknowledged.append((side, j))
            if generation != self._stop_generation:
                return 'Movement cancelled by stop; do not claim success.'
            status = self.send('p', collect_seconds=0.6)
            status = [line for line in status if re.match(r'^(RIGHT|LEFT)\s+(stepper|elbow|gripper):', line, re.I)]
            if status:
                print('Arm status:', ' | '.join(status))
                problems.extend(line for line in status if ('[STOPPED]' in line or 'feedback OFF' in line))
            if problems:
                return 'Command {} sent; Arduino warning/error: {}. Movement not confirmed.'.format(
                    sent, ' | '.join(problems[-4:]))
            if len(acknowledged) != len(expected):
                return 'Command {} sent; missing Arduino acknowledgement for {}. Movement not confirmed.'.format(
                    sent, ', '.join('{} {}'.format(*key) for key in expected if key not in acknowledged))
            return ('Arduino accepted command {}. Physical movement completion is NOT confirmed. '
                    'Say the target was accepted, not that the arm actually moved. Status: {}').format(
                        sent, ' | '.join(status) if status else 'no readback')

    def emergency_stop(self):
        self._stop_generation += 1
        try:
            return self.send('x', collect_seconds=0.3)
        except Exception as error:
            self.log('Arm: emergency stop could not be sent: {}'.format(error))
            return []


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
        log('Arm: not connected ({}). Voice and camera keep working.'.format(error))
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
    (1, r'1|один|одн[уа]|перв\w*|one|first|бір\w*|واحد'),
    (2, r'2|два|две|втор\w*|two|second|екі\w*|اتنين|اثنين'),
    (3, r'3|три|трет\w*|three|third|үш\w*|تلاتة|ثلاثة'),
    (4, r'4|четыре|четв[её]рт\w*|four|fourth|төрт\w*|اربعة|أربعة'),
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
    if STOP_WORDS.search(q):
        return _intent('stop')
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
