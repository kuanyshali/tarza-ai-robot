"""Aisana realtime speech-to-speech (Jetson Nano, Python 3.6).

Fast path: microphone PCM streams to the OpenAI Realtime API and spoken audio
streams back, without separate STT / chat / TTS round trips.
Tools: fresh camera frame, Arm team's Arduino arms over USB, live weather and
web search. A conversation starts with the name "Aisana" (AISANA_REALTIME_WAKE=1,
default); follow-ups need no name for AISANA_DIALOG_TIMEOUT seconds.
Uses websocket-client==1.3.1, arecord, aplay and Realtime GA.
Ctrl+C stops all subprocesses. Prefer headphones for first interruption tests.
"""
import base64
import json
import os
import queue
import requests
import re
import subprocess
import threading
import time

try:
    import websocket
except ImportError:          # tests and robot.py do not need websocket-client
    websocket = None

from aisana_arm import connect_from_environment, POSITION_WORD, NUMBER_WORDS, SHOULDER_WORDS, ELBOW_WORDS, ARM_WORDS
from sadam_features import WEB_ENABLED, get_weather, search_web

RATE = 24000
CHUNK = 1440  # 30 ms of mono S16_LE PCM
MODEL = os.environ.get('AISANA_REALTIME_MODEL', 'gpt-realtime-2.1-mini')
VOICE = os.environ.get('AISANA_REALTIME_VOICE', 'marin')
BARGE_IN = os.environ.get('AISANA_REALTIME_BARGE_IN', '0') == '1'
SILENCE_MS = int(os.environ.get('AISANA_REALTIME_SILENCE_MS', '850'))
ECHO_GUARD = os.environ.get('AISANA_ECHO_GUARD', '1') == '1'
ECHO_TAIL_SECONDS = float(os.environ.get('AISANA_ECHO_TAIL_SECONDS', '1.5'))
KEY = os.environ.get('OPENAI_API_KEY')
STT_MODEL = os.environ.get('AISANA_REALTIME_STT_MODEL', 'gpt-4o-mini-transcribe')
# Respond only after the name "Aisana" (then a timed follow-up window).
WAKE_GATE = os.environ.get('AISANA_REALTIME_WAKE', '1') == '1'
DIALOG_TIMEOUT = max(5.0, float(os.environ.get('AISANA_DIALOG_TIMEOUT', '20')))

WAKE_PATTERN = re.compile(
    r'^\W*(?:(?:ok|okay|окей|ок|эй|hey|привет)\W+)?'
    r'(?:айсан\w*|айсон\w*|эйсан\w*|aisan\w*|aysan\w*|aissan\w*|'
    r'آيسان\w*|ايسان\w*|أيسان\w*|عيسان\w*|إيسان\w*)', re.I)
STOP_PATTERN = re.compile(
    r'(?:^|\W)(?:стоп|остановись|остановитесь|замри|stop|freeze|тоқта\w*|اسكت|وقف)(?:\W|$)', re.I)
SLEEP_PATTERN = re.compile(
    r'(?:^|\W)(?:отбой|спи|хватит слушать|go to sleep|end conversation|болды)(?:\W|$)', re.I)


def has_wake_name(text):
    """True if the utterance starts with Aisana (a short preface is allowed)."""
    return bool(WAKE_PATTERN.match((text or '').strip().casefold()))


def _short(text, words=5):
    return len((text or '').split()) <= words


def is_stop_phrase(text):
    """Short "стоп" / "stop": freeze the arms and the speaker immediately."""
    return _short(text) and bool(STOP_PATTERN.search((text or '').casefold()))


def is_prompt_echo(text):
    """STT sometimes 'hears' its own hint on noise: 'Айсана, Айсана, Эйсана...'."""
    words = re.findall(r'[^\W\d_]+', (text or '').casefold())
    names = [w for w in words if WAKE_PATTERN.match(w)]
    return len(names) >= 3 and len(names) >= len(words) - 1


def is_sleep_phrase(text):
    return _short(text) and bool(SLEEP_PATTERN.search((text or '').casefold()))


# Numbered presets are absolute firmware positions, not relative degrees.
# Use the latest explicit target when speech contains a self-correction.
_POSITION_PATTERNS = []
for _number, _words in NUMBER_WORDS:
    _POSITION_PATTERNS.append((_number, re.compile(
        POSITION_WORD + r'\s*(?:номер\s*|number\s*|№\s*)?(?:' + _words + r')\b' +
        r'|\b(?:' + _words + r')\s*' + POSITION_WORD, re.I)))
_RIGHT = re.compile(r'\b(?:прав(?:ый|ая|ое|ую|ого|ой|ым|ом|ые|ых)|right|оң)\b', re.I)
_LEFT = re.compile(r'\b(?:лев(?:ый|ая|ое|ую|ого|ой|ым|ом|ые|ых)|left|сол)\b', re.I)
_BOTH = re.compile(r'\b(?:оба|обе|обоих|две|два|both|екеуін|екі)\b', re.I)
_WHOLE = re.compile(r'\b(?:рук[аиуеой]\w*|руками|arms?|hands?|қол\w*)\b', re.I)
_PLURAL = re.compile(r'\b(?:руки|плечи|локти|arms|shoulders|elbows)\b', re.I)
_NEGATED = re.compile(r'\b(?:не\s+(?:двигай|поднимай|опускай|переводи|ставь|надо|нужно|в)|не\s+трогай|don.t|do not|never)\b', re.I)
_QUESTION = re.compile(r'\b(?:какая|какой|какие|почему|зачем|что значит|расскажи|объясни|what is|which position|why|explain)\b', re.I)


def numbered_position_intent(text, previous=None):
    """Return (matched, intent); a matched but ambiguous request must not move."""
    q = (text or '').casefold()
    matches = [(m.start(), m.end(), n) for n, pattern in _POSITION_PATTERNS
               for m in pattern.finditer(q)]
    if not matches:
        return False, None
    start, end, number = max(matches, key=lambda m: (m[1], m[0]))
    # Do not move for a question about presets, a negation, or an unaddressed
    # standalone number with no remembered target.
    if _NEGATED.search(q) or _QUESTION.search(q):
        return True, None
    if re.search(r'\b(?:или|or)\b|\d\s*[-–/]\s*\d', q):
        return True, None
    if not ARM_WORDS.search(q) and previous is None:
        return True, None
    # The latest clause takes priority; preceding clauses supply missing context.
    cut = max(q.rfind('.', 0, start), q.rfind(';', 0, start), q.rfind('!', 0, start))
    clause = q[cut + 1:]
    # A correction may omit side/joint: inherit the nearest explicit context.
    def side_of(part):
        r, l, b = bool(_RIGHT.search(part)), bool(_LEFT.search(part)), bool(_BOTH.search(part))
        if b or (r and l):
            return 'both'
        if r:
            return 'right'
        if l:
            return 'left'
        return 'both' if _PLURAL.search(part) else None
    def joint_of(part):
        shoulder, elbow = bool(SHOULDER_WORDS.search(part)), bool(ELBOW_WORDS.search(part))
        if shoulder and elbow:
            return None
        if shoulder:
            return 'shoulder'
        if elbow:
            return 'elbow'
        return 'whole' if _WHOLE.search(part) else None
    if SHOULDER_WORDS.search(clause) and ELBOW_WORDS.search(clause):
        return True, None
    prev_text = q[:cut + 1] if cut >= 0 else ''
    # Walk backwards through earlier clauses instead of merging left/right.
    earlier = list(reversed(re.split(r'[.!;]', prev_text)))
    side = side_of(clause)
    joint = joint_of(clause)
    for part in earlier:
        side = side or side_of(part)
        joint = joint or joint_of(part)
    if previous:
        side = side or previous.get('arm')
        joint = joint or previous.get('joint')
    if side is None or joint is None:
        return True, None
    return True, {'action': 'position', 'arm': side, 'joint': joint, 'position': number}


def tool_definitions():
    tools = [
        {'type': 'function', 'name': 'inspect_camera',
         'description': 'Look at a fresh Jetson camera image to read or translate visible text, explain a shown formula, identify or compare shown objects, or answer any visual question including implicit this/that/here references.',
         'parameters': {'type': 'object', 'properties': {
             'question': {'type': 'string', 'description': 'User visual question in their original language'}},
             'required': ['question'], 'additionalProperties': False}},
        {'type': 'function', 'name': 'move_arm',
         'description': ('Control your two physical 2-DOF robotic arms (shoulder stepper, elbow servo, gripper) '
                         'on the Arm team Arduino. Use ONLY when the user explicitly asks to move an arm, '
                         'go to a numbered position, move a little up or down, open/close a gripper, '
                         'stop the arms, or report arm status. Any numbered position request MUST use '
                         'action position, NEVER nudge: right shoulder position 2 means '
                         'position/right/shoulder/2. A shoulder is the stepper, an elbow is the servo. '
                         'Presets are 0..4 but some are unset; the controller validates live presets. '
                         'Use nudge ONLY for explicitly relative movements with a specified direction. '
                         'Ask for missing side, target or direction instead of guessing.'),
         'parameters': {'type': 'object', 'properties': {
             'action': {'type': 'string', 'enum': ['position', 'nudge', 'gripper_open', 'gripper_close', 'stop', 'status']},
             'arm': {'type': 'string', 'enum': ['right', 'left', 'both'],
                     'description': 'Use both only if both arms are requested; otherwise use the explicit or conversationally established side'},
             'position': {'type': 'integer', 'minimum': 0, 'maximum': 4,
                          'description': 'Required for action=position'},
             'joint': {'type': 'string', 'enum': ['whole', 'shoulder', 'elbow'],
                       'description': 'position: whole = shoulder and elbow together (default). '
                                      'nudge: shoulder (default) or elbow'},
             'direction': {'type': 'string', 'enum': ['up', 'down'], 'description': 'Required for action=nudge'},
             'degrees': {'type': 'integer', 'minimum': 1, 'maximum': 45,
                         'description': 'For nudge: чуть-чуть/slightly 5, немного/a bit 10, no amount 15, '
                                        'сильно/a lot 30, or the number of degrees the user said'}},
             'required': ['action'], 'additionalProperties': False}},
        {'type': 'function', 'name': 'get_weather',
         'description': 'Live weather from Open-Meteo. Use for any weather or temperature question; never guess weather.',
         'parameters': {'type': 'object', 'properties': {
             'city': {'type': 'string', 'description': 'City name in English; empty means Astana'},
             'day': {'type': 'string', 'enum': ['now', 'tomorrow']},
             'language': {'type': 'string', 'enum': ['Russian', 'Kazakh', 'English', 'Egyptian Arabic']}},
             'required': ['city', 'day', 'language'], 'additionalProperties': False}},
    ]
    if WEB_ENABLED:
        tools.append(
            {'type': 'function', 'name': 'web_search',
             'description': 'Live web search for news, elections, results, prices, office holders and other facts that change. Never guess such facts.',
             'parameters': {'type': 'object', 'properties': {
                 'query': {'type': 'string', 'description': 'The question, in the user language'},
                 'language': {'type': 'string', 'enum': ['Russian', 'Kazakh', 'English', 'Egyptian Arabic']}},
                 'required': ['query', 'language'], 'additionalProperties': False}})
    return tools

PROMPT = (
    'You are Aisana (Айсана), a concise, intelligent female-voiced voice assistant. '
    'Your job is to answer the actual question directly, not to fill silence. '
    'LANGUAGE RULE: Identify the language of the latest USER speech, ignoring your own '
    'previous speech and ignoring proper names. Reply entirely in that same language. '
    'English question -> English answer. Russian question -> Russian answer. '
    'Kazakh question -> Kazakh answer. Never switch languages unless explicitly asked. '
    'If the user asks about an English technical topic in Russian, answer in Russian. '
    'LENGTH RULE: Default to ONE short sentence, maximum TWO short sentences. '
    'Give more detail only when the user explicitly requests explanation or detail. '
    'Do not add offers to help, follow-up questions, apologies, motivational language, '
    'reassurance, greetings, or filler unless explicitly relevant. '
    'Do not repeat yourself or continue talking after answering. '
    'When addressed as Aisana, wait for the complete question. '
    'In Russian use feminine grammatical forms when referring to yourself. '
    'You HAVE a working camera tool named inspect_camera. '
    'VISUAL INTENT RULE: Call inspect_camera BEFORE answering whenever the user refers '
    'to something physically in front of you, including implicit references like this, '
    'that, here, on the screen, on the paper, on the label, or what I am showing you. '
    'Examples: read this, what does this say, translate what is written here, '
    'pronounce this text, explain this formula, solve the problem on this sheet, '
    'identify this device, compare these phones, what model is this, and what do you see. '
    'This rule applies equally in Russian, English and Kazakh. '
    'For a request to translate or read text already SPOKEN aloud by the user, '
    'answer directly WITHOUT camera. If ambiguous but the user is showing an object, '
    'prefer the camera. When reading visible text, quote only legible words; '
    'say briefly when text is unreadable instead of inventing it. '
    'CAMERA BEHAVIOR: Never narrate your actions or plans. Never say you need to take '
    'a photo, will check the camera, are looking now, are analyzing, or will report back. '
    'Call the camera tool silently and wait for its result. Once the result arrives, '
    'speak directly as Aisana in FIRST PERSON: Я вижу..., Я заметила... / I can see... '
    'Do not refer to a photo, screenshot, image model, API, or a separate observer. '
    'Do not announce a future answer; give the actual answer. '
    'If the camera tool fails, briefly say in first person that you cannot see right now; '
    'do not invent objects. '
    'Do not say you have no camera or no visual access unless that tool actually fails. '
    'For other questions do not use the camera. '
    'ARMS: you have two physical arms with grippers, controlled by move_arm. '
    'Call move_arm FIRST and silently, without saying anything before the call; '
    'never say the arm is moving before move_arm has returned. '
    'Call move_arm only when the user explicitly asks to move an arm, go to a numbered '
    'position, open or close a gripper, stop the arms or report their status. '
    'POSITION PRIORITY: any numbered position (позиция 2, во вторую позицию, position two, '
    '2-қалып) means action=position, NOT nudge or degrees. If the user corrects a number, use only '
    'the latest intended position for that joint. Example: правое плечо в позицию 2 -> '
    'action position, arm right, joint shoulder, position 2. Left elbow position 3 -> position,left,elbow,3. '
    'Shoulder/плечо/иық is the stepper; elbow/локоть/шынтақ is the servo. '
    '0 is home; some presets are not defined; never assume position 4 exists or a number means higher. '
    'Use both only if the user explicitly asks for both; ambiguous singular side or direction requires clarification. '
    'For vague подвигай плечом, ask which direction or position, never invent a motion. '
    'RELATIVE MOVES: "подними руку", "чуть-чуть наверх", "немного ниже", "a bit higher", "опусти" '
    'are action nudge, direction up or down, joint shoulder unless the user mentions the elbow '
    '(локоть). Degrees: чуть-чуть/slightly 5, немного/a bit 10, no amount 15, сильно/a lot 30, '
    'or the number the user said (max 45). "Ещё" / "more" repeats the previous nudge. '
    'When asked to stop the arms, call move_arm with action stop at once. '
    'After move_arm returns, distinguish sent, accepted, rejected, and physically reached. '
    'Do NOT say I raised/moved/reached/stopped unless the controller has confirmed that fact. '
    'If only sent or accepted, say the command was sent or the target accepted; movement is unconfirmed. '
    'If missing acknowledgement, unset preset, limit, fault, or no connection, state the actual problem briefly. '
    'Never call a second movement automatically to compensate for a failure. '
    'WEATHER: always call get_weather for weather; never guess. '
    'LIVE FACTS: call web_search for news, elections, results, prices and other changing facts. '
)



def alsa_device(command, expected, description):
    output = subprocess.check_output([command, '-l'], stderr=subprocess.STDOUT).decode(
        'utf-8', errors='replace')
    matches = []
    for line in output.splitlines():
        m = re.search(r'^card\s+(\d+):\s*[^\[]+\[([^\]]+)\],\s*device\s+(\d+):', line)
        if m and expected.lower() in m.group(2).lower():
            matches.append('plughw:{},{}'.format(m.group(1), m.group(3)))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError('{} must match exactly one ALSA device; found {}.\n{}'.format(
            description, matches, output))
    return matches[0]


class RealtimeAisana(object):
    def __init__(self):
        self.ws = None
        self.send_lock = threading.Lock()
        self.play_lock = threading.Lock()
        self.stop = threading.Event()
        self.mic = None
        self.speaker = None
        self.playback = None
        self.responding = False
        self.playing = False
        self.speech_started_at = 0
        self.playback_interrupted = False
        self.audio_queue = queue.Queue(maxsize=256)
        self.audio_generation = 0
        self.mute_until = 0.0
        # True from response.created through the very last ALSA playback byte.
        # Covers the race between response.done and the playback worker.
        self.echo_guard_active = False
        self.camera_lock = threading.Lock()
        self.arm = None
        self.dialog_until = 0.0
        self.responded_items = set()
        self.handled_calls = set()
        self.calls_lock = threading.Lock()
        self.mic_device = None
        self.transcript_open = False
        self.last_position_target = None
        self.pending_position = None
        self.position_used = False

    def dialog_active(self):
        return time.monotonic() < self.dialog_until

    def respond(self, item_id=None, response=None):
        """Answer the latest user turn and keep the follow-up window open."""
        if item_id:
            self.responded_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        payload = {'type': 'response.create'}
        if response:
            payload['response'] = response
        self.send(payload)

    def ignore_item(self, item_id):
        """Drop unaddressed speech so it does not pollute the conversation."""
        if item_id:
            self.send({'type': 'conversation.item.delete', 'item_id': item_id})

    def send(self, payload):
        with self.send_lock:
            self.ws.send(json.dumps(payload))

    def stop_playback(self):
        with self.play_lock:
            self.audio_generation += 1
            p = self.playback
            self.playback = None
            self.playing = False
        if p:
            try:
                p.terminate()
            except Exception:
                pass

    def audio_worker(self):
        """Own ALSA writes outside the websocket receiver thread."""
        generation = -1
        p = None
        while not self.stop.is_set():
            try:
                item = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            kind, item_gen, chunk = item
            if item_gen != self.audio_generation:
                continue
            if kind == 'stop':
                if p:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                p = None
                continue
            if kind == 'done':
                if p:
                    try:
                        p.stdin.close()
                        p.wait(timeout=20)
                    except Exception:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                p = None
                with self.play_lock:
                    if item_gen == self.audio_generation:
                        self.playback = None
                        self.playing = False
                # Keep capture gated until the acoustic tail has dissipated.
                self.mute_until = time.monotonic() + ECHO_TAIL_SECONDS
                self.echo_guard_active = False
                if self.dialog_until:
                    # Follow-up window counts from the end of her speech.
                    self.dialog_until = self.mute_until + DIALOG_TIMEOUT
                continue
            if kind != 'audio':
                continue
            if p is None or generation != item_gen:
                p = subprocess.Popen(
                    ['aplay', '-q', '-D', self.speaker, '-t', 'raw', '-f', 'S16_LE',
                     '-r', str(RATE), '-c', '1'],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                generation = item_gen
                with self.play_lock:
                    self.playback = p
                    self.playing = True
            try:
                p.stdin.write(chunk)
            except (IOError, OSError, ValueError):
                pass
        if p:
            try:
                p.terminate()
            except Exception:
                pass

    def queue_audio(self, kind, chunk=b''):
        try:
            self.audio_queue.put_nowait((kind, self.audio_generation, chunk))
        except queue.Full:
            print('Audio queue overloaded; dropping audio frame')

    def capture(self):
        try:
            self.mic = subprocess.Popen(
                ['arecord', '-q', '-D', self.mic_device, '-t', 'raw',
                 '-f', 'S16_LE', '-r', str(RATE), '-c', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while not self.stop.is_set():
                pcm = self.mic.stdout.read(CHUNK)
                if len(pcm) != CHUNK:
                    raise RuntimeError('Microphone capture ended unexpectedly')
                if ECHO_GUARD and (self.echo_guard_active or self.responding or self.playing or time.monotonic() < self.mute_until):
                    continue
                self.send({'type': 'input_audio_buffer.append',
                           'audio': base64.b64encode(pcm).decode('ascii')})
        except Exception as exc:
            if not self.stop.is_set():
                print('Microphone/connection error:', exc)
                self.stop.set()

    def inspect_camera(self, question):
        """Capture one fresh IMX219 frame and analyze it with the vision API."""
        with self.camera_lock:
            # camera.py uses the Jetson GStreamer camera pipeline and writes photo.jpg.
            # Fail closed rather than analyzing a stale image if capture fails.
            if os.path.exists('photo.jpg'):
                os.remove('photo.jpg')
            started = time.monotonic()
            # GStreamer/Argus print ~30 lines per photo; show them only on failure.
            capture = subprocess.run(['/usr/bin/python3', 'camera.py'], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=20)
            if capture.returncode != 0 or not os.path.exists('photo.jpg'):
                tail = capture.stdout.decode('utf-8', 'replace').strip().splitlines()[-8:]
                raise RuntimeError('camera.py failed: ' + ' | '.join(tail))
            with open('photo.jpg', 'rb') as photo:
                encoded = base64.b64encode(photo.read()).decode('ascii')
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + KEY},
            json={'model': os.environ.get('AISANA_VISION_MODEL', 'gpt-4o-mini'),
                  'messages': [
                      {'role': 'system', 'content': (
                          'You are a visual observer. The user message includes a real camera image. '
                          'Inspect the attached image and describe only what is visible. '
                          'Do not claim you cannot access images or ask the user to describe it. '
                          'If the image is dark, blurry or unclear, describe what you can see and note the limitation. '
                          'Answer in at most two short sentences, most noticeable things first. '
                          'Never ask the user questions and never offer further help. '
                          'Answer in the language of the user question.')},
                      {'role': 'user', 'content': [
                      {'type': 'text', 'text': (
                          'Use this fresh camera photo to answer the user request, including reading visible text, translating signs, explaining visible formulas, or identifying objects as appropriate. '
                          'Answer this question if possible: ' + question +
                          '. Be concise, factual, and avoid guessing unseen details. '
                          'Use the same language as the question.')},
                      {'type': 'image_url', 'image_url': {
                          'url': 'data:image/jpeg;base64,' + encoded, 'detail': 'high'}}
                  ]}], 'max_tokens': 300},
            timeout=(10, 45))
        response.raise_for_status()
        data = response.json()
        choice = data['choices'][0]
        print('Camera: photo {} KB, answered in {:.1f}s ({} tokens)'.format(
            len(encoded) * 3 // 4 // 1024, time.monotonic() - started,
            data.get('usage', {}).get('total_tokens', '?')))
        content = choice.get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('Vision returned empty content (finish_reason={})'.format(
                choice.get('finish_reason')))
        return content.strip()

    def run_tool(self, name, args):
        """Execute one tool call; every result is plain text for the model."""
        if name == 'inspect_camera':
            question = str(args.get('question') or 'What is visible?')[:500]
            return self.inspect_camera(question)
        if name == 'move_arm':
            if self.pending_position is not None:
                if self.position_used:
                    return 'No new movement: this numbered command was already processed.'
                self.position_used = True
                # The local parser, not model inference, owns explicit numbered requests.
                args = dict(self.pending_position)
            if self.arm is None:
                return ('The arm controller (Arduino Mega) is not connected over USB. '
                        'Tell the user briefly; do not pretend the arm moved.')
            return self.arm.execute(args.get('action'), args.get('arm') or 'both',
                                    args.get('position'), args.get('joint') or 'whole',
                                    args.get('direction'), args.get('degrees'))
        if name == 'get_weather':
            city = str(args.get('city') or '').strip()[:60]
            phrase = 'weather' + (' in ' + city if city else '')
            if args.get('day') == 'tomorrow':
                phrase += ' tomorrow'
            return get_weather(phrase, language=args.get('language') or 'English')
        if name == 'web_search':
            return search_web(str(args.get('query') or '')[:500], args.get('language') or 'English',
                              [], {'Authorization': 'Bearer ' + (KEY or '')})
        raise ValueError('Unknown tool: ' + str(name))

    def start_tool(self, call_id, name, arguments):
        """Run each call_id once, off the websocket receiver thread."""
        if not call_id or not name:
            return
        with self.calls_lock:
            if call_id in self.handled_calls:
                return
            self.handled_calls.add(call_id)
        threading.Thread(target=self.handle_tool_call,
                         args=(call_id, name, arguments), daemon=True).start()

    def handle_tool_call(self, call_id, name, arguments):
        try:
            args = json.loads(arguments or '{}')
            result = self.run_tool(name, args)
            print('Tool {} {} -> {}'.format(name, args, ' '.join(str(result).split())[:300]))
        except Exception as exc:
            result = 'Tool {} failed: {}. Say briefly that it did not work; do not invent a result.'.format(name, exc)
            print('Tool {} FAILED: {}'.format(name, exc))
        if self.stop.is_set():
            return
        try:
            self.send({'type': 'conversation.item.create', 'item': {
                'type': 'function_call_output', 'call_id': call_id, 'output': str(result)[:4000]}})
            response = {'instructions': (
                'The tool has returned. Answer the user NOW, in first person as Aisana, '
                'using its result. No commentary about tools, photos or analysis. '
                'One or two short sentences in the language of the user question. '
                'No filler, no wishes, no offers of more help, no follow-up questions. '
                'If the tool failed, say briefly that it did not work. '
                'For arms, report only the actual controller result: sent or accepted is not proof of movement. '
                'Never claim a completed move or stop without feedback.')}
            if name == 'move_arm':
                response['tool_choice'] = 'none'
            self.send({'type': 'response.create', 'response': response})
        except Exception as exc:
            if not self.stop.is_set():
                print('Tool response error:', exc)

    def handle_transcript(self, item_id, transcript):
        print('You:', transcript)
        if item_id in self.responded_items:
            return
        if is_prompt_echo(transcript) or not transcript:
            print('Transcript looks like noise; ignored.')
            self.ignore_item(item_id)
            return
        active = self.dialog_active()
        if not active:
            self.last_position_target = None
        if is_stop_phrase(transcript):
            # Stop remains available even outside the conversation window.
            stop_result = self.arm.emergency_stop() if self.arm is not None else []
            self.pending_position = None
            self.stop_playback()
            if not WAKE_GATE or active or has_wake_name(transcript):
                confirmed = bool(stop_result and any('EMERGENCY STOP' in line.upper() for line in stop_result))
                self.respond(item_id, {'tool_choice': 'none', 'instructions':
                    'Reply in the user language. ' + ('Arduino confirmed the stop command.' if confirmed else
                    'The stop command is not confirmed; do not claim the arms stopped.')})
            else:
                self.ignore_item(item_id)
            return
        if is_sleep_phrase(transcript):
            self.dialog_until = 0.0
            self.last_position_target = None
            self.pending_position = None
            print('Dialogue closed; say Aisana to begin again.')
            self.ignore_item(item_id)
            return
        if WAKE_GATE and not (active or has_wake_name(transcript)):
            print('Not addressed to Aisana; ignored.')
            self.ignore_item(item_id)
            return
        self.pending_position = None
        self.position_used = False
        matched, intent = numbered_position_intent(transcript, self.last_position_target)
        if intent is not None:
            self.pending_position = intent
            self.last_position_target = dict(intent)
            print('Arm preset (transcript):', intent)
            self.respond(item_id, {'tool_choice': {'type': 'function', 'name': 'move_arm'},
                                  'instructions': 'Call move_arm once with exactly these arguments: ' +
                                                  json.dumps(intent) + '. Do not speak first.'})
        elif matched:
            self.respond(item_id, {'tool_choice': 'none', 'instructions':
                'Do not move anything. Answer a question about positions if asked, otherwise ask briefly '
                'which arm, joint and single numbered position the user means. Do not claim motion.'})
        else:
            self.respond(item_id)

    def handle(self, e):
        typ = e.get('type', '')
        if typ == 'session.updated':
            print('Realtime session ready. Talk to Aisana (Ctrl+C to stop).')
        elif typ == 'input_audio_buffer.speech_started':
            self.speech_started_at = time.monotonic()
            if BARGE_IN and (self.responding or self.playing):
                self.playback_interrupted = True
                self.stop_playback()
                # Realtime server_vad with interrupt_response=True already
                # cancels the active generation. A second response.cancel races
                # with response.done and triggers response_cancel_not_active.
        elif typ == 'response.created':
            self.echo_guard_active = True
            self.responding = True
            self.playback_interrupted = False
        elif typ == 'response.output_audio.delta':
            self.echo_guard_active = True
            if not self.playback_interrupted:
                self.queue_audio('audio', base64.b64decode(e.get('delta', '')))
        elif typ == 'response.output_audio_transcript.done':
            # One clean console line per reply (audio itself still streams).
            text = ' '.join((e.get('transcript') or '').split())
            if text:
                print('Aisana:', text)
        elif typ == 'input_audio_buffer.committed':
            # Always wait for the transcript: responding to raw audio early can
            # move the wrong joint before a correction or a number is understood.
            pass
        elif typ == 'conversation.item.input_audio_transcription.completed':
            self.handle_transcript(e.get('item_id'), (e.get('transcript') or '').strip())
        elif typ == 'conversation.item.input_audio_transcription.failed':
            print('Realtime transcription failed:', e.get('error', {}))
        elif typ == 'response.function_call_arguments.done':
            self.start_tool(e.get('call_id'), e.get('name'), e.get('arguments'))
        elif typ == 'response.output_item.done':
            item = e.get('item') or {}
            if item.get('type') == 'function_call':
                self.start_tool(item.get('call_id'), item.get('name'), item.get('arguments'))
        elif typ == 'response.done':
            self.responding = False
            # Do NOT lift echo guard here: playback can lag behind generation.
            self.queue_audio('done')
        elif typ == 'error':
            print('Realtime API error:', e.get('error', {}))

    def run(self):
        if not KEY:
            raise RuntimeError('OPENAI_API_KEY is missing')
        if websocket is None:
            raise RuntimeError('Realtime mode needs websocket-client: python3 -m pip install --user "websocket-client==1.3.1"')
        self.mic_device = alsa_device('arecord', 'KT USB Audio', 'KT microphone')
        self.speaker = alsa_device('aplay', 'USB Composite Device', 'Xiaomi speaker')
        print('Input:', self.mic_device, 'Output:', self.speaker)
        print('Model:', MODEL, 'Voice:', VOICE,
              'Barge-in:', 'ON (experimental)' if BARGE_IN else 'OFF',
              'Echo guard:', 'ON (no interruptions during playback)' if ECHO_GUARD else 'OFF',
              'Wake name:', 'required to start' if WAKE_GATE else 'OFF (answers any speech)')
        self.arm = connect_from_environment()
        self.ws = websocket.create_connection(
            'wss://api.openai.com/v1/realtime?model=' + MODEL,
            timeout=10, header=['Authorization: Bearer ' + KEY])
        try:
            self.send({
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'model': MODEL,
                    'instructions': PROMPT,
                    'tools': tool_definitions(),
                    'tool_choice': 'auto',
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            # Transcripts drive the wake-name gate and voice "stop".
                            'transcription': {
                                'model': STT_MODEL,
                                # A list of names here gets echoed back on noise.
                                'prompt': 'Speech may be Russian, Kazakh, English or Egyptian Arabic. The robot is called Aisana.'
                            },
                            'turn_detection': {
                                'type': 'server_vad',
                                'threshold': 0.5,
                                'prefix_padding_ms': 500,
                                'silence_duration_ms': max(500, min(3000, SILENCE_MS)),
                                'create_response': False,
                                'interrupt_response': BARGE_IN
                            }
                        },
                        'output': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            'voice': VOICE
                        }
                    }
                }
            })
            self.ws.settimeout(10)
            threading.Thread(target=self.audio_worker, daemon=True).start()
            worker = threading.Thread(target=self.capture, daemon=True)
            worker.start()
            while not self.stop.is_set():
                try:
                    event = json.loads(self.ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                self.handle(event)
        finally:
            self.stop.set()
            if self.mic:
                self.mic.terminate()
                try:
                    self.mic.wait(timeout=2)
                except Exception:
                    self.mic.kill()
                    self.mic.wait()
            self.stop_playback()
            if self.arm is not None:
                self.arm.close()
            if self.ws:
                self.ws.close()


if __name__ == '__main__':
    try:
        RealtimeAisana().run()
    except KeyboardInterrupt:
        print('\nAisana stopped.')
    except Exception as exc:
        print('Aisana realtime error:', exc)
        raise
