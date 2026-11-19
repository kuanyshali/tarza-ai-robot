"""Always-on camera and face/thing memory for Aisana (Python 3.6, JetPack OpenCV 4.1.1).

- CameraStream keeps the Jetson camera open and always holds a fresh frame,
  so "кто перед тобой?" no longer waits 2-3 s for the camera to start.
- FaceMemory finds faces with the OpenCV res10 SSD detector and turns each
  face into 128 numbers with OpenFace (both run on cv2.dnn, no pip installs).
  Only those numbers and the name are stored, never photos of people, and
  only when the person asks ("Айсана, запомни меня, я Али"). Unknown faces
  are never stored. "Айсана, забудь меня" deletes them.
- Things ("запомни, это моя кружка") are stored as a short description
  plus a small picture of the thing, for the vision model to recognize later.

Importable without cv2/numpy (tests, CI); they are loaded on first use.
"""
import hashlib
import os
import re
import sqlite3
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.expanduser(os.environ.get('AISANA_MODELS_DIR', os.path.join(HERE, 'models')))
DATA_DIR = os.path.expanduser(os.environ.get('AISANA_DATA_DIR', '~/.local/share/aisana'))
DB_PATH = os.path.expanduser(os.environ.get('AISANA_FACES_DB', os.path.join(DATA_DIR, 'faces.sqlite3')))
# L2 distance between normalized OpenFace vectors: same person is usually < 0.7.
THRESHOLD = float(os.environ.get('AISANA_FACE_THRESHOLD', '0.8'))
MIN_FACE_PX = int(os.environ.get('AISANA_FACE_MIN_PX', '40'))
MAX_VIEWS = 40                    # stored views per person (it keeps learning new angles)
ENABLED = os.environ.get('AISANA_FACES', '1') == '1'

MODELS = {
    'deploy.prototxt': (
        'https://raw.githubusercontent.com/opencv/opencv/4.1.1/samples/dnn/face_detector/deploy.prototxt',
        '1e5a51172abe1e4052c27a3488f99d86dfec9b962191bc999a98261990be5d64'),
    'res10_300x300_ssd_iter_140000.caffemodel': (
        'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/'
        'res10_300x300_ssd_iter_140000.caffemodel',
        '2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d'),
    'openface.nn4.small2.v1.t7': (
        'https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7',
        '9b72d54aeb24a64a8135dca8e792f7cc675c99a884a6940350a6cedcf7b7ba08'),
}

PIPELINE = (
    'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! '
    'nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! '
    'appsink drop=true max-buffers=1 sync=false')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_models(folder=MODELS_DIR, download=True):
    """Paths of the three model files; downloads (about 42 MB, once) if missing."""
    if not os.path.isdir(folder):
        os.makedirs(folder)
    paths = {}
    for name, (url, sha) in sorted(MODELS.items()):
        path = os.path.join(folder, name)
        if not (os.path.exists(path) and _sha256(path) == sha):
            if not download:
                raise RuntimeError('face model missing: ' + path)
            import requests
            print('Downloading face model {} ...'.format(name))
            part = path + '.part'
            with requests.get(url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with open(part, 'wb') as handle:
                    for block in response.iter_content(1 << 20):
                        handle.write(block)
            if _sha256(part) != sha:
                os.remove(part)
                raise RuntimeError('face model {} is corrupted (checksum)'.format(name))
            os.rename(part, path)
        paths[name] = path
    return paths


def rotate(frame, cv2, setting=None):
    code = {'90': cv2.ROTATE_90_CLOCKWISE, '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_COUNTERCLOCKWISE}.get(
                (setting if setting is not None else os.environ.get('TARZA_CAMERA_ROTATE', '180')).strip())
    return frame if code is None else cv2.rotate(frame, code)


class CameraStream(object):
    """Keeps the camera open and the newest frame (960x540, upright) in memory."""

    def __init__(self, pipeline=PIPELINE, opener=None):
        self.pipeline = pipeline
        self.opener = opener
        self.lock = threading.Lock()
        self.frame = None
        self.frame_at = 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def _open(self):
        import cv2
        if self.opener is not None:
            return self.opener()
        return cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    def _loop(self):
        import cv2
        reported = False
        while not self.stop_event.is_set():
            camera = self._open()
            if camera is None or not camera.isOpened():
                self.error = 'camera did not open'
                if not reported:
                    print('Camera stream: could not open the camera; retrying every 5 s.')
                    reported = True
                self.stop_event.wait(5)
                continue
            if reported:
                print('Camera stream: camera is back.')
            reported = False
            self.error = None
            failures = 0
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 30:
                        self.error = 'camera stopped giving frames'
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                if frame.shape[1] != 960:
                    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                frame = rotate(frame, cv2)
                with self.lock:
                    self.frame, self.frame_at = frame, time.monotonic()
                time.sleep(0.1)            # ~10 frames/s is plenty; the Nano CPU is for audio
            camera.release()

    def latest(self, max_age=1.5, wait=3.0):
        """A copy of the newest frame not older than max_age seconds, or None."""
        deadline = time.monotonic() + wait
        while True:
            with self.lock:
                if self.frame is not None and time.monotonic() - self.frame_at <= max_age:
                    return self.frame.copy()
            if time.monotonic() >= deadline or self.stop_event.is_set():
                return None
            time.sleep(0.05)

    def stop(self):
        self.stop_event.set()


def jpeg(frame, quality=80):
    import cv2
    ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return data.tobytes()


def square_crop(frame, box, margin=0.1):
    """The face as a square with a little margin: the same framing every time
    (the detector box changes shape from frame to frame)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    side = int(max(x2 - x1, y2 - y1) * (1 + margin))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = max(0, cx - side // 2), max(0, cy - side // 2)
    return frame[top:min(height, top + side), left:min(width, left + side)]


class FaceMemory(object):
    """Names for faces, stored as 128 numbers per face in SQLite."""

    def __init__(self, db_path=DB_PATH, threshold=THRESHOLD, models=None, embedder=None, detector=None):
        self.db_path = db_path
        self.threshold = threshold
        self.models = models
        self.embedder = embedder          # tests: fake functions instead of networks
        self.detector = detector
        self.nets = None
        self.lock = threading.RLock()
        self.learned_at = {}
        self.people = []                  # [(name, vector)]
        self.things = []                  # [(id, label, owner, description)]
        folder = os.path.dirname(db_path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'name TEXT NOT NULL, vector BLOB NOT NULL, created_at REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS things (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'label TEXT NOT NULL, owner TEXT, description TEXT NOT NULL, '
                       'picture TEXT, created_at REAL NOT NULL)')
        self._load()

    def _db(self):
        return sqlite3.connect(self.db_path, timeout=5)

    def _load(self):
        import numpy as np
        with self.lock, self._db() as db:
            self.people = [(name, np.frombuffer(blob, dtype=np.float32))
                           for name, blob in db.execute('SELECT name, vector FROM faces ORDER BY id')]
            self.things = list(db.execute('SELECT id, label, owner, description FROM things ORDER BY id'))

    # -- networks ---------------------------------------------------------
    def _load_nets(self):
        if self.nets is None:
            import cv2
            paths = self.models or ensure_models()
            detector = cv2.dnn.readNetFromCaffe(paths['deploy.prototxt'],
                                                paths['res10_300x300_ssd_iter_140000.caffemodel'])
            embedder = cv2.dnn.readNetFromTorch(paths['openface.nn4.small2.v1.t7'])
            self.nets = (detector, embedder)
        return self.nets

    def detect(self, frame, min_confidence=0.5):
        """Face boxes [(x1, y1, x2, y2, confidence)], biggest first."""
        if self.detector is not None:
            return self.detector(frame)
        import cv2
        detector = self._load_nets()[0]
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        with self.lock:
            detector.setInput(blob)
            found = detector.forward()
        boxes = []
        for i in range(found.shape[2]):
            confidence = float(found[0, 0, i, 2])
            if confidence < min_confidence:
                continue
            x1, y1, x2, y2 = (found[0, 0, i, 3:7] * [width, height, width, height]).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 >= MIN_FACE_PX and y2 - y1 >= MIN_FACE_PX:
                boxes.append((x1, y1, x2, y2, confidence))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes

    def embed(self, frame, box):
        import numpy as np
        if self.embedder is not None:
            vector = np.asarray(self.embedder(frame, box), dtype=np.float32)
        else:
            import cv2
            embedder = self._load_nets()[1]
            face = square_crop(frame, box)
            blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            with self.lock:
                embedder.setInput(blob)
                vector = embedder.forward().flatten().astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return vector / norm

    def identify(self, vector):
        """(name, distance) of the closest known face, name None if nobody is close."""
        import numpy as np
        best, best_distance = None, 9.0
        with self.lock:
            for name, known in self.people:
                distance = float(np.linalg.norm(known - vector))
                if distance < best_distance:
                    best, best_distance = name, distance
        return (best if best_distance <= self.threshold else None), best_distance

    def faces(self, frame):
        """Everyone in the frame: [{'box', 'vector', 'name', 'distance', 'where'}], biggest first."""
        width = frame.shape[1]
        result = []
        for box in self.detect(frame):
            vector = self.embed(frame, box)
            name, distance = self.identify(vector)
            centre = (box[0] + box[2]) / 2.0 / width
            where = 'left' if centre < 0.36 else 'right' if centre > 0.64 else 'center'
            result.append({'box': box, 'vector': vector, 'name': name, 'distance': distance, 'where': where})
        return result

    # -- memory -----------------------------------------------------------
    def remember(self, name, vectors):
        now = time.time()
        with self.lock, self._db() as db:
            for vector in vectors:
                db.execute('INSERT INTO faces (name, vector, created_at) VALUES (?, ?, ?)',
                           (name, vector.astype('float32').tobytes(), now))
        self._load()

    def learn(self, name, vector, distance):
        """Recognized but from a new angle/light: keep this view too (few per minute)."""
        now = time.monotonic()
        if not (0.4 < distance < self.threshold - 0.05):
            return False
        if now - self.learned_at.get(name, 0) < 15:
            return False
        with self.lock:
            views = sum(1 for known, _ in self.people if known == name)
        if views >= MAX_VIEWS:
            return False
        self.learned_at[name] = now
        self.remember(name, [vector])
        return True

    def forget(self, name):
        """Deletes a person or a thing by name; how many records went."""
        key = name.casefold()
        stem = key[:max(3, len(key) - 2)]          # "забудь кружку" -> кружка, "Алию" -> Алия

        def same(words):
            return key == words or any(w.startswith(stem) for w in words.split())
        with self.lock, self._db() as db:
            gone = 0
            for row_id, known in list(db.execute('SELECT id, name FROM faces')):
                if same(known.casefold()):
                    db.execute('DELETE FROM faces WHERE id = ?', (row_id,))
                    gone += 1
            for row_id, label, owner, picture in list(db.execute('SELECT id, label, owner, picture FROM things')):
                if same(label.casefold()):
                    db.execute('DELETE FROM things WHERE id = ?', (row_id,))
                    if picture and os.path.exists(picture):
                        os.remove(picture)
                    gone += 1
        self._load()
        return gone

    def names(self):
        with self.lock:
            seen = []
            for name, _ in self.people:
                if name not in seen:
                    seen.append(name)
            return seen

    def remember_thing(self, label, owner, description, picture_bytes=None):
        picture = None
        if picture_bytes:
            folder = os.path.join(os.path.dirname(self.db_path), 'things')
            if not os.path.isdir(folder):
                os.makedirs(folder)
            picture = os.path.join(folder, '{}.jpg'.format(int(time.time() * 1000)))
            with open(picture, 'wb') as handle:
                handle.write(picture_bytes)
        with self.lock, self._db() as db:
            db.execute('INSERT INTO things (label, owner, description, picture, created_at) VALUES (?, ?, ?, ?, ?)',
                       (label, owner, description, picture, time.time()))
        self._load()

    def things_text(self, limit=15):
        with self.lock:
            rows = self.things[-limit:]
        return '; '.join('{}{}: {}'.format(label, ' ({})'.format(owner) if owner else '', description)
                         for _, label, owner, description in rows)


# -- voice commands -------------------------------------------------------
REMEMBER = r'\b(?:запомни|запомнишь|запомните|remember)\b'
NAME = r'([A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі\-]{2,20}(?:\s+[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі\-]{2,20})?)'
SELF_NAME_RE = re.compile(r'(?:меня зовут|моё имя|мое имя|\bя\s*[-—,]?|my name is|\bi am|\bi\'m|\bim)\s+' + NAME, re.I)
OTHER_NAME_RE = re.compile(r'(?:(?:его|её|ее|их) зовут|\bэто|\bthis is|his name is|her name is)\s+' + NAME, re.I)
THING_WORDS = re.compile(r'\b(?:мой|моя|моё|мое|мои|наш|наша|наше|этот|эту|эта|предмет\w*|вещ\w*|my|our|this thing|this object)\b', re.I)
FORGET_RE = re.compile(r'\b(?:забудь|forget)\s+(?:про\s+|about\s+)?' + NAME, re.I)
LIST_RE = re.compile(r'(?:кого ты (?:знаешь|помнишь|запомнила)|кого ты уже знаешь|who do you (?:know|remember)|'
                     r'что ты (?:запомнила|помнишь из вещей))', re.I)
NOT_NAMES = {'тебя', 'тебе', 'тут', 'здесь', 'сейчас', 'пожалуйста', 'робот', 'айсана', 'это', 'что', 'кто',
             'please', 'here', 'now', 'aisana', 'не', 'же', 'ну', 'неважно', 'всё', 'все', 'об', 'этом',
             'it', 'that', 'this', 'about', 'и', 'а'}
SELF_WORDS = {'меня', 'me', 'myself'}


def _name(raw):
    words = [w.strip(' -—,.!?') for w in re.split(r'\s+', raw.strip(' -—,.!?'))]
    words = [w for w in words if w]
    words = [w for w in words if w.casefold() not in NOT_NAMES]
    if not words:
        return None
    # "я Али и ..." keeps the first word; two words only for a surname ("Али Куаныш").
    keep = words[:2] if len(words) > 1 and words[1][:1].isupper() else words[:1]
    return ' '.join(w[:1].upper() + w[1:] for w in keep)


def parse_memory_command(text):
    """A face/thing memory command in the user's words, or None.

    {'action': 'remember_me'|'remember_other'|'remember_thing'|'forget_me'|'forget'|'list',
     'name': str|None, 'label': str|None}
    """
    text = (text or '').strip()
    if LIST_RE.search(text):
        return {'action': 'list', 'name': None, 'label': None}
    forget = FORGET_RE.search(text)
    if forget:
        target = forget.group(1).split()[0].casefold()
        if target in SELF_WORDS:
            return {'action': 'forget_me', 'name': None, 'label': None}
        name = _name(forget.group(1))
        if name:
            return {'action': 'forget', 'name': name, 'label': None}
        return None
    remember = re.search(REMEMBER + r'(.*)$', text, re.I)
    if not remember:
        return None
    rest = remember.group(1)
    if re.search(r'\b(?:меня|me|моё лицо|мое лицо|my face)\b', rest, re.I):
        found = SELF_NAME_RE.search(rest) or SELF_NAME_RE.search(text[:remember.start()])
        return {'action': 'remember_me', 'name': _name(found.group(1)) if found else None, 'label': None}
    if THING_WORDS.search(rest):
        label = re.sub(r'^[\s,.:—-]*(?:(?:это|вот|this is|this|here is)\b)?\s*', '', rest, flags=re.I)
        label = label.strip(' ,.!?')
        return {'action': 'remember_thing', 'name': None, 'label': label or None}
    found = OTHER_NAME_RE.search(rest) or SELF_NAME_RE.search(rest)
    if found:
        action = 'remember_me' if SELF_NAME_RE.search(rest) and not OTHER_NAME_RE.search(rest) else 'remember_other'
        return {'action': action, 'name': _name(found.group(1)), 'label': None}
    if re.search(r'\b(?:его|её|ее|him|her|лицо|face)\b', rest, re.I):
        return {'action': 'remember_other', 'name': None, 'label': None}
    return None


def connect_from_environment(start_camera=True):
    """(CameraStream, FaceMemory) or (None, None) with a printed reason."""
    if not ENABLED:
        return None, None
    try:
        import cv2  # noqa: F401  (JetPack OpenCV)
        import numpy  # noqa: F401
    except ImportError as exc:
        print('Face memory off: {} (run with JetPack python / .venv with system packages)'.format(exc))
        return None, None
    camera = CameraStream().start() if start_camera else None
    try:
        memory = FaceMemory()
        memory._load_nets()
        print('Face memory ready: {} people, {} things.'.format(len(memory.names()), len(memory.things)))
    except Exception as exc:
        print('Face memory off: {}'.format(exc))
        memory = None
    return camera, memory
