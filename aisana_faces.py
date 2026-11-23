"""Always-on camera and face/thing memory for Aisana (Python 3.6, JetPack OpenCV 4.1.1).

- CameraStream keeps the Jetson camera open and always holds a fresh frame,
  so "кто перед тобой?" no longer waits 2-3 s for the camera to start.
- FaceMemory finds faces and 5 landmarks with YuNet, aligns them and turns
  each face into 128 numbers with SFace (both run on cv2.dnn, no pip installs).
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
# New file for SFace: the old OpenFace vectors cannot be compared with these.
DB_PATH = os.path.expanduser(os.environ.get('AISANA_FACES_DB', os.path.join(DATA_DIR, 'faces_sface.sqlite3')))
# L2 distance between normalized SFace vectors: same person ~0.2-0.6, others ~1.2+.
THRESHOLD = float(os.environ.get('AISANA_FACE_THRESHOLD', '1.0'))
MARGIN = 0.12                     # the best person must beat the second best by this much
MIN_FACE_PX = int(os.environ.get('AISANA_FACE_MIN_PX', '40'))
MAX_VIEWS = 40                    # stored views per person (it keeps learning new angles)
ENABLED = os.environ.get('AISANA_FACES', '1') == '1'

# YuNet (face + 5 landmarks, MIT) converted for JetPack OpenCV 4.1.1 by
# tools/convert_yunet.py and kept in the repo; SFace (Apache-2.0) is downloaded.
# Landmarks let the face be aligned before SFace: OpenFace on unaligned crops
# could not tell Куаныш from Шынгыс (Jetson log, Nov 20).
YUNET_PATH = os.path.join(HERE, 'face_models', 'yunet_320_opencv41.onnx')
MODELS = {
    'face_recognition_sface_2021dec.onnx': (
        'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/'
        'face_recognition_sface_2021dec.onnx',
        '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
}
YUNET_OUTPUTS = ['cls_8', 'cls_16', 'cls_32', 'obj_8', 'obj_16', 'obj_32',
                 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
# Where SFace expects the eyes, nose and mouth corners in its 112x112 input.
SFACE_TEMPLATE = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
                  (41.5493, 92.3655), (70.7299, 92.2041))

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
    """Paths of the model files; downloads SFace (about 39 MB, once) if missing."""
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
            with requests.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                total = int(response.headers.get('content-length') or 0)
                done, shown, started = 0, 0, time.monotonic()
                with open(part, 'wb') as handle:
                    for block in response.iter_content(1 << 16):
                        handle.write(block)
                        done += len(block)
                        if done - shown >= 4 << 20:     # progress every 4 MB
                            shown = done
                            print('  {} MB of {} MB ({:.0f} s)'.format(
                                done >> 20, total >> 20 if total else '?', time.monotonic() - started))
            if _sha256(part) != sha:
                os.remove(part)
                raise RuntimeError('face model {} is corrupted (checksum)'.format(name))
            os.rename(part, path)
        paths[name] = path
    paths['yunet'] = YUNET_PATH
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


def decode_yunet(outputs, size=320, threshold=0.6):
    """YuNet raw outputs -> [(x1, y1, x2, y2, score, landmarks)] in input pixels."""
    import numpy as np
    faces = []
    for k, stride in enumerate((8, 16, 32)):
        cls, obj = outputs[k][0][:, 0], outputs[3 + k][0][:, 0]
        bbox, kps = outputs[6 + k][0], outputs[9 + k][0]
        cols = size // stride
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for idx in np.where(score > threshold)[0]:
            row, col = divmod(int(idx), cols)
            cx, cy = (col + bbox[idx, 0]) * stride, (row + bbox[idx, 1]) * stride
            w, h = np.exp(bbox[idx, 2]) * stride, np.exp(bbox[idx, 3]) * stride
            points = [((kps[idx, 2 * p] + col) * stride, (kps[idx, 2 * p + 1] + row) * stride) for p in range(5)]
            faces.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(score[idx]), points))
    faces.sort(key=lambda f: -f[4])
    kept = []
    for face in faces:                 # drop overlapping duplicates
        if all(face[2] < k[0] or face[0] > k[2] or face[3] < k[1] or face[1] > k[3]
               or abs((face[0] + face[2]) - (k[0] + k[2])) > (k[2] - k[0]) * 0.5 for k in kept):
            kept.append(face)
    return kept


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
        self.ready = threading.Event()   # networks loaded (the download runs in the background)
        if embedder is not None or detector is not None:
            self.ready.set()
        self.people = []                  # [(name, vector)]
        self.titles = {}                  # name -> "President of Nazarbayev University" (guests)
        self.things = []                  # [(id, label, owner, description)]
        folder = os.path.dirname(db_path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'name TEXT NOT NULL, vector BLOB NOT NULL, created_at REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS titles (name TEXT PRIMARY KEY, title TEXT NOT NULL)')
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
            self.titles = dict(db.execute('SELECT name, title FROM titles'))

    # -- networks ---------------------------------------------------------
    def _load_nets(self):
        if self.nets is None:
            import cv2
            paths = self.models or ensure_models()
            detector = cv2.dnn.readNetFromONNX(paths['yunet'])
            embedder = cv2.dnn.readNetFromONNX(paths['face_recognition_sface_2021dec.onnx'])
            self.nets = (detector, embedder)
            self.ready.set()
        return self.nets

    def _yunet(self, image, threshold):
        """Faces in any image: padded to a square, run at 320x320, mapped back."""
        import cv2
        import numpy as np
        height, width = image.shape[:2]
        side = max(height, width)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[:height, :width] = image
        scale = side / 320.0
        detector = self._load_nets()[0]
        with self.lock:
            detector.setInput(cv2.dnn.blobFromImage(cv2.resize(square, (320, 320))))
            outputs = detector.forward(YUNET_OUTPUTS)
        return [(x1 * scale, y1 * scale, x2 * scale, y2 * scale, score, [(x * scale, y * scale) for x, y in points])
                for x1, y1, x2, y2, score, points in decode_yunet(outputs, 320, threshold)]

    def detect(self, frame, min_confidence=0.6):
        """Faces [(x1, y1, x2, y2, confidence, landmarks)], biggest first.

        A second pass on a crop around each face gives precise landmarks:
        at 320 px for the whole frame the same person drifted to distance 1.0."""
        if self.detector is not None:
            return self.detector(frame)
        height, width = frame.shape[:2]
        found = []
        for x1, y1, x2, y2, score, points in self._yunet(frame, min_confidence):
            if x2 - x1 < MIN_FACE_PX:
                continue
            side = int(max(x2 - x1, y2 - y1) * 1.8)
            left = int(max(0, (x1 + x2) / 2 - side / 2))
            top = int(max(0, (y1 + y2) / 2 - side / 2))
            crop = frame[top:min(height, top + side), left:min(width, left + side)]
            fine = self._yunet(crop, 0.5) if crop.size else []
            if fine:
                points = [(x + left, y + top) for x, y in fine[0][5]]
            box = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))
            found.append(box + (score, points))
        found.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return found

    def embed(self, frame, box):
        import numpy as np
        if self.embedder is not None:
            vector = np.asarray(self.embedder(frame, box), dtype=np.float32)
        else:
            import cv2
            embedder = self._load_nets()[1]
            matrix, _ = cv2.estimateAffinePartial2D(np.array(box[5], dtype=np.float32),
                                                    np.array(SFACE_TEMPLATE, dtype=np.float32),
                                                    method=cv2.LMEDS)
            face = cv2.warpAffine(frame, matrix, (112, 112))
            with self.lock:
                embedder.setInput(cv2.dnn.blobFromImage(face, 1.0, (112, 112), (0, 0, 0), True, False))
                vector = embedder.forward().flatten().astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return vector / norm

    def identify(self, vector):
        """(name, distance) of the closest known person; name None if nobody is
        close enough or two people are almost equally close (unsure)."""
        import numpy as np
        best = {}
        with self.lock:
            for name, known in self.people:
                distance = float(np.linalg.norm(known - vector))
                if distance < best.get(name, 9.0):
                    best[name] = distance
        if not best:
            return None, 9.0
        ranked = sorted(best.items(), key=lambda item: item[1])
        name, distance = ranked[0]
        if distance > self.threshold:
            return None, distance
        if len(ranked) > 1 and ranked[1][1] - distance < MARGIN:
            return None, distance
        return name, distance

    def faces(self, frame):
        """Everyone in the frame: [{'box', 'vector', 'name', 'distance', 'where'}], biggest first."""
        if not self.ready.is_set():
            return []                       # models still downloading
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
        if not (0.35 < distance < self.threshold - 0.25):
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

    def set_title(self, name, title):
        """A guest's role, for respectful greetings (enrolled from a photo with consent)."""
        with self.lock, self._db() as db:
            if title:
                db.execute('INSERT OR REPLACE INTO titles (name, title) VALUES (?, ?)', (name, title))
            else:
                db.execute('DELETE FROM titles WHERE name = ?', (name,))
        self._load()

    def title(self, name):
        return self.titles.get(name)

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
                    db.execute('DELETE FROM titles WHERE name = ?', (known,))
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
             'it', 'that', 'this', 'about', 'и', 'а', 'наш', 'наши', 'наш разговор', 'our'}
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
    except Exception as exc:
        print('Face memory off: {}'.format(exc))
        return camera, None

    def load():
        # In the background: a slow download must not keep Aisana from starting
        # (Jetson log, Nov 20: she sat silent while the model downloaded).
        try:
            memory._load_nets()
            print('Face memory ready: {} people, {} things.'.format(len(memory.names()), len(memory.things)))
            old = os.path.join(DATA_DIR, 'faces.sqlite3')
            if not memory.names() and os.path.exists(old):
                print('Face memory: the face model changed; please introduce yourselves again ("запомни меня, я ...").')
        except Exception as exc:
            print('Face memory off: {}'.format(exc))
    threading.Thread(target=load, daemon=True).start()
    return camera, memory
