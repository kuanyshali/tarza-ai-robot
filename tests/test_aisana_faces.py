"""Offline tests for face and thing memory; fake networks, no camera."""
import os
import shutil
import tempfile
import time
import unittest

import aisana_faces as faces

try:
    import numpy as np
except ImportError:          # pragma: no cover - CI installs numpy
    np = None


class ParseTests(unittest.TestCase):
    def check(self, text, action, name=None):
        command = faces.parse_memory_command(text)
        self.assertIsNotNone(command, text)
        self.assertEqual(command['action'], action, text)
        if name is not None:
            self.assertEqual(command['name'], name, text)
        return command

    def test_remember_and_forget(self):
        self.check('Айсана, запомни меня, я Али', 'remember_me', 'Али')
        self.check('Айсана, запомни меня. Меня зовут Айгерим', 'remember_me', 'Айгерим')
        self.check('Айсана, я Али, запомни меня', 'remember_me', 'Али')
        self.check("remember me, I'm Ali", 'remember_me', 'Ali')
        self.assertIsNone(self.check('запомни меня пожалуйста', 'remember_me')['name'])
        self.check('Айсана, запомни его, это Нурлан', 'remember_other', 'Нурлан')
        self.assertEqual(self.check('Айсана, запомни, это моя кружка', 'remember_thing')['label'], 'моя кружка')
        self.check('Айсана, забудь меня', 'forget_me')
        self.check('забудь кружку', 'forget', 'Кружку')
        self.check('Айсана, кого ты знаешь?', 'list')

    def test_ordinary_speech_is_not_a_memory_command(self):
        for text in ('ты запомнила что я сказал?', 'запомни, что завтра пара', 'забудь, неважно',
                     'Как дела?', 'Айсана, развернись', 'Айсана, кто перед тобой?'):
            with self.subTest(text=text):
                self.assertIsNone(faces.parse_memory_command(text))


@unittest.skipIf(np is None, 'numpy is needed')
class FaceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        # Fake networks: the "face" is the frame's mean colour turned into a vector.
        self.memory = faces.FaceMemory(
            db_path=os.path.join(self.folder, 'faces.sqlite3'),
            detector=lambda frame: [(0, 0, 100, 100, 0.99)] if frame.any() else [],
            embedder=lambda frame, box: frame.reshape(-1, 3).mean(axis=0).tolist() + [1.0])

    def tearDown(self):
        shutil.rmtree(self.folder)

    def frame(self, colour):
        return np.full((540, 960, 3), colour, dtype=np.uint8)

    def test_recognizes_only_remembered_people(self):
        ali = self.frame((200, 10, 10))
        self.assertIsNone(self.memory.faces(ali)[0]['name'])
        self.memory.remember('Али', [self.memory.faces(ali)[0]['vector']])
        self.assertEqual(self.memory.faces(self.frame((198, 12, 11)))[0]['name'], 'Али')
        self.assertIsNone(self.memory.faces(self.frame((10, 10, 200)))[0]['name'])
        self.assertEqual(self.memory.faces(self.frame((0, 0, 0))), [])

    def test_memory_survives_restart_and_forget_deletes(self):
        ali = self.frame((200, 10, 10))
        self.memory.remember('Али', [self.memory.faces(ali)[0]['vector']])
        again = faces.FaceMemory(db_path=self.memory.db_path, detector=self.memory.detector,
                                 embedder=self.memory.embedder)
        self.assertEqual(again.names(), ['Али'])
        self.assertEqual(again.forget('али'), 1)
        self.assertEqual(again.names(), [])

    def test_things(self):
        self.memory.remember_thing('кружка', 'Али', 'Белая кружка с синим логотипом.', b'jpg')
        self.assertIn('кружка (Али): Белая кружка', self.memory.things_text())
        self.assertEqual(self.memory.forget('кружку'), 1)
        self.assertEqual(self.memory.things_text(), '')


@unittest.skipIf(np is None, 'numpy is needed')
class CameraStreamTests(unittest.TestCase):
    def test_keeps_the_newest_frame(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 is needed')

        class FakeCapture(object):
            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((720, 1280, 3), dtype=np.uint8)

            def release(self):
                pass
        stream = faces.CameraStream(opener=FakeCapture).start()
        try:
            frame = stream.latest(wait=2)
            self.assertEqual(frame.shape, (540, 960, 3))
        finally:
            stream.stop()


if __name__ == '__main__':
    unittest.main()
