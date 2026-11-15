import os
import sqlite3
import tempfile
import unittest
from unittest import mock


class MemoryTests(unittest.TestCase):
    def test_persists_and_reloads_dialogue(self):
        with tempfile.TemporaryDirectory() as folder:
            import aisana_memory
            path = os.path.join(folder, 'dialogue.sqlite3')
            with mock.patch.object(aisana_memory, 'DB_PATH', path), \
                 mock.patch.object(aisana_memory, 'ENABLED', True), \
                 mock.patch.object(aisana_memory, 'MAX_TURNS', 2), \
                 mock.patch.object(aisana_memory, 'CONTEXT_MESSAGES', 4):
                aisana_memory.remember_turn('Hello', 'Hi')
                aisana_memory.remember_turn('What is my name?', 'You did not tell me')
                aisana_memory.remember_turn('My name is Dana', 'Nice to meet you')
                history = aisana_memory.load_history()
                self.assertEqual(len(history), 4)
                self.assertEqual(history[0]['content'], 'What is my name?')
                self.assertEqual(history[-1]['content'], 'Nice to meet you')
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute('SELECT count(*) FROM turns').fetchone()[0], 2)
                aisana_memory.clear_history()
                self.assertEqual(aisana_memory.load_history(), [])


if __name__ == '__main__':
    unittest.main()
