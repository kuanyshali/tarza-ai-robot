"""Persistent local dialogue history for Aisana (Python 3.6 compatible).

Only text turns are stored; camera images and audio are never saved here.
Set AISANA_MEMORY_ENABLED=0 to disable persistence.
"""
import os
import sqlite3
import threading
import time

_LOCK = threading.RLock()
DB_PATH = os.path.expanduser(os.environ.get('AISANA_MEMORY_DB', '~/.local/share/aisana/dialogue.sqlite3'))
MAX_TURNS = max(2, int(os.environ.get('AISANA_MEMORY_MAX_TURNS', '100')))
CONTEXT_MESSAGES = max(2, int(os.environ.get('AISANA_CONTEXT_MESSAGES', '24')))
ENABLED = os.environ.get('AISANA_MEMORY_ENABLED', '1') == '1'


def _connect():
    folder = os.path.dirname(DB_PATH)
    if folder and not os.path.isdir(folder):
        try:
            os.makedirs(folder)
        except OSError:
            if not os.path.isdir(folder):
                raise
    db = sqlite3.connect(DB_PATH, timeout=5)
    db.execute('CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, user_text TEXT NOT NULL, assistant_text TEXT NOT NULL)')
    return db


def load_history():
    if not ENABLED:
        return []
    with _LOCK:
        with _connect() as db:
            rows = db.execute('SELECT user_text, assistant_text FROM turns ORDER BY id DESC LIMIT ?', (max(1, CONTEXT_MESSAGES // 2),)).fetchall()
    result = []
    for user_text, assistant_text in reversed(rows):
        result.extend([{'role': 'user', 'content': user_text}, {'role': 'assistant', 'content': assistant_text}])
    return result


def recent_turns(limit=12):
    """[(created_at, user_text, assistant_text)], oldest first."""
    if not ENABLED:
        return []
    with _LOCK:
        with _connect() as db:
            rows = db.execute('SELECT created_at, user_text, assistant_text FROM turns ORDER BY id DESC LIMIT ?',
                              (max(1, limit),)).fetchall()
    return list(reversed(rows))


def remember_turn(question, answer):
    if not ENABLED or not question or not answer:
        return
    with _LOCK:
        with _connect() as db:
            db.execute('INSERT INTO turns (created_at, user_text, assistant_text) VALUES (?, ?, ?)', (time.time(), question, answer))
            db.execute('DELETE FROM turns WHERE id NOT IN (SELECT id FROM turns ORDER BY id DESC LIMIT ?)', (MAX_TURNS,))
            db.commit()


def clear_history():
    with _LOCK:
        with _connect() as db:
            db.execute('DELETE FROM turns')
            db.commit()
