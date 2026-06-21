"""
Session persistence per AI+.
Salva/carica conversazioni su disco per sopravvivere a riavvii.
"""

import os
import json
import time
import atexit
import threading
import uuid
from pathlib import Path

SESSIONS_DIR = Path.home() / ".config" / "hybrid-coder" / "sessions"

_CONV_MAX = 50


def _ensure_dir():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id):
    return SESSIONS_DIR / f"{session_id}.json"


def load_session(session_id):
    path = _session_path(session_id)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data.get("messages", [])
        except Exception:
            pass
    return []


def save_session(session_id, messages):
    _ensure_dir()
    path = _session_path(session_id)
    try:
        path.write_text(json.dumps({
            "session_id": session_id,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_count": len(messages),
            "messages": messages[-_CONV_MAX * 2:],
        }, ensure_ascii=False))
    except Exception:
        pass


def delete_session(session_id):
    path = _session_path(session_id)
    if path.exists():
        path.unlink()


def list_sessions():
    _ensure_dir()
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "session_id": data["session_id"],
                "updated": data.get("updated", ""),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            pass
    return sessions


# ── Conversation Store (in-memory + disk persistente) ──

class PersistentConversationStore:
    def __init__(self):
        self._conversations: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._dirty = set()
        self._load_all()

    def _load_all(self):
        _ensure_dir()
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                sid = data["session_id"]
                self._conversations[sid] = data.get("messages", [])
            except Exception:
                pass

    def get(self, session_id: str | None) -> tuple[str, list[dict]]:
        with self._lock:
            if not session_id or session_id not in self._conversations:
                session_id = session_id or uuid.uuid4().hex[:12]
                stored = load_session(session_id)
                self._conversations[session_id] = stored
            return session_id, self._conversations[session_id]

    def append(self, session_id: str, role: str, content: str):
        conv = self._conversations.get(session_id)
        if conv is not None:
            conv.append({"role": role, "content": content})
            if len(conv) > _CONV_MAX:
                self._conversations[session_id] = conv[-_CONV_MAX:]
            self._dirty.add(session_id)

    def flush(self, session_id: str | None = None):
        with self._lock:
            if session_id:
                if session_id in self._conversations:
                    save_session(session_id, self._conversations[session_id])
                self._dirty.discard(session_id)
            else:
                for sid in list(self._dirty):
                    if sid in self._conversations:
                        save_session(sid, self._conversations[sid])
                self._dirty.clear()

    def clear(self, session_id: str):
        with self._lock:
            self._conversations.pop(session_id, None)
            self._dirty.discard(session_id)
            delete_session(session_id)

    def messages(self, session_id: str) -> list[dict]:
        return self._conversations.get(session_id, [])


# Singolo globale
_store = None
_store_lock = threading.Lock()


def get_conversation_store():
    global _store
    with _store_lock:
        if _store is None:
            _store = PersistentConversationStore()
        return _store


# Auto-flush ogni 30s
def _auto_flush():
    store = get_conversation_store()
    store.flush()


_flush_timer = None


def start_auto_flush(interval=30):
    global _flush_timer
    if _flush_timer is not None:
        _flush_timer.cancel()

    def _run():
        _auto_flush()
        global _flush_timer
        _flush_timer = threading.Timer(interval, _run)
        _flush_timer.daemon = True
        _flush_timer.start()

    _flush_timer = threading.Timer(interval, _run)
    _flush_timer.daemon = True
    _flush_timer.start()

    # Flush su shutdown
    atexit.register(_auto_flush)
