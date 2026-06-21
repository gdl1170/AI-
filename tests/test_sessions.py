"""Tests per hycoder/sessions.py."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hycoder.sessions import (
    load_session,
    save_session,
    delete_session,
    list_sessions,
    PersistentConversationStore,
    get_conversation_store,
)


class TestSessionIO:
    def test_save_and_load(self, tmp_path):
        sid = "test-session-1"
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            save_session(sid, msgs)
            loaded = load_session(sid)
            assert loaded == msgs

    def test_load_nonexistent(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            loaded = load_session("nonexistent")
            assert loaded == []

    def test_save_truncates_long(self, tmp_path):
        sid = "trunc-test"
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(200)]
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            save_session(sid, msgs)
            loaded = load_session(sid)
            assert len(loaded) <= 100

    def test_delete_session(self, tmp_path):
        sid = "delete-me"
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            save_session(sid, [{"role": "user", "content": "hi"}])
            delete_session(sid)
            assert load_session(sid) == []

    def test_list_sessions(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            save_session("s1", [{"role": "user", "content": "a"}])
            save_session("s2", [{"role": "user", "content": "b"}])
            sessions = list_sessions()
            assert len(sessions) >= 2
            ids = [s["session_id"] for s in sessions]
            assert "s1" in ids
            assert "s2" in ids

    def test_list_empty(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            assert list_sessions() == []

    def test_corrupted_json(self, tmp_path):
        sid = "corrupted"
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir(parents=True)
        (sess_dir / f"{sid}.json").write_text("not json at all")
        with patch("hycoder.sessions.SESSIONS_DIR", sess_dir):
            assert load_session(sid) == []
            sessions = list_sessions()
            assert all(s["session_id"] != sid for s in sessions)


class TestPersistentConversationStore:
    def test_get_new_session(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid, msgs = store.get(None)
            assert len(sid) > 0
            assert msgs == []

    def test_get_existing_session(self, tmp_path):
        sid = "existing"
        expected = [{"role": "user", "content": "hello"}]
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            save_session(sid, expected)
            store = PersistentConversationStore()
            returned_sid, msgs = store.get(sid)
            assert returned_sid == sid
            assert msgs == expected

    def test_append_and_retrieve(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid, _ = store.get(None)
            store.append(sid, "user", "hello")
            store.append(sid, "assistant", "world")
            msgs = store.messages(sid)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["content"] == "world"

    def test_append_truncates(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            from hycoder.sessions import _CONV_MAX
            store = PersistentConversationStore()
            sid, _ = store.get(None)
            for i in range(_CONV_MAX + 20):
                store.append(sid, "user", f"msg-{i}")
            msgs = store.messages(sid)
            assert len(msgs) <= _CONV_MAX

    def test_clear(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid, _ = store.get(None)
            store.append(sid, "user", "hello")
            store.clear(sid)
            _, msgs = store.get(sid)
            assert msgs == []

    def test_flush_saves_to_disk(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid, _ = store.get(None)
            store.append(sid, "user", "persist me")
            store.flush(sid)
            assert (tmp_path / f"{sid}.json").exists()

    def test_flush_all(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid1, _ = store.get(None)
            sid2 = "other-session"
            store._conversations[sid2] = [{"role": "user", "content": "x"}]
            store._dirty.add(sid1)
            store._dirty.add(sid2)
            store.flush()
            assert (tmp_path / f"{sid1}.json").exists()
            assert (tmp_path / f"{sid2}.json").exists()

    def test_dirty_discard_after_flush(self, tmp_path):
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            sid, _ = store.get(None)
            store.append(sid, "user", "test")
            assert sid in store._dirty
            store.flush(sid)
            assert sid not in store._dirty

    def test_concurrent_get(self, tmp_path):
        import threading
        with patch("hycoder.sessions.SESSIONS_DIR", tmp_path):
            store = PersistentConversationStore()
            results = []

            def access():
                sid, msgs = store.get(None)
                store.append(sid, "user", "concurrent")
                results.append(sid)

            threads = [threading.Thread(target=access) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(results) == 10
            assert len(set(results)) == 10  # all unique session ids


class TestGetConversationStore:
    def test_singleton(self):
        store1 = get_conversation_store()
        store2 = get_conversation_store()
        assert store1 is store2

    def test_singleton_thread_safe(self):
        import threading
        stores = []

        def get():
            stores.append(get_conversation_store())

        threads = [threading.Thread(target=get) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(s is stores[0] for s in stores)
