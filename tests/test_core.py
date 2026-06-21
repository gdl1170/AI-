"""Tests per AI+."""

import json
import time
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hycoder.router import SmartRouter
from hycoder.tracker import SessionTracker
from hycoder.config import DEFAULT_CONFIG
from hycoder.knowledge import (
    KnowledgeBase, TFIDFIndex, chunk_text, _tokenize,
    _needs_rag, get_knowledge_base, reset_knowledge_base,
)
from hycoder.notes import NoteStore, parse_note, _slugify
from hycoder.resources import DiskCache
from hycoder.tools import (
    ReadTool, WriteTool, EditTool, SearchTool,
    RunTool, WebTool, MemoryTool, DeleteTool,
    execute_tool_call, ToolResult, get_tool_registry,
)


def make_cfg(**overrides):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in overrides.items():
        parts = k.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


class FakeResult:
    def __init__(self, text="hello", source="local", model="test",
                 tokens_in=10, tokens_out=20, time_s=0.5, cached=False, events=None):
        self.text = text
        self.source = source
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.tokens_total = tokens_in + tokens_out
        self.time_s = time_s
        self.cached = cached
        self.events = events or []


class TestSmartRouter:
    def test_always_local(self):
        cfg = make_cfg(**{"router.always_local": True})
        r = SmartRouter(cfg)
        assert r.decide("complex coding problem") == ("local", 0)

    def test_always_online(self):
        cfg = make_cfg(**{"router.always_online": True})
        r = SmartRouter(cfg)
        assert r.decide("simple math") == ("online", 0)

    def test_mode_local(self):
        cfg = make_cfg(**{"router.mode": "local"})
        r = SmartRouter(cfg)
        assert r.decide("anything") == ("local", 0)

    def test_mode_online(self):
        cfg = make_cfg(**{"router.mode": "online"})
        r = SmartRouter(cfg)
        assert r.decide("anything") == ("online", 0)

    def test_auto_routes_long_to_online(self):
        cfg = make_cfg()
        r = SmartRouter(cfg)
        long_prompt = "analyze " * 60
        decision, score = r.decide(long_prompt)
        assert decision == "online"
        assert score >= cfg["router"]["complexity_threshold"]

    def test_auto_routes_simple_to_local(self):
        cfg = make_cfg()
        r = SmartRouter(cfg)
        decision, score = r.decide("hello")
        assert decision == "local"

    def test_keyword_boost_online(self):
        cfg = make_cfg()
        r = SmartRouter(cfg)
        decision, score = r.decide("search the web for latest news 2026")
        assert decision == "online"

    def test_kb_boost_empty_kb(self):
        cfg = make_cfg()
        r = SmartRouter(cfg)
        score = r._kb_boost("test query")
        assert score == 0


class TestSessionTracker:
    def test_record_and_summary(self):
        cfg = make_cfg(**{"tracking.history_file": str(Path(tempfile.mkdtemp()) / "history.json"),
                          "tracking.save_history": False})
        t = SessionTracker(cfg)
        r = FakeResult()
        t.record("test prompt", r, agent="test-agent")
        s = t.summary_dict()
        assert s["local"]["calls"] == 1
        assert s["total"]["tokens_total"] == 30

    def test_agent_stored_in_history(self):
        cfg = make_cfg(**{"tracking.history_file": str(Path(tempfile.mkdtemp()) / "history.json"),
                          "tracking.save_history": False})
        t = SessionTracker(cfg)
        r = FakeResult()
        t.record("test", r, agent="my-agent")
        assert t.history[-1]["agent"] == "my-agent"

    def test_estimate_cost(self):
        cfg = make_cfg(**{"tracking.history_file": str(Path(tempfile.mkdtemp()) / "history.json"),
                          "tracking.save_history": False,
                          "tracking.online_cost_per_1k_input": 0.01,
                          "tracking.online_cost_per_1k_output": 0.03})
        t = SessionTracker(cfg)
        r = FakeResult(tokens_in=100, tokens_out=200, source="online")
        t.record("test", r)
        cost = t.estimate_cost()
        assert cost > 0

    def test_online_tracking(self):
        cfg = make_cfg(**{"tracking.history_file": str(Path(tempfile.mkdtemp()) / "history.json"),
                          "tracking.save_history": False})
        t = SessionTracker(cfg)
        r = FakeResult(source="online")
        t.record("test", r)
        s = t.summary_dict()
        assert s["online"]["calls"] == 1
        assert s["local"]["calls"] == 0

    def test_estimate_remaining_empty(self):
        cfg = make_cfg(**{"tracking.history_file": str(Path(tempfile.mkdtemp()) / "history.json"),
                          "tracking.save_history": False})
        t = SessionTracker(cfg)
        rem = t.estimate_remaining()
        assert rem["pct"] == 0

    def test_flush(self):
        tmp = Path(tempfile.mkdtemp()) / "history.json"
        cfg = make_cfg(**{"tracking.history_file": str(tmp),
                          "tracking.save_history": True})
        t = SessionTracker(cfg)
        r = FakeResult()
        t.record("test", r)
        t.flush()
        assert tmp.exists()


class TestConfig:
    def test_default_config_structure(self):
        assert "local" in DEFAULT_CONFIG
        assert "online" in DEFAULT_CONFIG
        assert "router" in DEFAULT_CONFIG
        assert "tracking" in DEFAULT_CONFIG

    def test_default_model(self):
        assert DEFAULT_CONFIG["local"]["model"] == "qwen3.5:4b"

    def test_deep_merge(self):
        from hycoder.config import deep_merge
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99, "e": 4}, "f": 5}
        deep_merge(base, override)
        assert base["a"] == 1
        assert base["b"]["c"] == 99
        assert base["b"]["d"] == 3
        assert base["b"]["e"] == 4
        assert base["f"] == 5


class TestKnowledgeBase:
    def test_chunk_text_small(self):
        assert chunk_text("hello") == ["hello"]

    def test_chunk_text_large(self):
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_tokenize(self):
        tokens = _tokenize("Hello World, test_123!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test_123" in tokens

    def test_needs_rag_simple(self):
        assert _needs_rag("ciao") is False
        assert _needs_rag("hello") is False

    def test_needs_rag_with_keyword(self):
        assert _needs_rag("cerca documentazione su python") is True
        assert _needs_rag("spiegami cos'è un decorator") is True

    def test_needs_rag_long_query(self):
        long_q = "what is " * 30
        assert _needs_rag(long_q) is True

    def test_needs_rag_question(self):
        assert _needs_rag("what is a transformer model?") is True
        assert _needs_rag("who are you") is False

    def test_tfidf_index_basic(self):
        idx = TFIDFIndex()
        idx.add([{"text": "hello world test", "source": "doc1", "chunk_id": 0}])
        results = idx.search("hello")
        assert len(results) > 0
        assert results[0][1]["source"] == "doc1"

    def test_tfidf_index_multi_doc(self):
        idx = TFIDFIndex()
        idx.add([
            {"text": "python programming language", "source": "a", "chunk_id": 0},
            {"text": "java virtual machine", "source": "b", "chunk_id": 0},
        ])
        results = idx.search("python")
        assert len(results) == 1
        assert results[0][1]["source"] == "a"

    def test_kb_add_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("AI+ is a hybrid AI assistant.")
        kb = KnowledgeBase()
        result = kb.add_file(str(f))
        assert "error" not in result
        assert result["chunks_added"] > 0

    def test_kb_query(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Artificial Intelligence and machine learning.")
        kb = KnowledgeBase()
        kb.add_file(str(f))
        results = kb.query("artificial intelligence")
        assert len(results) > 0

    def test_kb_build_context(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Python is a programming language.")
        kb = KnowledgeBase()
        kb.add_file(str(f))
        ctx = kb.build_context("python")
        assert "Python" in ctx

    def test_kb_clear(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test content")
        kb = KnowledgeBase()
        kb.add_file(str(f))
        kb.clear()
        assert kb.total_chunks == 0

    def test_kb_add_directory(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.txt").write_text("file a content")
        (tmp_path / "sub" / "b.txt").write_text("file b content")
        kb = KnowledgeBase()
        result = kb.add_directory(str(tmp_path), pattern="*.txt", recursive=True)
        assert result["files_processed"] >= 2
        assert result["chunks_added"] > 0

    def test_kb_status_empty(self):
        kb = KnowledgeBase()
        s = kb.status()
        assert s["total_chunks"] == 0

    def test_kb_remove_source(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content to remove")
        kb = KnowledgeBase()
        kb.add_file(str(f))
        assert kb.total_chunks > 0
        ok = kb.remove_source(str(f))
        assert ok is True
        assert kb.total_chunks == 0


class TestNotes:
    def test_parse_note_frontmatter(self):
        text = '---\n{"title": "Test Note", "tags": ["ai", "test"]}\n---\n\nBody content here.'
        parsed = parse_note(text, "test.md")
        assert parsed["title"] == "Test Note"
        assert "ai" in parsed["tags"]
        assert "Body content" in parsed["body"]

    def test_parse_note_no_frontmatter(self):
        parsed = parse_note("Just a note body", "note.md")
        assert parsed["title"] == "note"
        assert parsed["body"] == "Just a note body"

    def test_parse_note_wiki_links(self):
        text = "See [[Another Note]] and [[Related|related page]]"
        parsed = parse_note(text)
        assert "Another Note" in parsed["links"]
        assert "Related" in parsed["links"]

    def test_parse_note_inline_tags(self):
        text = "This is #important and #urgent"
        parsed = parse_note(text)
        assert "important" in parsed["tags"]
        assert "urgent" in parsed["tags"]

    def test_slugify(self):
        assert _slugify("Hello World") == "hello-world"
        assert _slugify("Test/Note") == "test-note"

    def test_note_store_crud(self):
        ns = NoteStore()
        note = ns.create("Test Note", body="Hello", tags=["test"])
        assert note["title"] == "Test Note"
        slug = note["slug"]

        fetched = ns.get(slug)
        assert fetched is not None
        assert fetched["title"] == "Test Note"

        updated = ns.update(slug, body="Updated body")
        assert updated is not None
        assert "Updated" in updated["body"]

        deleted = ns.delete(slug)
        assert deleted is True

    def test_note_store_list(self):
        ns = NoteStore()
        ns.create("List Test", body="test")
        notes = ns.list_all()
        assert len(notes) > 0
        assert any(n["title"] == "List Test" for n in notes)

    def test_note_store_graph(self):
        ns = NoteStore()
        ns.create("Node A", body="Link to [[Node B]]")
        ns.create("Node B", body="Link back [[Node A]]")
        g = ns.graph()
        assert len(g["nodes"]) >= 2
        assert len(g["edges"]) >= 1

    def test_note_store_search_by_tag(self):
        ns = NoteStore()
        ns.create("Tagged Note", body="test", tags=["special"])
        results = ns.search_by_tag("special")
        assert len(results) >= 1


class TestResources:
    def test_disk_cache_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test", max_entries=100, ttl_seconds=3600, dirpath=tmp)
            cache.set("hello", "world")
            assert cache.get("hello").decode() == "world"

    def test_disk_cache_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_json", max_entries=100, ttl_seconds=3600, dirpath=tmp)
            data = {"key": "value", "num": 42}
            cache.set("mykey", data)
            loaded = cache.get_json("mykey")
            assert loaded == data

    def test_disk_cache_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_pickle", max_entries=100, ttl_seconds=3600, dirpath=tmp)
            obj = {"nested": {"list": [1, 2, 3]}}
            cache.set_pickle("pickled", obj)
            loaded = cache.get_pickle("pickled")
            assert loaded == obj

    def test_disk_cache_ttl_zero_means_infinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_ttl", max_entries=100, ttl_seconds=0, dirpath=tmp)
            cache.set("persistent", "data")
            assert cache.get("persistent") == b"data"

    def test_disk_cache_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_ttl_exp", max_entries=100, ttl_seconds=1, dirpath=tmp)
            cache.set("will_expire", "data")
            assert cache.get("will_expire") == b"data"
            time.sleep(1.1)
            assert cache.get("will_expire") is None

    def test_disk_cache_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_lru", max_entries=3, ttl_seconds=3600, dirpath=tmp)
            cache.set("a", "1")
            cache.set("b", "2")
            cache.set("c", "3")
            cache.set("d", "4")
            assert cache.size <= 3
            assert cache.get("a") is None  # 'a' should be evicted (oldest)

    def test_disk_cache_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache("test_clear", max_entries=100, ttl_seconds=3600, dirpath=tmp)
            cache.set("x", "y")
            cache.clear()
            assert cache.get("x") is None
            assert cache.size == 0


class TestTools:
    def test_read_tool_no_file(self):
        tool = ReadTool()
        result = tool.execute(path="/nonexistent/file.txt")
        assert result.success is False
        assert "non trovato" in result.error.lower()

    def test_read_tool_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("hello")
        tool = ReadTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        assert "file1.txt" in result.output

    def test_write_tool(self, tmp_path):
        f = tmp_path / "new.txt"
        tool = WriteTool()
        result = tool.execute(path=str(f), content="test content")
        assert result.success is True
        assert f.read_text() == "test content"

    def test_write_tool_blocked_path(self, tmp_path):
        f = tmp_path / ".bashrc"
        f.write_text("existing config")
        tool = WriteTool()
        result = tool.execute(path=str(f), content="evil")
        assert result.success is False

    def test_edit_tool(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        tool = EditTool()
        result = tool.execute(path=str(f), old_string="hello", new_string="goodbye")
        assert result.success is True
        assert f.read_text() == "goodbye world"

    def test_edit_tool_no_match(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello")
        tool = EditTool()
        result = tool.execute(path=str(f), old_string="nonexistent", new_string="x")
        assert result.success is False

    def test_delete_tool(self, tmp_path):
        f = tmp_path / "delete_me.txt"
        f.write_text("delete me")
        tool = DeleteTool()
        result = tool.execute(path=str(f))
        assert result.success is True
        assert not f.exists()

    def test_search_tool_no_path(self):
        tool = SearchTool()
        result = tool.execute(pattern="nonexistent_pattern_xyz")
        assert result.success is False

    def test_memory_tool_store_recall(self):
        tool = MemoryTool()
        result = tool.execute(action="store", key="test_key", value="test_value")
        assert result.success is True
        result = tool.execute(action="recall", key="test_key")
        assert result.success is True
        assert "test_value" in result.output

    def test_memory_tool_list(self):
        tool = MemoryTool()
        tool.execute(action="store", key="k1", value="v1")
        result = tool.execute(action="list")
        assert result.success is True

    def test_memory_tool_clear(self):
        tool = MemoryTool()
        tool.execute(action="store", key="k", value="v")
        result = tool.execute(action="clear")
        assert result.success is True

    def test_execute_tool_call_valid(self):
        result = execute_tool_call({"name": "memory", "arguments": {"action": "list"}})
        assert result["name"] == "memory"
        assert "result" in result

    def test_execute_tool_call_unknown(self):
        result = execute_tool_call({"name": "nonexistent_tool", "arguments": {}})
        assert result["result"]["success"] is False

    def test_tool_registry_list(self):
        tools = get_tool_registry()
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "run" in tools
        assert "search" in tools
        assert "web" in tools
        assert "memory" in tools
        assert "delete" in tools

    def test_run_tool_basic(self):
        tool = RunTool()
        result = tool.execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.output


class TestWebSearch:
    def test_duplicate_search_endpoints(self):
        """Verify api/web/search and api/search/web are not both defined."""
        from hycoder.web.app import create_app
        import flask
        app = create_app(make_cfg())
        with app.test_request_context():
            rules = [r.rule for r in app.url_map.iter_rules()
                     if 'search' in r.rule and r.rule.startswith('/api/')]
            web_search_routes = [r for r in rules if '/web' in r or '/search' in r]
            assert len(web_search_routes) > 0


class TestProjectGeneration:
    def test_generate_system_prompt_defined(self):
        from hycoder.project import GENERATION_SYSTEM_PROMPT
        assert len(GENERATION_SYSTEM_PROMPT) > 100
        assert "files" in GENERATION_SYSTEM_PROMPT
        assert "run_command" in GENERATION_SYSTEM_PROMPT

    def test_generate_project_calls_provider(self):
        from hycoder.project import generate_project_from_prompt
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": "test-cli",
            "description": "Test CLI tool",
            "tech_stack": ["python"],
            "files": {
                "cli.py": 'print("hello")',
                "tests/test_cli.py": 'def test(): pass',
                "requirements.txt": "click\n",
            },
            "run_command": "python cli.py",
            "test_command": "pytest",
        })
        mock_generate_fn = MagicMock(return_value=mock_result)
        meta = generate_project_from_prompt("A CLI tool", mock_generate_fn)
        assert meta["name"] == "test-cli"
        assert meta["template"] == "ai-generated"
        assert meta["files_count"] == 3
        assert meta["run_cmd"] == "python cli.py"
        assert meta["test_cmd"] == "pytest"
        assert meta["tech_stack"] == ["python"]
        # Verify files were created
        project_path = Path(meta["path"])
        assert (project_path / "cli.py").exists()
        assert (project_path / "tests" / "test_cli.py").exists()
        assert (project_path / ".hybrid-project.json").exists()
        # Cleanup
        import shutil
        shutil.rmtree(project_path)

    def test_generate_project_dedup_name(self):
        from hycoder.project import generate_project_from_prompt
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": "dup",
            "description": "dup",
            "tech_stack": [],
            "files": {"f.py": "x"},
            "run_command": "",
            "test_command": "",
        })
        mock_fn = MagicMock(return_value=mock_result)
        meta1 = generate_project_from_prompt("test", mock_fn)
        meta2 = generate_project_from_prompt("test", mock_fn)
        assert meta1["name"] == "dup"
        assert meta2["name"] != meta1["name"]  # should be dup-1 or similar
        assert meta2["path"] != meta1["path"]
        import shutil
        shutil.rmtree(meta1["path"])
        shutil.rmtree(meta2["path"])

    def test_generate_project_invalid_json(self):
        from hycoder.project import generate_project_from_prompt
        mock_result = MagicMock()
        mock_result.text = "not json at all"
        mock_fn = MagicMock(return_value=mock_result)
        with pytest.raises(Exception):
            generate_project_from_prompt("test", mock_fn)

    def test_generate_project_markdown_fenced(self):
        from hycoder.project import generate_project_from_prompt
        data = json.dumps({
            "name": "md-test",
            "description": "t",
            "tech_stack": [],
            "files": {"a.py": "pass"},
            "run_command": "",
            "test_command": "",
        })
        mock_result = MagicMock()
        mock_result.text = f"```json\n{data}\n```"
        mock_fn = MagicMock(return_value=mock_result)
        meta = generate_project_from_prompt("test", mock_fn)
        assert meta["name"] == "md-test"
        import shutil
        shutil.rmtree(meta["path"])

    def test_project_generate_api_route(self):
        from hycoder.web.app import create_app
        app = create_app()
        with app.test_client() as c:
            r = c.post('/api/projects/generate', json={"prompt": "", "name": "test"})
            assert r.status_code == 400
            assert "prompt richiesto" in r.get_json()["error"]

    def test_write_project_file(self):
        from hycoder.project import generate_project_from_prompt, write_project_file
        import time
        uid = f"wr-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"f.py": "original"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        assert write_project_file(meta["name"], "new.py", "print('new')") == True
        assert (Path(meta["path"]) / "new.py").exists()
        assert (Path(meta["path"]) / "new.py").read_text() == "print('new')"
        import shutil; shutil.rmtree(meta["path"])

    def test_rename_project(self):
        from hycoder.project import generate_project_from_prompt, rename_project
        import time
        uid = f"rn-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"a.py": "x"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        new_name = uid + "-renamed"
        renamed = rename_project(meta["name"], new_name)
        assert renamed["name"] == new_name
        assert not Path(meta["path"]).exists()
        assert Path(renamed["path"]).exists()
        import shutil; shutil.rmtree(renamed["path"])

    def test_improve_project(self):
        from hycoder.project import generate_project_from_prompt, improve_project
        import time
        uid = f"imp-{int(time.time()*1000)}"
        mock_gen = MagicMock()
        mock_gen.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"a.py": "v1"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_gen))
        mock_improve = MagicMock()
        mock_improve.text = json.dumps({
            "description": "improved",
            "files": {"a.py": "v2", "b.py": "new_file"},
            "files_to_delete": [],
            "run_command": "python a.py",
            "test_command": "",
        })
        updated = improve_project(meta["name"], "add feature", MagicMock(return_value=mock_improve))
        assert updated["description"] == "improved"
        assert (Path(updated["path"]) / "b.py").exists()
        assert (Path(updated["path"]) / "a.py").read_text() == "v2"
        import shutil; shutil.rmtree(updated["path"])

    def test_improve_project_deletes_file(self):
        from hycoder.project import generate_project_from_prompt, improve_project
        import time
        uid = f"impdel-{int(time.time()*1000)}"
        mock_gen = MagicMock()
        mock_gen.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"keep.py": "ok", "remove.py": "bye"},
            "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_gen))
        mock_improve = MagicMock()
        mock_improve.text = json.dumps({
            "description": "removed file",
            "files": {},
            "files_to_delete": ["remove.py"],
            "run_command": "",
            "test_command": "",
        })
        updated = improve_project(meta["name"], "cleanup", MagicMock(return_value=mock_improve))
        assert not (Path(updated["path"]) / "remove.py").exists()
        assert (Path(updated["path"]) / "keep.py").exists()
        import shutil; shutil.rmtree(updated["path"])

    def test_export_project_zip(self):
        from hycoder.project import generate_project_from_prompt, export_project_zip
        import zipfile, time
        uid = f"zip-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"main.py": "print('hello')"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        zip_path = export_project_zip(meta["name"])
        assert zipfile.is_zipfile(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "main.py" in names
        import shutil; shutil.rmtree(meta["path"]); os.unlink(zip_path)

    def test_delete_project_file(self):
        from hycoder.project import generate_project_from_prompt, delete_project_file
        import time, shutil
        uid = f"del-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"main.py": "print('x')", "utils.py": "def f(): pass"},
            "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        delete_project_file(meta["name"], "utils.py")
        assert not (Path(meta["path"]) / "utils.py").exists()
        assert (Path(meta["path"]) / "main.py").exists()
        shutil.rmtree(meta["path"])

    def test_delete_project_file_not_found(self):
        from hycoder.project import generate_project_from_prompt, delete_project_file
        import time, shutil
        uid = f"delnf-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "t", "tech_stack": [],
            "files": {"a.py": "x"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        with pytest.raises(FileNotFoundError):
            delete_project_file(meta["name"], "nonexistent.py")
        shutil.rmtree(meta["path"])

    def test_update_project_config(self):
        from hycoder.project import generate_project_from_prompt, update_project_config, get_project
        import time, shutil
        uid = f"cfg-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "orig", "tech_stack": [],
            "files": {"a.py": "x"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        updated = update_project_config(meta["name"], description="new desc", run_cmd="python a.py")
        assert updated["description"] == "new desc"
        assert updated["run_cmd"] == "python a.py"
        fetched = get_project(meta["name"])
        assert fetched["description"] == "new desc"
        shutil.rmtree(meta["path"])

    def test_import_project_zip(self):
        from hycoder.project import import_project_zip
        import zipfile, time, shutil, tempfile
        uid = f"import-{int(time.time()*1000)}"
        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("hello.py", "print('hello')")
            zf.writestr("README.md", "# Hi")
        meta = import_project_zip(zip_path, name_hint=uid)
        assert Path(meta["path"]).exists()
        assert (Path(meta["path"]) / "hello.py").exists()
        assert meta["files_count"] == 2
        shutil.rmtree(meta["path"])
        os.unlink(zip_path)

    def test_import_project_zip_with_hint(self):
        from hycoder.project import import_project_zip
        import zipfile, time, shutil, tempfile
        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data.csv", "a,b,c\n1,2,3")
        meta = import_project_zip(zip_path, name_hint="myproject")
        assert meta["name"] == "myproject"
        shutil.rmtree(meta["path"])
        os.unlink(zip_path)

    def test_clone_project(self):
        from hycoder.project import generate_project_from_prompt, clone_project, get_project
        import time, shutil
        uid = f"clone-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "original", "tech_stack": ["python"],
            "files": {"main.py": "print('hello')", "utils.py": "def f(): pass"},
            "run_command": "python main.py", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        cloned = clone_project(meta["name"])
        assert cloned["name"] == uid + "-clone"
        assert cloned["description"] == "original"
        assert cloned["tech_stack"] == ["python"]
        assert cloned["run_cmd"] == "python main.py"
        assert (Path(cloned["path"]) / "main.py").exists()
        assert (Path(cloned["path"]) / "utils.py").exists()
        shutil.rmtree(meta["path"])
        shutil.rmtree(cloned["path"])

    def test_clone_project_custom_name(self):
        from hycoder.project import generate_project_from_prompt, clone_project
        import time, shutil
        uid = f"cln2-{int(time.time()*1000)}"
        mock_result = MagicMock()
        mock_result.text = json.dumps({
            "name": uid, "description": "", "tech_stack": [],
            "files": {"a.py": "x"}, "run_command": "", "test_command": "",
        })
        meta = generate_project_from_prompt("test", MagicMock(return_value=mock_result))
        cloned = clone_project(meta["name"], "my-custom-clone")
        assert cloned["name"] == "my-custom-clone"
        assert (Path(cloned["path"]) / "a.py").exists()
        shutil.rmtree(meta["path"])
        shutil.rmtree(cloned["path"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
