"""Tests per hycoder/web/app.py — API routes della web app."""

import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hycoder.web.app import create_app, _get_conv_store


WEB_CFG = {
    "local": {
        "enabled": True,
        "provider": "ollama",
        "model": "test-model",
        "ollama_base_url": "http://localhost:11434",
        "keep_alive": "5m",
        "context_length": 8192,
        "max_tokens": 2048,
        "temperature": 0.7,
    },
    "online": {
        "enabled": True,
        "provider": "opencode",
        "model": "openai/gpt-4o-mini",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_key": "sk-test",
        "auth_method": "api_key",
        "username": None,
        "password": None,
        "base_url": None,
        "max_tokens": 4096,
        "temperature": 0.7,
        "opencode_path": str(Path.home() / ".opencode" / "bin" / "opencode"),
    },
    "router": {
        "mode": "auto",
        "complexity_threshold": 4,
        "always_local": False,
        "always_online": False,
        "keyword_boost_local": ["hello", "ciao", "help"],
        "keyword_boost_online": ["search", "cerca", "news"],
    },
    "tracking": {
        "save_history": False,
        "history_file": "/tmp/test-history.json",
        "show_cost": True,
        "local_cost_per_1k_tokens": 0.0,
        "online_cost_per_1k_input": 0.01,
        "online_cost_per_1k_output": 0.03,
    },
    "workspace": {"default_path": "."},
    "backup": {"enabled": False, "interval_hours": 24, "destination": "/tmp/backups"},
    "language": "en",
}


@pytest.fixture
def app():
    cfg = WEB_CFG.copy()
    app = create_app(cfg)
    app.config["TESTING"] = True
    app.config["TRACKER"] = MagicMock()
    app.config["TRACKER"].summary_dict.return_value = {
        "local": {"calls": 1, "tokens_in": 10, "tokens_out": 20, "tokens_total": 30, "time_s": 0.5},
        "online": {"calls": 0, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "time_s": 0},
        "total": {"calls": 1, "tokens_total": 30, "time_s": 0.5, "tokens_in": 10, "tokens_out": 20},
        "cost": 0.0,
        "session_duration_s": 60,
        "progress_pct": 0,
        "total_estimate_s": 0,
        "remaining_s": 0,
    }
    app.config["TRACKER"].history = []
    def _make_result(text, source, model, tokens_in=5, tokens_out=10, cached=False):
        r = MagicMock()
        r.text = text
        r.source = source
        r.model = model
        r.tokens_in = tokens_in
        r.tokens_out = tokens_out
        r.tokens_total = tokens_in + tokens_out
        r.time_s = 0.3
        r.cached = cached
        r.events = None
        return r

    app.config["LOCAL"] = MagicMock()
    app.config["LOCAL"].generate_chat.return_value = _make_result(
        "Mock local response", "local", "test-model",
    )
    app.config["ONLINE"] = MagicMock()
    app.config["ONLINE"].generate_chat.return_value = _make_result(
        "Mock online response", "online", "gpt-4o", tokens_in=10, tokens_out=20,
    )
    app.config["ROUTER"] = MagicMock()
    app.config["ROUTER"].decide.return_value = ("local", 2)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestPageRoutes:
    def test_index_redirects(self, client):
        resp = client.get("/")
        assert resp.status_code in (302, 301)

    def test_chat_page(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 200

    def test_dashboard(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_agents_page(self, client):
        resp = client.get("/agents")
        assert resp.status_code == 200

    def test_commands_page(self, client):
        resp = client.get("/commands")
        assert resp.status_code == 200

    def test_settings_page(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200

    def test_help_page(self, client):
        resp = client.get("/help")
        assert resp.status_code == 200

    def test_knowledge_page(self, client):
        resp = client.get("/knowledge")
        assert resp.status_code == 200

    def test_notes_page(self, client):
        resp = client.get("/notes")
        assert resp.status_code == 200

    def test_projects_page(self, client):
        resp = client.get("/projects")
        assert resp.status_code == 200

    def test_prompts_page(self, client):
        resp = client.get("/prompts")
        assert resp.status_code == 200

    def test_not_found_page(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_not_found_api(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404
        assert resp.is_json


class TestApiStats:
    def test_stats_get(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "local" in data
        assert "online" in data
        assert "total" in data

    def test_stats_reset(self, client):
        resp = client.post("/api/stats/reset")
        assert resp.status_code == 200

    def test_stats_history(self, client):
        resp = client.get("/api/stats/history")
        assert resp.status_code == 200


class TestApiConfig:
    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "local_model" in data
        assert "router_mode" in data

    def test_update_config(self, client):
        resp = client.put("/api/config", json={"local_model": "llama3:8b", "router_mode": "local"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_update_config_with_api_key(self, client):
        resp = client.put("/api/config", json={"api_key": "sk-new-key"})
        assert resp.status_code == 200

    def test_config_test_ollama(self, client):
        resp = client.post("/api/config/test/ollama", json={"url": "http://localhost:1"})
        assert resp.status_code == 200

    def test_config_test_online_opencode(self, client):
        resp = client.post("/api/config/test/online", json={"provider": "opencode", "opencode_path": "/nonexistent"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False

    def test_config_reset(self, client):
        resp = client.post("/api/config/reset")
        assert resp.status_code == 200


class TestApiChat:
    def test_chat_missing_prompt(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 400

    def test_chat_basic(self, client):
        resp = client.post("/api/chat", json={"prompt": "hello"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data

    def test_chat_with_session(self, client):
        resp = client.post("/api/chat", json={"prompt": "test", "session_id": "test-session"})
        assert resp.status_code == 200

    def test_chat_clear(self, client):
        resp = client.post("/api/chat/clear", json={"session_id": "test-session"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


class TestApiAgents:
    def test_list_agents(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)

    def test_create_agent(self, client):
        name = f"agent-{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/agents", json={"name": name, "description": "Test agent"})
        assert resp.status_code == 201, resp.get_json()
        assert resp.get_json()["ok"] is True

    def test_create_agent_no_name(self, client):
        resp = client.post("/api/agents", json={})
        assert resp.status_code == 400

    def test_create_agent_duplicate(self, client):
        name = f"dup-{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/agents", json={"name": name})
        assert resp.status_code == 201
        resp = client.post("/api/agents", json={"name": name})
        assert resp.status_code == 409

    def test_get_agent_nonexistent(self, client):
        resp = client.get("/api/agents/nonexistent-xyz-123")
        assert resp.status_code == 404

    def test_delete_agent_nonexistent(self, client):
        resp = client.delete("/api/agents/nonexistent-xyz-123")
        assert resp.status_code == 404


class TestApiCommands:
    def test_list_commands(self, client):
        resp = client.get("/api/commands")
        assert resp.status_code == 200

    def test_create_command(self, client):
        name = f"cmd-{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/commands", json={"name": name, "prompt": "test prompt"})
        assert resp.status_code == 201, resp.get_json()

    def test_create_command_no_name(self, client):
        resp = client.post("/api/commands", json={})
        assert resp.status_code == 400


class TestApiKnowledge:
    def test_knowledge_status(self, client):
        resp = client.get("/api/knowledge")
        assert resp.status_code == 200

    def test_knowledge_clear(self, client):
        resp = client.post("/api/knowledge/clear")
        assert resp.status_code == 200

    def test_knowledge_query_no_query(self, client):
        resp = client.post("/api/knowledge/query", json={})
        assert resp.status_code == 400

    def test_knowledge_from_dir_no_path(self, client):
        resp = client.post("/api/knowledge/from-dir", json={})
        assert resp.status_code == 400

    def test_knowledge_from_url_no_url(self, client):
        resp = client.post("/api/knowledge/from-url", json={})
        assert resp.status_code == 400

    def test_knowledge_graph(self, client):
        resp = client.get("/api/knowledge/graph")
        assert resp.status_code == 200

    def test_knowledge_notebook_no_query(self, client):
        resp = client.post("/api/knowledge/notebook", json={})
        assert resp.status_code == 400

    def test_knowledge_briefing_no_chunks(self, client):
        resp = client.post("/api/knowledge/briefing", json={"topic": "test"})
        assert resp.status_code == 400


class TestApiSessions:
    def test_list_sessions(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200

    def test_get_session(self, client):
        resp = client.get("/api/sessions/test-session")
        assert resp.status_code == 200

    def test_delete_session(self, client):
        resp = client.delete("/api/sessions/test-session")
        assert resp.status_code == 200


class TestApiNotes:
    def test_list_notes(self, client):
        resp = client.get("/api/notes")
        assert resp.status_code == 200

    def test_create_note(self, client):
        resp = client.post("/api/notes", json={"title": "Test Note", "body": "Hello"})
        assert resp.status_code == 201

    def test_create_note_no_title(self, client):
        resp = client.post("/api/notes", json={})
        assert resp.status_code == 400

    def test_get_note_not_found(self, client):
        resp = client.get("/api/notes/nonexistent-slug")
        assert resp.status_code == 404

    def test_notes_graph(self, client):
        resp = client.get("/api/notes/graph")
        assert resp.status_code == 200

    def test_notes_search(self, client):
        resp = client.get("/api/notes/search")
        assert resp.status_code == 200


class TestApiProjects:
    def test_list_projects(self, client):
        resp = client.get("/api/projects")
        assert resp.status_code == 200

    def test_templates(self, client):
        resp = client.get("/api/project/templates")
        assert resp.status_code == 200

    def test_generate_no_prompt(self, client):
        resp = client.post("/api/projects/generate", json={"name": "test"})
        assert resp.status_code == 400


class TestApiModels:
    def test_list_models(self, client):
        resp = client.get("/api/models/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "local" in data
        assert "online" in data

    def test_set_active_model_no_kind(self, client):
        resp = client.post("/api/models/set-active", json={})
        assert resp.status_code == 400

    def test_set_active_model(self, client):
        resp = client.post("/api/models/set-active", json={"kind": "local", "model_id": "llama3:8b"})
        assert resp.status_code == 200


class TestApiWiki:
    def test_wiki_list(self, client):
        resp = client.get("/api/wiki")
        assert resp.status_code == 200

    def test_wiki_get_nonexistent(self, client):
        resp = client.get("/api/wiki/nonexistent")
        assert resp.status_code == 404

    def test_wiki_suggest(self, client):
        resp = client.get("/api/wiki/suggest")
        assert resp.status_code == 200

    def test_wiki_generate_no_topic(self, client):
        resp = client.post("/api/wiki/generate", json={})
        assert resp.status_code == 400


class TestApiSystem:
    def test_system_info(self, client):
        resp = client.get("/api/system/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "local_model" in data
        assert "online_model" in data
        assert "total_calls" in data

    def test_system_resources(self, client):
        resp = client.get("/api/system/resources")
        assert resp.status_code == 200

    def test_system_cleanup(self, client):
        resp = client.post("/api/system/cleanup")
        assert resp.status_code == 200

    def test_system_disk(self, client):
        resp = client.get("/api/system/disk")
        assert resp.status_code == 200


class TestApiTerminal:
    def test_terminal_exec_no_command(self, client):
        resp = client.post("/api/terminal/exec", json={})
        assert resp.status_code == 400

    def test_terminal_ai_cmd_no_name(self, client):
        resp = client.post("/api/terminal/ai-cmd", json={})
        assert resp.status_code == 400

    def test_terminal_ai_cmd_builtin(self, client):
        resp = client.post("/api/terminal/ai-cmd", json={"name": "help"})
        assert resp.status_code == 200
        assert resp.mimetype == "text/event-stream"


class TestApiWorkspace:
    def test_workspace_tree(self, client):
        resp = client.get("/api/workspace/tree")
        assert resp.status_code == 200

    def test_workspace_read_missing(self, client):
        resp = client.get("/api/workspace/read?path=/nonexistent-file-xyz")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data.get("error") in ("not a file", "path required")

    def test_workspace_mkdir(self, client):
        resp = client.post("/api/workspace/mkdir", json={"path": "/tmp/test-ws-dir-ai-plus"})
        assert resp.status_code == 200

    def test_workspace_delete(self, client):
        resp = client.post("/api/workspace/delete", json={"path": "/tmp/test-ws-dir-ai-plus"})
        assert resp.status_code == 200

    def test_workspace_info(self, client):
        resp = client.get("/api/workspace/info?path=/tmp")
        assert resp.status_code == 200

    def test_workspace_write(self, client):
        resp = client.post("/api/workspace/write", json={"path": "/tmp/test-write-ai-plus.txt", "content": "test"})
        assert resp.status_code == 200


class TestApiChatStream:
    def test_chat_stream_no_prompt(self, client):
        resp = client.post("/api/chat/stream", json={})
        assert resp.status_code == 400

    def test_chat_stream_basic(self, client):
        resp = client.post("/api/chat/stream", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert resp.mimetype == "text/event-stream"


class TestApiMisc:
    def test_agent_templates(self, client):
        resp = client.get("/api/agents/templates")
        assert resp.status_code == 200

    def test_commands_list_alias(self, client):
        resp = client.get("/api/commands/list")
        assert resp.status_code == 200


class TestErrorHandling:
    def test_404_api_json(self, client):
        resp = client.get("/api/unknown-route")
        assert resp.status_code == 404
        assert resp.is_json
        assert "error" in resp.get_json()

    def test_security_headers(self, client):
        resp = client.get("/chat")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
