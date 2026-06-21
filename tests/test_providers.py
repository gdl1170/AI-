"""Tests per hycoder/providers.py."""

import json
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from hycoder.providers import (
    ProviderResult,
    SessionPool,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    OpencodeProvider,
    get_local_provider,
    get_online_provider,
    close_sessions,
    _cache_key,
)
from hycoder.resources import reset_resource_manager


class TestProviderResult:
    def test_init_with_slots(self):
        r = ProviderResult(
            text="hello", model="test-model",
            tokens_in=10, tokens_out=20, time_s=0.5, source="local"
        )
        assert r.text == "hello"
        assert r.model == "test-model"
        assert r.tokens_in == 10
        assert r.tokens_out == 20
        assert r.tokens_total == 30
        assert r.time_s == 0.5
        assert r.source == "local"
        assert r.cached is False

    def test_cached_flag(self):
        r = ProviderResult("text", "m", 0, 0, 0, "local", cached=True)
        assert r.cached is True

    def test_repr(self):
        r = ProviderResult("text", "ollama/test", 10, 20, 2.5, "local")
        assert "ollama/test" in repr(r)
        assert "30tok" in repr(r)
        assert "2.5s" in repr(r)

    def test_repr_cached(self):
        r = ProviderResult("text", "m", 0, 0, 0, "local", cached=True)
        assert "cached" in repr(r)

    def test_events_field(self):
        r = ProviderResult("text", "m", 0, 0, 0, "local", events=[{"type": "text"}])
        assert r.events == [{"type": "text"}]


class TestSessionPool:
    def test_get_session(self):
        session = SessionPool.get("http://test.local")
        assert session is not None
        assert session.headers["Connection"] == "keep-alive"

    def test_get_caches_sessions(self):
        s1 = SessionPool.get("http://test.local")
        s2 = SessionPool.get("http://test.local")
        assert s1 is s2

    def test_different_urls_different_sessions(self):
        s1 = SessionPool.get("http://url1.local")
        s2 = SessionPool.get("http://url2.local")
        assert s1 is not s2

    def test_close_all(self):
        SessionPool.get("http://close-test.local")
        SessionPool.close_all()
        assert len(SessionPool._instances) == 0

    def test_close_all_empty(self):
        SessionPool.close_all()
        assert len(SessionPool._instances) == 0


class TestCacheKey:
    def test_consistent_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs)
        k2 = _cache_key("model-a", msgs)
        assert k1 == k2

    def test_different_model_different_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs)
        k2 = _cache_key("model-b", msgs)
        assert k1 != k2

    def test_different_messages_different_key(self):
        k1 = _cache_key("m", [{"role": "user", "content": "hello"}])
        k2 = _cache_key("m", [{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_deterministic(self):
        msgs1 = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        msgs2 = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        assert _cache_key("m", msgs1) == _cache_key("m", msgs2)

    def test_returns_hex_string(self):
        key = _cache_key("m", [{"role": "user", "content": "test"}])
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hex


class TestOllamaProvider:
    def test_init(self):
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "llama3:8b"}
        provider = OllamaProvider(cfg)
        assert provider.model == "llama3:8b"
        assert provider.base_url == "http://ollama:11434"
        assert provider._chat_url == "http://ollama:11434/api/chat"

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Hello from Ollama!"},
            "prompt_eval_count": 15,
            "eval_count": 10,
        }
        mock_post.return_value = mock_resp
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "test-model"}
        provider = OllamaProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert result.text == "Hello from Ollama!"
        assert result.source == "local"
        assert result.tokens_in == 15
        assert result.tokens_out == 10

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_connection_error(self, mock_post):
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("refused")
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "test"}
        provider = OllamaProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert "ERRORE" in result.text
        assert "offline" in result.model

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_timeout(self, mock_post):
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout("timed out")
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "test"}
        provider = OllamaProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert "ERRORE" in result.text
        assert "timeout" in result.model

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_stream_yields_done(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"message": {"content": "Hello"}, "done": false}',
            b'{"message": {"content": " world"}, "done": false}',
            b'{"done": true, "prompt_eval_count": 10, "eval_count": 5}',
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "test"}
        provider = OllamaProvider(cfg)
        events = list(provider.generate_chat_stream([{"role": "user", "content": "hi"}]))
        tokens = [e for e in events if not e.get("done")]
        done_events = [e for e in events if e.get("done")]
        assert len(tokens) >= 1
        assert len(done_events) == 1

    def test_generate_chat_stream_cached(self):
        cfg = {"ollama_base_url": "http://ollama:11434", "model": "test"}
        provider = OllamaProvider(cfg)
        cached_result = ProviderResult("cached text", "ollama/test", 10, 20, 0.001, "local", cached=True)
        provider.cache.set_pickle(_cache_key("test", [{"role": "user", "content": "cached"}]), cached_result)
        events = list(provider.generate_chat_stream([{"role": "user", "content": "cached"}]))
        assert any(e.get("done") for e in events)


class TestOpenAIProvider:
    def test_init(self):
        cfg = {"api_key": "sk-test", "model": "gpt-4o-mini", "base_url": None}
        provider = OpenAIProvider(cfg)
        assert provider.model == "gpt-4o-mini"
        assert provider.api_key == "sk-test"

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenAI!"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        mock_post.return_value = mock_resp
        cfg = {"api_key": "sk-test", "model": "gpt-4o", "base_url": None}
        provider = OpenAIProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert result.text == "Hello from OpenAI!"
        assert result.source == "online"
        assert result.tokens_in == 20
        assert result.tokens_out == 15

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_timeout(self, mock_post):
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout("timed out")
        cfg = {"api_key": "sk-test", "model": "gpt-4o", "base_url": None}
        provider = OpenAIProvider(cfg)
        provider.cache.clear()
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert "ERRORE" in result.text
        assert "timeout" in result.model

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_custom_base_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        mock_post.return_value = mock_resp
        cfg = {"api_key": "sk-test", "model": "custom-model", "base_url": "https://custom.api/v1"}
        provider = OpenAIProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert result.text == "ok"


class TestOpenRouterProvider:
    def test_init(self):
        cfg = {"api_key": "sk-test", "model": "openai/gpt-4o-mini"}
        provider = OpenRouterProvider(cfg)
        assert provider.model == "openai/gpt-4o-mini"
        assert "openrouter" in provider.base_url

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "OpenRouter response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        mock_post.return_value = mock_resp
        cfg = {"api_key": "sk-test", "model": "openai/gpt-4o-mini"}
        provider = OpenRouterProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert result.text == "OpenRouter response"
        assert result.source == "online"
        assert "openrouter" in result.model

    @patch("hycoder.providers.requests.Session.post")
    def test_generate_chat_error_response(self, mock_post):
        from requests.exceptions import HTTPError
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = HTTPError("401 Unauthorized")
        mock_post.return_value = mock_resp
        cfg = {"api_key": "bad-key", "model": "openai/gpt-4o-mini"}
        provider = OpenRouterProvider(cfg)
        provider.cache.clear()
        result = provider.generate_chat([{"role": "user", "content": "hi"}])
        assert "ERRORE" in result.text


class TestOpencodeProvider:
    def test_init(self):
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        assert provider.binary == "/usr/local/bin/opencode"
        assert provider.model == "opencode"

    def test_init_default_path(self):
        cfg = {}
        provider = OpencodeProvider(cfg)
        assert "opencode" in provider.binary

    @patch("hycoder.providers.subprocess.run")
    def test_generate_chat_success_with_ndjson(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.stdout = (
            '{"type": "text", "part": {"text": "Hello from"}}\n'
            '{"type": "text", "part": {"text": " opencode!"}}\n'
            '{"type": "step_finish", "part": {"tokens": {"input": 50, "output": 30}}}\n'
        )
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hello"}])
        assert "Hello from" in result.text
        assert "opencode!" in result.text
        assert result.tokens_in >= 50
        assert result.tokens_out >= 30

    @patch("hycoder.providers.subprocess.run")
    def test_generate_chat_fallback_to_stdout(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.stdout = "Plain text response"
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hello"}])
        assert result.text == "Plain text response"

    @patch("hycoder.providers.subprocess.run")
    def test_generate_chat_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="opencode", timeout=300)
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hello"}])
        assert "ERRORE" in result.text

    @patch("hycoder.providers.subprocess.run")
    def test_generate_chat_file_not_found(self, mock_run):
        import subprocess
        mock_run.side_effect = FileNotFoundError("opencode not found")
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        result = provider.generate_chat([{"role": "user", "content": "hello"}])
        assert "ERRORE" in result.text

    def test_generate_chat_no_user_message(self):
        cfg = {"opencode_path": "/usr/local/bin/opencode"}
        provider = OpencodeProvider(cfg)
        result = provider.generate_chat([{"role": "system", "content": "sys"}])
        assert "ERRORE" in result.text


class TestFactoryFunctions:
    def test_get_local_provider(self):
        cfg = {"local": {"provider": "ollama", "model": "test", "ollama_base_url": "http://localhost:11434"}}
        provider = get_local_provider(cfg)
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "test"

    def test_get_local_provider_cached(self):
        cfg = {"local": {"provider": "ollama", "model": "cached-test", "ollama_base_url": "http://localhost:11434"}}
        p1 = get_local_provider(cfg)
        p2 = get_local_provider(cfg)
        assert p1 is p2

    def test_get_online_provider_opencode(self):
        cfg = {"online": {"provider": "opencode", "model": "opencode", "opencode_path": "/bin/echo"}}
        provider = get_online_provider(cfg)
        assert isinstance(provider, OpencodeProvider)

    def test_get_online_provider_openrouter(self):
        cfg = {"online": {"provider": "openrouter", "model": "gpt-4o", "api_key": "sk-test"}}
        provider = get_online_provider(cfg)
        assert isinstance(provider, OpenRouterProvider)

    def test_get_online_provider_openai_default(self):
        cfg = {"online": {"provider": "unknown", "model": "gpt-4o", "api_key": "sk-test"}}
        provider = get_online_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_clear_caches(self):
        from hycoder.providers import clear_caches
        cfg = {"local": {"provider": "ollama", "model": "cc-test", "ollama_base_url": "http://localhost:11434"}}
        get_local_provider(cfg)
        reset_resource_manager()
        clear_caches()

    def test_close_sessions(self):
        SessionPool.get("http://close-factory.local")
        close_sessions()
        assert len(SessionPool._instances) == 0
