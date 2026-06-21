"""
Provider per modelli AI locali e online.
Ottimizzato: DiskCache persistente, session pool con keep-alive, retry, timeouts.
"""

import time
import json
import hashlib
import threading
import os
import subprocess
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .resources import get_resource_manager


# ─── HTTP Session Pool (ottimizzato) ────────────────────────────────────

class SessionPool:
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, base_url, pool_connections=4, pool_maxsize=8, retries=1, timeout=60):
        with cls._lock:
            key = f"{base_url}|{pool_connections}|{pool_maxsize}"
            if key not in cls._instances:
                s = requests.Session()
                retry = Retry(
                    total=retries,
                    backoff_factor=0.5,
                    allowed_methods={"GET", "POST"},
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(
                    pool_connections=pool_connections,
                    pool_maxsize=pool_maxsize,
                    max_retries=retry,
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                s.headers.update({
                    "Connection": "keep-alive",
                    "Keep-Alive": f"timeout={timeout}, max=100",
                })
                cls._instances[key] = s
            return cls._instances[key]

    @classmethod
    def close_all(cls):
        with cls._lock:
            for s in cls._instances.values():
                try:
                    s.close()
                except Exception:
                    pass
            cls._instances.clear()


# ─── Provider Result (__slots__ per risparmio memoria) ──────────────────

class ProviderResult:
    __slots__ = ("text", "model", "tokens_in", "tokens_out",
                 "tokens_total", "time_s", "source", "cached", "events")

    def __init__(self, text, model, tokens_in, tokens_out, time_s, source, cached=False, events=None):
        self.text = text
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.tokens_total = tokens_in + tokens_out
        self.time_s = time_s
        self.source = source
        self.cached = cached
        self.events = events

    def __repr__(self):
        tag = " [cached]" if self.cached else ""
        return f"<{self.source}/{self.model} {self.tokens_total}tok {self.time_s:.1f}s{tag}>"


# ─── Helper cache ───────────────────────────────────────────────────────

def _cache_key(model, messages):
    """Chiave hash per messaggi chat."""
    raw = f"{model}|{json.dumps(messages, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─── Timeout di default ─────────────────────────────────────────────────

_LOCAL_TIMEOUT = int(os.environ.get("HYBRID_LOCAL_TIMEOUT", "120"))
_ONLINE_TIMEOUT = int(os.environ.get("HYBRID_ONLINE_TIMEOUT", "120"))


# ─── Ollama Provider (chat-aware, persistente) ──────────────────────────

class OllamaProvider:
    def __init__(self, cfg):
        self.base_url = cfg["ollama_base_url"].rstrip("/")
        self.model = cfg["model"]
        self.session = SessionPool.get(self.base_url, pool_connections=2, pool_maxsize=4)
        self._chat_url = f"{self.base_url}/api/chat"
        mgr = get_resource_manager()
        self.cache = mgr.response_cache
        self.timeout = _LOCAL_TIMEOUT

    def generate_chat(self, messages):
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            return ProviderResult(
                cached.text, cached.model,
                cached.tokens_in, cached.tokens_out,
                0.001, cached.source, cached=True,
            )

        t0 = time.time()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": 4096,
                "temperature": 0.7,
            }
        }

        try:
            resp = self.session.post(self._chat_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return ProviderResult(
                text="[ERRORE] Ollama non raggiungibile. `ollama serve` è in esecuzione?",
                model=f"{self.model} (offline)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="local",
            )
        except requests.exceptions.Timeout:
            return ProviderResult(
                text=f"[ERRORE] Timeout ({self.timeout}s): il modello locale non risponde.",
                model=f"{self.model} (timeout)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="local",
            )
        except requests.exceptions.TooManyRedirects:
            return ProviderResult(
                text="[ERRORE] Troppi redirect nella risposta Ollama.",
                model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="local",
            )
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 404:
                return ProviderResult(
                    text="[ERRORE] Ollama API non trovata. Verifica che la versione di Ollama sia aggiornata.",
                    model=f"{self.model} (error)",
                    tokens_in=0, tokens_out=0,
                    time_s=time.time() - t0, source="local",
                )
            return ProviderResult(
                text=f"[ERRORE LOCALE] HTTP {code}: {e}",
                model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="local",
            )
        except Exception as e:
            return ProviderResult(
                text=f"[ERRORE LOCALE] {e}",
                model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="local",
            )

        elapsed = time.time() - t0
        text = data.get("message", {}).get("content", "")
        tok_in = data.get("prompt_eval_count", max(1, sum(len(m.get("content", "")) for m in messages) // 4))
        tok_out = data.get("eval_count", max(1, len(text) // 4))

        result = ProviderResult(text, f"ollama/{self.model}", tok_in, tok_out, elapsed, "local")

        if not text.startswith("[ERRORE"):
            self.cache.set_pickle(ck, result)

        return result

    def generate_chat_stream(self, messages):
        """Yield token dicts: {'token': str, 'done': bool} + final 'result' key."""
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            yield {"token": cached.text, "done": True, "result": cached, "source": "local"}
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": "10m",
            "options": {"num_predict": 4096, "temperature": 0.7},
        }
        t0 = time.time()
        text_parts = []
        try:
            resp = self.session.post(self._chat_url, json=payload, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    tok_in = data.get("prompt_eval_count", max(1, sum(len(m.get("content", "")) for m in messages) // 4))
                    tok_out = data.get("eval_count", max(1, sum(len(t) for t in text_parts) // 4))
                    elapsed = time.time() - t0
                    full = "".join(text_parts)
                    result = ProviderResult(full, f"ollama/{self.model}", tok_in, tok_out, elapsed, "local")
                    self.cache.set_pickle(ck, result)
                    yield {"token": "", "done": True, "result": result, "source": "local"}
                    return
                token = data.get("message", {}).get("content", "")
                if token:
                    text_parts.append(token)
                    yield {"token": token, "done": False}
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            msg = "Ollama API non trovata. Verifica che la versione di Ollama sia aggiornata." if code == 404 else f"HTTP {code}: {e}"
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE LOCALE] {msg}", model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0, time_s=time.time() - t0, source="local",
            ), "source": "local"}
        except Exception as e:
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE LOCALE] {e}", model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0, time_s=time.time() - t0, source="local",
            ), "source": "local"}


# ─── OpenAI Provider ────────────────────────────────────────────────────

class OpenAIProvider:
    def __init__(self, cfg):
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.session = SessionPool.get(self.base_url, pool_connections=2, pool_maxsize=4, retries=2)
        mgr = get_resource_manager()
        self.cache = mgr.response_cache
        self.timeout = _ONLINE_TIMEOUT

    def generate_chat(self, messages):
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            return ProviderResult(
                cached.text, cached.model,
                cached.tokens_in, cached.tokens_out,
                0.001, cached.source, cached=True,
            )

        t0 = time.time()
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return ProviderResult(
                text=f"[ERRORE] Timeout ({self.timeout}s) chiamata API OpenAI.",
                model=f"{self.model} (timeout)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )
        except Exception as e:
            return ProviderResult(
                text=f"[ERRORE ONLINE] {e}",
                model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )

        elapsed = time.time() - t0
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        tok_in = usage.get("prompt_tokens", max(1, sum(len(m.get("content", "")) for m in messages) // 4))
        tok_out = usage.get("completion_tokens", max(1, len(text) // 4))

        result = ProviderResult(text, f"openai/{self.model}", tok_in, tok_out, elapsed, "online")

        if not text.startswith("[ERRORE"):
            self.cache.set_pickle(ck, result)

        return result

    def generate_chat_stream(self, messages):
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            yield {"token": cached.text, "done": True, "result": cached, "source": "online"}
            return

        t0 = time.time()
        text_parts = []
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith(":") or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        text_parts.append(token)
                        yield {"token": token, "done": False}
        except Exception as e:
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE ONLINE] {e}", model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0, time_s=time.time() - t0, source="online",
            ), "source": "online"}
            return

        full = "".join(text_parts)
        tok_in = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        tok_out = max(1, len(full) // 4)
        elapsed = time.time() - t0
        result = ProviderResult(full, f"openai/{self.model}", tok_in, tok_out, elapsed, "online")
        self.cache.set_pickle(ck, result)
        yield {"token": "", "done": True, "result": result, "source": "online"}


# ─── OpenRouter Provider ────────────────────────────────────────────────

class OpenRouterProvider:
    def __init__(self, cfg):
        self.api_key = cfg["api_key"]
        self.model = cfg.get("model", "openai/gpt-4o-mini")
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = SessionPool.get(self.base_url, pool_connections=2, pool_maxsize=4, retries=2)
        mgr = get_resource_manager()
        self.cache = mgr.response_cache
        self.timeout = _ONLINE_TIMEOUT

    def generate_chat(self, messages):
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            return ProviderResult(
                cached.text, cached.model,
                cached.tokens_in, cached.tokens_out,
                0.001, cached.source, cached=True,
            )

        t0 = time.time()
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/ai-plus",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return ProviderResult(
                text=f"[ERRORE] Timeout ({self.timeout}s) chiamata OpenRouter.",
                model=f"{self.model} (timeout)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )
        except Exception as e:
            return ProviderResult(
                text=f"[ERRORE ONLINE] {e}",
                model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )

        elapsed = time.time() - t0
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        tok_in = usage.get("prompt_tokens", max(1, sum(len(m.get("content", "")) for m in messages) // 4))
        tok_out = usage.get("completion_tokens", max(1, len(text) // 4))

        result = ProviderResult(text, f"openrouter/{self.model}", tok_in, tok_out, elapsed, "online")

        if not text.startswith("[ERRORE"):
            self.cache.set_pickle(ck, result)

        return result

    def generate_chat_stream(self, messages):
        ck = _cache_key(self.model, messages)
        cached = self.cache.get_pickle(ck)
        if cached is not None:
            yield {"token": cached.text, "done": True, "result": cached, "source": "online"}
            return

        t0 = time.time()
        text_parts = []
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/ai-plus",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith(":") or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        text_parts.append(token)
                        yield {"token": token, "done": False}
        except Exception as e:
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE ONLINE] {e}", model=f"{self.model} (error)",
                tokens_in=0, tokens_out=0, time_s=time.time() - t0, source="online",
            ), "source": "online"}
            return

        full = "".join(text_parts)
        tok_in = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        tok_out = max(1, len(full) // 4)
        elapsed = time.time() - t0
        result = ProviderResult(full, f"openrouter/{self.model}", tok_in, tok_out, elapsed, "online")
        self.cache.set_pickle(ck, result)
        yield {"token": "", "done": True, "result": result, "source": "online"}


# ─── Opencode Provider (esegue prompt su opencode CLI) ─────────────────

class OpencodeProvider:
    """Invia il prompt a opencode via `opencode run --format json`.
    Restituisce il testo della risposta + eventi NDJSON grezzi."""

    def __init__(self, cfg):
        self.binary = cfg.get("opencode_path") or str(Path.home() / ".opencode" / "bin" / "opencode")
        self.model = "opencode"
        self.timeout = cfg.get("opencode_timeout", 300)

    def generate_chat(self, messages):
        t0 = time.time()
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if not last_user:
            return ProviderResult(
                text="[ERRORE] Nessun messaggio utente trovato.",
                model="opencode/error", tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )

        cmd = [self.binary, "run", "--format", "json"]
        cmd.append(last_user)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.timeout,
                env={**os.environ, "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")},
            )
        except subprocess.TimeoutExpired:
            return ProviderResult(
                text=f"[ERRORE] Timeout opencode ({self.timeout}s).",
                model="opencode/timeout", tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )
        except FileNotFoundError:
            return ProviderResult(
                text="[ERRORE] opencode non trovato. Installalo o aggiorna il percorso.",
                model="opencode/error", tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )
        except Exception as e:
            return ProviderResult(
                text=f"[ERRORE OPENCODE] {e}",
                model="opencode/error", tokens_in=0, tokens_out=0,
                time_s=time.time() - t0, source="online",
            )

        elapsed = time.time() - t0
        events = []
        text_parts = []
        total_input = 0
        total_output = 0

        # opencode stderr può contenere eventi NDJSON, stdout può contenere eventi o testo
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                events.append(ev)
                if ev.get("type") == "text":
                    part_text = ev.get("part", {}).get("text", "")
                    if part_text:
                        text_parts.append(part_text)
                if ev.get("type") == "step_finish":
                    toks = ev.get("part", {}).get("tokens", {})
                    total_input = toks.get("input", 0)
                    total_output = toks.get("output", 0)
            except (json.JSONDecodeError, KeyError):
                text_parts.append(line)

        # Se nessun evento NDJSON, usa stdout intero
        if not events:
            text_parts = [proc.stdout.strip()]

        text = "\n".join(text_parts) if text_parts else "(risposta vuota)"
        inp = max(1, total_input or (len(last_user) // 4))
        out = max(1, total_output or (len(text) // 4))

        return ProviderResult(
            text=text, model="opencode/agent",
            tokens_in=inp, tokens_out=out,
            time_s=round(elapsed, 2), source="online",
            events=events[:200],  # massimo 200 eventi
        )

    def generate_chat_stream(self, messages):
        t0 = time.time()
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if not last_user:
            yield {"token": "", "done": True, "result": ProviderResult(
                text="[ERRORE] Nessun messaggio utente.", model="opencode/error",
                tokens_in=0, tokens_out=0, time_s=time.time() - t0, source="online",
            ), "source": "online"}
            return

        cmd = [self.binary, "run", "--format", "json", last_user]
        text_parts = []
        events = []
        total_input = 0
        total_output = 0

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")},
            )
            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    events.append(ev)
                    if ev.get("type") == "text":
                        part_text = ev.get("part", {}).get("text", "")
                        if part_text:
                            text_parts.append(part_text)
                            yield {"token": part_text, "done": False}
                    if ev.get("type") == "step_finish":
                        toks = ev.get("part", {}).get("tokens", {})
                        total_input = toks.get("input", 0)
                        total_output = toks.get("output", 0)
                except json.JSONDecodeError:
                    text_parts.append(line)
                    yield {"token": line, "done": False}

            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            elapsed = time.time() - t0
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE] Timeout opencode ({self.timeout}s).", model="opencode/timeout",
                tokens_in=0, tokens_out=0, time_s=elapsed, source="online",
            ), "source": "online"}
            return
        except Exception as e:
            elapsed = time.time() - t0
            yield {"token": "", "done": True, "result": ProviderResult(
                text=f"[ERRORE OPENCODE] {e}", model="opencode/error",
                tokens_in=0, tokens_out=0, time_s=elapsed, source="online",
            ), "source": "online"}
            return

        elapsed = time.time() - t0
        text = "".join(text_parts) if text_parts else "(risposta vuota)"
        inp = max(1, total_input or (len(last_user) // 4))
        out = max(1, total_output or (len(text) // 4))
        result = ProviderResult(text, "opencode/agent", inp, out, elapsed, "online", events=events[:200])
        yield {"token": "", "done": True, "result": result, "source": "online"}


# ─── Factory con caching provider ───────────────────────────────────────

_local_providers = {}
_online_providers = {}


def get_local_provider(cfg):
    key = f"{cfg['local']['provider']}:{cfg['local']['model']}"
    if key not in _local_providers:
        _local_providers[key] = OllamaProvider(cfg["local"])
    return _local_providers[key]


def get_online_provider(cfg):
    key = f"{cfg['online']['provider']}:{cfg['online']['model']}"
    if key not in _online_providers:
        pn = cfg["online"]["provider"]
        if pn == "opencode":
            _online_providers[key] = OpencodeProvider(cfg["online"])
        elif pn == "openrouter":
            _online_providers[key] = OpenRouterProvider(cfg["online"])
        else:
            _online_providers[key] = OpenAIProvider(cfg["online"])
    return _online_providers[key]


def clear_caches():
    mgr = get_resource_manager()
    mgr.response_cache.clear()
    for p in _local_providers.values():
        if hasattr(p, 'cache'):
            p.cache.clear()
    for p in _online_providers.values():
        if hasattr(p, 'cache'):
            p.cache.clear()


def close_sessions():
    SessionPool.close_all()
