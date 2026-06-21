"""
Gestione risorse hardware: cache persistente su disco,
monitoraggio memoria/CPU, thread pool globale, limiti.
"""

import os
import gc
import json
import time
import pickle
import resource
import threading
import platform
from pathlib import Path

# ─── Percorsi dati ──────────────────────────────────────────────────────

DATA_DIR = Path.home() / ".config" / "hybrid-coder"
CACHE_DIR = DATA_DIR / "cache"
KB_DIR = DATA_DIR / "knowledge"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)


# ─── Disk Cache (LRU + TTL + persistente) ──────────────────────────────

class DiskCache:
    """Cache chiave→valore su disco con LRU e TTL. Thread-safe."""

    def __init__(self, name="default", max_entries=512, ttl_seconds=3600, dirpath=None):
        _ensure_dirs()
        self.dir = Path(dirpath or CACHE_DIR) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._meta_file = self.dir / "_meta.json"
        self._meta_lock = threading.Lock()
        self._meta: dict[str, dict] = {}
        self._load_meta()
        self._housekeeping()

    def _meta_path(self):
        return self._meta_file

    def _load_meta(self):
        try:
            data = json.loads(self._meta_file.read_text())
            self._meta = data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._meta = {}

    def _save_meta(self):
        tmp = self._meta_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._meta, separators=(",", ":")))
            tmp.rename(self._meta_file)
        except OSError:
            pass

    def _key_path(self, key):
        safe = key.encode("utf-8").hex()
        return self.dir / safe[:2] / safe

    def _is_expired(self, entry):
        expires = entry.get("expires", float("inf"))
        if expires == 0:
            return False
        return time.time() > expires

    def get(self, key):
        entry = self._meta.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self.delete(key)
            return None
        path = self._key_path(key)
        try:
            val = path.read_bytes()
            # Aggiorna LRU
            with self._meta_lock:
                if key in self._meta:
                    self._meta[key]["atime"] = time.time()
            return val
        except (FileNotFoundError, OSError):
            self.delete(key)
            return None

    def get_json(self, key):
        raw = self.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode())

    def get_pickle(self, key):
        raw = self.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)

    def set(self, key, value, ttl=None):
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            raw = value.encode()
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raw = json.dumps(value).encode()
        path.write_bytes(raw)

        ttl_actual = ttl if ttl is not None else self.ttl
        expires = 0 if ttl_actual == 0 else time.time() + ttl_actual
        with self._meta_lock:
            self._meta[key] = {
                "expires": expires,
                "atime": time.time(),
                "size": len(raw),
            }
            # LRU eviction se supera max_entries
            if len(self._meta) > self.max_entries:
                sorted_k = sorted(self._meta, key=lambda k: self._meta[k].get("atime", 0))
                for k in sorted_k[:len(self._meta) - self.max_entries]:
                    del self._meta[k]
            self._save_meta()

    def set_pickle(self, key, value, ttl=None):
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        ttl_actual = ttl if ttl is not None else self.ttl
        expires = 0 if ttl_actual == 0 else time.time() + ttl_actual
        with self._meta_lock:
            self._meta[key] = {
                "expires": expires,
                "atime": time.time(),
                "size": len(raw),
            }
            self._save_meta()

    def delete(self, key):
        with self._meta_lock:
            self._meta.pop(key, None)
            self._save_meta()
        path = self._key_path(key)
        try:
            path.unlink()
        except OSError:
            pass

    def clear(self):
        with self._meta_lock:
            self._meta.clear()
            self._save_meta()
        for p in self.dir.rglob("*"):
            if p.is_file() and p.name != "_meta.json":
                try:
                    p.unlink()
                except OSError:
                    pass

    def _housekeeping(self):
        """Elimina entry scadute e riduce se oltre max_entries."""
        now = time.time()
        with self._meta_lock:
            expired = [k for k, v in self._meta.items()
                       if v.get("expires", 0) not in (0, None) and v.get("expires", 0) < now]
            for k in expired:
                self._meta.pop(k, None)
            # LRU eviction
            if len(self._meta) > self.max_entries:
                sorted_keys = sorted(self._meta, key=lambda k: self._meta[k].get("atime", 0))
                for k in sorted_keys[: len(self._meta) - self.max_entries]:
                    self._meta.pop(k, None)
            self._save_meta()

    @property
    def size(self):
        return len(self._meta)

    @property
    def disk_usage_bytes(self):
        total = 0
        for p in self.dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total


# ─── Resource Monitor ───────────────────────────────────────────────────

class ResourceMonitor:
    """Monitoraggio risorse di sistema (CPU, RAM, process)."""

    def __init__(self):
        self._start = time.time()
        self._samples = []

    @staticmethod
    def cpu_percent():
        """CPU usage del processo corrente (approssimato, macOS/Linux)."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            total_cpu = usage.ru_utime + usage.ru_stime
            uptime = max(time.time() - _PROC_START, 1)
            return total_cpu / uptime * 100.0 / max(os.cpu_count() or 1, 1)
        except Exception:
            return 0.0

    @staticmethod
    def memory_bytes():
        """Memoria RSS del processo corrente in byte."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss = usage.ru_maxrss
            if rss < 1024:  # Impossibile: < 1KB = errore unità
                return 0
            return rss
        except Exception:
            return 0

    @staticmethod
    def memory_mb():
        return ResourceMonitor.memory_bytes() / (1024 * 1024)

    @staticmethod
    def ollama_status(base_url="http://localhost:11434"):
        """Stato di Ollama: modelli caricati, memoria usata."""
        try:
            import requests
            r = requests.get(f"{base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return {"running": False}
            models = r.json().get("models", [])
            loaded = [m for m in models if m.get("size") or m.get("details", {}).get("family")]
            total_size = sum(m.get("size", 0) or 0 for m in loaded)
            return {
                "running": True,
                "models_count": len(loaded),
                "models": [m["name"] for m in loaded],
                "total_size_gb": round(total_size / (1024**3), 2),
            }
        except Exception:
            return {"running": False}

    def snapshot(self, ollama_base_url="http://localhost:11434"):
        now = time.time()
        s = {
            "timestamp": now,
            "uptime_s": round(now - self._start),
            "process_memory_mb": round(self.memory_mb(), 1),
            "cpu_percent": round(self.cpu_percent(), 1),
            "ollama": self.ollama_status(ollama_base_url),
            "cpu_count": os.cpu_count(),
        }
        self._samples.append(s)
        if len(self._samples) > 3600:  # max 1h di campioni ogni secondo
            self._samples = self._samples[-1800:]
        return s


_PROC_START = time.time()


# ─── Global Thread Pool ─────────────────────────────────────────────────

class ManagedPool:
    """Pool thread globale con limite configurabile."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
        self._lock = threading.Lock()
        self._active = 0
        self._threads = []

    def submit(self, fn, *args, **kwargs):
        """Avvia un thread con limitazione concorrenza."""
        t = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        with self._lock:
            self._active += 1
            self._threads.append(t)
        t.start()
        return t

    def _run(self, fn, args, kwargs):
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
        finally:
            with self._lock:
                self._active -= 1
                self._threads = [t for t in self._threads if t.is_alive()]

    @property
    def active_count(self):
        with self._lock:
            return self._active

    def _drain_idle(self):
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]

    def wait(self, timeout=None):
        for t in self._threads:
            t.join(timeout=timeout)


# ─── Global Resource Manager ────────────────────────────────────────────

class ResourceManager:
    """Gestore centralizzato risorse per tutta l'app."""

    def __init__(self, cfg=None):
        _ensure_dirs()
        self.cfg = cfg or {}
        # Limiti configurabili (default)
        self.max_memory_mb = int(self.cfg.get("max_memory_mb", 512))
        self.max_cache_entries = int(self.cfg.get("max_cache_entries", 1024))
        self.max_conversation_messages = int(self.cfg.get("max_conversation_messages", 100))
        self.        max_conversation_tokens = int(self.cfg.get("max_conversation_tokens", 64000))
        self.knowledge_max_chunks = int(self.cfg.get("knowledge_max_chunks", 10000))

        # Caching — TTL=0 = persistenza indefinita (solo clear manuale)
        self.response_cache = DiskCache("responses", max_entries=self.max_cache_entries, ttl_seconds=0)
        self.embedding_cache = DiskCache("embeddings", max_entries=4096, ttl_seconds=0)

        # Monitor
        self.monitor = ResourceMonitor()

        # Pool thread
        self.pool = ManagedPool(max_workers=max(1, (os.cpu_count() or 4) // 2))

        self._lock = threading.Lock()
        self._gc_interval = 300  # 5 min
        self._last_gc = time.time()

    def check_memory(self):
        """True se sotto il limite di memoria."""
        mb = self.monitor.memory_mb()
        return mb < self.max_memory_mb

    def snapshot(self):
        s = self.monitor.snapshot()
        s["limits"] = {
            "max_memory_mb": self.max_memory_mb,
            "cache_entries": self.response_cache.size,
            "cache_disk_mb": round(self.response_cache.disk_usage_bytes / (1024 * 1024), 2),
            "conversation_max_msgs": self.max_conversation_messages,
            "pool_active": self.pool.active_count,
        }
        # GC
        s["gc_objects"] = len(gc.get_objects())
        return s

    def cleanup(self, force=False):
        """Pulizia periodica risorse. Con force=True, GC completo."""
        now = time.time()
        if not force and now - self._last_gc < self._gc_interval:
            return
        self.response_cache._housekeeping()
        self.embedding_cache._housekeeping()
        collected = gc.collect()
        self._last_gc = now
        return collected

    def optimize_memory(self):
        """Riduce uso memoria: GC forzato, cache trimming, pool drain."""
        n = self.cleanup(force=True) or 0
        # Pool: riduci worker inattivi
        self.pool._drain_idle()
        return {"gc_collected": n, "cache_entries": self.response_cache.size, "pool_active": self.pool.active_count}

    def close(self):
        """Chiusura ordinata: salva cache."""
        self.response_cache._save_meta()
        self.embedding_cache._save_meta()


# ─── Singolo globale ────────────────────────────────────────────────────

_resource_manager = None
_rm_lock = threading.Lock()


def get_resource_manager(cfg=None):
    global _resource_manager
    with _rm_lock:
        if _resource_manager is None:
            _resource_manager = ResourceManager(cfg)
        return _resource_manager


def reset_resource_manager():
    global _resource_manager
    with _rm_lock:
        if _resource_manager is not None:
            _resource_manager.close()
        _resource_manager = ResourceManager()
