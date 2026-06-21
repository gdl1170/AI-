import os
import time
from copy import deepcopy

import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "local": {
        "enabled": True,
        "provider": "ollama",
        "model": "qwen3.5:4b",
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
        "api_key": None,
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
        "keyword_boost_local": ["hello", "ciao", "help", "aiuto", "define", "cos'è", "what is", "simple", "basic"],
        "keyword_boost_online": ["search", "cerca", "news", "notizie", "latest", "2025", "2026", "current",
                                 "web", "internet", "optimize", "ottimizza", "refactor", "architettura",
                                 "deploy", "production", "security", "sicurezza", "complex", "complesso",
                                 "large", "grande"],
    },
    "tracking": {
        "save_history": True,
        "history_file": "~/.ai-plus-history.json",
        "show_cost": True,
        "local_cost_per_1k_tokens": 0.0,
        "online_cost_per_1k_input": 0.01,
        "online_cost_per_1k_output": 0.03,
    },
    "workspace": {
        "default_path": ".",
    },
    "backup": {
        "enabled": False,
        "interval_hours": 24,
        "destination": str(Path.home() / ".config" / "ai-plus" / "backups"),
    },
    "language": "en",
}

CONFIG_DIR = Path.home() / ".config" / "hybrid-coder"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

_config_cache = {}
_config_cache_time = 0
_config_cache_ttl = 2


def load_config(force=False):
    global _config_cache, _config_cache_time

    now = time.time()
    if not force and _config_cache and (now - _config_cache_time) < _config_cache_ttl:
        return _config_cache

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user = yaml.safe_load(f) or {}
        cfg = deepcopy(DEFAULT_CONFIG)
        deep_merge(cfg, user)

        if cfg["online"]["api_key"] is None:
            env_key = cfg["online"]["api_key_env"]
            cfg["online"]["api_key"] = os.environ.get(env_key)
    else:
        cfg = deepcopy(DEFAULT_CONFIG)
        env_key = cfg["online"]["api_key_env"]
        cfg["online"]["api_key"] = os.environ.get(env_key)

    _config_cache = cfg
    _config_cache_time = now
    return cfg


def deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    safe = cfg.copy()
    if safe["online"].get("api_key"):
        safe["online"]["api_key"] = "***"
    if safe["online"].get("password"):
        safe["online"]["password"] = "***"
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(safe, f, default_flow_style=False, sort_keys=False)
    load_config(force=True)
