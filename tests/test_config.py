"""Tests per hycoder/config.py."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from hycoder.config import (
    DEFAULT_CONFIG,
    deep_merge,
    load_config,
    save_config,
    CONFIG_DIR,
    CONFIG_FILE,
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


class TestDeepMerge:
    def test_merge_simple(self):
        base = {"a": 1, "b": 2}
        deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        deep_merge(base, {"a": {"y": 99, "z": 100}})
        assert base["a"]["x"] == 1
        assert base["a"]["y"] == 99
        assert base["a"]["z"] == 100
        assert base["b"] == 3

    def test_merge_new_section(self):
        base = {"a": 1}
        deep_merge(base, {"new_section": {"nested": "value"}})
        assert base["new_section"]["nested"] == "value"

    def test_merge_overwrite_non_dict(self):
        base = {"a": {"nested": "old"}}
        deep_merge(base, {"a": "new_value"})
        assert base["a"] == "new_value"

    def test_merge_empty_override(self):
        base = {"a": 1, "b": 2}
        deep_merge(base, {})
        assert base == {"a": 1, "b": 2}

    def test_merge_none_value(self):
        base = {"a": 1}
        deep_merge(base, {"a": None})
        assert base["a"] is None


class TestLoadConfig:
    def test_load_default_when_no_file(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
        ):
            cfg = load_config(force=True)
            assert cfg["local"]["model"] == DEFAULT_CONFIG["local"]["model"]
            assert cfg["router"]["mode"] == "auto"

    def test_load_merges_user_config(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        cfg_dir.mkdir(parents=True)
        user_cfg = {"local": {"model": "llama3:8b"}, "router": {"mode": "online"}}
        cfg_file.write_text(yaml.dump(user_cfg))
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
        ):
            cfg = load_config(force=True)
            assert cfg["local"]["model"] == "llama3:8b"
            assert cfg["router"]["mode"] == "online"
            assert cfg["online"]["model"] == DEFAULT_CONFIG["online"]["model"]

    def test_load_respects_env_api_key(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"}),
        ):
            cfg = load_config(force=True)
            assert cfg["online"]["api_key"] == "sk-test-key"

    def test_load_empty_yaml(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        cfg_dir.mkdir(parents=True)
        cfg_file.write_text("")
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
        ):
            cfg = load_config(force=True)
            assert cfg["local"]["model"] == DEFAULT_CONFIG["local"]["model"]

    def test_cache_works(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        cfg_dir.mkdir(parents=True)
        cfg_file.write_text(yaml.dump({"local": {"model": "first"}}))
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
        ):
            from hycoder.config import _config_cache, _config_cache_time
            _config_cache.clear()
            _config_cache_time = 0
            cfg1 = load_config(force=True)
            cfg_file.write_text(yaml.dump({"local": {"model": "second"}}))
            cfg2 = load_config(force=False)
            assert cfg2["local"]["model"] == "first"

    def test_force_reload(self, tmp_path):
        cfg_dir = tmp_path / ".config" / "hybrid-coder"
        cfg_file = cfg_dir / "config.yaml"
        cfg_dir.mkdir(parents=True)
        cfg_file.write_text(yaml.dump({"local": {"model": "first"}}))
        with (
            patch("hycoder.config.CONFIG_DIR", cfg_dir),
            patch("hycoder.config.CONFIG_FILE", cfg_file),
        ):
            from hycoder.config import _config_cache, _config_cache_time
            _config_cache.clear()
            _config_cache_time = 0
            cfg1 = load_config(force=True)
            cfg_file.write_text(yaml.dump({"local": {"model": "second"}}))
            cfg2 = load_config(force=True)
            assert cfg2["local"]["model"] == "second"


class TestSaveConfig:
    def test_save_creates_dir(self, tmp_path):
        cfg = make_cfg()
        config_dir = tmp_path / ".config" / "hybrid-coder"
        config_file = config_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", config_dir),
            patch("hycoder.config.CONFIG_FILE", config_file),
        ):
            save_config(cfg)
            assert config_file.exists()
            loaded = yaml.safe_load(config_file.read_text())
            assert loaded["local"]["model"] == cfg["local"]["model"]

    def test_save_masks_api_key(self, tmp_path):
        cfg = make_cfg(**{"online.api_key": "sk-secret-123"})
        config_dir = tmp_path / ".config" / "hybrid-coder"
        config_file = config_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", config_dir),
            patch("hycoder.config.CONFIG_FILE", config_file),
        ):
            save_config(cfg)
            loaded = yaml.safe_load(config_file.read_text())
            assert loaded["online"]["api_key"] == "***"
            assert "sk-secret" not in config_file.read_text()

    def test_save_masks_password(self, tmp_path):
        cfg = make_cfg(**{"online.password": "supersecret"})
        config_dir = tmp_path / ".config" / "hybrid-coder"
        config_file = config_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", config_dir),
            patch("hycoder.config.CONFIG_FILE", config_file),
        ):
            save_config(cfg)
            loaded = yaml.safe_load(config_file.read_text())
            assert loaded["online"]["password"] == "***"

    def test_save_roundtrip(self, tmp_path):
        cfg = make_cfg(**{"local.model": "custom-model", "router.mode": "local"})
        config_dir = tmp_path / ".config" / "hybrid-coder"
        config_file = config_dir / "config.yaml"
        with (
            patch("hycoder.config.CONFIG_DIR", config_dir),
            patch("hycoder.config.CONFIG_FILE", config_file),
        ):
            save_config(cfg)
            loaded = yaml.safe_load(config_file.read_text())
            assert loaded["local"]["model"] == "custom-model"
            assert loaded["router"]["mode"] == "local"


class TestDefaultConfig:
    def test_has_all_sections(self):
        assert "local" in DEFAULT_CONFIG
        assert "online" in DEFAULT_CONFIG
        assert "router" in DEFAULT_CONFIG
        assert "tracking" in DEFAULT_CONFIG
        assert "workspace" in DEFAULT_CONFIG
        assert "backup" in DEFAULT_CONFIG
        assert "language" in DEFAULT_CONFIG

    def test_router_has_expected_keys(self):
        assert "mode" in DEFAULT_CONFIG["router"]
        assert "complexity_threshold" in DEFAULT_CONFIG["router"]
        assert "always_local" in DEFAULT_CONFIG["router"]
        assert "always_online" in DEFAULT_CONFIG["router"]
        assert "keyword_boost_local" in DEFAULT_CONFIG["router"]
        assert "keyword_boost_online" in DEFAULT_CONFIG["router"]

    def test_default_model(self):
        model = DEFAULT_CONFIG["local"]["model"]
        assert isinstance(model, str) and len(model) > 0

    def test_default_online_provider(self):
        assert DEFAULT_CONFIG["online"]["provider"] == "opencode"

    def test_default_language(self):
        assert DEFAULT_CONFIG["language"] == "en"

    def test_backup_disabled_by_default(self):
        assert DEFAULT_CONFIG["backup"]["enabled"] is False
