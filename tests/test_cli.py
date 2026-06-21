"""Tests per hycoder/cli.py — CLI commands via Click test runner."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from hycoder.cli import cli


BASE_CFG = {
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
def runner():
    return CliRunner()


class TestCliBasic:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "AI+" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AI+" in result.output

    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    def test_no_subcommand_with_json(self, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        mock_router.return_value.decide.return_value = ("local", 0)
        result = runner.invoke(cli, ["--json"])
        assert result.exit_code == 0


class TestCliConfig:
    @patch("hycoder.cli.load_config")
    def test_config_show(self, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "local.model" in result.output

    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.save_config")
    def test_set_config(self, mock_save, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG.copy()
        result = runner.invoke(cli, ["set", "local.model", "llama3:8b"])
        assert result.exit_code == 0
        assert "impostato" in result.output

    @patch("hycoder.cli.clear_caches")
    def test_clearcache(self, mock_clear, runner):
        result = runner.invoke(cli, ["clearcache"])
        assert result.exit_code == 0
        assert "Cache" in result.output


class TestCliAgent:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    @patch("hycoder.cli.SessionTracker")
    def test_agent_command(self, mock_tracker, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        mock_tracker.return_value.summary_dict.return_value = {
            "local": {"calls": 5, "tokens_in": 100, "tokens_out": 200,
                      "tokens_total": 300, "time_s": 1.0},
            "online": {"calls": 3, "tokens_in": 50, "tokens_out": 150,
                       "tokens_total": 200, "time_s": 2.0},
            "total": {"calls": 8, "tokens_total": 500, "time_s": 3.0,
                      "tokens_in": 150, "tokens_out": 350},
            "cost": 0.0,
            "session_duration_s": 60,
            "progress_pct": 0,
            "total_estimate_s": 0,
            "remaining_s": 0,
        }
        result = runner.invoke(cli, ["agent"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "local_model" in data
        assert "online_avail" in data


class TestCliLearn:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    def test_learn_status_empty(self, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["learn", "status"])
        assert result.exit_code == 0

    def test_learn_help(self, runner):
        result = runner.invoke(cli, ["learn", "--help"])
        assert result.exit_code == 0


class TestCliNote:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    def test_note_create_and_list(self, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["note", "create", "Test Note", "-b", "Hello world", "-t", "test,cli"])
        assert result.exit_code == 0
        assert "creata" in result.output.lower()

    def test_note_help(self, runner):
        result = runner.invoke(cli, ["note", "--help"])
        assert result.exit_code == 0


class TestCliProject:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    def test_project_create(self, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["project", "create", "test-proj", "-t", "python", "-p", "."])
            assert result.exit_code == 0

    def test_project_list_templates(self, runner):
        result = runner.invoke(cli, ["project", "list-templates"])
        assert result.exit_code == 0
        assert "python" in result.output

    def test_project_help(self, runner):
        result = runner.invoke(cli, ["project", "--help"])
        assert result.exit_code == 0


class TestCliResources:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    def test_resources(self, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["resources"])
        assert result.exit_code == 0


class TestCliPack:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    @patch("hycoder.install_pack.generate_pack")
    def test_pack(self, mock_gen_pack, mock_local, mock_router, mock_cfg, runner, tmp_path):
        import zipfile
        mock_cfg.return_value = BASE_CFG
        zip_path = tmp_path / "test-pack.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "hello")
        mock_gen_pack.return_value = zip_path
        result = runner.invoke(cli, ["pack", "-o", str(zip_path)])
        assert result.exit_code == 0, result.output
        assert "Pacchetto" in result.output


class TestCliGenerate:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    @patch("hycoder.cli.get_online_provider")
    def test_generate_command(self, mock_online, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        mock_online.return_value.generate_chat.return_value = MagicMock(
            text=json.dumps({"response": "test"}),
            source="online", cached=False, tokens_total=10, tokens_in=5, tokens_out=5,
            time_s=0.5, model="gpt-4o",
        )
        mock_local.return_value.generate_chat.return_value = MagicMock(
            text="local response", source="local", cached=False,
            tokens_total=10, tokens_in=5, tokens_out=5, time_s=0.5, model="test",
        )
        mock_router.return_value.decide.return_value = ("local", 0)
        result = runner.invoke(cli, ["generate", "test prompt"])
        assert result.exit_code == 0


class TestCliWeb:
    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    @patch("hycoder.cli._serve_web")
    def test_web(self, mock_serve, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["web", "--port", "9999"])
        assert result.exit_code == 0

    @patch("hycoder.cli.load_config")
    @patch("hycoder.cli.SmartRouter")
    @patch("hycoder.cli.get_local_provider")
    @patch("hycoder.cli._serve_web")
    def test_serve_alias(self, mock_serve, mock_local, mock_router, mock_cfg, runner):
        mock_cfg.return_value = BASE_CFG
        result = runner.invoke(cli, ["serve", "--port", "9999"])
        assert result.exit_code == 0
