"""Tests per hycoder/install_pack.py."""

import os
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from hycoder.install_pack import (
    generate_setup_script,
    generate_bat_script,
    generate_install_sh,
    generate_install_bat,
    generate_pack,
    PACK_VERSION,
    WEB_PORT,
)


class TestGenerateSetupScript:
    def test_returns_content_when_source_exists(self, tmp_path):
        setup_source = tmp_path / "setupAI+"
        setup_source.write_text('VERSION = "0.4.0"\nWEB_PORT = 8081\nprint("hello")')
        with (
            patch("hycoder.install_pack.Path") as MockPath,
        ):
            mock_file = MagicMock(spec=Path)
            mock_file.resolve.return_value.parent.parent = tmp_path
            MockPath.return_value = mock_file
            MockPath.home = Path.home
            content = generate_setup_script()
            assert content is not None
            assert 'VERSION = "2.0.0"' in content

    def test_returns_none_when_missing(self, tmp_path):
        with (
            patch("hycoder.install_pack.Path") as MockPath,
        ):
            mock_file = MagicMock(spec=Path)
            mock_file.resolve.return_value.parent.parent = tmp_path
            MockPath.return_value = mock_file
            MockPath.home = Path.home
            content = generate_setup_script()
            assert content is None


class TestGenerateBatScript:
    def test_returns_content_when_exists(self, tmp_path):
        bat_source = tmp_path / "setupAI+.bat"
        bat_source.write_text("@echo off\necho Hello")
        with (
            patch("hycoder.install_pack.Path") as MockPath,
        ):
            mock_file = MagicMock(spec=Path)
            mock_file.resolve.return_value.parent.parent = tmp_path
            MockPath.return_value = mock_file
            MockPath.home = Path.home
            content = generate_bat_script()
            assert content is not None


class TestGenerateInstallSh:
    def test_contains_setup_check(self):
        script = generate_install_sh()
        assert "#!/usr/bin/env bash" in script
        assert "setupAI+" in script

    def test_zip_extraction_code(self):
        script = generate_install_sh()
        assert "zipfile" in script


class TestGenerateInstallBat:
    def test_contains_windows_commands(self):
        script = generate_install_bat()
        assert "@echo off" in script
        assert "setupAI+.bat" in script


class TestGeneratePack:
    def _setup_source_dir(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "setup.py").write_text("# setup")
        (src / "requirements.txt").write_text("click\nrich\n")
        (src / "README.md").write_text("# AI+")
        (src / "setupAI+").write_text("# setup script")
        (src / "setupAI+.bat").write_text("@echo off")
        hy = src / "hycoder"
        hy.mkdir()
        (hy / "__init__.py").write_text('__version__ = "0.4.0"')
        (hy / "cli.py").write_text("# cli")
        bin_d = src / "bin"
        bin_d.mkdir()
        (bin_d / "ai-plus").write_text("#!/usr/bin/env python3")
        os.chmod(src / "setupAI+", 0o755)
        return src

    def test_generates_zip_with_content(self, tmp_path):
        dest = tmp_path / "output"
        src = self._setup_source_dir(tmp_path)
        with (
            patch("hycoder.install_pack.Path") as MockPath,
        ):
            mock_file = MagicMock(spec=Path)
            mock_file.resolve.return_value.parent.parent = src
            MockPath.return_value = mock_file
            MockPath.home = Path.home
            MockPath.side_effect = lambda *a, **kw: Path(*a, **kw) if a or kw else mock_file

            zip_path = generate_pack(str(dest))
            assert zip_path.exists()
            assert zipfile.is_zipfile(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                assert "setup.py" in names
                assert "requirements.txt" in names
                assert "README.md" in names
                assert "setupAI+" in names
                assert "install.sh" in names
                assert "hycoder/__init__.py" in names

    def test_skips_pycache(self, tmp_path):
        dest = tmp_path / "output"
        src = tmp_path / "source"
        src.mkdir()
        (src / "setup.py").write_text("")
        (src / "requirements.txt").write_text("")
        (src / "README.md").write_text("")
        hy = src / "hycoder"
        hy.mkdir()
        (hy / "__init__.py").write_text("")
        (hy / "__pycache__").mkdir()
        (hy / "__pycache__" / "cache.pyc").write_text("")
        with (
            patch("hycoder.install_pack.Path") as MockPath,
        ):
            mock_file = MagicMock(spec=Path)
            mock_file.resolve.return_value.parent.parent = src
            MockPath.return_value = mock_file
            MockPath.home = Path.home
            MockPath.side_effect = lambda *a, **kw: Path(*a, **kw) if a or kw else mock_file

            zip_path = generate_pack(str(dest))
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                assert not any("__pycache__" in n for n in names)
