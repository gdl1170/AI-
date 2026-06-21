"""
Install pack generator for AI+.
Creates a cross-platform, single-command installation package.
"""
import io
import os
import re
import json
import zipfile
import platform
from pathlib import Path

WEB_PORT = 8081
PACK_VERSION = "2.0.0"


def generate_setup_script(port=WEB_PORT, version=PACK_VERSION):
    """Read canonical setupAI+ from source, substitute runtime values."""
    src = Path(__file__).resolve().parent.parent / "setupAI+"
    if not src.exists():
        return None
    content = src.read_text(encoding="utf-8")
    content = re.sub(r'^VERSION\s*=\s*"[^"]*"', f'VERSION = "{version}"', content, count=1)
    content = re.sub(r'^WEB_PORT\s*=\s*\d+', f"WEB_PORT = {port}", content, count=1)
    return content


def generate_bat_script():
    """Read canonical setupAI+.bat from source."""
    src = Path(__file__).resolve().parent.parent / "setupAI+.bat"
    if not src.exists():
        return None
    return src.read_text(encoding="utf-8")


def generate_install_sh(port=WEB_PORT):
    """Generate single-command install.sh wrapper (Unix)."""
    return f'''#!/usr/bin/env bash
#
# AI+ — Single-command installer
# Extracts and runs setupAI+, then starts the web UI.
#
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            AI+  —  Install (single command)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
cd "$DIR"

# If source is inside a zip, extract first
if [ ! -f "$DIR/setupAI+" ]; then
    echo "[INFO] Estrazione pacchetto..."
    python3 -c "
import zipfile, os
here = os.path.dirname(os.path.abspath(__file__))
for fn in os.listdir(here):
    if fn.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(here, fn)) as z:
            z.extractall(here)
        print('[OK] Estratto: ' + fn)
        break
"
fi

if [ -f "$DIR/setupAI+" ]; then
    chmod +x "$DIR/setupAI+"
    exec python3 "$DIR/setupAI+"
else
    echo "[ERR] setupAI+ non trovato. Assicurati di aver estratto tutto il pacchetto."
    exit 1
fi
'''


def generate_install_bat(port=WEB_PORT):
    """Generate single-command install.bat wrapper (Windows)."""
    return f'''@echo off
title AI+ Installer
chcp 65001 >nul

echo.
echo ======================================================================
echo           AI+  --  Install (single command, Windows)
echo ======================================================================
echo.

if not exist "%~dp0setupAI+.bat" (
    echo [INFO] Estrazione pacchetto...
    for %%f in ("%~dp0*.zip") do (
        python -c "import zipfile, sys; zipfile.ZipFile(r'%%f').extractall(r'%~dp0')"
        echo [OK] Estratto: %%~nxf
    )
)

if exist "%~dp0setupAI+.bat" (
    call "%~dp0setupAI+.bat"
) else (
    echo [ERR] setupAI+.bat non trovato.
    pause
    exit /b 1
)
'''


def generate_pack(dest_path, port=WEB_PORT):
    """Generate the full install pack zip with fresh files."""
    root = Path(__file__).resolve().parent.parent
    dest = Path(dest_path).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    setup_content = generate_setup_script(port=port)
    bat_content = generate_bat_script()
    install_sh = generate_install_sh(port=port)
    install_bat = generate_install_bat(port=port)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        def add_file(content, arcname, executable=False):
            if content is None:
                return
            info = zipfile.ZipInfo(arcname)
            info.external_attr = 0o100755 << 16 if executable else 0o100644 << 16
            zf.writestr(info, content)

        def add_disk_file(src_path, arcname=None, extra_skip=None):
            p = Path(src_path)
            if not p.exists():
                return
            name = arcname or p.name
            is_exec = name in ("setupAI+", "install.sh") or name.startswith("bin/")
            if is_exec:
                st = p.stat()
                zi = zipfile.ZipInfo.from_file(p, name)
                zi.external_attr = (st.st_mode | 0o111) << 16
                zf.writestr(zi, p.read_bytes())
            else:
                zf.write(p, arcname=name)

        def skip(name):
            return (name == "__pycache__" or name.endswith(".pyc")
                    or ".egg-info" in name or name == ".pytest_cache"
                    or name == "__init__.pyc" or name.startswith("."))

        # 1. Freshly generated install scripts
        add_file(setup_content, "setupAI+", executable=True)
        add_file(bat_content, "setupAI+.bat")
        add_file(install_sh, "install.sh", executable=True)
        add_file(install_bat, "install.bat")

        # 2. Core project files
        for name in ["setup.py", "requirements.txt", "README.md"]:
            add_disk_file(root / name)

        # 3. bin/
        bin_dir = root / "bin"
        if bin_dir.exists():
            for p in sorted(bin_dir.iterdir()):
                if p.is_file():
                    add_disk_file(p, arcname=f"bin/{p.name}")

        # 4. hycoder/ source tree
        for p in sorted((root / "hycoder").rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(skip(x) for x in rel.parts):
                continue
            add_disk_file(p, arcname=str(rel))

    zip_name = "ai-plus-install-pack.zip"
    zip_path = dest / zip_name
    zip_path.write_bytes(buf.getvalue())
    return zip_path
