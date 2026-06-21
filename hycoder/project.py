"""
AI+ project module: scaffold, test, run, preview.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import json
import time
from pathlib import Path

PROJECTS_DIR = Path.home() / ".config" / "hybrid-coder" / "projects"


def _projects_dir():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


PYTHON = sys.executable  # python3 on macOS/python on Linux

PROJECT_TEMPLATES = {
    "python": {
        "files": {
            "main.py": 'def main():\n    print("Hello from AI+!")\n\nif __name__ == "__main__":\n    main()\n',
            "tests/test_main.py": 'from main import *\n\ndef test_placeholder():\n    assert True\n',
        },
        "dirs": ["tests"],
        "run": f"{PYTHON} main.py",
        "test": f"{PYTHON} -m pytest tests/ -v",
    },
    "python-flask": {
        "files": {
            "app.py": 'from flask import Flask\n\napp = Flask(__name__)\n\n@app.route("/")\ndef hello():\n    return "Hello from AI+!"\n\nif __name__ == "__main__":\n    app.run(debug=True, port=5000)\n',
            "tests/test_app.py": 'from app import app\n\ndef test_hello():\n    with app.test_client() as c:\n        rv = c.get("/")\n        assert rv.status_code == 200\n',
            "requirements.txt": "flask\npytest\n",
        },
        "dirs": ["tests"],
        "run": f"{PYTHON} app.py",
        "test": f"{PYTHON} -m pytest tests/ -v",
    },
    "html": {
        "files": {
            "index.html": "<!DOCTYPE html>\n<html lang=\"it\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>Progetto AI+</title>\n    <style>\n        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }\n        h1 { color: #3fb950; }\n    </style>\n</head>\n<body>\n    <h1>Ciao da AI+!</h1>\n    <p>Progetto generato dall\'AI ibrida.</p>\n</body>\n</html>\n",
            "style.css": "body {\n    background: #0d1117;\n    color: #c9d1d9;\n}\n",
        },
        "dirs": [],
        "run": "python -m http.server 8000",
        "test": "",
    },
    "node-js": {
        "files": {
            "index.js": 'console.log("Hello from AI+!");\n\nfunction main() {\n    return "OK";\n}\n\nmodule.exports = { main };\n',
            "tests/test.js": 'const { main } = require("../index");\nconst assert = require("assert");\n\ndescribe("main", () => {\n    it("should return OK", () => {\n        assert.strictEqual(main(), "OK");\n    });\n});\n',
            "package.json": '{\n  "name": "hybrid-project",\n  "version": "1.0.0",\n  "scripts": {\n    "start": "node index.js",\n    "test": "mocha tests/test.js"\n  },\n  "devDependencies": {\n    "mocha": "^10.0.0"\n  }\n}\n',
        },
        "dirs": ["tests"],
        "run": "node index.js",
        "test": "npm test",
    },
}


def list_templates():
    return list(PROJECT_TEMPLATES.keys())


def get_template(name):
    t = PROJECT_TEMPLATES.get(name)
    if not t:
        raise ValueError(f"Template '{name}' non trovato. Disponibili: {', '.join(list_templates())}")
    return t


def create_project(name, template="python", path=None):
    if path is None:
        path = _projects_dir() / name
    else:
        path = Path(path)

    if path.exists():
        raise FileExistsError(f"'{path}' esiste già")

    t = get_template(template)

    path.mkdir(parents=True, exist_ok=True)
    for d in t.get("dirs", []):
        (path / d).mkdir(parents=True, exist_ok=True)

    for filepath, content in t["files"].items():
        fp = path / filepath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

    meta = {
        "name": name,
        "template": template,
        "path": str(path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_cmd": t.get("run", ""),
        "test_cmd": t.get("test", ""),
    }
    (path / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))

    return meta


def list_projects():
    d = _projects_dir()
    projects = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and (p / ".hybrid-project.json").exists():
            meta = json.loads((p / ".hybrid-project.json").read_text())
            projects.append(meta)
    return projects


def get_project(name_or_path):
    projects = list_projects()
    for p in projects:
        if p["name"] == name_or_path:
            return p
    p = Path(name_or_path)
    if p.exists() and (p / ".hybrid-project.json").exists():
        return json.loads((p / ".hybrid-project.json").read_text())
    raise FileNotFoundError(f"Progetto '{name_or_path}' non trovato")


def run_project(name_or_path, file=None):
    project = get_project(name_or_path)
    cwd = project["path"]

    if file:
        cmd = [sys.executable, str(Path(cwd) / file)]
    else:
        cmd = project.get("run_cmd", "").split()
        if not cmd:
            raise ValueError(f"Nessun comando di run per '{project['name']}'")

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    return {
        "project": project["name"],
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }


def test_project(name_or_path):
    project = get_project(name_or_path)
    cwd = project["path"]
    cmd = project.get("test_cmd", "").split()
    if not cmd:
        return {
            "project": project["name"],
            "command": "",
            "stdout": "",
            "stderr": "Nessun comando di test configurato",
            "returncode": -1,
            "success": False,
        }

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    return {
        "project": project["name"],
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }


def delete_project(name_or_path):
    project = get_project(name_or_path)
    path = Path(project["path"])
    shutil.rmtree(path)
    return project


def preview_project(name_or_path, file=None):
    project = get_project(name_or_path)
    cwd = Path(project["path"])

    if file:
        files_to_show = [cwd / file]
    else:
        files_to_show = sorted(cwd.rglob("*"))
        files_to_show = [f for f in files_to_show if f.is_file() and f.name != ".hybrid-project.json"]

    sources = {}
    for fp in files_to_show:
        try:
            rel = fp.relative_to(cwd)
            sources[str(rel)] = fp.read_text()
        except Exception:
            pass

    run_result = None
    try:
        run_result = run_project(name_or_path)
    except Exception:
        pass

    return {
        "project": project,
        "sources": sources,
        "output": run_result,
    }


# ── AI-powered project generation ──

GENERATION_SYSTEM_PROMPT = """You are a project generator AI. Given a description, generate a complete working project.

Respond with ONLY valid JSON. No markdown fences, no explanation, no commentary.

JSON structure:
{
  "name": "kebab-case-project-name",
  "description": "One-line description",
  "tech_stack": ["python", "flask"],
  "files": {
    "relative/path/to/file.py": "complete file contents",
    "relative/path/to/test_file.py": "complete test file contents",
    "requirements.txt": "dependencies"
  },
  "run_command": "command to run the project",
  "test_command": "command to run tests"
}

Rules:
- Include ALL files needed for a complete, working project
- Include config files (requirements.txt, package.json, etc.)
- Include at least basic tests
- File contents MUST be complete working code with real implementations
- Use relative paths from project root
- Name should be kebab-case"""


def generate_project_from_prompt(prompt, generate_fn, name_hint=None):
    """Generate a complete project using an AI provider.
    
    Args:
        prompt: Natural language project description
        generate_fn: Callable(messages) -> ProviderResult with .text
        name_hint: Optional preferred project name
    
    Returns: Project metadata dict
    """
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt + (f"\n\nPreferred name: {name_hint}" if name_hint else "")}
    ]

    result = generate_fn(messages)
    text = result.text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    project_def = json.loads(text)

    name = project_def.get("name", name_hint or "ai-project")
    files = project_def.get("files", {})

    if not files:
        raise ValueError("AI did not return any files. Response: " + text[:500])

    # Ensure unique project path
    projects_dir = _projects_dir()
    path = projects_dir / name
    counter = 1
    while path.exists():
        path = projects_dir / f"{name}-{counter}"
        counter += 1
    name = path.name

    path.mkdir(parents=True, exist_ok=True)
    for filepath, content in files.items():
        fp = path / filepath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

    meta = {
        "name": name,
        "template": "ai-generated",
        "path": str(path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": project_def.get("description", ""),
        "tech_stack": project_def.get("tech_stack", []),
        "run_cmd": project_def.get("run_command", ""),
        "test_cmd": project_def.get("test_command", ""),
        "files_count": len(files),
    }
    (path / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))

    return meta


IMPROVE_SYSTEM_PROMPT = """You are a project improver AI. Given an existing project and a modification request, update the project files.

Respond with ONLY valid JSON. No markdown fences.

The project currently has these files:
{file_list}

JSON structure:
{json_schema}

Rules:
- Include ONLY files that need to change. Omitted files stay as-is.
- For modified files, include the COMPLETE updated content, not just the diff
- For new files, include full content
- File contents MUST be complete working code
- Use relative paths from project root"""

IMPROVE_JSON_SCHEMA = r"""{
  "description": "Brief summary of changes",
  "files": {
    "relative/path/to/new_or_existing_file.py": "updated/complete file contents"
  },
  "files_to_delete": ["relative/path/to/file_to_delete"],
  "run_command": "updated run command (or leave empty to keep current)",
  "test_command": "updated test command (or leave empty to keep current)"
}"""


def improve_project(name, prompt, generate_fn):
    """Use AI to modify/improve an existing project.
    
    Returns updated project metadata.
    """
    project = get_project(name)
    cwd = Path(project["path"])

    # Build file list for context
    file_list = []
    for f in sorted(cwd.rglob("*")):
        if f.is_file() and f.name != ".hybrid-project.json":
            rel = f.relative_to(cwd)
            file_list.append(str(rel))

    sys_prompt = IMPROVE_SYSTEM_PROMPT.replace("{file_list}", "\n".join(file_list)).replace("{json_schema}", IMPROVE_JSON_SCHEMA)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Project: {project['name']}\nDescription: {project.get('description', '')}\n\nRequest: {prompt}"}
    ]

    result = generate_fn(messages)
    text = result.text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    changes = json.loads(text)

    # Update/delete files
    files = changes.get("files", {})
    for filepath, content in files.items():
        fp = cwd / filepath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

    for filepath in changes.get("files_to_delete", []):
        fp = cwd / filepath
        if fp.exists():
            fp.unlink()

    # Update metadata
    meta = get_project(name)
    if changes.get("description"):
        meta["description"] = changes["description"]
    if changes.get("run_command"):
        meta["run_cmd"] = changes["run_command"]
    if changes.get("test_command"):
        meta["test_cmd"] = changes["test_command"]

    # Re-count files
    file_count = 0
    for f in cwd.rglob("*"):
        if f.is_file() and f.name != ".hybrid-project.json":
            file_count += 1
    meta["files_count"] = file_count

    (cwd / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))

    return meta


def write_project_file(name, filepath, content):
    """Write/update a single file in a project."""
    project = get_project(name)
    cwd = Path(project["path"])
    fp = cwd / filepath
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)

    # Re-count files
    file_count = 0
    for f in cwd.rglob("*"):
        if f.is_file() and f.name != ".hybrid-project.json":
            file_count += 1

    meta = get_project(name)
    meta["files_count"] = file_count
    (cwd / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))
    return True


def delete_project_file(name, filepath):
    """Delete a single file from a project."""
    project = get_project(name)
    cwd = Path(project["path"])
    fp = cwd / filepath
    if not fp.exists():
        raise FileNotFoundError(f"File '{filepath}' not found in project")
    if not fp.is_file() or fp.name == ".hybrid-project.json":
        raise ValueError(f"Cannot delete '{filepath}'")
    fp.unlink()

    # Remove empty parent dirs
    parent = fp.parent
    while parent != cwd:
        if any(parent.iterdir()):
            break
        parent.rmdir()
        parent = parent.parent

    meta = get_project(name)
    file_count = 0
    for f in cwd.rglob("*"):
        if f.is_file() and f.name != ".hybrid-project.json":
            file_count += 1
    meta["files_count"] = file_count
    (cwd / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))
    return True


def run_project_stream(name, file=None):
    """Run project command and yield output lines in real-time."""
    import shlex as _shlex
    project = get_project(name)
    cwd = Path(project["path"])

    if file:
        cmd = [sys.executable, str(cwd / file)]
    else:
        cmd_str = project.get("run_cmd", "")
        if not cmd_str:
            raise ValueError(f"No run command for '{project['name']}'")
        cmd = _shlex.split(cmd_str)

    yield {"type": "status", "message": f"Running: {' '.join(cmd)}"}

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        for line in iter(proc.stdout.readline, ""):
            if line:
                yield {"type": "stdout", "line": line.rstrip()}
        for line in iter(proc.stderr.readline, ""):
            if line:
                yield {"type": "stderr", "line": line.rstrip()}
        proc.wait()
        yield {"type": "done", "returncode": proc.returncode}
    except FileNotFoundError as e:
        yield {"type": "error", "message": f"Command not found: {e}"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}


def clone_project(name, new_name=None):
    """Clone an existing project."""
    project = get_project(name)
    src = Path(project["path"])
    new_name = new_name or f"{name}-clone"
    projects_dir = _projects_dir()
    dest = projects_dir / new_name
    counter = 1
    while dest.exists():
        dest = projects_dir / f"{new_name}-{counter}"
        counter += 1

    # Copy all files
    for f in src.rglob("*"):
        if f.is_file() and f.name != ".hybrid-project.json":
            rel = f.relative_to(src)
            fp = dest / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(f.read_bytes())

    # Create metadata
    file_count = 0
    for f in dest.rglob("*"):
        if f.is_file():
            file_count += 1

    meta = {
        "name": dest.name,
        "template": project.get("template", "cloned"),
        "path": str(dest),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": project.get("description", ""),
        "tech_stack": list(project.get("tech_stack", [])),
        "run_cmd": project.get("run_cmd", ""),
        "test_cmd": project.get("test_cmd", ""),
        "files_count": file_count,
    }
    (dest / ".hybrid-project.json").write_text(json.dumps(meta, indent=2))
    return meta


def update_project_config(name, **kwargs):
    """Update project metadata fields."""
    project = get_project(name)
    cwd = Path(project["path"])
    meta_path = cwd / ".hybrid-project.json"
    meta = json.loads(meta_path.read_text())
    for key, val in kwargs.items():
        if val is not None and key in ("description", "run_cmd", "test_cmd"):
            meta[key] = str(val)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def rename_project(name, new_name):
    """Rename a project directory and update metadata."""
    project = get_project(name)
    old_path = Path(project["path"])
    new_path = old_path.parent / new_name

    if new_path.exists():
        raise FileExistsError(f"Project '{new_name}' already exists")

    # Update metadata in-place before rename so get_project still works for path lookup
    meta_path = old_path / ".hybrid-project.json"
    meta = json.loads(meta_path.read_text())
    meta["name"] = new_name
    meta["path"] = str(new_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    old_path.rename(new_path)
    return json.loads((new_path / ".hybrid-project.json").read_text())


def export_project_zip(name, output_path=None):
    """Create a zip archive of the project directory."""
    import zipfile
    project = get_project(name)
    cwd = Path(project["path"])

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".zip")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in cwd.rglob("*"):
            if f.is_file():
                rel = f.relative_to(cwd)
                zf.write(f, str(rel))

    return output_path


def import_project_zip(zip_path, name_hint=None):
    """Import a project from a zip archive."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get project name from the zip entries
        names = zf.namelist()
        if not names:
            raise ValueError("Empty zip archive")

        # Determine root: if all files share a common prefix dir, use that as name
        common_root = Path(names[0]).parts[0] if '/' in names[0] else None
        if common_root and all(n.startswith(common_root + '/') or n == common_root for n in names):
            base_name = name_hint or common_root
        else:
            base_name = name_hint or Path(zip_path).stem

        projects_dir = _projects_dir()
        dest = projects_dir / base_name
        counter = 1
        while dest.exists():
            dest = projects_dir / f"{base_name}-{counter}"
            counter += 1

        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name.endswith('/'):
                continue
            # Strip common root if present
            if common_root and name.startswith(common_root + '/'):
                rel = Path(name).relative_to(common_root)
            else:
                rel = Path(name)
            fp = dest / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(zf.read(name))

        # Create metadata if not present
        meta_path = dest / ".hybrid-project.json"
        if not meta_path.exists():
            # Determine tech stack
            tech = []
            if (dest / "requirements.txt").exists():
                tech.append("python")
            if (dest / "package.json").exists():
                tech.append("node")
            if (dest / "index.html").exists():
                tech.append("html")
            if (dest / "Cargo.toml").exists():
                tech.append("rust")
            if (dest / "go.mod").exists():
                tech.append("go")

            file_count = 0
            for f in dest.rglob("*"):
                if f.is_file() and f.name != ".hybrid-project.json":
                    file_count += 1

            meta = {
                "name": dest.name,
                "template": "imported",
                "path": str(dest),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "description": f"Imported project ({file_count} files)",
                "tech_stack": tech,
                "run_cmd": "",
                "test_cmd": "",
                "files_count": file_count,
            }
            meta_path.write_text(json.dumps(meta, indent=2))

        return json.loads(meta_path.read_text())


# ── Run arbitrary code (sandboxed for web) ──

# Languages that can be executed
EXECUTABLE_LANGS = {
    ".py": sys.executable,
    ".sh": "/bin/bash",
    ".js": "node",
    ".go": "go run",
}

RUNNABLE_LANG_NAMES = {
    "python": sys.executable,
    "python3": sys.executable,
    "bash": "/bin/bash",
    "sh": "/bin/bash",
    "shell": "/bin/bash",
    "javascript": "node",
    "js": "node",
    "typescript": "npx tsx",
    "ts": "npx tsx",
    "ruby": "ruby",
    "php": "php",
    "go": "go run",
}

def _cmd_for(language, fpath, code):
    interp = RUNNABLE_LANG_NAMES.get(language)
    if interp == "go run":
        fpath.write_text(f"package main\n\nimport \"fmt\"\n\nfunc main() {{\n{code}\n}}")
        return ["go", "run", str(fpath)]
    if interp == "npx tsx":
        return ["npx", "tsx", str(fpath)]
    return [interp, str(fpath)]


def run_code(code, language="python"):
    interpreter = RUNNABLE_LANG_NAMES.get(language)
    if not interpreter:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Linguaggio '{language}' non supportato. Supportati: {', '.join(RUNNABLE_LANG_NAMES.keys())}",
            "returncode": -1,
        }

    import time as _time
    t0 = _time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = ".py" if language in ("python", "python3") else ".sh" if language in ("bash", "sh", "shell") else ".js" if language in ("javascript", "js") else ".ts" if language in ("typescript", "ts") else ".rb" if language == "ruby" else ".php" if language == "php" else ".txt"
        fpath = Path(tmpdir) / f"code{ext}"
        fpath.write_text(code)
        os.chmod(fpath, 0o755)

        cmd = _cmd_for(language, fpath, code)

        try:
            result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, timeout=30)
            elapsed = int((_time.time() - t0) * 1000)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "time_ms": elapsed,
            }
        except subprocess.TimeoutExpired:
            elapsed = int((_time.time() - t0) * 1000)
            return {
                "success": False,
                "stdout": "",
                "stderr": "Timeout (30s)",
                "returncode": -1,
                "time_ms": elapsed,
            }
        except FileNotFoundError as e:
            elapsed = int((_time.time() - t0) * 1000)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Interprete non trovato: {e}",
                "returncode": -1,
                "time_ms": elapsed,
            }
