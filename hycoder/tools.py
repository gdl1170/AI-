"""
AI+ Agent Tools — sistema di tool per agenti AI.
Ispirato a opencode, Claude Code, ChatGPT Code Interpreter.
"""

import os
import re
import json
import time
import uuid
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict


# ─── Configuration ───────────────────────────────────────────────

WORKSPACE_DIR = Path.home() / "hybrid-coder-workspace"
MAX_OUTPUT_CHARS = 10000
MAX_COMMAND_TIMEOUT = 60
ALLOWED_COMMANDS = [
    "python", "python3", "node", "npm", "npx",
    "ls", "cat", "head", "tail", "grep", "find",
    "echo", "pwd", "mkdir", "cp", "mv", "rm",
    "git", "curl", "wget", "pip", "pip3",
    "which", "file", "du", "df", "wc",
    "sort", "uniq", "diff", "patch", "tar", "zip", "unzip",
]
ALLOWED_PATTERNS = [
    re.compile(r"^python[23]?\s"),
    re.compile(r"^node\s"),
    re.compile(r"^npm\s"),
    re.compile(r"^npx\s"),
    re.compile(r"^git\s"),
    re.compile(r"^ls\s"),
    re.compile(r"^cat\s"),
    re.compile(r"^head\s"),
    re.compile(r"^tail\s"),
    re.compile(r"^grep\s"),
    re.compile(r"^find\s"),
    re.compile(r"^echo\s"),
    re.compile(r"^pwd\s"),
    re.compile(r"^mkdir\s"),
    re.compile(r"^cp\s"),
    re.compile(r"^mv\s"),
    re.compile(r"^rm\s"),
    re.compile(r"^curl\s"),
    re.compile(r"^wget\s"),
    re.compile(r"^pip\s"),
    re.compile(r"^which\s"),
    re.compile(r"^file\s"),
    re.compile(r"^du\s"),
    re.compile(r"^df\s"),
    re.compile(r"^wc\s"),
    re.compile(r"^sort\s"),
    re.compile(r"^uniq\s"),
    re.compile(r"^diff\s"),
    re.compile(r"^patch\s"),
    re.compile(r"^tar\s"),
    re.compile(r"^zip\s"),
    re.compile(r"^unzip\s"),
]


# ─── Data Types ──────────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    data: Any = None
    duration: float = 0.0

    def dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output[:MAX_OUTPUT_CHARS] if self.output else "",
            "error": self.error[:2000] if self.error else "",
            "data": self.data,
            "duration": round(self.duration, 3),
        }


class ToolError(Exception):
    pass


# ─── Tool base ───────────────────────────────────────────────────

class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


# ─── ReadTool ────────────────────────────────────────────────────

class ReadTool(Tool):
    name = "read"
    description = "Legge il contenuto di un file o directory"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Percorso del file o directory"},
            "offset": {"type": "integer", "description": "Linea da cui iniziare (opzionale)"},
            "limit": {"type": "integer", "description": "Numero massimo di linee (opzionale)"},
        },
        "required": ["path"],
    }

    def execute(self, path: str, offset: int = 0, limit: int = 2000, **kwargs) -> ToolResult:
        start = time.time()
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, error=f"File non trovato: {path}", duration=time.time() - start)
            if p.is_dir():
                entries = sorted(p.iterdir())
                lines = []
                for e in entries:
                    suffix = "/" if e.is_dir() else ""
                    size = e.stat().st_size if e.is_file() else 0
                    lines.append(f"{e.name}{suffix}  ({size:,} bytes)" if size else e.name + suffix)
                return ToolResult(success=True, output="\n".join(lines), data={"type": "directory", "entries": len(lines)}, duration=time.time() - start)

            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if offset > 0 or limit < 100000:
                lines = content.splitlines()
                start_line = max(0, offset - 1)
                end_line = start_line + limit if limit else len(lines)
                selected = lines[start_line:end_line]
                output = "\n".join(selected)
                meta = {"total_lines": len(lines), "start_line": start_line + 1, "end_line": end_line, "showing": len(selected)}
            else:
                output = content
                meta = {"total_lines": content.count("\n") + 1, "size_bytes": len(content)}

            return ToolResult(success=True, output=output, data=meta, duration=time.time() - start)

        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── WriteTool ───────────────────────────────────────────────────

class WriteTool(Tool):
    name = "write"
    description = "Scrive contenuto in un file (crea o sovrascrive)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Percorso del file"},
            "content": {"type": "string", "description": "Contenuto da scrivere"},
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        start = time.time()
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            # Safety: prevent overwriting critical files
            if p.name in (".bashrc", ".zshrc", ".profile", ".ssh/config") and p.exists():
                return ToolResult(success=False, error=f"Per sicurezza, non puoi sovrascrivere {p.name}")

            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Scritti {len(content)} bytes in {p}", data={"path": str(p), "size": len(content)}, duration=time.time() - start)
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── EditTool ────────────────────────────────────────────────────

class EditTool(Tool):
    name = "edit"
    description = "Modifica un file esistente: trova oldString e la sostituisce con newString"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Percorso del file"},
            "old_string": {"type": "string", "description": "Testo da trovare (deve corrispondere esattamente)"},
            "new_string": {"type": "string", "description": "Testo sostitutivo"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def execute(self, path: str, old_string: str, new_string: str, **kwargs) -> ToolResult:
        start = time.time()
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, error=f"File non trovato: {path}", duration=time.time() - start)

            content = p.read_text(encoding="utf-8")
            if old_string not in content:
                return ToolResult(success=False, error=f"oldString non trovato in {path}", duration=time.time() - start)

            count = content.count(old_string)
            if count > 1:
                return ToolResult(success=False, error=f"Trovate {count} occorrenze. Fornisci più contesto.", duration=time.time() - start)

            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding="utf-8")
            return ToolResult(success=True, output=f"Modificato {p}", data={"path": str(p), "replaced": len(old_string), "new_size": len(new_content)}, duration=time.time() - start)

        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── DeleteTool ──────────────────────────────────────────────────

class DeleteTool(Tool):
    name = "delete"
    description = "Elimina un file o directory"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Percorso del file/directory da eliminare"},
            "recursive": {"type": "boolean", "description": "Se true, elimina ricorsivamente le directory"},
        },
        "required": ["path"],
    }

    def execute(self, path: str, recursive: bool = False, **kwargs) -> ToolResult:
        start = time.time()
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, error=f"Non trovato: {path}", duration=time.time() - start)

            safe_paths = [Path.home() / ".config" / "hybrid-coder"]

            # Safety checks
            if any(s in p.parents for s in safe_paths):
                return ToolResult(success=False, error=f"Per sicurezza, non puoi eliminare file di sistema AI+", duration=time.time() - start)

            if p.is_dir():
                if not recursive:
                    return ToolResult(success=False, error=f"Use recursive=True per eliminare directory", duration=time.time() - start)
                shutil.rmtree(p)
                return ToolResult(success=True, output=f"Directory eliminata: {p}", duration=time.time() - start)
            else:
                p.unlink()
                return ToolResult(success=True, output=f"File eliminato: {p}", duration=time.time() - start)

        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── RunTool ─────────────────────────────────────────────────────

class RunTool(Tool):
    name = "run"
    description = "Esegue un comando shell. Limite: 60 secondi, output 10K caratteri."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Comando da eseguire"},
            "workdir": {"type": "string", "description": "Directory di lavoro (opzionale)"},
            "timeout": {"type": "integer", "description": "Timeout in secondi (default: 30, max: 60)"},
        },
        "required": ["command"],
    }

    def execute(self, command: str, workdir: str = None, timeout: int = 30, **kwargs) -> ToolResult:
        start = time.time()
        try:
            # Security: check allowed commands
            cmd_trimmed = command.strip()
            allowed = any(pattern.match(cmd_trimmed) for pattern in ALLOWED_PATTERNS)
            if not allowed:
                return ToolResult(success=False, error=f"Comando non consentito. Comandi permessi: {', '.join(ALLOWED_COMMANDS[:10])}...", duration=time.time() - start)

            timeout = min(timeout, MAX_COMMAND_TIMEOUT)
            cwd = Path(workdir).expanduser().resolve() if workdir else None

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or os.getcwd(),
            )

            output = ""
            if result.stdout:
                output += result.stdout[:MAX_OUTPUT_CHARS]
            if result.stderr:
                if output:
                    output += "\n--- stderr ---\n"
                output += result.stderr[:MAX_OUTPUT_CHARS]

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                data={
                    "returncode": result.returncode,
                    "stdout": (result.stdout or "")[:MAX_OUTPUT_CHARS],
                    "stderr": (result.stderr or "")[:2000],
                },
                duration=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Comando terminato per timeout ({timeout}s)", duration=time.time() - start)
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── SearchTool ──────────────────────────────────────────────────

class SearchTool(Tool):
    name = "search"
    description = "Cerca testo nei file del workspace (grep/find combinato)"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Pattern regex da cercare"},
            "path": {"type": "string", "description": "Directory di ricerca (default: workspace)"},
            "include": {"type": "string", "description": "Glob pattern per file (es. *.py, *.{ts,tsx})"},
            "max_results": {"type": "integer", "description": "Massimo risultati (default: 20)"},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = None, include: str = None, max_results: int = 20, **kwargs) -> ToolResult:
        start = time.time()
        try:
            search_path = Path(path).expanduser().resolve() if path else WORKSPACE_DIR.expanduser().resolve()
            if not search_path.exists():
                return ToolResult(success=False, error=f"Directory non trovata: {search_path}", duration=time.time() - start)

            # Build grep command
            cmd = f"grep -rn --binary-files=without-match"
            if include:
                cmd += f" --include='{include}'"
            cmd += f" -m {max_results} '{pattern}' {search_path}"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)

            if result.returncode > 1:
                return ToolResult(success=False, error=f"Errore ricerca: {result.stderr[:500]}", duration=time.time() - start)

            lines = [l for l in result.stdout.split("\n") if l.strip()][:max_results]
            return ToolResult(success=True, output="\n".join(lines), data={"matches": len(lines), "pattern": pattern}, duration=time.time() - start)

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Ricerca terminata per timeout (15s)", duration=time.time() - start)
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── WebTool ─────────────────────────────────────────────────────

class WebTool(Tool):
    name = "web"
    description = "Cerca informazioni sul web o scarica una pagina"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query di ricerca web o URL completo"},
            "type": {"type": "string", "enum": ["search", "fetch"], "description": "search=google, fetch=scarica URL"},
        },
        "required": ["query"],
    }

    def execute(self, query: str, type: str = "search", **kwargs) -> ToolResult:
        start = time.time()
        try:
            if type == "fetch" and (query.startswith("http://") or query.startswith("https://")):
                import urllib.request as req
                r = req.urlopen(query, timeout=10)
                content = r.read().decode("utf-8", errors="replace")
                import re as _re
                text = _re.sub(r"<[^>]+>", " ", content)
                text = _re.sub(r"\s+", " ", text).strip()[:6000]
                return ToolResult(success=True, output=text, data={"url": query, "chars": len(text)}, duration=time.time() - start)
            else:
                # Simple DuckDuckGo-style search via curl
                q = query.replace(" ", "+")
                url = f"https://html.duckduckgo.com/html/?q={q}"
                import urllib.request as req
                r = req.urlopen(url, timeout=10)
                html = r.read().decode("utf-8", errors="replace")
                import re as _re
                results = _re.findall(r'class="result__snippet">(.*?)</a>', html, _re.DOTALL)
                if not results:
                    results = _re.findall(r'class="result__body".*?>(.*?)</div>', html, _re.DOTALL)
                clean = [_re.sub(r"<[^>]+>", "", r).strip()[:300] for r in results[:5]]
                output = "\n---\n".join(clean) if clean else "Nessun risultato trovato"
                return ToolResult(success=True, output=output, data={"results": len(clean), "query": query}, duration=time.time() - start)

        except Exception as e:
            return ToolResult(success=False, error=f"Web error: {str(e)[:200]}", duration=time.time() - start)


# ─── MemoryTool ──────────────────────────────────────────────────

class MemoryTool(Tool):
    name = "memory"
    description = "Salva e recupera informazioni nella memoria a breve termine dell'agente"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["store", "recall", "list", "clear"], "description": "store=salva, recall=recupera, list=elenca, clear=pulisci"},
            "key": {"type": "string", "description": "Chiave per store/recall"},
            "value": {"type": "string", "description": "Valore da salvare (solo per action=store)"},
        },
        "required": ["action"],
    }

    _memory: dict = {}
    _lock = threading.Lock()

    def execute(self, action: str, key: str = None, value: str = None, **kwargs) -> ToolResult:
        start = time.time()
        try:
            with self._lock:
                if action == "store":
                    if not key or value is None:
                        return ToolResult(success=False, error="Servono key e value per store", duration=time.time() - start)
                    self._memory[key] = {"value": value, "time": time.time()}
                    return ToolResult(success=True, output=f"Memorizzato: {key}", data={"key": key}, duration=time.time() - start)

                elif action == "recall":
                    if not key:
                        return ToolResult(success=False, error="Serve key per recall", duration=time.time() - start)
                    entry = self._memory.get(key)
                    if entry is None:
                        return ToolResult(success=False, error=f"Nessun dato per: {key}", duration=time.time() - start)
                    return ToolResult(success=True, output=entry["value"], data={"key": key, "age_s": time.time() - entry["time"]}, duration=time.time() - start)

                elif action == "list":
                    if not self._memory:
                        return ToolResult(success=True, output="Memoria vuota", data={"keys": []}, duration=time.time() - start)
                    lines = [f"• {k}  ({len(v['value'])} chars, {time.time()-v['time']:.0f}s ago)" for k, v in self._memory.items()]
                    return ToolResult(success=True, output="\n".join(lines), data={"keys": list(self._memory.keys()), "count": len(self._memory)}, duration=time.time() - start)

                elif action == "clear":
                    count = len(self._memory)
                    self._memory.clear()
                    return ToolResult(success=True, output=f"Memoria pulita ({count} entry rimosse)", data={"cleared": count}, duration=time.time() - start)

                return ToolResult(success=False, error=f"Azione sconosciuta: {action}", duration=time.time() - start)

        except Exception as e:
            return ToolResult(success=False, error=str(e), duration=time.time() - start)


# ─── ToolRegistry ────────────────────────────────────────────────

_registry = None
_registry_lock = threading.Lock()


def get_tool_registry() -> dict[str, Tool]:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = {
                    "read": ReadTool(),
                    "write": WriteTool(),
                    "edit": EditTool(),
                    "delete": DeleteTool(),
                    "run": RunTool(),
                    "search": SearchTool(),
                    "web": WebTool(),
                    "memory": MemoryTool(),
                }
    return _registry


def list_tools() -> list[dict]:
    reg = get_tool_registry()
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in reg.values()
    ]


def execute_tool(name: str, **kwargs) -> ToolResult:
    reg = get_tool_registry()
    tool = reg.get(name)
    if not tool:
        return ToolResult(success=False, error=f"Tool sconosciuto: {name}. Disponibili: {', '.join(reg.keys())}")
    return tool.execute(**kwargs)


def execute_tool_call(tool_call: dict) -> dict:
    """
    tool_call: {"name": "read", "arguments": {"path": "..."}}
    Returns: {"name": "read", "result": {...}}
    """
    name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    result = execute_tool(name, **arguments)
    return {
        "name": name,
        "arguments": arguments,
        "result": result.dict(),
    }
