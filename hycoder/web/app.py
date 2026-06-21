import os
import re
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import requests as _requests

from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, send_from_directory, send_file, Response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ai-plus")

from ..config import load_config, CONFIG_FILE, save_config
from ..router import SmartRouter
from ..providers import get_local_provider, get_online_provider, clear_caches
from ..tracker import SessionTracker
from ..knowledge import get_knowledge_base, build_rag_context
from ..resources import get_resource_manager, DATA_DIR, CACHE_DIR, KB_DIR
from ..notes import get_note_store, NOTES_DIR
from ..project import (create_project, list_projects, get_project,
                       run_project, test_project, delete_project,
                       preview_project, list_templates, run_code)
from ..sessions import (get_conversation_store, start_auto_flush,
                         list_sessions, delete_session as del_session)
from ..tools import (list_tools, execute_tool, execute_tool_call,
                     get_tool_registry, ToolResult)
from ..agent_executor import AgentExecutor, build_agent_messages
from ..knowledge import (build_knowledge_graph, list_wiki_pages,
                         get_wiki_page, delete_wiki_page,
                         suggest_wiki_topics, generate_wiki_page,
                         auto_generate_wiki)

# ─── Conversation Store (persistente su disco) ─────────────────────────

_CONV_MAX = 50
_conv_store = None


def _get_conv_store():
    global _conv_store
    if _conv_store is None:
        _conv_store = get_conversation_store()
    return _conv_store


def _get_conv(session_id: str | None) -> tuple[str, list[dict]]:
    store = _get_conv_store()
    return store.get(session_id)


def _append_conv(session_id: str, role: str, content: str):
    store = _get_conv_store()
    store.append(session_id, role, content)


def _trim_conv(messages: list[dict], max_tokens: int = 32000) -> list[dict]:
    """Tronca la storia conservando system message e messaggi recenti."""
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    total = sum(len(m["content"]) for m in others)
    while others and total > max_tokens:
        removed = others.pop(0)
        total -= len(removed["content"])
    return system + others


# ─── Agent Storage ──────────────────────────────────────────────────────

AGENTS_DIR = Path.home() / ".config" / "hybrid-coder" / "agents"
COMMANDS_DIR = Path.home() / ".config" / "hybrid-coder" / "commands"
SKILLS_DIR = Path.home() / ".config" / "hybrid-coder" / "skills"


def _ensure_dirs():
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def list_agents():
    _ensure_dirs()
    agents = []
    for f in sorted(AGENTS_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["file"] = f.name
                agents.append(data)
        except Exception:
            pass
    return agents


def get_agent(name):
    path = AGENTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as fp:
            return json.load(fp)
    return None


def save_agent(name, data):
    _ensure_dirs()
    path = AGENTS_DIR / f"{name}.json"
    data["name"] = name
    data["updated"] = datetime.now().isoformat()
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    return path


def delete_agent(name):
    path = AGENTS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_commands():
    _ensure_dirs()
    cmds = []
    for f in sorted(COMMANDS_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["file"] = f.name
                cmds.append(data)
        except Exception:
            pass
    return cmds


def get_command(name):
    path = COMMANDS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as fp:
            return json.load(fp)
    return None


def save_command(name, data):
    _ensure_dirs()
    path = COMMANDS_DIR / f"{name}.json"
    data["name"] = name
    data["updated"] = datetime.now().isoformat()
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    return path


def delete_command(name):
    path = COMMANDS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ─── App Factory ─────────────────────────────────────────────────────────

# ─── Backup Scheduler ────────────────────────────────────────────────
_BACKUP_LAST_RUN_FILE = Path.home() / ".config" / "ai-plus" / "backup_last_run.txt"

def _load_last_backup_time():
    try:
        if _BACKUP_LAST_RUN_FILE.exists():
            return float(_BACKUP_LAST_RUN_FILE.read_text().strip())
    except: pass
    return 0

def _save_last_backup_time(t=None):
    if t is None:
        t = time.time()
    _BACKUP_LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BACKUP_LAST_RUN_FILE.write_text(str(t))

def _run_auto_backup(app):
    """Create a backup if the schedule says it's due."""
    from ..config import load_config
    cfg = load_config()
    bk = cfg.get("backup", {})
    if not bk.get("enabled", False):
        return
    interval = float(bk.get("interval_hours", 24))
    last = _load_last_backup_time()
    if time.time() - last < interval * 3600:
        return
    try:
        from .. import __version__
        bid = time.strftime("auto_%Y%m%d_%H%M%S")
        mdir = Path(bk.get("destination", "")) / bid
        mdir.mkdir(parents=True, exist_ok=True)
        manifest = {"id": bid, "version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": "auto", "size_bytes": 0}
        _save_backup_manifest(mdir, manifest)
        zip_path = mdir / "backup.zip"
        _build_backup_zip(zip_path)
        manifest["size_bytes"] = zip_path.stat().st_size
        _save_backup_manifest(mdir, manifest)
        _save_last_backup_time()
        log.info(f"Auto-backup created: {bid}")
    except Exception as e:
        log.error(f"Auto-backup failed: {e}")

def start_backup_scheduler(app, check_interval=60):
    def _loop():
        while True:
            try:
                _run_auto_backup(app)
            except Exception as e:
                log.debug(f"Backup scheduler check: {e}")
            time.sleep(check_interval)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    log.info("Backup scheduler started")

_BACKUPS_DIR = Path.home() / ".config" / "ai-plus" / "backups"

def _backup_meta_dir(bid):
    return _BACKUPS_DIR / bid

def _load_backup_manifest(mdir):
    mfile = mdir / "manifest.json"
    if mfile.exists():
        return json.loads(mfile.read_text())
    return None

def _save_backup_manifest(mdir, meta):
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def _build_backup_zip(zip_path):
    """Build the actual backup zip content (reused by auto and manual)."""
    import io, zipfile
    from ..config import CONFIG_FILE
    from ..knowledge import get_knowledge_base
    from ..notes import NOTES_DIR
    from ..sessions import SESSIONS_DIR
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        cfg_path = Path(CONFIG_FILE)
        if cfg_path.exists():
            zf.write(cfg_path, "config.yaml")
        agents = list_agents()
        zf.writestr("agents/index.json", json.dumps(agents, ensure_ascii=False, indent=2))
        for a in agents:
            name = a.get("name", "unknown")
            agent_data = get_agent(name)
            if agent_data:
                zf.writestr(f"agents/{name}.json", json.dumps(agent_data, ensure_ascii=False, indent=2))
        sdir = Path(SESSIONS_DIR)
        if sdir.exists():
            for f in sdir.glob("*.json"):
                zf.write(f, f"sessions/{f.name}")
        kb = get_knowledge_base()
        kdata = {"total_chunks": kb.total_chunks, "sources": kb.sources}
        if hasattr(kb.index, 'documents'):
            kdata["documents"] = kb.index.documents
        zf.writestr("knowledge/index.json", json.dumps(kdata, ensure_ascii=False, indent=2))
        wiki_dir = Path.home() / ".config" / "hybrid-coder" / "llmwiki"
        if wiki_dir.exists():
            for f in wiki_dir.glob("*.md"):
                zf.write(f, f"wiki/{f.name}")
        ndir = Path(NOTES_DIR)
        if ndir.exists():
            for f in ndir.glob("*.md"):
                zf.write(f, f"notes/{f.name}")
        prompts_file = Path.home() / ".config" / "hybrid-coder" / "prompts.json"
        if prompts_file.exists():
            zf.write(prompts_file, "prompts.json")
    zip_path.write_bytes(buf.getvalue())


class FakeResult:
    """Mock result object per l'agent executor."""
    def __init__(self, text="", source="local", tokens_total=0, time_s=0, model="", cached=False):
        self.text = text
        self.source = source
        self.tokens_in = 0
        self.tokens_out = tokens_total
        self.tokens_total = tokens_total
        self.time_s = time_s
        self.model = model
        self.cached = cached
        self.events = []


def create_app(cfg=None):
    if cfg is None:
        cfg = load_config()

    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.secret_key = os.urandom(24)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("error.html", code=404, message="Pagina non trovata"), 404

    @app.errorhandler(500)
    def server_error(e):
        log.error(f"500 error: {e}")
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal server error"}), 500
        return render_template("error.html", code=500, message="Errore interno del server"), 500

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        return resp

    router = SmartRouter(cfg)
    local_provider = get_local_provider(cfg)
    online_provider = get_online_provider(cfg)
    online_avail = online_provider is not None
    tracker = SessionTracker(cfg)

    app.config.update(
        CFG=cfg,
        ROUTER=router,
        LOCAL=local_provider,
        ONLINE=online_provider,
        ONLINE_AVAIL=online_avail,
        TRACKER=tracker,
    )

    # Auto-flush conversazioni ogni 30s (persistenza su disco)
    start_auto_flush(30)

    # Backup scheduler
    start_backup_scheduler(app)

    # ─── Routes ──────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return redirect(url_for("chat_page"))

    @app.route("/chat")
    def chat_page():
        return render_template("chat.html")

    @app.route("/dashboard")
    def dashboard_page():
        return render_template("dashboard.html")

    @app.route("/agents")
    def agents_page():
        return render_template("agents.html", active_page="agents")

    @app.route("/commands")
    def commands_page():
        return render_template("commands.html")

    @app.route("/settings")
    def settings_page():
        return render_template("settings.html")

    @app.route("/profile")
    def profile_page():
        return render_template("profile.html", active_page="profile")

    @app.route("/help")
    def help_page():
        return render_template("help.html", active_page="help")

    # ─── API: Chat Clear ────────────────────────────────────────────

    @app.route("/api/chat/clear", methods=["POST"])
    def api_chat_clear():
        data = request.get_json() or {}
        sid = data.get("session_id", "")
        store = _get_conv_store()
        if sid:
            store.clear(sid)
        return jsonify({"ok": True})

    # ─── API: Stats ──────────────────────────────────────────────────

    @app.route("/api/stats")
    def api_stats():
        t = app.config["TRACKER"]
        s = t.summary_dict()
        return jsonify(s)

    @app.route("/api/stats/reset", methods=["POST"])
    def api_stats_reset():
        from ..config import load_config
        cfg = load_config()
        hf = Path(cfg["tracking"]["history_file"]).expanduser()
        if hf.exists():
            hf.unlink()
        app.config["TRACKER"] = SessionTracker(cfg)
        return jsonify({"ok": True})

    @app.route("/api/stats/history")
    def api_stats_history():
        t = app.config["TRACKER"]
        return jsonify(t.history[-100:])

    # ─── API: Terminal ─────────────────────────────────────────────

    @app.route("/api/terminal/exec", methods=["POST"])
    def api_terminal_exec():
        """Esegue un comando shell e restituisce output in SSE."""
        data = request.get_json() or {}
        command = (data.get("command") or "").strip()
        cwd = (data.get("cwd") or os.path.expanduser("~")).strip()

        if not command:
            return jsonify({"error": "Comando vuoto"}), 400

        import subprocess as sp, shlex

        def generate():
            yield f"event: start\ndata: {json.dumps({'cwd': cwd})}\n\n"
            try:
                proc = sp.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=True,
                    env={**os.environ, "TERM": "xterm-256color"},
                )
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    yield f"data: {json.dumps({'text': line.rstrip()})}\n\n"
                proc.wait()
                yield f"event: exit\ndata: {json.dumps({'code': proc.returncode})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'text': f'[ERRORE] {e}'})}\n\n"
                yield f"event: exit\ndata: {json.dumps({'code': -1})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    def _get_builtin_commands():
        """Restituisce dict {nome: prompt_template} per comandi built-in AI+."""
        return {
            "ai-plus": "Sei AI+. Mostra un breve messaggio di benvenuto e spiega cosa puoi fare.",
            "help": "Sei un assistente AI+. Elenca brevemente i comandi disponibili: /mode, /model, /cache, /reset, /chat, /deploy, /test. Inoltre puoi usare 'ai-plus <comando>' per eseguire comandi CLI.",
            "mode": "Cambia la modalità router a: {args}. Opzioni: auto, local, online. Se {args} è vuoto, mostra la modalità corrente.",
            "model": "Cambia il modello locale a: {args}. Se {args} è vuoto, mostra il modello corrente.",
            "cache": "Pulisce la cache delle risposte AI.",
            "reset": "Resetta tutte le statistiche di sessione.",
            "/mode": "Cambia la modalità router a: {args}. Opzioni: auto, local, online. Se {args} è vuoto, mostra la modalità corrente.",
            "/model": "Cambia il modello locale a: {args}. Se {args} è vuoto, mostra il modello corrente.",
            "/cache": "Pulisce la cache delle risposte AI.",
            "/reset": "Resetta tutte le statistiche di sessione.",
        }

    @app.route("/api/terminal/ai-cmd", methods=["POST"])
    def api_terminal_ai_cmd():
        """Esegue un comando AI+ (custom o slash) e streamma risposta SSE."""
        data = request.get_json() or {}
        cmd_name = (data.get("name") or "").strip().lstrip("/")
        cmd_args = (data.get("args") or "").strip()
        cwd = (data.get("cwd") or os.path.expanduser("~")).strip()

        if not cmd_name:
            return jsonify({"error": "Nome comando vuoto"}), 400

        # 1. Cerca comando personalizzato
        custom = get_command(cmd_name)
        prompt = None
        source = None
        if custom:
            prompt = custom.get("prompt", "")
            source = custom.get("source", "auto")
        else:
            # 2. Cerca nei comandi built-in: slash e CLI
            builtins = _get_builtin_commands()
            if cmd_name in builtins:
                prompt = builtins[cmd_name]
            elif f"/{cmd_name}" in builtins:
                prompt = builtins[f"/{cmd_name}"]
            else:
                # 3. Fallback: usa cmd_name come prompt diretto
                prompt = cmd_name

        # Se il prompt ha {args}, sostituisci
        if prompt and cmd_args:
            prompt = prompt.replace("{args}", cmd_args).replace("{cwd}", cwd)

        online = app.config.get("ONLINE")
        local = app.config.get("LOCAL")
        provider = online if online else local
        source_name = "online" if online else "local"

        if not provider:
            def no_provider():
                yield f"data: {json.dumps({'text': '[ERRORE] Nessun provider AI disponibile (configura un provider online o avvia Ollama).'})}\n\n"
                yield f"event: exit\ndata: {json.dumps({'code': 1})}\n\n"
            return Response(no_provider(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        def generate():
            yield f"event: start\ndata: {json.dumps({'cwd': cwd, 'source': source_name})}\n\n"
            yield f"data: {json.dumps({'text': f'[AI+ /{cmd_name}] Esecuzione...'})}\n\n"
            try:
                result = provider.generate_chat([{"role": "user", "content": prompt or cmd_name}])
                text = result.text if isinstance(result, object) and hasattr(result, 'text') else str(result)
                for line in text.split("\n"):
                    if line.strip():
                        yield f"data: {json.dumps({'text': line})}\n\n"
                yield f"event: exit\ndata: {json.dumps({'code': 0})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'text': f'[ERRORE] {e}'})}\n\n"
                yield f"event: exit\ndata: {json.dumps({'code': 1})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ─── API: Agents ─────────────────────────────────────────────────

    @app.route("/api/agents")
    def api_agents():
        return jsonify(list_agents())

    @app.route("/api/agents/<name>")
    def api_agent_get(name):
        a = get_agent(name)
        if a is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(a)

    @app.route("/api/agents", methods=["POST"])
    def api_agent_create():
        data = request.get_json()
        name = (data or {}).get("name", "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        if get_agent(name):
            return jsonify({"error": "already exists"}), 409
        payload = {k: v for k, v in (data or {}).items() if k != "name"}
        if "description" not in payload:
            payload["description"] = ""
        if "model" not in payload:
            payload["model"] = "auto"
        if "mode" not in payload:
            payload["mode"] = "auto"
        if "instructions" not in payload:
            payload["instructions"] = ""
        if "permissions" not in payload:
            payload["permissions"] = {"read": True, "edit": False, "bash": False}
        if "tags" not in payload:
            payload["tags"] = []
        payload["created"] = datetime.now().isoformat()
        save_agent(name, payload)
        return jsonify({"ok": True, "name": name}), 201

    @app.route("/api/agents/<name>", methods=["PUT"])
    def api_agent_update(name):
        data = request.get_json()
        if not data:
            return jsonify({"error": "no data"}), 400
        existing = get_agent(name)
        if existing is None:
            return jsonify({"error": "not found"}), 404
        existing.update({k: v for k, v in data.items() if k != "name"})
        save_agent(name, existing)
        return jsonify({"ok": True})

    @app.route("/api/agents", methods=["DELETE"])
    def api_agents_delete_all():
        count = 0
        for f in AGENTS_DIR.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return jsonify({"ok": True, "deleted": count})

    @app.route("/api/agents/<name>", methods=["DELETE"])
    def api_agent_delete(name):
        if delete_agent(name):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    # ─── API: Commands ───────────────────────────────────────────────

    @app.route("/api/commands")
    def api_commands():
        return jsonify(list_commands())

    @app.route("/api/commands/<name>")
    def api_command_get(name):
        c = get_command(name)
        if c is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(c)

    @app.route("/api/commands", methods=["POST"])
    def api_command_create():
        data = request.get_json()
        name = (data or {}).get("name", "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        if get_command(name):
            return jsonify({"error": "already exists"}), 409
        payload = {k: v for k, v in (data or {}).items() if k != "name"}
        if "description" not in payload:
            payload["description"] = ""
        if "prompt" not in payload:
            payload["prompt"] = ""
        if "source" not in payload:
            payload["source"] = "auto"
        payload["created"] = datetime.now().isoformat()
        save_command(name, payload)
        return jsonify({"ok": True, "name": name}), 201

    @app.route("/api/commands/<name>", methods=["PUT"])
    def api_command_update(name):
        data = request.get_json()
        if not data:
            return jsonify({"error": "no data"}), 400
        existing = get_command(name)
        if existing is None:
            return jsonify({"error": "not found"}), 404
        existing.update({k: v for k, v in data.items() if k != "name"})
        save_command(name, existing)
        return jsonify({"ok": True})

    @app.route("/api/commands/<name>", methods=["DELETE"])
    def api_command_delete(name):
        if delete_command(name):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    # ─── API: Config ─────────────────────────────────────────────────

    @app.route("/api/config")
    def api_config():
        c = load_config(force=True)
        safe = {
            "local_model": c["local"]["model"],
            "local_provider": c["local"]["provider"],
            "local_ollama_url": c["local"].get("ollama_base_url", "http://localhost:11434"),
            "local_temperature": c["local"].get("temperature", 0.7),
            "local_max_tokens": c["local"].get("max_tokens", 2048),
            "local_context_length": c["local"].get("context_length", 8192),
            "online_model": c["online"]["model"],
            "online_provider": c["online"]["provider"],
            "online_api_key": bool(c["online"]["api_key"]),
            "online_auth_method": c["online"].get("auth_method", "api_key"),
            "online_username": c["online"].get("username", ""),
            "online_password": bool(c["online"].get("password")),
            "online_base_url": c["online"].get("base_url", ""),
            "online_temperature": c["online"].get("temperature", 0.7),
            "online_max_tokens": c["online"].get("max_tokens", 4096),
            "opencode_path": c["online"].get("opencode_path", ""),
            "online_avail": bool(c["online"]["api_key"]) or c["online"]["provider"] == "opencode",
            "opencode_avail": Path(c["online"].get("opencode_path", "")).exists() if c["online"]["provider"] == "opencode" else False,
            "router_mode": c["router"]["mode"],
            "router_threshold": c["router"]["complexity_threshold"],
            "save_history": c["tracking"]["save_history"],
            "show_cost": c["tracking"].get("show_cost", True),
            "local_cost_per_1k": c["tracking"].get("local_cost_per_1k_tokens", 0.0),
            "online_cost_per_1k_input": c["tracking"].get("online_cost_per_1k_input", 0.01),
            "online_cost_per_1k_output": c["tracking"].get("online_cost_per_1k_output", 0.03),
            "keep_alive": c["local"].get("keep_alive", "5m"),
            "workspace_default_path": c.get("workspace", {}).get("default_path", "."),
            "backup_enabled": c.get("backup", {}).get("enabled", False),
            "backup_interval_hours": c.get("backup", {}).get("interval_hours", 24),
            "backup_destination": c.get("backup", {}).get("destination", ""),
            "language": c.get("language", "en"),
        }
        return jsonify(safe)

    @app.route("/api/config", methods=["PUT"])
    def api_config_update():
        data = request.get_json() or {}
        c = load_config(force=True)

        field_map = {
            "local_model": ("local", "model"),
            "local_ollama_url": ("local", "ollama_base_url"),
            "local_temperature": ("local", "temperature"),
            "local_max_tokens": ("local", "max_tokens"),
            "local_context_length": ("local", "context_length"),
            "online_model": ("online", "model"),
            "online_provider": ("online", "provider"),
            "online_auth_method": ("online", "auth_method"),
            "online_username": ("online", "username"),
            "online_base_url": ("online", "base_url"),
            "online_temperature": ("online", "temperature"),
            "online_max_tokens": ("online", "max_tokens"),
            "opencode_path": ("online", "opencode_path"),
            "router_mode": ("router", "mode"),
            "router_threshold": ("router", "complexity_threshold"),
            "save_history": ("tracking", "save_history"),
            "show_cost": ("tracking", "show_cost"),
            "local_cost_per_1k": ("tracking", "local_cost_per_1k_tokens"),
            "online_cost_per_1k_input": ("tracking", "online_cost_per_1k_input"),
            "online_cost_per_1k_output": ("tracking", "online_cost_per_1k_output"),
            "keep_alive": ("local", "keep_alive"),
            "workspace_default_path": ("workspace", "default_path"),
            "backup_enabled": ("backup", "enabled"),
            "backup_interval_hours": ("backup", "interval_hours"),
            "backup_destination": ("backup", "destination"),
            "language": ("language", None),
        }

        for field, (section, key) in field_map.items():
            if field in data:
                val = data[field]
                if field in ("router_threshold", "local_max_tokens", "local_context_length", "online_max_tokens"):
                    val = int(val)
                elif field in ("local_temperature", "online_temperature"):
                    val = float(val)
                elif field in ("local_cost_per_1k", "online_cost_per_1k_input", "online_cost_per_1k_output"):
                    val = float(val)
                elif field in ("save_history", "show_cost", "backup_enabled"):
                    val = bool(val)
                elif field in ("backup_interval_hours",):
                    val = float(val)
                if key is None:
                    c[section] = val
                else:
                    c[section][key] = val

        # Handle API key (masked in GET)
        if "api_key" in data and data["api_key"] and data["api_key"] != "••••••••":
            c["online"]["api_key"] = data["api_key"]
        # Handle password (masked in GET)
        if "online_password" in data and data["online_password"] and data["online_password"] != "••••••••":
            c["online"]["password"] = data["online_password"]

        # If online provider changed, re-init
        if "online_provider" in data or "online_model" in data:
            app.config["ONLINE"] = get_online_provider(c)

        save_config(c)
        app.config["ROUTER"] = SmartRouter(c)
        return jsonify({"ok": True})

    # ─── API: Models (locali + online) ───────────────────────────────

    @app.route("/api/models/list")
    def api_models_list():
        """Restituisce modelli disponibili (locale Ollama + online OpenRouter)."""
        cfg = load_config()
        local_models = []
        try:
            import requests
            r = requests.get(f"{cfg['local']['ollama_base_url']}/api/tags", timeout=5)
            if r.status_code == 200:
                for m in (r.json().get("models") or []):
                    name = m.get("name", m.get("model", "?"))
                    local_models.append({"id": name, "name": name})
        except Exception:
            pass

        online_models = []
        if cfg["online"].get("api_key"):
            try:
                import requests
                headers = {"Authorization": f"Bearer {cfg['online']['api_key']}"}
                r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
                if r.status_code == 200:
                    for m in (r.json().get("data") or []):
                        mid = m.get("id", "")
                        if mid:
                            online_models.append({"id": mid, "name": mid})
            except Exception:
                pass

        # fallback se non arriva nulla
        if not local_models:
            local_models = [{"id": "qwen3.5:4b", "name": "qwen3.5:4b"},
                            {"id": "llama3.2:3b", "name": "llama3.2:3b"},
                            {"id": "mistral:7b", "name": "mistral:7b"}]
        if not online_models:
            online_models = [{"id": "openai/gpt-4o-mini", "name": "openai/gpt-4o-mini"},
                             {"id": "openai/gpt-4o", "name": "openai/gpt-4o"},
                             {"id": "anthropic/claude-3.5-sonnet", "name": "anthropic/claude-3.5-sonnet"},
                             {"id": "google/gemini-2.0-flash-001", "name": "google/gemini-2.0-flash-001"},
                             {"id": "mistralai/mistral-7b-instruct", "name": "mistralai/mistral-7b-instruct"}]

        return jsonify({"local": local_models, "online": online_models})

    @app.route("/api/models/set-active", methods=["POST"])
    def api_models_set_active():
        """Cambia il modello attivo locale o online."""
        data = request.get_json() or {}
        c = load_config(force=True)
        kind = data.get("kind")  # "local" o "online"
        model_id = data.get("model_id", "").strip()
        if kind not in ("local", "online") or not model_id:
            return jsonify({"ok": False, "message": "kind (local/online) e model_id richiesti"}), 400
        c[kind]["model"] = model_id
        save_config(c)
        app.config["ROUTER"] = SmartRouter(c)
        if kind == "online":
            app.config["ONLINE"] = get_online_provider(c)
        return jsonify({"ok": True, "message": f"Modello {kind}: {model_id}"})

    @app.route("/api/models/pull", methods=["POST"])
    def api_models_pull():
        """Esegue ollama pull <model> e restituisce output in streaming."""
        data = request.get_json() or {}
        model = data.get("model", "").strip()
        if not model:
            return jsonify({"ok": False, "message": "model richiesto"}), 400
        def generate():
            yield "data: " + json.dumps({"type": "status", "message": f"Scarico {model}..."}) + "\n\n"
            try:
                proc = subprocess.Popen(
                    ["ollama", "pull", model],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        yield "data: " + json.dumps({"type": "output", "message": line}) + "\n\n"
                proc.wait()
                if proc.returncode == 0:
                    yield "data: " + json.dumps({"type": "done", "message": f"{model} scaricato con successo"}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"type": "error", "message": f"Errore durante il download (codice {proc.returncode})"}) + "\n\n"
            except FileNotFoundError:
                yield "data: " + json.dumps({"type": "error", "message": "Ollama non trovato. Installalo con: curl -fsSL https://ollama.com/install.sh | sh"}) + "\n\n"
            except Exception as e:
                yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"
        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/models/delete", methods=["POST"])
    def api_models_delete():
        """Elimina un modello Ollama."""
        data = request.get_json() or {}
        model = data.get("model", "").strip()
        if not model:
            return jsonify({"ok": False, "message": "model richiesto"}), 400
        try:
            proc = subprocess.run(["ollama", "rm", model], capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                return jsonify({"ok": True, "message": f"Modello {model} eliminato"})
            else:
                return jsonify({"ok": False, "message": proc.stderr.strip() or f"Errore (codice {proc.returncode})"})
        except FileNotFoundError:
            return jsonify({"ok": False, "message": "Ollama non trovato."})
        except Exception as e:
            return jsonify({"ok": False, "message": str(e)})

    @app.route("/api/config/test/ollama", methods=["POST"])
    def api_config_test_ollama():
        data = request.get_json() or {}
        url = data.get("url", "http://localhost:11434")
        try:
            import requests
            r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name", "?") for m in (r.json().get("models") or [])]
                return jsonify({"ok": True, "models": models, "message": f"Ollama attivo. {len(models)} modelli trovati."})
            return jsonify({"ok": False, "message": f"Status: {r.status_code}"})
        except Exception as e:
            return jsonify({"ok": False, "message": f"Connessione fallita: {e}"})

    @app.route("/api/config/test/online", methods=["POST"])
    def api_config_test_online():
        data = request.get_json() or {}
        provider = data.get("provider", "opencode")
        api_key = data.get("api_key", "")
        if provider == "opencode":
            path = data.get("opencode_path", "")
            if path and Path(path).exists():
                return jsonify({"ok": True, "message": "Opencode trovato."})
            return jsonify({"ok": False, "message": f"Opencode non trovato: {path}"})
        if not api_key:
            return jsonify({"ok": False, "message": "API Key richiesta."})
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"}
            r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            if r.status_code == 200:
                models = [m.get("id", "?") for m in (r.json().get("data") or [])]
                return jsonify({"ok": True, "models": models, "message": f"Connessione OK. {len(models)} modelli disponibili."})
            return jsonify({"ok": False, "message": f"API error: {r.status_code}"})
        except Exception as e:
            return jsonify({"ok": False, "message": f"Connessione fallita: {e}"})

    @app.route("/api/config/reset", methods=["POST"])
    def api_config_reset():
        from ..config import DEFAULT_CONFIG
        save_config(DEFAULT_CONFIG)
        app.config["ROUTER"] = SmartRouter(DEFAULT_CONFIG)
        app.config["ONLINE"] = get_online_provider(DEFAULT_CONFIG)
        return jsonify({"ok": True})

    # ─── API: Knowledge ──────────────────────────────────────────────

    @app.route("/knowledge")
    def knowledge_page():
        return render_template("knowledge.html", active_page="knowledge")

    @app.route("/notes")
    def notes_page():
        return render_template("notes.html", active_page="notes")

    @app.route("/notes/<slug>")
    def note_page(slug):
        ns = get_note_store()
        note = ns.get(slug)
        if note is None:
            return render_template("notes.html", active_page="notes", error="Nota non trovata")
        return render_template("note_view.html", note=note, active_page="notes")

    @app.route("/projects")
    def projects_page():
        return render_template("projects.html", active_page="projects")

    @app.route("/prompts")
    def prompts_page():
        return render_template("prompts.html", active_page="prompts")

    @app.route("/api/knowledge")
    def api_knowledge():
        kb = get_knowledge_base()
        return jsonify(kb.status())

    _kb_import_jobs = {}

    @app.route("/api/knowledge/from-dir", methods=["POST"])
    def api_knowledge_from_dir():
        data = request.get_json() or {}
        path = data.get("path", "").strip()
        if not path:
            return jsonify({"error": "path required"}), 400
        pattern = data.get("pattern", "*")
        recursive = data.get("recursive", True)
        import uuid
        job_id = uuid.uuid4().hex[:8]
        _kb_import_jobs[job_id] = {"status": "running", "progress": 0, "message": "Avvio..."}
        kb = get_knowledge_base()
        import threading as _thr
        def _run(jid):
            try:
                result = kb.add_directory(path, pattern=pattern, recursive=recursive, progress_dict=_kb_import_jobs[jid])
                if "error" in result:
                    _kb_import_jobs[jid] = {"status": "error", "message": result["error"]}
                else:
                    files = result.get("files_processed", 0)
                    chunks = result.get("chunks_added", 0)
                    errs = result.get("errors", [])
                    msg = f"✅ {files} file, {chunks} chunk"
                    if errs:
                        msg += f", {len(errs)} errori"
                    kb._save_index()
                    _kb_import_jobs[jid] = {"status": "done", "progress": 100, "message": msg}
            except Exception as e:
                _kb_import_jobs[jid] = {"status": "error", "message": str(e)}
        _thr.Thread(target=_run, args=(job_id,), daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id, "message": f"Indicizzazione avviata: {path}"})

    @app.route("/api/knowledge/import-status/<job_id>")
    def api_knowledge_import_status(job_id):
        job = _kb_import_jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)

    @app.route("/api/knowledge/preview-dir", methods=["POST"])
    def api_knowledge_preview_dir():
        data = request.get_json() or {}
        path = data.get("path", "").strip()
        if not path:
            return jsonify({"error": "path required"}), 400
        pattern = data.get("pattern", "*")
        recursive = data.get("recursive", True)
        kb = get_knowledge_base()
        result = kb.preview_directory(path, pattern=pattern, recursive=recursive)
        return jsonify(result)

    @app.route("/api/knowledge/from-file", methods=["POST"])
    def api_knowledge_from_file():
        if "file" not in request.files:
            return jsonify({"error": "file required"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "file required"}), 400
        import tempfile
        suffix = Path(f.filename).suffix or ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        try:
            kb = get_knowledge_base()
            result = kb.add_file(tmp_path)
            if "error" in result:
                return jsonify(result), 400
            if result.get("chunks_added", 0) > 0:
                kb._save_index()
            result["filename"] = f.filename
            return jsonify(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @app.route("/api/knowledge/from-url", methods=["POST"])
    def api_knowledge_from_url():
        data = request.get_json() or {}
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"error": "url required"}), 400
        kb = get_knowledge_base()
        result = kb.add_url(url)
        if "error" in result:
            return jsonify(result), 400
        if result.get("chunks_added", 0) > 0:
            kb._save_index()
        return jsonify(result)

    @app.route("/api/knowledge/from-file-path", methods=["POST"])
    def api_knowledge_from_file_path():
        """Index a single file by its filesystem path."""
        data = request.get_json() or {}
        path = data.get("path", "").strip()
        if not path:
            return jsonify({"error": "path required"}), 400
        fp = Path(path).expanduser()
        if not fp.is_file():
            return jsonify({"error": f"File non trovato: {path}"}), 400
        kb = get_knowledge_base()
        result = kb.add_file(str(fp))
        if "error" in result:
            return jsonify(result), 400
        if result.get("chunks_added", 0) > 0:
            kb._save_index()
        result["filename"] = fp.name
        return jsonify(result)

    @app.route("/api/knowledge/query", methods=["POST"])
    def api_knowledge_query():
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400
        kb = get_knowledge_base()
        results = kb.query(query)
        return jsonify({"results": results})

    @app.route("/api/knowledge/clear", methods=["POST"])
    def api_knowledge_clear():
        from ..knowledge import reset_knowledge_base
        reset_knowledge_base()
        return jsonify({"ok": True})

    @app.route("/api/knowledge/graph")
    def api_knowledge_graph():
        kb = get_knowledge_base()
        return jsonify(build_knowledge_graph(kb))

    @app.route("/api/knowledge/source/delete", methods=["POST"])
    def api_knowledge_source_delete():
        data = request.get_json() or {}
        source = data.get("source", "").strip()
        if not source:
            return jsonify({"error": "source required"}), 400
        kb = get_knowledge_base()
        ok = kb.remove_source(source)
        return jsonify({"ok": ok})

    @app.route("/api/knowledge/notebook", methods=["POST"])
    def api_knowledge_notebook():
        """KB-grounded Q&A: search KB, send context + query to LLM, return answer + sources."""
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "")
        if not query:
            return jsonify({"error": "Query vuota"}), 400

        kb = get_knowledge_base()
        results = kb.query(query, top_k=8)
        if not results:
            return jsonify({
                "response": "Non ho trovato informazioni nella knowledge base per questa domanda.",
                "sources": [],
            })

        context_lines = []
        for r in results[:6]:
            label = r.get("source", "?") or "?"
            cid = r.get("chunk_id", 0)
            score = r.get("score", 0)
            text = r.get("text", "")[:500]
            context_lines.append(f"[Fonte: {label} | chunk {cid} | rilevanza {score:.2f}]\n{text}")
        context = "\n\n---\n\n".join(context_lines)

        system = (
            "Sei un assistente che risponde basandosi ESCLUSIVAMENTE sul contesto fornito qui sotto. "
            "Ogni affermazione deve essere supportata dal contesto. "
            "Cita sempre la fonte tra [Fonte: ...]. "
            "Se il contesto non contiene informazioni sufficienti, dillo chiaramente."
        )

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {query}"})

        t_local = app.config["LOCAL"]
        t_online = app.config["ONLINE"]
        result = None
        try:
            result = t_local.generate_chat(messages)
        except Exception:
            try:
                result = t_online.generate_chat(messages)
            except Exception as e:
                return jsonify({"error": str(e), "response": None, "sources": results[:5]}), 500

        return jsonify({
            "response": result.text,
            "sources": results[:5],
            "source": result.source,
            "model": result.model,
            "tokens": result.tokens_total,
            "time_s": round(result.time_s, 1),
        })

    @app.route("/api/knowledge/briefing", methods=["POST"])
    def api_knowledge_briefing():
        """Generate a structured briefing (summary, FAQ, glossary) from KB sources."""
        data = request.get_json() or {}
        source_filter = data.get("sources")  # optional list
        topic = data.get("topic", "knowledge base")

        kb = get_knowledge_base()
        all_docs = kb.index.documents
        if source_filter:
            src_set = set(source_filter)
            chunks = [d for d in all_docs if d.get("source") in src_set]
        else:
            chunks = all_docs

        if not chunks:
            return jsonify({"error": "Nessun documento disponibile"}), 400

        # Build context from top chunks
        texts = [d.get("text", "")[:400] for d in chunks[:200]]
        full_text = "\n\n".join(texts)[:6000]

        system = (
            "Genera un briefing strutturato in italiano basato esclusivamente sul testo fornito. "
            "Il briefing deve includere:\n\n"
            "## Riepilogo\nUn paragrafo che riassume i punti principali.\n\n"
            "## Argomenti chiave\nElenco puntato dei temi principali.\n\n"
            "## FAQ\n3-5 domande frequenti con risposte basate sul testo.\n\n"
            "## Glossario\n termini importanti con definizioni.\n\n"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Testo di riferimento (argomento: {topic}):\n\n{full_text}"},
        ]

        t_local = app.config["LOCAL"]
        result = None
        try:
            result = t_local.generate_chat(messages)
        except Exception:
            t_online = app.config["ONLINE"]
            try:
                result = t_online.generate_chat(messages)
            except Exception:
                return jsonify({"error": "Nessun provider disponibile"}), 500

        return jsonify({
            "briefing": result.text,
            "source": result.source,
            "model": result.model,
            "tokens": result.tokens_total,
            "time_s": round(result.time_s, 1),
        })

    # ─── API: LLM Wiki ─────────────────────────────────────────────

    @app.route("/api/wiki")
    def api_wiki_list():
        return jsonify(list_wiki_pages())

    @app.route("/api/wiki/<slug>")
    def api_wiki_get(slug):
        page = get_wiki_page(slug)
        if page is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(page)

    @app.route("/api/wiki/suggest")
    def api_wiki_suggest():
        kb = get_knowledge_base()
        return jsonify(suggest_wiki_topics(kb))

    @app.route("/api/wiki/generate", methods=["POST"])
    def api_wiki_generate():
        data = request.get_json() or {}
        topic = data.get("topic", "").strip()
        if not topic:
            return jsonify({"error": "topic required"}), 400
        kb = get_knowledge_base()
        page = generate_wiki_page(topic, kb)
        if page is None:
            return jsonify({"error": "Impossibile generare pagina"}), 400
        return jsonify(page), 201

    @app.route("/api/wiki/<slug>", methods=["DELETE"])
    def api_wiki_delete(slug):
        if delete_wiki_page(slug):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    # ─── API: Sessions ─────────────────────────────────────────────

    @app.route("/api/sessions")
    def api_sessions():
        return jsonify(list_sessions())

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def api_session_get(session_id):
        store = _get_conv_store()
        msgs = store.messages(session_id)
        return jsonify({"session_id": session_id, "messages": msgs})

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def api_session_delete(session_id):
        del_session(session_id)
        return jsonify({"ok": True})

    # ─── API: Chat RAG integration ────────────────────────────────────

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json() or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt vuoto"}), 400

        session_id = data.get("session_id", "")
        agent_name = data.get("agent")
        force_source = data.get("source")

        # RAG: arricchisce prompt
        ctx = build_rag_context(prompt)
        enriched_prompt = prompt
        rag_used = False
        if ctx:
            enriched_prompt = f"{prompt}\n\n{ctx}"
            rag_used = True

        # Carica/crea conversazione (persistente su disco)
        store = _get_conv_store()
        sid, messages = store.get(session_id)
        system = None

        # Se agente selezionato, applica le sue impostazioni
        agent_def = None
        if agent_name:
            agent_def = get_agent(agent_name)
            if agent_def:
                system = agent_def.get("instructions", "")
                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": system})

        t_router = app.config["ROUTER"]
        t_tracker = app.config["TRACKER"]
        t_local = app.config["LOCAL"]
        t_online = app.config["ONLINE"]
        t_online_avail = app.config["ONLINE_AVAIL"]

        # Pre-check disponibilità provider
        def _is_local_avail():
            try:
                import requests
                cfg = load_config()
                r = requests.get(f"{cfg['local']['ollama_base_url']}/api/tags", timeout=3)
                return r.status_code == 200
            except Exception:
                return False

        t_local_avail = _is_local_avail()

        # Routing: priorità a sorgente forzata, poi agente, poi router smart
        if force_source == "online" and t_online_avail:
            decision = "online"
            score = 99
        elif force_source == "local" and t_local_avail:
            decision = "local"
            score = -99
        elif agent_def:
            agent_mode = agent_def.get("mode", "auto")
            if agent_mode == "local" and t_local_avail:
                decision = "local"
                score = -99
            elif agent_mode == "online" and t_online_avail:
                decision = "online"
                score = 99
            else:
                decision, score = t_router.decide(prompt)
                if decision == "online" and not t_online_avail:
                    decision = "local"
                if decision == "local" and not t_local_avail and t_online_avail:
                    decision = "online"
        else:
            decision, score = t_router.decide(prompt)
            if decision == "online" and not t_online_avail:
                decision = "local"
            if decision == "local" and not t_local_avail and t_online_avail:
                decision = "online"

        provider = t_online if decision == "online" else t_local

        # Per opencode: includiamo il prompt corrente + history ridotta
        if decision == "online" and type(provider).__name__ == "OpencodeProvider":
            chat_messages_raw = [m for m in store.messages(sid) if m["role"] in ("user", "assistant")]
            chat_messages = (chat_messages_raw[-4:] if len(chat_messages_raw) > 4 else chat_messages_raw)
            chat_messages.append({"role": "user", "content": enriched_prompt})
        else:
            _append_conv(sid, "user", enriched_prompt)
            chat_messages = _trim_conv(store.messages(sid))

        t0 = time.time()
        fallback_used = False

        def _try_provider(prov, prov_name):
            try:
                res = prov.generate_chat(chat_messages)
                if res.text.startswith("[ERRORE"):
                    return None, res
                return res, None
            except Exception:
                return None, None

        result = None
        err_text = None
        result, err_text = _try_provider(provider, decision)
        if (result is None or result.text.startswith("[ERRORE")) and decision == "local" and t_online_avail:
            log.info(f"Fallback: da locale a online ({err_text or result.text[:50]})")
            provider = t_online
            decision = "online"
            fallback_used = True
            result, err_text = _try_provider(provider, "online")
        elif (result is None or result.text.startswith("[ERRORE")) and decision == "online" and t_local_avail:
            log.info(f"Fallback: da online a locale ({err_text or result.text[:50]})")
            provider = t_local
            decision = "local"
            fallback_used = True
            result, err_text = _try_provider(provider, "local")

        if result is None:
            return jsonify({"error": err_text or "Provider non disponibile"}), 503

        if not result.text.startswith("[ERRORE"):
            _append_conv(sid, "assistant", result.text)

        t_tracker.record(prompt, result, agent=agent_name)
        store.flush(sid)

        resp = {
            "response": result.text,
            "source": result.source,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "tokens_total": result.tokens_total,
            "time_s": round(result.time_s, 2),
            "model": result.model,
            "cached": result.cached,
            "complexity_score": score,
            "agent": agent_name,
            "agent_params": {
                "temperature": agent_def.get("temperature") if agent_def else None,
                "max_tokens": agent_def.get("max_tokens") if agent_def else None,
                "top_p": agent_def.get("top_p") if agent_def else None,
            } if agent_def else None,
            "session_id": sid,
            "rag_used": rag_used,
            "history_len": len(store.messages(sid)),
            "fallback_used": fallback_used,
        }
        if result.events:
            resp["opencode_events"] = result.events

        # Background: auto-genera pagine wiki da risposte significative
        if result.text and len(result.text) > 600 and not result.text.startswith("[ERRORE"):
            threading.Thread(
                target=auto_generate_wiki,
                args=(result.text,),
                daemon=True,
            ).start()

        return jsonify(resp)

    @app.route("/api/chat/stream", methods=["POST"])
    def api_chat_stream():
        """SSE streaming endpoint: restituisce token uno per uno."""
        data = request.get_json() or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt vuoto"}), 400

        session_id = data.get("session_id", "")
        agent_name = data.get("agent")
        force_source = data.get("source")

        ctx = build_rag_context(prompt)
        enriched_prompt = prompt
        if ctx:
            enriched_prompt = f"{prompt}\n\n{ctx}"

        store = _get_conv_store()
        sid, messages = store.get(session_id)
        agent_def = None
        if agent_name:
            agent_def = get_agent(agent_name)
            if agent_def:
                system = agent_def.get("instructions", "")
                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": system})

        t_router = app.config["ROUTER"]
        t_local = app.config["LOCAL"]
        t_online = app.config["ONLINE"]
        t_online_avail = app.config["ONLINE_AVAIL"]

        # Pre-check disponibilità provider
        def _is_local_avail():
            try:
                import requests
                cfg = load_config()
                r = requests.get(f"{cfg['local']['ollama_base_url']}/api/tags", timeout=3)
                return r.status_code == 200
            except Exception:
                return False

        t_local_avail = _is_local_avail()

        score = 0

        if force_source == "online" and t_online_avail:
            decision = "online"
            score = 99
        elif force_source == "local" and t_local_avail:
            decision = "local"
            score = -99
        elif agent_def:
            agent_mode = agent_def.get("mode", "auto")
            if agent_mode == "local" and t_local_avail:
                decision = "local"
                score = -99
            elif agent_mode == "online" and t_online_avail:
                decision = "online"
                score = 99
            else:
                decision, score = t_router.decide(prompt)
                if decision == "online" and not t_online_avail:
                    decision = "local"
                if decision == "local" and not t_local_avail and t_online_avail:
                    decision = "online"
        else:
            decision, score = t_router.decide(prompt)
            if decision == "online" and not t_online_avail:
                decision = "local"
            if decision == "local" and not t_local_avail and t_online_avail:
                decision = "online"

        provider = t_online if decision == "online" else t_local
        fallback_used = False

        if decision == "online" and type(provider).__name__ == "OpencodeProvider":
            chat_messages_raw = [m for m in store.messages(sid) if m["role"] in ("user", "assistant")]
            chat_messages = (chat_messages_raw[-4:] if len(chat_messages_raw) > 4 else chat_messages_raw)
            chat_messages.append({"role": "user", "content": enriched_prompt})
        else:
            _append_conv(sid, "user", enriched_prompt)
            chat_messages = _trim_conv(store.messages(sid))

        def generate():
            from flask import stream_with_context

            # Se un agente è selezionato, usa l'agent executor con tool calling
            use_agent_executor = agent_def is not None and agent_def.get("permissions", {}).get("bash", False)

            # Se nessun provider è disponibile, errore immediato
            if not t_local_avail and not t_online_avail:
                yield f"event: error\ndata: {json.dumps('Nessun provider disponibile (locale e online non raggiungibili)')}\n\n"
                return

            yield f"event: meta\ndata: {json.dumps({'source': decision, 'agent': agent_name or '', 'fallback': fallback_used})}\n\n"
            full_text_parts = []
            try:
                if use_agent_executor:
                    # Agent executor loop
                    executor = AgentExecutor(provider, {})
                    if system:
                        messages = build_agent_messages(prompt, history=None, system_prompt=system, agent_config=agent_def)
                    else:
                        messages = build_agent_messages(prompt, history=None, agent_config=agent_def)

                    for event in executor.execute(prompt, history=None, system_prompt=system or None, agent_config=agent_def):
                        if event["type"] == "token":
                            token = event.get("text", "")
                            if token:
                                full_text_parts.append(token)
                                yield f"event: token\ndata: {json.dumps(token)}\n\n"
                        elif event["type"] == "tool_call":
                            call = event.get("call", {})
                            yield f"event: tool_call\ndata: {json.dumps(call)}\n\n"
                        elif event["type"] == "tool_result":
                            yield f"event: tool_result\ndata: {json.dumps(event.get('result', {}))}\n\n"
                        elif event["type"] == "done":
                            result_text = event.get("text", "")
                            event_source = event.get("source", decision)
                            if result_text:
                                _append_conv(sid, "assistant", result_text)
                            app.config["TRACKER"].record(prompt, FakeResult(
                                text=result_text,
                                source=event_source,
                                tokens_total=event.get("tokens", 0),
                                time_s=event.get("time_s", 0),
                                model=event.get("model", ""),
                                cached=event.get("cached", False),
                            ), agent=agent_name)
                            store.flush(sid)

                            final = {
                                "text": result_text,
                                "source": event_source,
                                "tokens_total": event.get("tokens", 0),
                                "time_s": round(event.get("time_s", 0), 2),
                                "model": event.get("model", ""),
                                "cached": event.get("cached", False),
                                "agent": agent_name,
                                "session_id": sid,
                                "score": score,
                                "history_len": len(store.messages(sid)),
                            }
                            yield f"event: done\ndata: {json.dumps(final)}\n\n"
                            return
                        elif event["type"] == "info":
                            yield f"event: info\ndata: {json.dumps(event)}\n\n"
                        elif event["type"] == "error":
                            yield f"event: error\ndata: {json.dumps(event.get('message', 'Errore'))}\n\n"
                else:
                    # Normal streaming (no agent tools)
                    for chunk in provider.generate_chat_stream(chat_messages):
                        if chunk.get("done"):
                            result = chunk["result"]
                            if not result.text.startswith("[ERRORE"):
                                _append_conv(sid, "assistant", result.text)
                            app.config["TRACKER"].record(prompt, result, agent=agent_name)
                            store.flush(sid)

                            # Background: auto-genera wiki
                            if result.text and len(result.text) > 600 and not result.text.startswith("[ERRORE"):
                                threading.Thread(
                                    target=auto_generate_wiki,
                                    args=(result.text,),
                                    daemon=True,
                                ).start()

                            final = {
                                "text": result.text,
                                "source": result.source,
                                "tokens_in": result.tokens_in,
                                "tokens_out": result.tokens_out,
                                "tokens_total": result.tokens_total,
                                "time_s": round(result.time_s, 2),
                                "model": result.model,
                                "cached": result.cached,
                                "agent": agent_name,
                                "session_id": sid,
                                "score": score,
                                "history_len": len(store.messages(sid)),
                                "fallback_used": fallback_used,
                            }
                            if result.events:
                                final["opencode_events"] = result.events
                            yield f"event: done\ndata: {json.dumps(final)}\n\n"
                            return
                        token = chunk.get("token", "")
                        if token:
                            full_text_parts.append(token)
                            yield f"event: token\ndata: {json.dumps(token)}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

        from flask import stream_with_context
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ─── API: System ─────────────────────────────────────────────────

    @app.route("/api/system/info")
    def api_system():
        cfg = load_config()
        mgr = get_resource_manager()
        snap = mgr.snapshot()
        is_opencode = cfg["online"]["provider"] == "opencode"
        online_ok = bool(cfg["online"]["api_key"]) or is_opencode
        tracker = app.config["TRACKER"]
        s = tracker.summary_dict()

        bf = Path.home() / ".config" / "hybrid-coder" / "build_counter"
        build_number = int(bf.read_text().strip()) if bf.exists() else 0

        return jsonify({
            "ollama_running": _check_ollama(cfg),
            "online_configured": online_ok,
            "build_number": build_number,
            "local_model": cfg["local"]["model"],
            "online_model": cfg["online"]["model"],
            "online_provider": cfg["online"]["provider"],
            "opencode_avail": Path(cfg["online"].get("opencode_path", "")).exists() if is_opencode else False,
            "cache_entries": mgr.response_cache.size,
            "cache_disk_mb": snap["limits"]["cache_disk_mb"],
            "process_memory_mb": snap["process_memory_mb"],
            "cpu_percent": snap["cpu_percent"],
            "uptime_s": snap["uptime_s"],
            "total_calls": s["total"]["calls"] if "total" in s else 0,
            "total_tokens": s["total"]["tokens_total"] if "total" in s else 0,
            "total_time_s": s["total"]["time_s"] if "total" in s else 0,
            "local_calls": s["local"]["calls"] if "local" in s else 0,
            "online_calls": s["online"]["calls"] if "online" in s else 0,
            "history_count": len(getattr(tracker, "history", [])),
        })

    @app.route("/api/system/resources")
    def api_system_resources():
        mgr = get_resource_manager()
        return jsonify(mgr.snapshot())

    @app.route("/api/system/cleanup", methods=["POST"])
    def api_system_cleanup():
        mgr = get_resource_manager()
        mgr.cleanup()
        import gc
        gc.collect()
        return jsonify({"ok": True})

    # ─── API: Disk ───────────────────────────────────────────────────

    @app.route("/api/system/disk")
    def api_system_disk():
        """Mostra occupazione disco delle directory di AI+.
        Se query param ?path=..., elenca file e directory."""
        browse = request.args.get("path")
        if browse:
            bp = Path(browse).expanduser().resolve()
            if not bp.exists():
                return jsonify({"error": "percorso non trovato"}), 404
            if bp.is_file():
                st = bp.stat()
                return jsonify({"path": str(bp), "type": "file", "size": st.st_size, "mtime": st.st_mtime})
            dirs = sorted([str(p) for p in bp.iterdir() if p.is_dir()])
            files = sorted([str(p) for p in bp.iterdir() if p.is_file()])
            file_info = []
            for f in (Path(p) for p in files):
                try:
                    st = f.stat()
                    file_info.append({"path": str(f), "size": st.st_size, "mtime": st.st_mtime})
                except Exception:
                    file_info.append({"path": str(f), "size": 0, "mtime": 0})
            return jsonify({
                "path": str(bp),
                "type": "dir",
                "dirs": dirs[:200],
                "files": [fi["path"] for fi in file_info[:200]],
                "file_info": file_info[:200],
            })

        import shutil
        info = {}
        for name, d in [("config", DATA_DIR), ("cache", CACHE_DIR), ("notes", NOTES_DIR),
                        ("knowledge", Path(DATA_DIR) / "knowledge")]:
            d = Path(d)
            if d.exists():
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                info[name] = {
                    "path": str(d),
                    "size_bytes": total,
                    "size_mb": round(total / (1024 * 1024), 2),
                    "files": sum(1 for _ in d.rglob("*") if _.is_file()),
                }
            else:
                info[name] = {"path": str(d), "size_bytes": 0, "size_mb": 0, "files": 0}
        home = Path.home()
        du = shutil.disk_usage(home)
        info["system"] = {
            "total_gb": round(du.total / (1024**3), 1),
            "used_gb": round(du.used / (1024**3), 1),
            "free_gb": round(du.free / (1024**3), 1),
            "percent": round(du.used / du.total * 100, 1),
        }
        return jsonify(info)

    # ─── API: Notes ──────────────────────────────────────────────────

    @app.route("/api/notes")
    def api_notes_list():
        ns = get_note_store()
        return jsonify(ns.list_all())

    @app.route("/api/notes/<slug>")
    def api_note_get(slug):
        ns = get_note_store()
        note = ns.get(slug)
        if note is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(note)

    @app.route("/api/notes", methods=["POST"])
    def api_note_create():
        data = request.get_json() or {}
        title = (data.get("title") or "").strip()
        if not title:
            return jsonify({"error": "title required"}), 400
        ns = get_note_store()
        note = ns.create(title, body=data.get("body", ""), tags=data.get("tags"))
        return jsonify(note), 201

    @app.route("/api/notes/<slug>", methods=["PUT"])
    def api_note_update(slug):
        data = request.get_json() or {}
        ns = get_note_store()
        note = ns.update(slug,
                         body=data.get("body"),
                         tags=data.get("tags"),
                         title=data.get("title"))
        if note is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(note)

    @app.route("/api/notes/<slug>", methods=["DELETE"])
    def api_note_delete(slug):
        ns = get_note_store()
        if ns.delete(slug):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    @app.route("/api/notes/graph")
    def api_notes_graph():
        ns = get_note_store()
        return jsonify(ns.graph())

    @app.route("/api/notes/search")
    def api_notes_search():
        q = request.args.get("q", "").strip()
        tag = request.args.get("tag", "").strip()
        ns = get_note_store()
        if tag:
            results = ns.search_by_tag(tag)
        elif q:
            results = ns.search(q)
        else:
            results = ns.list_all()
        return jsonify(results)

    # ─── API: Commands (lista comandi) ──────────────────────────────

    @app.route("/api/commands/list")
    def api_commands_list():
        """Restituisce tutti i comandi disponibili (CLI + slash + web)."""
        return jsonify({
            "cli": [
                {"name": "chat", "desc": "Avvia chat interattiva o invia prompt singolo", "usage": "ai-plus chat [prompt]"},
                {"name": "generate", "desc": "Invia prompt, ricevi JSON strutturato (per opencode)", "usage": "ai-plus generate [--system] <prompt>"},
                {"name": "serve", "desc": "Avvia la web UI", "usage": "ai-plus serve [--port]"},
                {"name": "learn from-dir", "desc": "Indicizza file per RAG", "usage": "ai-plus learn from-dir <path>"},
                {"name": "learn from-url", "desc": "Indicizza pagina web per RAG", "usage": "ai-plus learn from-url <url>"},
                {"name": "learn status", "desc": "Stato knowledge base RAG", "usage": "ai-plus learn status"},
                {"name": "resources", "desc": "Mostra CPU, RAM, cache, pool", "usage": "ai-plus resources"},
                {"name": "note list", "desc": "Elenco note", "usage": "ai-plus note list"},
                {"name": "note create", "desc": "Crea nuova nota", "usage": "ai-plus note create <title>"},
                {"name": "note graph", "desc": "Mostra grafo connessioni note", "usage": "ai-plus note graph"},
                {"name": "config", "desc": "Mostra configurazione", "usage": "ai-plus config"},
                {"name": "set", "desc": "Imposta chiave di configurazione", "usage": "ai-plus set <key> <value>"},
                {"name": "project create", "desc": "Crea nuovo progetto da template", "usage": "ai-plus project create <nome> --template python"},
                {"name": "project list", "desc": "Elenca progetti", "usage": "ai-plus project list"},
                {"name": "project run", "desc": "Esegue progetto e mostra output", "usage": "ai-plus project run <nome>"},
                {"name": "project test", "desc": "Esegue test del progetto", "usage": "ai-plus project test <nome>"},
                {"name": "project preview", "desc": "Mostra codice + output split-view", "usage": "ai-plus project preview <nome>"},
                {"name": "project delete", "desc": "Elimina progetto", "usage": "ai-plus project delete <nome>"},
                {"name": "stats", "desc": "Statistiche sessione (in chat)", "usage": "/stats o ai-plus agent"},
                {"name": "agent", "desc": "Output JSON stato per opencode", "usage": "ai-plus agent"},
            ],
            "slash": [
                {"name": "/mode", "desc": "Cambia modalità router (auto|local|online)"},
                {"name": "/model", "desc": "Cambia modello locale"},
                {"name": "/cache", "desc": "Pulisce cache risposte"},
                {"name": "/reset", "desc": "Resetta statistiche sessione"},
            ],
        })

    # ─── API: Project ──────────────────────────────────────────────

    @app.route("/api/projects")
    def api_projects_list():
        return jsonify(list_projects())

    @app.route("/api/projects", methods=["POST"])
    def api_project_create():
        data = request.get_json() or {}
        name = data.get("name", "")
        if not name:
            return jsonify({"error": "name richiesto"}), 400
        template = data.get("template", "python")
        path = data.get("path")
        try:
            meta = create_project(name, template=template, path=path)
            return jsonify(meta), 201
        except (FileExistsError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>")
    def api_project_get(name):
        try:
            p = get_project(name)
            return jsonify(p)
        except FileNotFoundError:
            return jsonify({"error": "Progetto non trovato"}), 404

    @app.route("/api/projects/<name>/run", methods=["POST"])
    def api_project_run(name):
        data = request.get_json(silent=True) or {}
        try:
            r = run_project(name, file=data.get("file"))
            return jsonify(r)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/test", methods=["POST"])
    def api_project_test(name):
        try:
            r = test_project(name)
            return jsonify(r)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/preview")
    def api_project_preview(name):
        file = request.args.get("file")
        try:
            p = preview_project(name, file=file)
            return jsonify(p)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>", methods=["DELETE"])
    def api_project_delete(name):
        try:
            p = delete_project(name)
            return jsonify(p)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/project/templates")
    def api_project_templates():
        return jsonify(list_templates())

    @app.route("/api/project/run-code", methods=["POST"])
    def api_project_run_code():
        data = request.get_json() or {}
        code = data.get("code", "")
        language = data.get("language", "python")
        try:
            r = run_code(code, language=language)
            return jsonify(r)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/generate", methods=["POST"])
    def api_project_generate():
        """Generate a project from a natural language description using AI."""
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt richiesto"}), 400
        name_hint = data.get("name", "").strip() or None
        source = data.get("source", app.config.get("DEFAULT_SOURCE", "online"))

        def generate():
            yield "data: " + json.dumps({"type": "status", "message": "Analizzo descrizione..."}) + "\n\n"
            try:
                # Select provider
                cfg = load_config()
                if source == "local":
                    provider = get_local_provider(cfg)
                else:
                    provider = get_online_provider(cfg)

                yield "data: " + json.dumps({"type": "status", "message": "Genero struttura progetto..."}) + "\n\n"

                from ..project import generate_project_from_prompt
                meta = generate_project_from_prompt(prompt, provider.generate_chat, name_hint)

                yield "data: " + json.dumps({
                    "type": "done",
                    "message": f"Progetto '{meta['name']}' creato con {meta['files_count']} file",
                    "project": meta
                }) + "\n\n"

            except json.JSONDecodeError as e:
                yield "data: " + json.dumps({"type": "error", "message": f"Errore nel parsing della risposta AI: {e}"}) + "\n\n"
            except Exception as e:
                log.exception("Project generation failed")
                yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/projects/<name>/file", methods=["PUT"])
    def api_project_write_file(name):
        """Write/update a single file in a project."""
        data = request.get_json() or {}
        filepath = data.get("filepath", "")
        content = data.get("content", "")
        if not filepath:
            return jsonify({"error": "filepath richiesto"}), 400
        try:
            from ..project import write_project_file
            write_project_file(name, filepath, content)
            return jsonify({"ok": True, "filepath": filepath})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/improve", methods=["POST"])
    def api_project_improve(name):
        """AI improvement of an existing project."""
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt richiesto"}), 400
        source = data.get("source", "online")

        def generate():
            yield "data: " + json.dumps({"type": "status", "message": "Analizzo progetto..."}) + "\n\n"
            try:
                cfg = load_config()
                if source == "local":
                    provider = get_local_provider(cfg)
                else:
                    provider = get_online_provider(cfg)

                yield "data: " + json.dumps({"type": "status", "message": "Applico modifiche..."}) + "\n\n"

                from ..project import improve_project
                meta = improve_project(name, prompt, provider.generate_chat)

                yield "data: " + json.dumps({
                    "type": "done",
                    "message": f"Progetto '{meta['name']}' aggiornato ({meta['files_count']} file)",
                    "project": meta
                }) + "\n\n"

            except json.JSONDecodeError as e:
                yield "data: " + json.dumps({"type": "error", "message": f"Errore parsing risposta AI: {e}"}) + "\n\n"
            except Exception as e:
                log.exception("Project improve failed")
                yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/projects/<name>/rename", methods=["POST"])
    def api_project_rename(name):
        """Rename a project."""
        data = request.get_json() or {}
        new_name = data.get("name", "").strip()
        if not new_name:
            return jsonify({"error": "name richiesto"}), 400
        try:
            from ..project import rename_project
            meta = rename_project(name, new_name)
            return jsonify(meta)
        except FileExistsError as e:
            return jsonify({"error": str(e)}), 409
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/download")
    def api_project_download(name):
        """Download a project as a zip archive."""
        try:
            from ..project import export_project_zip
            zip_path = export_project_zip(name)
            return send_file(
                zip_path,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{name}.zip"
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/delete-file", methods=["POST"])
    def api_project_delete_file(name):
        """Delete a file from a project."""
        data = request.get_json() or {}
        filepath = data.get("filepath", "")
        if not filepath:
            return jsonify({"error": "filepath richiesto"}), 400
        try:
            from ..project import delete_project_file
            delete_project_file(name, filepath)
            return jsonify({"ok": True, "filepath": filepath})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/run-stream")
    def api_project_run_stream(name):
        """Run project with real-time output streaming."""
        file = request.args.get("file")
        try:
            from ..project import run_project_stream
            def generate():
                for event in run_project_stream(name, file=file):
                    yield "data: " + json.dumps(event) + "\n\n"
            return Response(generate(), mimetype="text/event-stream")
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/config", methods=["PUT"])
    def api_project_update_config(name):
        """Update project metadata (description, run_cmd, test_cmd)."""
        data = request.get_json() or {}
        try:
            from ..project import update_project_config
            meta = update_project_config(
                name,
                description=data.get("description"),
                run_cmd=data.get("run_cmd"),
                test_cmd=data.get("test_cmd"),
            )
            return jsonify(meta)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/<name>/clone", methods=["POST"])
    def api_project_clone(name):
        """Clone a project."""
        data = request.get_json() or {}
        new_name = data.get("name", "").strip() or None
        try:
            from ..project import clone_project
            meta = clone_project(name, new_name)
            return jsonify(meta), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/projects/import", methods=["POST"])
    def api_project_import():
        """Import a project from an uploaded zip file."""
        if "file" not in request.files:
            return jsonify({"error": "file zip richiesto"}), 400
        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return jsonify({"error": "Solo file .zip"}), 400
        try:
            tmp = tempfile.mktemp(suffix=".zip")
            file.save(tmp)
            from ..project import import_project_zip
            meta = import_project_zip(tmp, name_hint=request.form.get("name"))
            os.unlink(tmp)
            return jsonify(meta), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # ─── API: Agents export/import ─────────────────────────────────

    @app.route("/api/agents/<name>/export")
    def api_agent_export(name):
        a = get_agent(name)
        if a is None:
            return jsonify({"error": "not found"}), 404
        resp = jsonify(a)
        resp.headers["Content-Disposition"] = f'attachment; filename="{name}.agent.json"'
        return resp

    @app.route("/api/agents/import", methods=["POST"])
    def api_agent_import():
        data = request.get_json()
        if not data or not data.get("name"):
            return jsonify({"error": "name required"}), 400
        name = data["name"].strip()
        if get_agent(name):
            return jsonify({"error": "already exists"}), 409
        payload = {k: v for k, v in data.items() if k != "name"}
        payload.setdefault("description", "")
        payload.setdefault("model", "auto")
        payload.setdefault("mode", "auto")
        payload.setdefault("instructions", "")
        payload.setdefault("permissions", {"read": True, "edit": False, "bash": False})
        payload.setdefault("tags", [])
        payload["created"] = datetime.now().isoformat()
        save_agent(name, payload)
        return jsonify({"ok": True, "name": name}), 201

    # ─── API: Agents migliorata ─────────────────────────────────────

    @app.route("/api/agents/stats")
    def api_agents_stats():
        """Statistiche per agente: quante chiamate, token, tempo."""
        agents = list_agents()
        tracker = app.config["TRACKER"]
        history = getattr(tracker, "history", [])
        stats = {}
        for a in agents:
            name = a.get("name", "")
            calls = [h for h in history if h.get("agent") == name]
            last_time = calls[-1].get("time_s") if calls else None
            stats[name] = {
                "calls": len(calls),
                "total_tokens": sum(h.get("tokens_total", 0) for h in calls),
                "total_time_s": round(sum(h.get("time_s", 0) for h in calls), 2),
                "last_used": calls[-1].get("timestamp") if calls else None,
            }
        return jsonify(stats)

    AGENT_TEMPLATES = [
        {
            "name": "",
            "description": "AI+ Coding — assistente completo con tool (read/write/edit/run/search/web). Come opencode.",
            "mode": "online",
            "model": "auto",
            "instructions": "Sei AI+, un assistente di programmazione esperto in tutti i linguaggi. Hai accesso a strumenti per leggere, scrivere, modificare file, eseguire comandi e cercare sul web. Usali per aiutare l'utente in modo proattivo. Scrivi codice pulito, testalo, e itera fino a ottenere il risultato giusto.",
            "tags": ["coding", "opencode", "agent", "tools", "template"],
            "permissions": {"read": True, "edit": True, "bash": True, "websearch": True, "webfetch": True},
            "temperature": 0.3,
            "max_tokens": 16384,
            "top_p": 0.9,
            "favorite": True,
        },
        {
            "name": "",
            "description": "Code reviewer — analisi codice statico, bug detection, suggerimenti (sola lettura)",
            "mode": "online",
            "model": "auto",
            "instructions": "Sei un code reviewer esperto. Analizza il codice fornito, trova bug, vulnerabilità e opportunità di refactoring. Fornisci suggerimenti concreti con esempi di codice. Puoi leggere file ma non modificarli.",
            "tags": ["coding", "review", "template"],
            "permissions": {"read": True, "edit": False, "bash": False, "websearch": False, "webfetch": False},
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "favorite": False,
        },
        {
            "name": "",
            "description": "DevOps engineer — esegue comandi, gestisce server, docker, deploy",
            "mode": "online",
            "model": "auto",
            "instructions": "Sei un ingegnere DevOps. Puoi eseguire comandi shell, leggere configurazioni, gestire docker, analisi di sistema. Aiuta l'utente con deploy, troubleshooting infrastruttura e automazione.",
            "tags": ["devops", "docker", "server", "template"],
            "permissions": {"read": True, "edit": True, "bash": True, "websearch": True, "webfetch": True},
            "temperature": 0.2,
            "max_tokens": 8192,
            "top_p": 0.9,
            "favorite": False,
        },
        {
            "name": "",
            "description": "Traduttore — traduce testo tra lingue mantenendo contesto",
            "mode": "auto",
            "model": "auto",
            "instructions": "Sei un traduttore professionista. Traduci il testo fornito nella lingua richiesta mantenendo tono, registro e sfumature culturali. Spiega le scelte traduttive quando utile.",
            "tags": ["translation", "template"],
            "permissions": {"read": False, "edit": False, "bash": False, "websearch": False, "webfetch": False},
            "temperature": 0.3,
            "max_tokens": 2048,
            "top_p": 0.9,
            "favorite": False,
        },
        {
            "name": "",
            "description": "Summarizer — riassume articoli, documenti e conversazioni (locale, veloce)",
            "mode": "local",
            "model": "auto",
            "instructions": "Sei un assistente specializzato in riassunti. Leggi il contenuto fornito e produci un riassunto conciso ma completo, strutturato in punti chiave. Mantieni i fatti importanti senza opinioni personali.",
            "tags": ["summarization", "template"],
            "permissions": {"read": True, "edit": False, "bash": False, "websearch": False, "webfetch": False},
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.9,
            "favorite": False,
        },
        {
            "name": "",
            "description": "Web researcher — cerca informazioni online, analizza pagine, riassume risultati",
            "mode": "online",
            "model": "auto",
            "instructions": "Sei un ricercatore web. Cerca informazioni online, analizza pagine, riassume risultati. Usa web search e webfetch per trovare informazioni aggiornate. Cita sempre le fonti.",
            "tags": ["web", "research", "search", "template"],
            "permissions": {"read": True, "edit": False, "bash": False, "websearch": True, "webfetch": True},
            "temperature": 0.3,
            "max_tokens": 4096,
            "top_p": 0.9,
            "favorite": False,
        },
        {
            "name": "",
            "description": "Local fast — risposte rapide con modello locale, senza tool esterni",
            "mode": "local",
            "model": "auto",
            "instructions": "Sei un assistente AI amichevole e diretto. Rispondi in modo conciso e pertinente. Non hai accesso a internet o strumenti esterni.",
            "tags": ["fast", "local", "template"],
            "permissions": {"read": False, "edit": False, "bash": False, "websearch": False, "webfetch": False},
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "favorite": False,
        },
    ]

    @app.route("/api/agents/templates")
    def api_agent_templates():
        return jsonify(AGENT_TEMPLATES)

    @app.route("/api/agents/<name>/duplicate", methods=["POST"])
    def api_agent_duplicate(name):
        a = get_agent(name)
        if a is None:
            return jsonify({"error": "not found"}), 404
        new_name = f"{name}-copy"
        base = new_name
        counter = 1
        while get_agent(new_name):
            new_name = f"{base}-{counter}"
            counter += 1
        payload = {k: v for k, v in a.items() if k != "name"}
        payload["name"] = new_name
        payload["created"] = datetime.now().isoformat()
        payload["description"] = (payload.get("description", "") or "") + " (copia)"
        save_agent(new_name, payload)
        return jsonify({"ok": True, "name": new_name}), 201

    # ─── API: Workspace (files + git, like opencode) ──────────────────

    @app.route("/workspace")
    def workspace_page():
        return render_template("workspace.html", active_page="workspace")

    @app.route("/api/workspace/tree", methods=["GET"])
    def api_workspace_tree():
        path = request.args.get("path", ".")
        try:
            p = Path(path).expanduser().resolve()
            if not p.is_dir():
                return jsonify({"error": "not a directory"}), 400
            entries = []
            for child in sorted(p.iterdir()):
                name = child.name
                if name.startswith("."):
                    continue
                entries.append({
                    "name": name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "size": child.stat().st_size if child.is_file() else 0,
                })
            return jsonify({"path": str(p), "entries": entries, "cwd": str(Path.cwd()), "home": str(Path.home())})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/read", methods=["GET"])
    def api_workspace_read():
        path = request.args.get("path", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(path).resolve()
            if not p.is_file():
                return jsonify({"error": "not a file"}), 400
            content = p.read_text(encoding="utf-8", errors="replace")
            return jsonify({"path": str(p), "content": content, "size": p.stat().st_size})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/write", methods=["POST"])
    def api_workspace_write():
        data = request.get_json() or {}
        path = data.get("path", "")
        content = data.get("content", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log.info(f"Written {len(content)} bytes to {p}")
            return jsonify({"path": str(p), "size": len(content)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/mkdir", methods=["POST"])
    def api_workspace_mkdir():
        data = request.get_json() or {}
        path = data.get("path", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(path).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return jsonify({"ok": True, "path": str(p)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/delete", methods=["POST"])
    def api_workspace_delete():
        data = request.get_json() or {}
        path = data.get("path", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(path).resolve()
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                return jsonify({"error": "not found"}), 404
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/rename", methods=["POST"])
    def api_workspace_rename():
        data = request.get_json() or {}
        src = data.get("src", "")
        dst = data.get("dst", "")
        if not src or not dst:
            return jsonify({"error": "src and dst required"}), 400
        try:
            sp = Path(src).resolve()
            dp = Path(dst).resolve()
            dp.parent.mkdir(parents=True, exist_ok=True)
            sp.rename(dp)
            return jsonify({"ok": True, "src": str(sp), "dst": str(dp)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/info", methods=["GET"])
    def api_workspace_info():
        path = request.args.get("path", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(path).resolve()
            if not p.exists():
                return jsonify({"error": "not found"}), 404
            import mimetypes
            mt, _ = mimetypes.guess_type(str(p))
            st = p.stat()
            is_binary = False
            if p.is_file() and p.stat().st_size > 0:
                try:
                    with open(p, "rb") as f:
                        chunk = f.read(1024)
                    is_binary = b"\0" in chunk
                except Exception:
                    pass
            return jsonify({
                "path": str(p),
                "name": p.name,
                "is_dir": p.is_dir(),
                "is_file": p.is_file(),
                "size": st.st_size,
                "mtime": st.st_mtime,
                "mimetype": mt or "application/octet-stream",
                "is_binary": is_binary,
                "permissions": oct(st.st_mode)[-3:],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workspace/search", methods=["GET"])
    def api_workspace_search():
        q = request.args.get("q", "")
        path = request.args.get("path", ".")
        case = request.args.get("case", "0") == "1"
        if not q:
            return jsonify({"results": []})
        try:
            results = []
            p = Path(path).resolve()
            q_comp = q if case else q.lower()
            for f in p.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        fname_comp = f.name if case else f.name.lower()
                        if q_comp in fname_comp:
                            results.append({"path": str(f), "name": f.name, "match": "filename", "line": "", "lineno": 0})
                            if len(results) >= 100:
                                break
                            continue
                    except Exception:
                        pass
                    try:
                        if f.stat().st_size < 1024 * 100:
                            text = f.read_text(encoding="utf-8", errors="replace")
                            lines = text.split("\n")
                            for i, line in enumerate(lines, 1):
                                line_comp = line if case else line.lower()
                                if q_comp in line_comp:
                                    results.append({
                                        "path": str(f), "name": f.name, "match": "content",
                                        "line": line.strip()[:200], "lineno": i,
                                    })
                                    if len(results) >= 100:
                                        break
                    except Exception:
                        pass
                    if len(results) >= 100:
                        break
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── API: Tools (Agent Tool System) ─────────────────────────────

    @app.route("/api/tools", methods=["GET"])
    def api_tools_list():
        return jsonify({"tools": list_tools()})

    @app.route("/api/tools/execute", methods=["POST"])
    def api_tools_execute():
        data = request.get_json() or {}
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if not name:
            return jsonify({"error": "tool name required"}), 400
        result = execute_tool(name, **arguments)
        return jsonify(result.dict())

    @app.route("/api/tools/batch", methods=["POST"])
    def api_tools_batch():
        """Esegue una sequenza di tool call (per agenti)."""
        data = request.get_json() or {}
        calls = data.get("calls", [])
        if not calls:
            return jsonify({"error": "calls required"}), 400
        results = []
        for call in calls:
            results.append(execute_tool_call(call))
        return jsonify({"results": results})

    @app.route("/api/workspace/run", methods=["POST"])
    def api_workspace_run():
        """Esegue un comando nel workspace (sandboxed)."""
        data = request.get_json() or {}
        command = data.get("command", "")
        workdir = data.get("workdir", ".")
        if not command:
            return jsonify({"error": "command required"}), 400
        from ..tools import RunTool
        result = RunTool().execute(command=command, workdir=workdir)
        return jsonify(result.dict())

    @app.route("/api/git/status", methods=["GET"])
    def api_git_status():
        path = request.args.get("path", ".")
        try:
            r = subprocess.run(
                ["git", "-C", path, "status", "--porcelain", "--branch"],
                capture_output=True, text=True, timeout=10,
            )
            return jsonify({
                "status": r.stdout,
                "error": r.stderr.strip() or None,
                "returncode": r.returncode,
            })
        except FileNotFoundError:
            return jsonify({"error": "git not found"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/diff", methods=["GET"])
    def api_git_diff():
        path = request.args.get("path", ".")
        staged = request.args.get("staged", "0") == "1"
        file = request.args.get("file")
        try:
            cmd = ["git", "-C", path, "diff"]
            if staged:
                cmd.append("--staged")
            if file:
                cmd.append("--")
                cmd.append(file)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return jsonify({
                "diff": r.stdout,
                "error": r.stderr.strip() or None,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/add", methods=["POST"])
    def api_git_add():
        data = request.get_json() or {}
        path = data.get("path", ".")
        files = data.get("files", None)
        all_files = data.get("all", True)
        try:
            cmd = ["git", "-C", path, "add"]
            if all_files or not files:
                cmd.append(".")
            else:
                cmd.extend(files)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return jsonify({"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/unstage", methods=["POST"])
    def api_git_unstage():
        data = request.get_json() or {}
        path = data.get("path", ".")
        files = data.get("files", ["."])
        try:
            cmd = ["git", "-C", path, "restore", "--staged"] + files
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return jsonify({"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/commit", methods=["POST"])
    def api_git_commit():
        data = request.get_json() or {}
        path = data.get("path", ".")
        message = data.get("message", "")
        add_first = data.get("add", True)
        if not message:
            return jsonify({"error": "commit message required"}), 400
        try:
            if add_first:
                subprocess.run(["git", "-C", path, "add", "."], capture_output=True, text=True, timeout=10)
            r = subprocess.run(
                ["git", "-C", path, "commit", "-m", message],
                capture_output=True, text=True, timeout=10,
            )
            return jsonify({
                "stdout": r.stdout,
                "stderr": r.stderr,
                "returncode": r.returncode,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/log", methods=["GET"])
    def api_git_log():
        path = request.args.get("path", ".")
        count = int(request.args.get("count", "10"))
        try:
            r = subprocess.run(
                ["git", "-C", path, "log", f"--max-count={count}",
                 "--pretty=format:%h|%an|%ar|%s", "--date=relative"],
                capture_output=True, text=True, timeout=10,
            )
            commits = []
            for line in r.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({"hash": parts[0], "author": parts[1], "date": parts[2], "message": parts[3]})
            return jsonify({"commits": commits})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/git/branch", methods=["POST"])
    def api_git_branch():
        data = request.get_json() or {}
        path = data.get("path", ".")
        name = data.get("name", "")
        action = data.get("action", "create")  # create | checkout | delete
        try:
            if action == "create" and name:
                cmd = ["git", "-C", path, "checkout", "-b", name]
            elif action == "checkout" and name:
                cmd = ["git", "-C", path, "checkout", name]
            elif action == "delete" and name:
                cmd = ["git", "-C", path, "branch", "-D", name]
            else:
                return jsonify({"error": "invalid action or missing name"}), 400
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return jsonify({"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── Export System ───────────────────────────────────────────

    @app.route("/api/export/conversations", methods=["GET"])
    def api_export_conversations():
        """Export all conversations as JSON."""
        from ..sessions import SESSIONS_DIR
        sessions_dir = Path(SESSIONS_DIR)
        if not sessions_dir.exists():
            return jsonify({"conversations": []})
        all_convs = []
        for f in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text())
                all_convs.append(data)
            except Exception:
                pass
        fmt = request.args.get("format", "json")
        if fmt == "json":
            return jsonify({"conversations": all_convs, "total": len(all_convs), "exported": time.strftime("%Y-%m-%dT%H:%M:%S")})
        elif fmt == "markdown":
            lines = ["# AI+ — Conversazioni esportate", f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Totale: {len(all_convs)} sessioni", ""]
            for conv in all_convs:
                sid = conv.get("session_id", "unknown")
                updated = conv.get("updated", "")
                count = conv.get("message_count", 0)
                lines.append(f"## Sessione: {sid}")
                lines.append(f"_Aggiornata: {updated} · {count} messaggi_")
                lines.append("")
                for msg in conv.get("messages", []):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    lines.append(f"### {role.upper()}")
                    lines.append(content)
                    lines.append("")
                lines.append("---")
                lines.append("")
            return Response("\n".join(lines), mimetype="text/markdown", headers={"Content-Disposition": "attachment; filename=conversations.md"})
        return jsonify({"error": "formato non supportato"}), 400

    @app.route("/api/export/conversation/<session_id>", methods=["GET"])
    def api_export_conversation(session_id):
        """Export a single conversation."""
        from ..sessions import load_session as _load
        messages = _load(session_id)
        if not messages:
            return jsonify({"error": "Sessione non trovata"}), 404
        fmt = request.args.get("format", "markdown")
        if fmt == "json":
            return jsonify({"session_id": session_id, "messages": messages, "count": len(messages)})
        lines = [f"# Conversazione: {session_id}", f"Data export: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Messaggi: {len(messages)}", ""]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"> _Istruzioni di sistema:_ {content}")
            else:
                lines.append(f"## {role.upper()}")
                lines.append(content)
            lines.append("")
        return Response("\n".join(lines), mimetype="text/markdown", headers={"Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.md"})

    @app.route("/api/export/knowledge", methods=["GET"])
    def api_export_knowledge():
        """Export knowledge base content."""
        kb = get_knowledge_base()
        fmt = request.args.get("format", "markdown")
        if fmt == "json":
            docs = kb.index.documents if hasattr(kb.index, 'documents') else []
            return jsonify({"total_chunks": kb.total_chunks, "sources": kb.sources, "documents": docs})
        lines = ["# Knowledge Base — Esportazione", f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Totale chunk: {kb.total_chunks}", f"Totale fonti: {len(kb.sources)}", ""]
        for src_key, src_info in kb.sources.items():
            lines.append(f"## Fonte: {src_key}")
            lines.append(f"_Tipo: {src_info.get('type', '?')} · Chunk: {src_info.get('chunks', 0)}_")
            lines.append("")
            chunks = kb.get_source_chunks(src_key) if hasattr(kb, 'get_source_chunks') else []
            for c in chunks:
                lines.append(f"### Chunk {c.get('chunk_id', 0)}")
                lines.append(c.get("text", ""))
                lines.append("")
            lines.append("---")
            lines.append("")
        return Response("\n".join(lines), mimetype="text/markdown", headers={"Content-Disposition": "attachment; filename=knowledge_base.md"})

    @app.route("/api/export/agents", methods=["GET"])
    def api_export_agents():
        """Export all agents as JSON."""
        agents_list = list_agents()
        return jsonify({"agents": agents_list, "total": len(agents_list), "exported": time.strftime("%Y-%m-%dT%H:%M:%S")})

    @app.route("/api/export/notes", methods=["GET"])
    def api_export_notes():
        """Export all notes."""
        from ..notes import get_note_store
        ns = get_note_store()
        fmt = request.args.get("format", "markdown")
        if fmt == "json":
            all_notes = [ns.get(s) for s in ns.list_all() if ns.get(s)]
            return jsonify({"notes": all_notes, "total": len(all_notes)})
        lines = ["# Note — Esportazione", f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for slug in sorted(ns.list_all()):
            note = ns.get(slug)
            if note:
                lines.append(f"# {note.get('title', slug)}")
                lines.append("")
                lines.append(note.get("body", ""))
                lines.append("")
                lines.append("---")
                lines.append("")
        return Response("\n".join(lines), mimetype="text/markdown", headers={"Content-Disposition": "attachment; filename=notes.md"})

    @app.route("/api/export/wiki", methods=["GET"])
    def api_export_wiki():
        """Export all LLM wiki pages."""
        fmt = request.args.get("format", "markdown")
        pages = list_wiki_pages()
        if fmt == "json":
            full = []
            for p in pages:
                page = get_wiki_page(p.get("slug", ""))
                if page:
                    full.append(page)
            return jsonify({"pages": full, "total": len(full)})
        lines = ["# LLM Wiki — Esportazione", f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for p in pages:
            page = get_wiki_page(p.get("slug", ""))
            if page:
                lines.append(page.get("content", ""))
                lines.append("")
                lines.append("---")
                lines.append("")
        return Response("\n".join(lines), mimetype="text/markdown", headers={"Content-Disposition": "attachment; filename=wiki.md"})

    @app.route("/api/export/restore", methods=["POST"])
    def api_export_restore():
        """Restore platform from a backup ZIP."""
        if 'backup' not in request.files:
            return jsonify({"error": "Nessun file fornito"}), 400
        file = request.files['backup']
        import io, zipfile, shutil
        try:
            with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zf:
                for name in zf.namelist():
                    if name.startswith("/") or ".." in name:
                        continue
                    # Extract to appropriate locations
                    if name == "config.yaml":
                        zf.extract(name, DATA_DIR)
                    elif name.startswith("agents/") and name.endswith(".json"):
                        zf.extract(name, AGENTS_DIR)
                    elif name.startswith("sessions/") and name.endswith(".json"):
                        from ..sessions import SESSIONS_DIR
                        zf.extract(name, SESSIONS_DIR)
                    elif name.startswith("knowledge/"):
                        zf.extract(name, KB_DIR)
                    elif name.startswith("wiki/") and name.endswith(".md"):
                        wiki_dir = Path.home() / ".config" / "hybrid-coder" / "llmwiki"
                        zf.extract(name, wiki_dir)
                    elif name.startswith("notes/") and name.endswith(".md"):
                        from ..notes import NOTES_DIR
                        zf.extract(name, NOTES_DIR)
                    elif name == "prompts.json":
                        prompts_file = Path.home() / ".config" / "hybrid-coder" / "prompts.json"
                        zf.extract(name, prompts_file.parent)
            return jsonify({"ok": True, "message": "Backup ripristinato. Riavvia l'app per applicare le modifiche."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/export/backup", methods=["POST"])
    def api_export_backup():
        """Create a full platform backup as ZIP."""
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Config
            cfg_path = Path(CONFIG_FILE)
            if cfg_path.exists():
                zf.write(cfg_path, "config.yaml")
            # Agents
            agents = list_agents()
            zf.writestr("agents/index.json", json.dumps(agents, ensure_ascii=False, indent=2))
            for a in agents:
                name = a.get("name", "unknown")
                agent_data = get_agent(name)
                if agent_data:
                    zf.writestr(f"agents/{name}.json", json.dumps(agent_data, ensure_ascii=False, indent=2))
            # Sessions
            from ..sessions import SESSIONS_DIR
            sdir = Path(SESSIONS_DIR)
            if sdir.exists():
                for f in sdir.glob("*.json"):
                    zf.write(f, f"sessions/{f.name}")
            # Knowledge
            kb = get_knowledge_base()
            kdata = {
                "total_chunks": kb.total_chunks,
                "sources": kb.sources,
            }
            if hasattr(kb.index, 'documents'):
                kdata["documents"] = kb.index.documents
            zf.writestr("knowledge/index.json", json.dumps(kdata, ensure_ascii=False, indent=2))
            # Wiki
            wiki_dir = Path.home() / ".config" / "hybrid-coder" / "llmwiki"
            if wiki_dir.exists():
                for f in wiki_dir.glob("*.md"):
                    zf.write(f, f"wiki/{f.name}")
            # Notes
            from ..notes import NOTES_DIR
            ndir = Path(NOTES_DIR)
            if ndir.exists():
                for f in ndir.glob("*.md"):
                    zf.write(f, f"notes/{f.name}")
            # Prompts
            prompts_file = Path.home() / ".config" / "hybrid-coder" / "prompts.json"
            if prompts_file.exists():
                zf.write(prompts_file, "prompts.json")
        buf.seek(0)
        return Response(
            buf.getvalue(),
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename=ai-plus-backup-{time.strftime('%Y%m%d')}.zip"},
        )

    # ─── Web Search ────────────────────────────────────────────────

    # ─── Prompt Library ────────────────────────────────────────────

    PROMPTS_DIR = KB_DIR / "prompts"
    PROMPTS_FILE = PROMPTS_DIR / "index.json"

    def _load_prompts():
        if PROMPTS_FILE.exists():
            try:
                return json.loads(PROMPTS_FILE.read_text())
            except Exception:
                pass
        return []

    def _sync_prompt_file(prompt):
        """Salva ogni prompt come file .md nella cartella prompts/ per indicizzazione KB."""
        slug = prompt.get("name", "untitled").lower().replace(" ", "-").replace("/", "-")
        slug = slug + "-" + prompt["id"][-6:]
        md = f"""---
id: {prompt['id']}
title: {prompt['name']}
tags: {json.dumps(prompt.get('tags', []))}
created: {prompt['created']}
updated: {prompt['updated']}
type: prompt
---

# {prompt['name']}

{prompt['content']}
"""
        (PROMPTS_DIR / f"{slug}.md").write_text(md)

    def _remove_prompt_file(prompt_id):
        for f in PROMPTS_DIR.glob(f"*-{prompt_id[-6:]}.md"):
            f.unlink()

    def _save_prompts(prompts):
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        PROMPTS_FILE.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
        for p in prompts:
            _sync_prompt_file(p)

    @app.route("/api/prompts", methods=["GET"])
    def api_prompts_list():
        prompts = _load_prompts()
        return jsonify({"prompts": prompts, "total": len(prompts)})

    @app.route("/api/prompts", methods=["POST"])
    def api_prompts_create():
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        content = data.get("content", "").strip()
        if not name or not content:
            return jsonify({"error": "name e content richiesti"}), 400
        prompts = _load_prompts()
        new_id = str(int(time.time()))
        new_prompt = {
            "id": new_id,
            "name": name,
            "content": content,
            "tags": data.get("tags", []),
            "variables": data.get("variables", []),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        prompts.append(new_prompt)
        _save_prompts(prompts)
        return jsonify(new_prompt), 201

    @app.route("/api/prompts/<prompt_id>", methods=["PUT"])
    def api_prompts_update(prompt_id):
        data = request.get_json() or {}
        prompts = _load_prompts()
        for p in prompts:
            if p.get("id") == prompt_id:
                if "name" in data:
                    p["name"] = data["name"].strip()
                if "content" in data:
                    p["content"] = data["content"].strip()
                if "tags" in data:
                    p["tags"] = data["tags"]
                if "variables" in data:
                    p["variables"] = data["variables"]
                p["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_prompts(prompts)
                return jsonify(p)
        return jsonify({"error": "Prompt non trovato"}), 404

    @app.route("/api/prompts/<prompt_id>", methods=["DELETE"])
    def api_prompts_delete(prompt_id):
        prompts = _load_prompts()
        deleted = [p for p in prompts if p.get("id") == prompt_id]
        new_prompts = [p for p in prompts if p.get("id") != prompt_id]
        if not deleted:
            return jsonify({"error": "Prompt non trovato"}), 404
        _remove_prompt_file(prompt_id)
        _save_prompts(new_prompts)
        return jsonify({"ok": True})

    # ─── Version ──────────────────────────────────────────────────

    @app.route("/api/system/version")
    def api_system_version():
        from .. import __version__
        return jsonify({
            "version": __version__,
            "name": "AI+",
            "build": time.strftime("%Y%m%d"),
        })

    # ─── Versioned Backup System ────────────────────────────────

    @app.route("/api/backup/create", methods=["POST"])
    def api_backup_create():
        """Create a versioned backup snapshot with metadata."""
        body = request.get_json() or {}
        label = body.get("label", "")
        bid = time.strftime("backup_%Y%m%d_%H%M%S")
        mdir = _backup_meta_dir(bid)
        mdir.mkdir(parents=True, exist_ok=True)
        from .. import __version__
        manifest = {
            "id": bid,
            "version": __version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": label,
            "size_bytes": 0,
        }
        zip_path = mdir / "backup.zip"
        _build_backup_zip(zip_path)
        manifest["size_bytes"] = zip_path.stat().st_size
        _save_backup_manifest(mdir, manifest)
        return jsonify({"ok": True, "backup": manifest})

    @app.route("/api/backup/list")
    def api_backup_list():
        """List all versioned backups."""
        _BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        backups = []
        for d in sorted(_BACKUPS_DIR.iterdir(), reverse=True):
            if d.is_dir():
                m = _load_backup_manifest(d)
                if m:
                    backups.append(m)
        return jsonify({"backups": backups, "total": len(backups)})

    @app.route("/api/backup/restore/<bid>", methods=["POST"])
    def api_backup_restore(bid):
        """Restore from a specific versioned backup."""
        mdir = _backup_meta_dir(bid)
        zip_path = mdir / "backup.zip"
        if not zip_path.exists():
            return jsonify({"error": "Backup non trovato"}), 404
        import io, zipfile, shutil
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.startswith("/") or ".." in name:
                        continue
                    if name == "config.yaml":
                        zf.extract(name, DATA_DIR)
                    elif name.startswith("agents/") and name.endswith(".json"):
                        zf.extract(name, AGENTS_DIR)
                    elif name.startswith("sessions/") and name.endswith(".json"):
                        from ..sessions import SESSIONS_DIR
                        zf.extract(name, SESSIONS_DIR)
                    elif name.startswith("knowledge/"):
                        zf.extract(name, KB_DIR)
                    elif name.startswith("wiki/") and name.endswith(".md"):
                        wiki_dir = Path.home() / ".config" / "hybrid-coder" / "llmwiki"
                        zf.extract(name, wiki_dir)
                    elif name.startswith("notes/") and name.endswith(".md"):
                        from ..notes import NOTES_DIR
                        zf.extract(name, NOTES_DIR)
                    elif name == "prompts.json":
                        prompts_file = Path.home() / ".config" / "hybrid-coder" / "prompts.json"
                        zf.extract(name, prompts_file.parent)
            return jsonify({"ok": True, "message": "Backup ripristinato. Riavvia l'app."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/backup/delete/<bid>", methods=["DELETE"])
    def api_backup_delete(bid):
        """Delete a specific backup."""
        mdir = _backup_meta_dir(bid)
        if not mdir.exists():
            return jsonify({"error": "Backup non trovato"}), 404
        import shutil
        shutil.rmtree(mdir)
        return jsonify({"ok": True})

    # ─── Backup Schedule ────────────────────────────────────────────

    @app.route("/api/backup/schedule", methods=["GET"])
    def api_backup_schedule_get():
        from ..config import load_config
        cfg = load_config()
        bk = cfg.get("backup", {})
        return jsonify({
            "enabled": bk.get("enabled", False),
            "interval_hours": bk.get("interval_hours", 24),
            "destination": bk.get("destination", ""),
            "last_run": _load_last_backup_time(),
        })

    @app.route("/api/backup/schedule", methods=["PUT"])
    def api_backup_schedule_set():
        from ..config import load_config, save_config
        data = request.get_json() or {}
        cfg = load_config(force=True)
        if "enabled" in data:
            cfg.setdefault("backup", {})["enabled"] = bool(data["enabled"])
        if "interval_hours" in data:
            cfg.setdefault("backup", {})["interval_hours"] = float(data["interval_hours"])
        if "destination" in data:
            cfg.setdefault("backup", {})["destination"] = str(data["destination"])
        save_config(cfg)
        return jsonify({"ok": True})

    # ─── Export Install Pack ────────────────────────────────────────

    @app.route("/api/export/install-pack", methods=["POST"])
    def api_export_install_pack():
        """Genera il pacchetto di installazione. Se 'destination' è specificato, lo salva su disco."""
        import tempfile
        import shutil

        from ..install_pack import generate_pack, WEB_PORT

        build = _next_build_number()
        body = request.get_json() or {}
        dest = body.get("destination", "").strip()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = generate_pack(tmp, port=WEB_PORT)
                size = zip_path.stat().st_size
                log.info("Install pack generato (build #%d): %s (%d bytes)", build, zip_path, size)
                if dest:
                    dest_path = Path(dest).expanduser().resolve()
                    dest_path.mkdir(parents=True, exist_ok=True)
                    out = dest_path / f"ai-plus-install-pack-v{build}.zip"
                    shutil.copy2(zip_path, out)
                    return jsonify({"ok": True, "build": build, "path": str(out), "size": size})
                resp = send_file(
                    zip_path,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=zip_path.name,
                )
                resp.headers["X-Build-Number"] = str(build)
                return resp
        except Exception as e:
            log.exception("Errore generazione install pack")
            return jsonify({"error": str(e)}), 500

    def _next_build_number():
        bf = Path.home() / ".config" / "hybrid-coder" / "build_counter"
        bf.parent.mkdir(parents=True, exist_ok=True)
        cur = 0
        if bf.exists():
            cur = int(bf.read_text().strip() or "0")
        cur += 1
        bf.write_text(str(cur))
        return cur

    @app.route("/api/export/install-pack/build", methods=["GET"])
    def api_install_pack_build():
        bf = Path.home() / ".config" / "hybrid-coder" / "build_counter"
        cur = int(bf.read_text().strip()) if bf.exists() else 0
        return jsonify({"build": cur})

    # ─── Skills System ───────────────────────────────────────────

    @app.route("/skills")
    def page_skills():
        return render_template("skills.html", active_page="skills")

    @app.route("/api/skills", methods=["GET"])
    def api_skills_list():
        from ..skill_manager import list_skills
        return jsonify({"skills": list_skills()})

    @app.route("/api/skills/<skill_id>", methods=["GET"])
    def api_skill_get(skill_id):
        from ..skill_manager import get_skill
        skill = get_skill(skill_id)
        if not skill:
            return jsonify({"error": "Skill non trovata"}), 404
        return jsonify(skill)

    @app.route("/api/skills", methods=["POST"])
    def api_skill_create():
        from ..skill_manager import create_skill
        data = request.get_json() or {}
        if not data.get("name"):
            return jsonify({"error": "Nome richiesto"}), 400
        result = create_skill(data)
        return jsonify(result), 201

    @app.route("/api/skills/upload", methods=["POST"])
    def api_skill_upload():
        """Upload a skill file and auto-adapt it."""
        from ..skill_manager import create_skill, adapt_skill, detect_platform
        if "file" not in request.files:
            return jsonify({"error": "file required"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "file required"}), 400
        content = f.read().decode("utf-8", errors="replace")
        platform = request.form.get("platform", "")
        if not platform:
            platform = detect_platform(f.filename, content)
        name = request.form.get("name", "") or Path(f.filename).stem
        adapted = adapt_skill(content, platform)
        if "error" in adapted:
            return jsonify({"error": adapted["error"]}), 400
        skill_data = {
            "name": name,
            "description": adapted.get("description", ""),
            "source_platform": platform,
            "instructions": content,
            "adapted_instructions": adapted.get("adapted_instructions", ""),
            "system_prompt": adapted.get("system_prompt", adapted.get("instructions", "")),
            "tools": adapted.get("tools", []),
            "tags": adapted.get("tags", []),
        }
        result = create_skill(skill_data)
        return jsonify(result), 201

    @app.route("/api/skills/<skill_id>", methods=["DELETE"])
    def api_skill_delete(skill_id):
        from ..skill_manager import delete_skill
        if delete_skill(skill_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Skill non trovata"}), 404

    @app.route("/api/skills/<skill_id>/export", methods=["GET"])
    def api_skill_export(skill_id):
        from ..skill_manager import get_skill
        skill = get_skill(skill_id)
        if not skill:
            return jsonify({"error": "Skill non trovata"}), 404
        return jsonify(skill)

    # ─── Profiles & Login ────────────────────────────────────────

    PROFILES_FILE = Path.home() / ".config" / "hybrid-coder" / "profiles.json"
    SALT = "hycoder"  # simple salt for hashing

    def _load_profiles():
        if PROFILES_FILE.exists():
            return json.loads(PROFILES_FILE.read_text())
        return {"profiles": {}, "active": None}

    def _save_profiles(data):
        PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROFILES_FILE.write_text(json.dumps(data, indent=2))

    def _hash_pwd(password):
        import hashlib
        return hashlib.sha256((SALT + password).encode()).hexdigest()

    def _current_profile():
        return session.get("profile")

    @app.route("/api/profiles", methods=["GET"])
    def api_profiles_list():
        data = _load_profiles()
        return jsonify({
            "profiles": list(data["profiles"].keys()),
            "active": data["active"],
            "current": _current_profile(),
        })

    @app.route("/api/profiles/login", methods=["POST"])
    def api_profiles_login():
        body = request.get_json() or {}
        username = body.get("username", "").strip()
        password = body.get("password", "")
        if not username:
            return jsonify({"error": "Inserisci un nome utente"}), 400
        data = _load_profiles()
        profile = data["profiles"].get(username)
        if not profile:
            return jsonify({"error": "Profilo non trovato"}), 401
        if profile.get("password_hash") and profile["password_hash"] != _hash_pwd(password):
            return jsonify({"error": "Password errata"}), 401
        session["profile"] = username
        data["profiles"][username]["last_login"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["active"] = username
        _save_profiles(data)
        return jsonify({"ok": True, "profile": username})

    @app.route("/api/profiles/logout", methods=["POST"])
    def api_profiles_logout():
        session.pop("profile", None)
        return jsonify({"ok": True})

    @app.route("/api/profiles/create", methods=["POST"])
    def api_profiles_create():
        body = request.get_json() or {}
        username = body.get("username", "").strip()
        password = body.get("password", "")
        if not username:
            return jsonify({"error": "Inserisci un nome utente"}), 400
        data = _load_profiles()
        if username in data["profiles"]:
            return jsonify({"error": "Profilo già esistente"}), 409
        data["profiles"][username] = {
            "password_hash": _hash_pwd(password) if password else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_login": None,
        }
        _save_profiles(data)
        return jsonify({"ok": True, "profile": username})

    @app.route("/api/profiles/delete", methods=["POST"])
    def api_profiles_delete():
        body = request.get_json() or {}
        username = body.get("username", "").strip()
        if not username:
            return jsonify({"error": "Inserisci un nome utente"}), 400
        data = _load_profiles()
        if username not in data["profiles"]:
            return jsonify({"error": "Profilo non trovato"}), 404
        del data["profiles"][username]
        if data.get("active") == username:
            data["active"] = None
        _save_profiles(data)
        return jsonify({"ok": True})

    @app.route("/api/profiles/switch", methods=["POST"])
    def api_profiles_switch():
        body = request.get_json() or {}
        username = body.get("username", "").strip()
        if not username:
            return jsonify({"error": "Inserisci un nome utente"}), 400
        data = _load_profiles()
        if username not in data["profiles"]:
            return jsonify({"error": "Profilo non trovato"}), 404
        session["profile"] = username
        data["active"] = username
        _save_profiles(data)
        return jsonify({"ok": True, "profile": username})

    # ─── Auto-Diagnosis ─────────────────────────────────────────

    @app.route("/api/system/diagnose")
    def api_system_diagnose():
        """Run full platform diagnosis."""
        cfg = load_config()
        from .. import __version__
        results = []
        # 1. Config file exists
        config_ok = Path(CONFIG_FILE).exists()
        results.append({
            "check": "File configurazione",
            "status": "ok" if config_ok else "warn",
            "detail": str(CONFIG_FILE) if config_ok else "Non trovato",
        })
        # 2. Ollama connectivity
        base_url = cfg["local"].get("ollama_base_url", "").strip()
        if not base_url:
            results.append({
                "check": "Ollama",
                "status": "error",
                "detail": "URL Ollama non configurato",
                "fix": "set_local_ollama_url",
            })
        else:
            try:
                import requests
                r = requests.get(f"{base_url}/api/tags", timeout=5)
                ollama_ok = r.status_code == 200
                results.append({
                    "check": "Ollama",
                    "status": "ok" if ollama_ok else "error",
                    "detail": f"{base_url} → {'OK' if ollama_ok else 'Non raggiungibile'}",
                    "fix": "check_ollama_service" if not ollama_ok else None,
                })
            except Exception as e:
                results.append({"check": "Ollama", "status": "error", "detail": str(e), "fix": "check_ollama_service"})
        # 3. Online provider
        is_opencode = cfg["online"]["provider"] == "opencode"
        online_key = bool(cfg["online"]["api_key"])
        opencode_path = Path(cfg["online"].get("opencode_path", "")).exists() if is_opencode else False
        online_ok = online_key or opencode_path
        results.append({
            "check": "Provider Online",
            "status": "ok" if online_ok else "warn",
            "detail": f"{cfg['online']['provider']} → {'Configurato' if online_ok else 'Non configurato'}",
        })
        # 4. Local model
        results.append({
            "check": "Modello Locale",
            "status": "ok" if cfg["local"].get("model") else "warn",
            "detail": cfg["local"].get("model") or "Nessun modello configurato",
            "fix": None if cfg["local"].get("model") else "set_default_model",
        })
        # 5. Data directories
        dirs = {
            "Agenti": AGENTS_DIR,
            "Sessioni": Path.home() / ".config" / "hybrid-coder" / "sessions",
            "Note": Path.home() / ".config" / "hybrid-coder" / "notes",
            "Knowledge": KB_DIR,
        }
        for label, d in dirs.items():
            exists = d.exists()
            results.append({
                "check": f"Directory {label}",
                "status": "ok" if exists else "info",
                "detail": str(d) + (" ✓" if exists else " (da creare)"),
            })
        # 6. Backup system
        _BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        bk_count = len(list(_BACKUPS_DIR.iterdir()))
        results.append({
            "check": "Backup disponibili",
            "status": "ok" if bk_count > 0 else "info",
            "detail": f"{bk_count} backup salvati",
        })
        # 7. Disk space
        try:
            import shutil
            _, _, free = shutil.disk_usage(Path.home())
            free_gb = free / (1024**3)
            results.append({
                "check": "Spazio disco",
                "status": "ok" if free_gb > 1 else "error",
                "detail": f"{free_gb:.1f} GB liberi",
            })
        except Exception as e:
            results.append({"check": "Spazio disco", "status": "warn", "detail": str(e)})
        # 8. Version
        results.append({
            "check": "Versione",
            "status": "ok",
            "detail": f"AI+ v{__version__}",
        })
        errors = sum(1 for r in results if r["status"] == "error")
        warnings = sum(1 for r in results if r["status"] == "warn")
        return jsonify({
            "checks": results,
            "total": len(results),
            "errors": errors,
            "warnings": warnings,
            "healthy": errors == 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    @app.route("/api/system/diagnose/fix", methods=["POST"])
    def api_diagnose_fix():
        """Auto-correzione per problemi rilevati dalla diagnosi."""
        body = request.get_json() or {}
        fix = body.get("fix", "")
        cfg = load_config()
        from ..config import save_config, DEFAULT_CONFIG
        if fix == "set_local_ollama_url":
            cfg["local"]["ollama_base_url"] = DEFAULT_CONFIG["local"]["ollama_base_url"]
            save_config(cfg)
            return jsonify({"ok": True, "message": "URL Ollama reimpostato a http://localhost:11434"})
        if fix == "set_default_model":
            cfg["local"]["model"] = DEFAULT_CONFIG["local"]["model"]
            save_config(cfg)
            return jsonify({"ok": True, "message": f"Modello locale reimpostato a {DEFAULT_CONFIG['local']['model']}"})
        if fix == "check_ollama_service":
            return jsonify({"ok": True, "message": "Verifica che 'ollama serve' sia in esecuzione e raggiungibile."})
        return jsonify({"ok": False, "error": f"Fix sconosciuto: {fix}"}), 400

    # ─── Dependency check ───────────────────────────────────────

    @app.route("/api/system/deps-check")
    def api_system_deps_check():
        """Check installed packages vs PyPI latest versions."""
        import subprocess, sys, json, re
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return jsonify({"ok": False, "error": result.stderr.strip()})
            outdated = json.loads(result.stdout.strip() or "[]")
            return jsonify({
                "ok": True,
                "outdated": [{"name": p["name"], "installed": p["version"], "latest": p["latest_version"]} for p in outdated],
                "total": len(outdated),
            })
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "error": "Timeout"}), 500
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # ─── Self-Update ─────────────────────────────────────────────

    @app.route("/api/system/update", methods=["POST"])
    def api_system_update():
        """Self-update: pip upgrade or AI-assisted self-improvement."""
        import subprocess, sys
        body = request.get_json() or {}
        mode = body.get("mode", "pip")

        if mode == "ai":
            try:
                online = app.config.get("ONLINE")
                if not online:
                    return jsonify({"ok": False, "mode": "ai", "error": "Nessun provider online configurato"}), 400
                prompt = """Sei un assistente di auto-miglioramento. Analizza questo progetto AI+ e suggerisci:
1. Dipendenze obsolete da aggiornare (guarda setup.py)
2. Miglioramenti alla struttura del codice
3. Funzionalità mancanti rispetto a tool simili
4. Bug potenziali o problemi di sicurezza
5. Ottimizzazioni di performance

Formatta la risposta in Markdown con sezioni chiare.
"""
                response = online.generate_chat([{"role": "user", "content": prompt}])
                text = response.get("text", "") if isinstance(response, dict) else str(response)
                return jsonify({
                    "ok": True,
                    "mode": "ai",
                    "analysis": text[:5000],
                    "message": "Analisi completata. Esamina i suggerimenti sopra.",
                })
            except Exception as e:
                return jsonify({"ok": False, "mode": "ai", "error": str(e)}), 500

        if mode == "pip-upgrade-all":
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    return jsonify({"ok": False, "mode": "pip-upgrade-all", "error": result.stderr.strip()})
                outdated = json.loads(result.stdout.strip() or "[]")
                if not outdated:
                    return jsonify({"ok": True, "mode": "pip-upgrade-all", "message": "Tutte le librerie sono già aggiornate."})
                names = [p["name"] for p in outdated]
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + names
                upg_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                success = upg_result.returncode == 0
                return jsonify({
                    "ok": success,
                    "mode": "pip-upgrade-all",
                    "updated": len(names),
                    "stdout": upg_result.stdout[-2000:] if upg_result.stdout else "",
                    "stderr": upg_result.stderr[-2000:] if upg_result.stderr else "",
                    "message": f"Aggiornate {len(names)} librerie." if success else "Aggiornamento fallito.",
                })
            except subprocess.TimeoutExpired:
                return jsonify({"ok": False, "mode": "pip-upgrade-all", "error": "Timeout durante l'aggiornamento"}), 500
            except Exception as e:
                return jsonify({"ok": False, "mode": "pip-upgrade-all", "error": str(e)}), 500

        # Default: pip upgrade (solo ai-plus)
        try:
            from .. import __version__
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "ai-plus"],
                capture_output=True, text=True, timeout=120,
            )
            success = result.returncode == 0
            return jsonify({
                "ok": success,
                "mode": "pip",
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "message": "Aggiornato con successo! Riavvia l'app." if success else "Aggiornamento fallito.",
            })
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "mode": "pip", "error": "Timeout durante l'aggiornamento"}), 500
        except Exception as e:
            return jsonify({"ok": False, "mode": "pip", "error": str(e)}), 500

    # ─── Cache / Memory Management ───────────────────────────────

    @app.route("/api/system/cache", methods=["GET"])
    def api_cache_status():
        mgr = get_resource_manager()
        return jsonify({
            "response_cache": mgr.response_cache.size,
            "response_cache_disk_mb": round(mgr.response_cache.disk_usage_bytes / (1024*1024), 2),
            "embedding_cache": mgr.embedding_cache.size,
            "embedding_cache_disk_mb": round(mgr.embedding_cache.disk_usage_bytes / (1024*1024), 2),
            "pool_active": mgr.pool.active_count,
        })

    @app.route("/api/system/cache/clear", methods=["POST"])
    def api_cache_clear():
        """Cancella solo cache temporanee (response + embedding), NON i dati utente."""
        mgr = get_resource_manager()
        mgr.response_cache.clear()
        mgr.embedding_cache.clear()
        log.info("🗑 Cache cancellata manualmente")
        return jsonify({"ok": True, "message": "Cache cancellata. I dati utente (sessioni, KB, note) sono intatti."})

    @app.route("/api/system/optimize-memory", methods=["POST"])
    def api_optimize_memory():
        """GC forzato + pool drain + report."""
        mgr = get_resource_manager()
        result = mgr.optimize_memory()
        from ..sessions import get_conversation_store
        store = get_conversation_store()
        store.flush()
        result["conversations_flushed"] = True
        log.info(f"🧹 Memory optimized: {result}")
        return jsonify(result)

    # ─── API: Web Search ──────────────────────────────────────────

    @app.route("/api/web/search", methods=["POST"])
    @app.route("/api/search/web", methods=["POST"])
    def api_search_web():
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query richiesta"}), 400

        headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-plus/1.0; +https://github.com/G-DL/ai-plus)"}
        try:
            resp = _requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            return jsonify({"error": f"Errore ricerca: {str(e)}"}), 502

        html = resp.text
        results = []
        for m in re.finditer(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]*)</a>',
            html, re.DOTALL,
        ):
            url = m.group(1)
            title = re.sub(r'<[^>]+>', "", m.group(2)).strip()
            snippet = re.sub(r'<[^>]+>', "", m.group(3)).strip()
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= 10:
                break

        return jsonify({"results": results, "query": query, "count": len(results)})

    @app.route("/api/search/save", methods=["POST"])
    def api_search_save():
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        results = data.get("results", [])
        if not query or not results:
            return jsonify({"error": "Dati insufficienti"}), 400

        from ..resources import KB_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^a-zA-Z0-9_]+', "_", query[:40])
        filename = f"websearch_{safe_name}_{ts}.md"

        lines = [
            f"# Web Search: {query}",
            "",
            "---",
            f"query: \"{query}\"",
            f"date: {datetime.now().isoformat()}",
            f"results: {len(results)}",
            "type: websearch",
            "---",
            "",
        ]
        for i, r in enumerate(results, 1):
            title = r.get("title", "N/A")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"## {i}. [{title}]({url})")
            lines.append("")
            lines.append(snippet)
            lines.append("")

        content = "\n".join(lines)
        filepath = KB_DIR / "wiki" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

        kb = get_knowledge_base()
        kb.add_file(str(filepath))
        kb._save_index()

        return jsonify({
            "saved": True,
            "file": str(filepath),
            "chunks": len(results),
            "title": f"Web Search: {query}",
        })

    @app.route("/api/system/restart", methods=["POST"])
    def api_system_restart():
        """Restart the AI+ service."""
        import time, threading, sys, subprocess
        def _restart():
            time.sleep(1.5)
            log.info("🔄 Restarting AI+ service...")
            try:
                args = [sys.executable, "-m", "hycoder.cli"] + sys.argv[1:]
                log.info(f"Spawning: {args}")
                subprocess.Popen(args, close_fds=True)
            except Exception as e:
                log.error(f"Restart spawn failed: {e}")
            os._exit(0)
        threading.Thread(target=_restart, daemon=False).start()
        return jsonify({"ok": True, "message": "Restarting..."})

    @app.route("/api/github/status", methods=["POST"])
    def api_github_status():
        """Check git status of a directory."""
        data = request.get_json() or {}
        path = data.get("path", ".").strip()
        p = Path(path).expanduser().resolve()

        result = {"git_installed": False, "is_repo": False}

        # Check if git is installed
        import shutil
        if not shutil.which("git"):
            return jsonify(result)

        result["git_installed"] = True
        try:
            r = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
            result["git_version"] = r.stdout.strip()
        except Exception:
            pass

        # Check if path is a git repo
        try:
            r = subprocess.run(["git", "-C", str(p), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, text=True, timeout=5)
            result["is_repo"] = r.returncode == 0 and r.stdout.strip() == "true"
        except Exception:
            pass

        if result["is_repo"]:
            try:
                r = subprocess.run(["git", "-C", str(p), "remote", "get-url", "origin"],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    result["remote_url"] = r.stdout.strip()
            except Exception:
                pass
            try:
                r = subprocess.run(["git", "-C", str(p), "rev-parse", "--abbrev-ref", "HEAD"],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    result["branch"] = r.stdout.strip()
            except Exception:
                pass
            try:
                r = subprocess.run(["git", "-C", str(p), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5)
                result["files_to_commit"] = len([l for l in r.stdout.split("\n") if l.strip()])
            except Exception:
                pass
            try:
                r = subprocess.run(["git", "-C", str(p), "log", "-1", "--oneline"],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    result["last_commit"] = r.stdout.strip()
            except Exception:
                pass

        return jsonify(result)

    @app.route("/api/github/export", methods=["POST"])
    def api_github_export():
        """Export (init + add + commit + push) a directory to GitHub."""
        data = request.get_json() or {}
        repo_url = data.get("repo_url", "").strip()
        branch = data.get("branch", "main").strip()
        token = data.get("token", "").strip()
        export_path = data.get("export_path", "").strip() or "."
        commit_msg = data.get("commit_msg", "auto: aggiornamento AI+")

        if not repo_url:
            return jsonify({"error": "URL repository richiesto"}), 400

        p = Path(export_path).expanduser().resolve()
        if not p.is_dir():
            return jsonify({"error": f"Directory non trovata: {p}"}), 400

        import shutil
        if not shutil.which("git"):
            return jsonify({"error": "Git non installato. Installa git per usare questa funzione."}), 400

        log_lines = []
        def log_msg(m):
            log_lines.append(m)
            log.info(f"[github-export] {m}")

        try:
            # 1. Init repo if not already
            r = subprocess.run(["git", "-C", str(p), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or r.stdout.strip() != "true":
                log_msg("📦 Inizializzazione repository git...")
                subprocess.run(["git", "-C", str(p), "init"], capture_output=True, timeout=10, check=True)
                log_msg("✅ git init completato")
            else:
                log_msg("✅ Repository già inizializzato")

            # 2. Set remote origin
            r = subprocess.run(["git", "-C", str(p), "remote", "get-url", "origin"],
                               capture_output=True, text=True, timeout=5)
            current_remote = r.stdout.strip() if r.returncode == 0 else ""
            if current_remote != repo_url:
                if current_remote:
                    log_msg("🔄 Aggiornamento remote origin...")
                    subprocess.run(["git", "-C", str(p), "remote", "remove", "origin"],
                                   capture_output=True, timeout=10)
                log_msg("🔗 Aggiunta remote origin...")
                subprocess.run(["git", "-C", str(p), "remote", "add", "origin", repo_url],
                               capture_output=True, timeout=10, check=True)
                log_msg("✅ Remote origin configurata")
            else:
                log_msg("✅ Remote origin già configurata")

            # 3. Configure git user (for commits in fresh repos)
            for key, val in [("user.name", "AI+ Export"), ("user.email", "ai-plus@local")]:
                subprocess.run(["git", "-C", str(p), "config", key, val],
                               capture_output=True, timeout=5)

            # 4. Add all files
            log_msg("📄 Aggiunta file...")
            subprocess.run(["git", "-C", str(p), "add", "-A", "--force"],
                           capture_output=True, timeout=30, check=True)

            # 5. Always commit (allow-empty so re-runs also push)
            log_msg("💾 Commit in corso...")
            r = subprocess.run(
                ["git", "-C", str(p), "commit", "--allow-empty", "-m", commit_msg],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                log_msg(f"⚠ Commit fallito: {r.stderr.strip() or r.stdout.strip()[:200]}")
            else:
                log_msg("✅ Commit creato")

            # 7. Push
            log_msg("⬆ Push su GitHub...")
            # Prevent git from hanging on auth prompts
            git_env = {**os.environ,
                       "GIT_TERMINAL_PROMPT": "0",
                       "GIT_ASKPASS": "echo"}
            push_url = repo_url
            masked_url = repo_url
            if token and repo_url.startswith("https://"):
                push_url = repo_url.replace("https://", f"https://{token}@")
                masked_url = repo_url.replace("https://", "https://***@")
            elif token and repo_url.startswith("http://"):
                push_url = repo_url.replace("http://", f"http://{token}@")
                masked_url = repo_url.replace("http://", "http://***@")
            else:
                masked_url = push_url

            log_msg(f"Push URL: {masked_url}")
            log_msg(f"Branch: {branch}")

            r = subprocess.run(
                ["git", "-C", str(p), "push", "-u", push_url, branch],
                capture_output=True, text=True, timeout=120,
                env=git_env,
            )
            if r.returncode != 0:
                err = r.stderr.strip() or r.stdout.strip()
                log_msg(f"❌ Push fallito (exit {r.returncode}): {err[:500]}")
                # Provide user-friendly hints
                hint = ""
                if "Repository not found" in err:
                    hint = " — Il repository non esiste o non hai accesso. Verifica l'URL."
                elif "Authentication failed" in err or "auth" in err.lower():
                    hint = " — Autenticazione fallita. Per HTTPS usa un Personal Access Token."
                elif "Permission denied" in err:
                    hint = " — Permesso negato. Per SSH verifica la chiave pubblica, per HTTPS usa un token."
                elif "could not read Username" in err or "could not read Password" in err:
                    hint = " — Git sta chiedendo credenziali interattive. Per HTTPS usa un Personal Access Token."
                elif "timeout" in err.lower() or "timed out" in err.lower():
                    hint = " — Timeout di rete. Verifica la connessione a github.com."
                log_msg(hint.strip() if hint else "")
                return jsonify({"error": f"Push fallito: {err[:300]}{hint}", "log": "\n".join(log_lines)}), 400

            log_msg("✅ Push completato con successo!")
            return jsonify({"ok": True, "message": f"Repository esportato su {branch}", "log": "\n".join(log_lines)})

        except subprocess.TimeoutExpired as e:
            return jsonify({"error": f"Timeout: {e.cmd}", "log": "\n".join(log_lines)}), 400
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "")
            return jsonify({"error": f"Git error: {err[:500]}", "log": "\n".join(log_lines)}), 400
        except Exception as e:
            return jsonify({"error": str(e)[:500], "log": "\n".join(log_lines)}), 400

    return app


def _check_ollama(cfg):
    import requests
    try:
        r = requests.get(f"{cfg['local']['ollama_base_url']}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _count_cache():
    from ..providers import _local_providers, _online_providers
    count = 0
    for p in list(_local_providers.values()):
        count += len(p.cache._cache)
    for p in list(_online_providers.values()):
        count += len(p.cache._cache)
    return count
