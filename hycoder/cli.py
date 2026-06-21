#!/usr/bin/env python3
"""
AI+: CLI intelligente con AI locale (Ollama) e online (OpenAI/OpenRouter).
Routing automatico ottimizzato, cache, keep-alive, metriche in tempo reale.
Integrato con opencode come sub-agente @AI+.
"""

import sys
import os
import signal
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from .config import load_config, CONFIG_FILE, save_config
from .router import SmartRouter
from .providers import get_local_provider, get_online_provider, clear_caches
from .tracker import SessionTracker
from .knowledge import get_knowledge_base, build_rag_context
from .project import (create_project, list_projects, get_project,
                      run_project, test_project, delete_project,
                      preview_project, list_templates, run_code,
                      generate_project_from_prompt)

console = Console()
ERR = Console(stderr=True)


# ─── Web server ──────────────────────────────────────────────────────────
def _serve_web(cfg):
    from .web.app import create_app
    import webbrowser
    import socket

    app = create_app(cfg)
    port = int(os.environ.get("PORT", 8081))

    url = f"http://127.0.0.1:{port}"

    console.print(f"[bold cyan]  AI+ web[/]")
    console.print(f"  [dim]Dashboard:[/] {url}")
    console.print(f"  [dim]Ollama:[/] {cfg['local']['model']} {'✅' if _check_ollama(cfg) else '❌'}")
    console.print(f"  [dim]Online:[/] {cfg['online']['model'] or '-'} {'✅' if cfg['online']['api_key'] else '❌'}")
    console.print(f"  [dim]Ctrl+C[/] per fermare\n")

    webbrowser.open(url)
    app.run(debug=False, host="0.0.0.0", port=port)


def _check_ollama(cfg):
    import requests as r
    try:
        return r.get(f"{cfg['local']['ollama_base_url']}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def format_tokens(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def render_stats(tracker, router_decision=None, router_score=None):
    s = tracker.summary_dict()
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Proprietà", style="dim")
    table.add_column("Valore")

    table.add_row("[bold]Sessione[/]", f"{format_time(s['session_duration_s'])}")
    table.add_row("", "")

    l = s["local"]
    table.add_row("[green]Locale (Ollama)[/]", f"{l['calls']} chiamate")
    table.add_row("", f"  Tokens in:  {format_tokens(l['tokens_in'])}")
    table.add_row("", f"  Tokens out: {format_tokens(l['tokens_out'])}")
    table.add_row("", f"  Tokens tot: {format_tokens(l['tokens_total'])}")
    table.add_row("", f"  Tempo:      {format_time(l['time_s'])}")
    table.add_row("", "")

    o = s["online"]
    table.add_row("[blue]Online (Cloud)[/]", f"{o['calls']} chiamate")
    table.add_row("", f"  Tokens in:  {format_tokens(o['tokens_in'])}")
    table.add_row("", f"  Tokens out: {format_tokens(o['tokens_out'])}")
    table.add_row("", f"  Tokens tot: {format_tokens(o['tokens_total'])}")
    table.add_row("", f"  Tempo:      {format_time(o['time_s'])}")
    table.add_row("", "")

    t = s["total"]
    table.add_row("[bold yellow]Totali[/]", f"{t['calls']} chiamate")
    table.add_row("", f"  Tokens tot: [bold]{format_tokens(t['tokens_total'])}[/]")
    table.add_row("", f"  Tempo tot:  [bold]{format_time(t['time_s'])}[/]")
    if s["cost"] > 0:
        table.add_row("", f"  Costo:      ${s['cost']:.4f}")
    table.add_row("", "")

    table.add_row("[magenta]Progresso[/]", f"{s['progress_pct']}%")
    table.add_row("", f"  Stimato:    {format_time(s['total_estimate_s'])} totali")
    table.add_row("", f"  Rimasto:    {format_time(s['remaining_s'])}")

    if router_decision:
        label = "ONLINE" if router_decision == "online" else "LOCALE"
        color = "blue" if router_decision == "online" else "green"
        table.add_row("", "")
        table.add_row(f"[{color}]Ultima rotta[/]", f"[{color}]{label}[/] (score: {router_score})")

    return table


def handle_sigint(sig, frame):
    console.print("\n[yellow]Uscita...[/]")
    sys.exit(0)


@click.group(invoke_without_command=True)
@click.option("--local", "force_local", is_flag=True, help="Forza solo AI locale")
@click.option("--online", "force_online", is_flag=True, help="Forza solo AI online")
@click.option("--model", "-m", help="Modello locale (es. qwen3.5:4b)")
@click.option("--online-model", help="Modello online (es. gpt-4o-mini)")
@click.option("--mode", type=click.Choice(["auto", "local", "online"]), help="Modalità router")
@click.option("--no-cache", is_flag=True, help="Disabilita cache risposte")
@click.option("--json", "json_out", is_flag=True, help="Output JSON (per integrazione)")
@click.version_option(version="0.4.0", prog_name="AI+", message="AI+ v%(version)s — AI ibrida locale + cloud")
@click.pass_context
def cli(ctx, force_local, force_online, model, online_model, mode, no_cache, json_out):
    """AI+: AI ibrida locale + cloud con routing intelligente."""
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config()

    if force_local:
        cfg["router"]["always_local"] = True
    if force_online:
        cfg["router"]["always_online"] = True
    if model:
        cfg["local"]["model"] = model
    if online_model:
        cfg["online"]["model"] = online_model
    if mode:
        cfg["router"]["mode"] = mode
    if no_cache:
        clear_caches()

    router = SmartRouter(cfg)
    local_provider = get_local_provider(cfg)

    online_avail = bool(cfg["online"]["api_key"])
    if online_avail:
        online_provider = get_online_provider(cfg)
    else:
        online_provider = None
        if cfg["router"]["mode"] in ("auto", "online") and not cfg["router"]["always_local"]:
            cfg["router"]["always_local"] = True
            ERR.print("[yellow]Nessuna API key configurata. Solo modalità locale.[/]")
            ERR.print(f"[dim]Usa: ai-plus set online.api_key <chiave>[/]")

    tracker = SessionTracker(cfg)

    ctx.obj = {
        "cfg": cfg,
        "router": router,
        "local": local_provider,
        "online": online_provider,
        "tracker": tracker,
        "online_avail": online_avail,
        "json_out": json_out,
    }

    if ctx.invoked_subcommand is None:
        if json_out:
            sys.exit(0)
        interactive(ctx.obj)


@cli.command()
@click.argument("prompt", nargs=-1, required=False)
@click.option("--system", "-s", help="System prompt / istruzioni agente")
@click.pass_context
def chat(ctx, prompt, system):
    """Avvia chat interattiva o invia un singolo prompt."""
    state = ctx.obj
    if prompt:
        handle_single(state, " ".join(prompt), system=system)
    else:
        interactive(state)


@cli.command()
@click.option("--port", default=8081, help="Porta del server web")
@click.pass_context
def web(ctx, port):
    """Avvia la piattaforma web AI+."""
    os.environ["PORT"] = str(port)
    state = ctx.obj
    _serve_web(state["cfg"])


@cli.command()
@click.option("--port", default=8081, help="Porta del server web")
@click.pass_context
def serve(ctx, port):
    """Alias per 'web'. Avvia la piattaforma web AI+."""
    os.environ["PORT"] = str(port)
    state = ctx.obj
    _serve_web(state["cfg"])


@cli.command()
@click.option("--output", "-o", default="ai-plus-install-pack.zip",
              help="Percorso del file zip da generare")
@click.pass_context
def pack(ctx, output):
    """Genera il pacchetto di installazione (setupAI+ + sorgenti)."""
    from .install_pack import generate_pack, WEB_PORT
    out = Path(output).expanduser().resolve()
    zip_path = generate_pack(out.parent, port=WEB_PORT)

    console.print(f"[bold green]✓[/] Pacchetto generato: [bold]{zip_path}[/]")
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        count = len(zf.namelist())
    console.print(f"  {count} file inclusi")
    sz = zip_path.stat().st_size
    console.print(f"  Dimensione: {sz / 1024:.1f} KB ({sz / 1024 / 1024:.2f} MB)")
    console.print(f"\n  [bold]Per installare su un'altra macchina:[/]")
    console.print(f"    1. Copia {zip_path.name} sulla macchina di destinazione")
    console.print(f"    2. Estrai:  [bold cyan]unzip {zip_path.name}[/]")
    console.print(f"    3. Esegui:  [bold cyan]./install.sh[/] (Linux/macOS)  oppure  [bold cyan]install.bat[/] (Windows)")
    console.print(f"")
    console.print(f"  [bold]Oppure usa setupAI+ direttamente:[/]")
    console.print(f"    [bold cyan]./setupAI+[/] (Linux/macOS)  oppure  [bold cyan]setupAI+.bat[/] (Windows)")
    console.print(f"")
    console.print(f"  [bold]L'installer automatico:[/]")
    console.print(f"    - Crea ambiente virtuale isolato")
    console.print(f"    - Installa AI+ e tutte le librerie")
    console.print(f"    - Configura CLI: hy e ai-plus")
    console.print(f"    - Avvia server web su http://localhost:{WEB_PORT}/")


@cli.command()
def config():
    """Mostra configurazione corrente."""
    cfg = load_config(force=True)
    if CONFIG_FILE.exists():
        console.print(f"[dim]File: {CONFIG_FILE}[/]\n")
    else:
        console.print("[yellow]File config non trovato. Uso valori predefiniti.[/]")

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Chiave", style="dim")
    table.add_column("Valore")

    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, key)
            else:
                val = str(v)
                if "key" in k.lower() and v:
                    val = "***" + str(v)[-4:]
                table.add_row(key, val)

    flatten(cfg)
    console.print(table)


@cli.command()
@click.argument("key", nargs=1)
@click.argument("value", nargs=1)
@click.pass_context
def set(ctx, key, value):
    """Imposta una chiave di configurazione (es. online.api_key sk-xxx)."""
    cfg = load_config(force=True)
    keys = key.split(".")
    target = cfg
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value
    save_config(cfg)
    console.print(f"[green]✓[/] {key} impostato")


@cli.command()
def clearcache():
    """Pulisce la cache delle risposte."""
    clear_caches()
    console.print("[green]✓ Cache pulita[/]")


@cli.command()
@click.option("--system", "-s", help="System prompt / istruzioni agente")
@click.argument("prompt", nargs=-1, required=True)
@click.pass_context
def generate(ctx, prompt, system):
    """Invia un prompt e ricevi JSON strutturato (per opencode/tool)."""
    state = ctx.obj
    state["json_out"] = True
    handle_single(state, " ".join(prompt), system=system)


@cli.command()
@click.pass_context
def agent(ctx):
    """Output JSON per integrazione con opencode."""
    state = ctx.obj
    tracker = state["tracker"]
    s = tracker.summary_dict()
    s["mode"] = state["cfg"]["router"]["mode"]
    s["local_model"] = state["cfg"]["local"]["model"]
    s["online_model"] = state["cfg"]["online"]["model"]
    s["online_avail"] = state["online_avail"]
    s["cache_active"] = True
    print(json.dumps(s, indent=2))


# ─── Learn (RAG Knowledge Base) ─────────────────────────────────────────

@cli.group()
def learn():
    """Gestisci la knowledge base (RAG). Aggiungi documenti o URL da cui AI+ può imparare."""


@learn.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--pattern", default="*.*", help="Glob pattern (es. *.md, *.py)")
@click.option("--no-recursive", is_flag=True, help="Non ricorsivo")
@click.pass_context
def from_dir(ctx, path, pattern, no_recursive):
    """Indicizza tutti i file in una directory."""
    from .knowledge import get_knowledge_base
    kb = get_knowledge_base()
    with console.status("[cyan]Indicizzazione file..."):
        result = kb.add_directory(path, pattern=pattern, recursive=not no_recursive)
    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return
    console.print(f"[green]✓[/] {result['files_processed']} file elaborati, {result['chunks_added']} chunk aggiunti")
    if result["errors"]:
        console.print(f"[yellow]  {len(result['errors'])} errori[/]")


@learn.command()
@click.argument("url")
@click.pass_context
def from_url(ctx, url):
    """Scarica e indicizza una risorsa web."""
    from .knowledge import get_knowledge_base
    kb = get_knowledge_base()
    with console.status(f"[cyan]Scaricamento {url}..."):
        result = kb.add_url(url)
    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return
    if result.get("fallback"):
        console.print(f"[yellow]↳[/] Scaricamento non riuscito, indicizzati risultati ricerca web")
    console.print(f"[green]✓[/] {result['chunks_added']} chunk da '{result.get('title', url)}'")


@learn.command()
def status():
    """Mostra lo stato della knowledge base."""
    from .knowledge import get_knowledge_base
    from rich.table import Table
    kb = get_knowledge_base()
    s = kb.status()

    if s["total_chunks"] == 0:
        console.print("[yellow]Knowledge base vuota.[/]")
        console.print("  [dim]Usa: ai-plus learn from-dir <path>[/]")
        console.print("  [dim]Usa: ai-plus learn from-url <url>[/]")
        return

    console.print(f"\n[bold]Knowledge Base[/]")
    console.print(f"  [dim]Chunk totali:[/] {s['total_chunks']}")
    console.print(f"  [dim]Fonti:[/] {s['total_sources']}")

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Fonte", style="cyan", no_wrap=True)
    table.add_column("Tipo")
    table.add_column("Chunk")
    table.add_column("Anteprima")

    for src in s["sources"]:
        src_path = src["source"]
        type_ = src["type"]
        chunks = str(src["chunks"])
        preview = src_path.split("/")[-1] if "/" in src_path else src_path[:50]
        table.add_row(src_path[:60], type_, chunks, preview)

    console.print(table)
    console.print()


@learn.command()
def clear():
    """Pulisce la knowledge base."""
    from .knowledge import reset_knowledge_base
    reset_knowledge_base()
    console.print("[green]✓ Knowledge base pulita[/]")


@cli.command()
@click.pass_context
def kb(ctx):
    """Alias: mostra stato knowledge base (come 'learn status')."""
    from .knowledge import get_knowledge_base
    from rich.table import Table
    kb = get_knowledge_base()
    s = kb.status()
    if s["total_chunks"] == 0:
        console.print("[yellow]Knowledge base vuota.[/]")
    else:
        console.print(f"[green]{s['total_chunks']}[/] chunk da [green]{s['total_sources']}[/] fonti")


@cli.command()
@click.option("--gc", "do_gc", is_flag=True, help="Esegue garbage collection")
@click.pass_context
def resources(ctx, do_gc):
    """Mostra risorse hardware (CPU, RAM, cache, pool)."""
    from .resources import get_resource_manager
    from rich.table import Table
    mgr = get_resource_manager()
    snap = mgr.snapshot()

    if do_gc:
        import gc
        before = len(gc.get_objects())
        gc.collect()
        after = len(gc.get_objects())
        console.print(f"[green]GC: {before} → {after} oggetti[/]")

    console.print(f"\n[bold]Risorse Sistema[/]")
    console.print(f"  [dim]Memoria processo:[/] {snap['process_memory_mb']:.1f} MB")
    console.print(f"  [dim]CPU:[/] {snap['cpu_percent']:.1f}%")
    console.print(f"  [dim]Uptime:[/] {snap['uptime_s']:.0f}s")
    console.print(f"  [dim]CPU cores:[/] {snap['cpu_count']}")
    console.print(f"  [dim]Oggetti Python:[/] {snap['gc_objects']}")

    console.print(f"\n[bold]Cache e Pool[/]")
    console.print(f"  [dim]Cache risposte:[/] {snap['limits']['cache_entries']} entry ({snap['limits']['cache_disk_mb']} MB su disco)")
    console.print(f"  [dim]Thread pool attivi:[/] {snap['limits']['pool_active']}")

    o = snap.get("ollama", {})
    if o.get("running"):
        console.print(f"\n[green]Ollama:[/] {o['models_count']} modelli, {o.get('total_size_gb', 0)} GB")
        for m in o.get("models", []):
            console.print(f"  · {m}")
    else:
        console.print(f"\n[yellow]Ollama:[/] non raggiungibile")


# ─── Project ───────────────────────────────────────────────────────────

@cli.group()
def project():
    """Gestisci progetti (scaffold, test, esegui, preview)."""


@project.command()
@click.argument("name")
@click.option("--template", "-t", default="python", help="Template (python, html, node-js, python-flask)")
@click.option("--path", "-p", help="Percorso personalizzato (default: ~/.config/ai-plus/projects/<name>)")
def create(name, template, path):
    """Crea un nuovo progetto da template."""
    try:
        meta = create_project(name, template=template, path=path)
        console.print(f"[green]✓[/] Progetto creato: [cyan]{meta['name']}[/]")
        console.print(f"  [dim]Percorso:[/] {meta['path']}")
        console.print(f"  [dim]Template:[/] {meta['template']}")
        console.print(f"  [dim]Run:[/] {meta.get('run_cmd', '-')}")
        console.print(f"  [dim]Test:[/] {meta.get('test_cmd', '-')}")
    except FileExistsError as e:
        ERR.print(f"[red]{e}[/]")
    except ValueError as e:
        ERR.print(f"[red]{e}[/]")


@project.command()
@click.argument("name", required=False, default=None)
def list(name):
    """Elenca progetti o mostra dettagli di uno specifico."""
    if name:
        try:
            p = get_project(name)
            console.print(f"[bold]{p['name']}[/]")
            console.print(f"  [dim]Template:[/] {p['template']}")
            console.print(f"  [dim]Percorso:[/] {p['path']}")
            console.print(f"  [dim]Creato:[/] {p['created']}")
            console.print(f"  [dim]Run:[/] {p.get('run_cmd', '-')}")
            console.print(f"  [dim]Test:[/] {p.get('test_cmd', '-')}")
        except FileNotFoundError:
            ERR.print(f"[red]Progetto '{name}' non trovato[/]")
        return

    projects = list_projects()
    if not projects:
        console.print("[dim]Nessun progetto. Crealo con: ai-plus project create <nome>[/]")
        return
    table = Table(show_header=True, box=box.SIMPLE)
    table.add_column("Nome", style="cyan")
    table.add_column("Template", style="dim")
    table.add_column("Percorso")
    for p in projects:
        table.add_row(p["name"], p["template"], p["path"])
    console.print(table)


@project.command()
@click.argument("name")
def run(name):
    """Esegue il progetto e mostra output."""
    try:
        r = run_project(name)
        console.print(f"[bold]Esecuzione:[/] {r['command']}")
        if r["stdout"]:
            console.print(r["stdout"].rstrip())
        if r["stderr"]:
            ERR.print(f"[yellow]{r['stderr'].rstrip()}[/]")
        console.print(f"[dim]→ exit code: {r['returncode']}[/]")
        if r["success"]:
            console.print("[green]✓ Successo[/]")
        else:
            ERR.print("[red]✗ Fallito[/]")
    except Exception as e:
        ERR.print(f"[red]{e}[/]")


@project.command()
@click.argument("name")
def test(name):
    """Esegue i test del progetto."""
    try:
        r = test_project(name)
        console.print(f"[bold]Test:[/] {r['command']}")
        if r["stdout"]:
            console.print(r["stdout"].rstrip())
        if r["stderr"]:
            ERR.print(f"[yellow]{r['stderr'].rstrip()}[/]")
        if r["success"]:
            console.print("[green]✓ Test passati[/]")
        else:
            ERR.print("[red]✗ Test falliti[/]")
    except Exception as e:
        ERR.print(f"[red]{e}[/]")


@project.command()
@click.argument("name")
@click.option("--file", "-f", help="Mostra solo un file specifico (relativo)")
def preview(name, file):
    """Mostra preview di progetto: codice + output in stile split-view."""
    try:
        p = preview_project(name, file=file)
        console.print(f"[bold cyan]{p['project']['name']}[/] ([dim]{p['project']['template']}[/])\n")

        for rel, src in p["sources"].items():
            console.print(f"[underline]{rel}[/]")
            from rich.syntax import Syntax
            s = Syntax(src, "python" if rel.endswith(".py") else "javascript" if rel.endswith(".js") else "html" if rel.endswith(".html") else "css" if rel.endswith(".css") else "bash", theme="monokai", line_numbers=True)
            console.print(s)

        if p["output"]:
            console.print(f"\n[bold]Output:[/]")
            if p["output"]["stdout"]:
                console.print(f"[green]{p['output']['stdout'].rstrip()}[/]")
            if p["output"]["stderr"]:
                console.print(f"[red]{p['output']['stderr'].rstrip()}[/]")
            console.print(f"[dim]→ exit code: {p['output']['returncode']}[/]")
    except Exception as e:
        ERR.print(f"[red]{e}[/]")


@project.command()
@click.argument("name")
def delete(name):
    """Elimina un progetto."""
    try:
        p = delete_project(name)
        console.print(f"[green]✓[/] Progetto [cyan]{p['name']}[/] eliminato")
    except Exception as e:
        ERR.print(f"[red]{e}[/]")


@project.command(name="list-templates")
def list_templates_cmd():
    """Elenca i template disponibili."""
    for t in list_templates():
        console.print(f"  · [cyan]{t}[/]")


@project.command()
@click.argument("prompt")
@click.option("--name", "-n", help="Nome suggerito per il progetto")
@click.option("--source", "-s", default="auto", help="Provider: local, online, auto")
def generate(prompt, name, source):
    """Genera un progetto AI da descrizione testuale."""
    import json as _json
    cfg = load_config()
    provider = get_online_provider(cfg)
    if source == "local":
        provider = get_local_provider(cfg)

    console.print("[dim]Analizzo descrizione e genero progetto...[/]")
    try:
        meta = generate_project_from_prompt(prompt, provider.generate_chat, name)
        console.print(f"[green]✓[/] Progetto generato: [cyan]{meta['name']}[/]")
        console.print(f"  [dim]Descrizione:[/] {meta.get('description', '-')}")
        if meta.get("tech_stack"):
            console.print(f"  [dim]Tecnologie:[/] {', '.join(meta['tech_stack'])}")
        console.print(f"  [dim]File creati:[/] {meta['files_count']}")
        console.print(f"  [dim]Percorso:[/] {meta['path']}")
        console.print(f"  [dim]Run:[/] {meta.get('run_cmd', '-')}")
        console.print(f"  [dim]Test:[/] {meta.get('test_cmd', '-')}")
    except _json.JSONDecodeError as e:
        ERR.print(f"[red]Errore risposta AI (JSON non valido): {e}[/]")
    except Exception as e:
        ERR.print(f"[red]{e}[/]")

# ─── Notes (Obsidian-style) ─────────────────────────────────────────────

@cli.group()
def note():
    """Gestisci note in stile Obsidian (wiki-link, tag, grafo)."""


@note.command()
@click.argument("title")
@click.option("--body", "-b", help="Testo della nota")
@click.option("--tags", "-t", help="Tag separati da virgola")
@click.pass_context
def create(ctx, title, body, tags):
    """Crea una nuova nota."""
    from .notes import get_note_store
    ns = get_note_store()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    note = ns.create(title, body=body or "", tags=tag_list)
    console.print(f"[green]✓[/] Nota creata: [cyan]{note['title']}[/] ({note['slug']})")


@note.command()
@click.argument("slug", required=False)
@click.pass_context
def list(ctx, slug):
    """Elenco note o dettaglio."""
    from .notes import get_note_store
    from rich.table import Table
    ns = get_note_store()
    if slug:
        note = ns.get(slug)
        if not note:
            console.print("[red]Nota non trovata[/]")
            return
        console.print(f"\n[bold]{note['title']}[/]")
        console.print(f"  [dim]Slug:[/] {note['slug']}")
        console.print(f"  [dim]Tag:[/] {', '.join(note.get('tags', [])) or '-'}")
        console.print(f"  [dim]Link uscenti:[/] {', '.join(note.get('links', [])) or '-'}")
        console.print(f"  [dim]Backlink:[/] {len(note.get('backlinks', []))}")
        if note.get("body"):
            from rich.markdown import Markdown
            console.print()
            console.print(Markdown(note["body"][:1000]))
        return

    notes = ns.list_all()
    if not notes:
        console.print("[yellow]Nessuna nota. Crea con: ai-plus note create <titolo>[/]")
        return
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Titolo", style="cyan")
    table.add_column("Tag", style="dim")
    table.add_column("Link")
    table.add_column("Modificato")
    for n in notes:
        table.add_row(
            n["title"][:40],
            ", ".join(n.get("tags", [])[:3]),
            str(len(n.get("links", []))),
            n.get("modified", "")[:10],
        )
    console.print(table)


@note.command()
@click.argument("slug")
@click.argument("body", required=False)
@click.option("--tags", "-t", help="Tag separati da virgola")
@click.pass_context
def edit(ctx, slug, body, tags):
    """Modifica una nota (body o tag)."""
    from .notes import get_note_store
    ns = get_note_store()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    note = ns.update(slug, body=body, tags=tag_list)
    if note:
        console.print(f"[green]✓[/] Nota aggiornata: {note['title']}")
    else:
        console.print("[red]Nota non trovata[/]")


@note.command()
@click.argument("slug")
@click.pass_context
def delete(ctx, slug):
    """Elimina una nota."""
    from .notes import get_note_store
    ns = get_note_store()
    if ns.delete(slug):
        console.print(f"[green]✓[/] Nota eliminata: {slug}")
    else:
        console.print("[red]Nota non trovata[/]")


@note.command()
def graph():
    """Mostra grafo connessioni tra note."""
    from .notes import get_note_store
    from rich.table import Table
    ns = get_note_store()
    g = ns.graph()
    console.print(f"\n[bold]Grafo Note[/]")
    console.print(f"  [dim]Nodi:[/] {len(g['nodes'])}  [dim]Archi:[/] {len(g['edges'])}")
    if g["edges"]:
        table = Table(show_header=True, box=None)
        table.add_column("Da", style="cyan")
        table.add_column("→", style="dim")
        table.add_column("A", style="green")
        for e in g["edges"][:20]:
            table.add_row(e["source"], "→", e["target"])
        console.print(table)


# ─── Interactive ─────────────────────────────────────────────────────────

def interactive(state):
    console.clear()
    console.print(banner())
    console.print()

    if state["online_avail"]:
        console.print("  [dim]Locale:[/] [green]Ollama[/]  ·  [dim]Online:[/] [blue]disponibile[/]")
    else:
        console.print("  [dim]Locale:[/] [green]Ollama[/]  ·  [dim]Online:[/] [yellow]non configurato[/]")
    console.print(f"  [dim]Modello locale:[/] {state['cfg']['local']['model']}")
    if state["online_avail"]:
        console.print(f"  [dim]Modello online:[/] {state['cfg']['online']['model']}")
    console.print(f"  [dim]Router:[/] {state['cfg']['router']['mode']}  ·  [dim]Cache:[/] attiva  ·  [dim]I/O:[/] differito")
    console.print()
    console.print("  [dim]La conversazione mantiene la storia tra un messaggio e l'altro.[/]")
    console.print()

    # Storia conversazione persistente
    history = []

    while True:
        try:
            prompt = console.input("[bold cyan]┃[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q", "esci"):
            break
        if prompt.lower() in ("stats", "s"):
            show_stats(state)
            continue
        if prompt.lower() in ("clear", "c"):
            console.clear()
            console.print(banner())
            history.clear()
            console.print("  [dim]Conversazione resettata.[/]")
            console.print()
            continue
        if prompt.lower() in ("help", "h", "?"):
            show_help()
            continue
        if prompt.lower().startswith("/"):
            handle_slash(prompt[1:], state)
            continue

        handle_single_chat(state, prompt, history)

    state["tracker"].flush()
    show_stats(state)
    console.print("\n[green]Arrivederci![/]")


def handle_single_chat(state, prompt, history):
    """Invia prompt con storia conversazione, aggiorna history."""
    router = state["router"]
    tracker = state["tracker"]
    json_out = state.get("json_out", False)

    # RAG: arricchisce prompt con contesto
    ctx = build_rag_context(prompt)
    enriched_prompt = prompt
    rag_used = False
    if ctx:
        enriched_prompt = f"{prompt}\n\n{ctx}"
        rag_used = True

    decision, score = router.decide(prompt)

    if decision == "online" and not state["online_avail"]:
        decision = "local"

    provider = state["online"] if decision == "online" else state["local"]

    messages = list(history)
    messages.append({"role": "user", "content": enriched_prompt})

    spinner_colors = {"online": "blue", "local": "green"}

    with console.status(f"[{spinner_colors[decision]}]Elaborazione {decision}...", spinner="dots"):
        result = provider.generate_chat(messages)

    # Aggiorna storia
    history.append({"role": "user", "content": prompt})
    if not result.text.startswith("[ERRORE"):
        history.append({"role": "assistant", "content": result.text})
    if len(history) > 40:
        history[:] = history[-40:]

    tracker.record(prompt, result)

    if json_out:
        s = tracker.summary_dict()
        click.echo(json.dumps({
            "response": result.text,
            "source": result.source,
            "cached": result.cached,
            "tokens": result.tokens_total,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "time_s": result.time_s,
            "model": result.model,
            "complexity_score": score,
            "routing_decision": decision,
            "rag_used": rag_used,
            "history_len": len(history),
            "session": s,
        }, indent=2))
        return

    rag_tag = " [dim](RAG)[/]" if rag_used else ""
    cache_tag = " [dim](cache)[/]" if result.cached else ""
    ctx_tag = f" [dim]({len(history)//2} msg)[/]" if len(history) > 2 else ""

    console.print()
    md = Markdown(result.text)
    title = f"[{spinner_colors[decision]}]{decision.upper()} · {format_tokens(result.tokens_total)} token · {result.time_s:.1f}s{_src_icon(result.source)}{cache_tag}{rag_tag}{ctx_tag}[/]"
    console.print(Panel(md, border_style=spinner_colors[decision], title=title))
    console.print()


def handle_single(state, prompt, system=None):
    router = state["router"]
    tracker = state["tracker"]
    json_out = state.get("json_out", False)

    # RAG: arricchisce prompt con contesto dalla knowledge base
    ctx = build_rag_context(prompt)
    enriched_prompt = prompt
    rag_used = False
    if ctx:
        enriched_prompt = f"{prompt}\n\n{ctx}"
        rag_used = True

    decision, score = router.decide(prompt)

    if decision == "online" and not state["online_avail"]:
        decision = "local"

    if decision == "online":
        provider = state["online"]
        label = "[bold blue]ONLINE[/]"
    else:
        provider = state["local"]
        label = "[bold green]LOCALE[/]"

    # Costruisce messaggi chat
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": enriched_prompt})

    spinner_colors = {"online": "blue", "local": "green"}

    with console.status(f"[{spinner_colors[decision]}]Elaborazione {decision}...", spinner="dots"):
        result = provider.generate_chat(messages)

    tracker.record(prompt, result)

    if json_out:
        s = tracker.summary_dict()
        click.echo(json.dumps({
            "response": result.text,
            "source": result.source,
            "cached": result.cached,
            "tokens": result.tokens_total,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "time_s": result.time_s,
            "model": result.model,
            "complexity_score": score,
            "routing_decision": decision,
            "system": system,
            "rag_used": rag_used,
            "session": s,
        }, indent=2))
        return

    rag_tag = " [dim](RAG)[/]" if rag_used else ""
    cache_tag = " [dim](cache)[/]" if result.cached else ""

    console.print()
    md = Markdown(result.text)
    title = f"[{spinner_colors[decision]}]{decision.upper()} · {format_tokens(result.tokens_total)} token · {result.time_s:.1f}s{_src_icon(result.source)}{cache_tag}{rag_tag}[/]"
    console.print(Panel(md, border_style=spinner_colors[decision], title=title))
    console.print()


def _src_icon(source):
    return " " if source == "local" else " ☁"


def show_stats(state):
    tracker = state["tracker"]
    table = render_stats(tracker)
    console.print()
    console.print(Panel(table, title="[bold]Statistiche Sessione[/]", border_style="cyan"))
    console.print()


def show_help():
    help_text = """
[bold]Comandi:[/]
  [cyan]help[/]              Questo aiuto
  [cyan]stats[/]             Statistiche sessione
  [cyan]clear[/]             Pulisci schermo
  [cyan]exit[/]              Esci
  [cyan]/mode auto|local|online[/]  Cambia modalità router
  [cyan]/model <nome>[/]     Cambia modello locale
  [cyan]/cache[/]            Pulisci cache risposte
  [cyan]/reset[/]            Resetta statistiche

[bold]CLI:[/]
  [cyan]ai-plus chat <prompt>[/]           Singolo prompt
  [cyan]ai-plus chat --system <sys> <p>[/] Prompt con istruzioni
  [cyan]ai-plus generate <prompt>[/]       Output JSON (per tool/API)
  [cyan]ai-plus generate -s <sys> <p>[/]   JSON con system prompt
  [cyan]ai-plus --json chat <prompt>[/]    Output JSON

[bold]Knowledge Base (RAG):[/]
  [cyan]ai-plus learn from-dir <path>[/]     Indicizza una directory
  [cyan]ai-plus learn from-url <url>[/]      Indicizza una pagina web
  [cyan]ai-plus learn status[/]              Stato della knowledge base
  [cyan]ai-plus learn clear[/]               Pulisce la knowledge base
  [cyan]ai-plus kb[/]                        Alias per status rapido

  Il contesto RAG viene automaticamente iniettato nelle chat
  sia da CLI che da web. Ogni prompt viene arricchito con i
  chunk più rilevanti dalla knowledge base.

[bold]Progetti (scaffold + test + run):[/]
  [cyan]ai-plus project create <nome> --template python[/]  Crea progetto
  [cyan]ai-plus project list[/]                               Elenca progetti
  [cyan]ai-plus project run <nome>[/]                        Esegui
  [cyan]ai-plus project test <nome>[/]                       Test
  [cyan]ai-plus project preview <nome>[/]                    Preview split-view
  [cyan]ai-plus project delete <nome>[/]                     Elimina

  Template: python, python-flask, html, node-js

[bold]Integrazione opencode:[/]
  [cyan]ai-plus generate[/] produce JSON strutturato che opencode
  può consumare come tool locale. Usa [cyan]--json[/] per output machine-readable.
  La web app su [cyan]ai-plus serve[/] fornisce UI alternativa a opencode.

  Esempio in opencode (bash):
    ai-plus generate --json "analizza questo codice"
"""
    console.print(help_text)


def handle_slash(cmd, state):
    parts = cmd.split(maxsplit=1)
    c = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if c == "mode" and arg:
        state["cfg"]["router"]["mode"] = arg
        state["router"] = SmartRouter(state["cfg"])
        console.print(f"[green]Router → {arg}[/]")
    elif c == "model" and arg:
        state["cfg"]["local"]["model"] = arg
        state["local"] = get_local_provider(state["cfg"])
        console.print(f"[green]Modello locale → {arg}[/]")
    elif c == "online-model" and arg:
        state["cfg"]["online"]["model"] = arg
        if state["online_avail"]:
            state["online"] = get_online_provider(state["cfg"])
        console.print(f"[green]Modello online → {arg}[/]")
    elif c == "reset":
        state["tracker"] = SessionTracker(state["cfg"])
        console.print("[green]Statistiche resettate.[/]")
    elif c == "cache":
        clear_caches()
        console.print("[green]Cache pulita.[/]")
    else:
        console.print(f"[red]Sconosciuto: /{c}[/]")


def banner():
    return """\
[bold cyan]  ┌─┐┌─┐┌┐┌┌─┐┌─┐┬ ┬┌─┐┌─┐┌─┐[/]
[bold cyan]  │  │ ││││└─┐│  └┬┘├┤ ├┤ ├─┘[/]
[bold cyan]  └─┘└─┘┘└┘└─┘└─┘ ┴ └─┘└─┘┴  [/]
[dim]  AI ibrida locale + cloud · routing intelligente[/]"""


def main():
    cli()


if __name__ == "__main__":
    main()
