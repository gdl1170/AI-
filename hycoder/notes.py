"""
Note in stile Obsidian: wiki-links, tag, grafo, markdown.
Integrato con la knowledge base RAG per ricerca semantica.
"""

import os
import re
import json
import time
import glob
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .resources import DATA_DIR

NOTES_DIR = DATA_DIR / "notes"


def _ensure():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


_WIKI_LINK_RE = re.compile(r'\[\[([^\]]+)\]\]')
_TAG_RE = re.compile(r'(?:^|\s)#([a-zA-Z0-9_/\-]+)')
_FRONT_MATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def parse_note(text: str, source: str = ""):
    """Analizza un file markdown, estrae frontmatter, wiki-link, tag."""
    text = text or ""
    front = {}
    body = text

    m = _FRONT_MATTER_RE.match(text)
    if m:
        try:
            front = json.loads(m.group(1))
        except json.JSONDecodeError:
            front = _parse_yaml_like(m.group(1))
        body = text[m.end():]

    title = front.get("title") or Path(source).stem if source else "Untitled"
    tags = front.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Tag inline #tag
    body_tags = _TAG_RE.findall(body)
    tags = list(set(tags + body_tags))

    # Wiki-link [[Link]] → salviamo slug e testo
    raw_links = _WIKI_LINK_RE.findall(body)
    links = []
    for l in raw_links:
        parts = l.split("|")
        target = parts[0].strip()
        links.append(target)  # testo originale
    link_slugs = [_slugify(l.split("|")[0].strip()) for l in raw_links]

    return {
        "title": title,
        "tags": tags,
        "links": links,
        "link_slugs": link_slugs,
        "body": body.strip(),
        "frontmatter": front,
    }


def _parse_yaml_like(text):
    """Parser YAML minimale per frontmatter semplice."""
    data = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v.startswith("[") and v.endswith("]"):
                v = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",")]
            data[k] = v
    return data


def render_wiki_links(body, notes_index: dict) -> str:
    """Converte [[Link]] in HTML con link alla nota corrispondente."""
    def _replace(m):
        target = m.group(1).split("|")[0].strip()
        label = m.group(1).split("|")[1].strip() if "|" in m.group(1) else target
        slug = _slugify(target)
        exists = slug in notes_index
        cls = "wiki-link" if exists else "wiki-link broken"
        href = f"/notes/{slug}"
        return f'<a href="{href}" class="{cls}" data-slug="{slug}">{label}</a>'
    return _WIKI_LINK_RE.sub(_replace, body)


def _slugify(name):
    return name.lower().replace(" ", "-").replace("/", "-")


class NoteStore:
    """Gestisce note in stile Obsidian su filesystem."""

    def __init__(self):
        _ensure()
        self._lock = threading.Lock()

    # ── CRUD ─────────────────────────────────────────────────────────

    def list_all(self) -> list[dict]:
        _ensure()
        notes = []
        for f in sorted(NOTES_DIR.glob("*.md")):
            info = self._info(f)
            if info:
                notes.append(info)
        return notes

    def get(self, slug: str) -> dict | None:
        path = self._path_by_slug(slug)
        if not path or not path.exists():
            return None
        info = self._info(path, include_body=True)
        if not info:
            return None
        # Arricchisci con backlinks (confronto su slug)
        all_notes = self.list_all()
        info["backlinks"] = [n for n in all_notes if slug in n.get("link_slugs", [])]
        return info

    def create(self, title: str, body: str = "", tags: list[str] | None = None) -> dict:
        _ensure()
        tags = tags or []
        slug = _slugify(title)
        path = NOTES_DIR / f"{slug}.md"
        if path.exists():
            # Slug alternativo
            i = 1
            while path.exists():
                path = NOTES_DIR / f"{slug}-{i}.md"
                i += 1

        front = {
            "title": title,
            "tags": tags,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
        }
        content = f"---\n{json.dumps(front, indent=2)}\n---\n\n{body}\n"
        path.write_text(content)
        return self._info(path, include_body=True) or {}

    def update(self, slug: str, body: str | None = None, tags: list[str] | None = None,
               title: str | None = None) -> dict | None:
        path = self._path_by_slug(slug)
        if not path or not path.exists():
            return None
        text = path.read_text()
        parsed = parse_note(text, str(path))
        front = parsed["frontmatter"]
        if title:
            front["title"] = title
        if tags is not None:
            front["tags"] = tags
        front["modified"] = datetime.now().isoformat()
        new_body = body if body is not None else parsed["body"]
        content = f"---\n{json.dumps(front, indent=2)}\n---\n\n{new_body}\n"
        path.write_text(content)
        return self._info(path, include_body=True)

    def delete(self, slug: str) -> bool:
        path = self._path_by_slug(slug)
        if path and path.exists():
            path.unlink()
            return True
        return False

    # ── Grafo ────────────────────────────────────────────────────────

    def graph(self) -> dict:
        """Restituisce nodi e archi per visualizzazione grafo."""
        notes = self.list_all()
        nodes = []
        edges = []
        slug_map = {}

        for n in notes:
            slug = n["slug"]
            slug_map[slug] = n["title"]
            nodes.append({
                "id": slug,
                "label": n["title"],
                "tags": n.get("tags", []),
                "links_count": len(n.get("link_slugs", [])),
            })

        for n in notes:
            for i, link_slug in enumerate(n.get("link_slugs", [])):
                if link_slug in slug_map:
                    label = n.get("links", [])[i] if i < len(n.get("links", [])) else link_slug
                    edges.append({
                        "source": n["slug"],
                        "target": link_slug,
                        "label": label,
                    })

        return {"nodes": nodes, "edges": edges}

    # ── Ricerca ──────────────────────────────────────────────────────

    def search(self, query: str) -> list[dict]:
        """Ricerca full-text semplice nelle note."""
        q = query.lower()
        results = []
        for n in self.list_all():
            text = Path(NOTES_DIR / f"{n['slug']}.md").read_text().lower()
            if q in text:
                results.append({**n, "match": True})
        return results

    def search_by_tag(self, tag: str) -> list[dict]:
        return [n for n in self.list() if tag in n.get("tags", [])]

    # ── Helper ───────────────────────────────────────────────────────

    def _path_by_slug(self, slug: str) -> Path:
        _ensure()
        p = NOTES_DIR / f"{slug}.md"
        if p.exists():
            return p
        # Cerca con slug diverso
        for f in NOTES_DIR.glob("*.md"):
            if _slugify(f.stem) == slug:
                return f
        return p  # ritorna comunque il path predefinito

    def _info(self, path: Path, include_body=False) -> dict | None:
        if not path.exists():
            return None
        try:
            text = path.read_text()
        except Exception:
            return None
        parsed = parse_note(text, str(path))
        slug = _slugify(parsed["title"] or path.stem)
        info = {
            "slug": slug,
            "title": parsed["title"],
            "tags": parsed["tags"],
            "links": parsed["links"],
            "link_slugs": parsed.get("link_slugs", []),
            "created": parsed["frontmatter"].get("created", ""),
            "modified": parsed["frontmatter"].get("modified", ""),
            "file": path.name,
            "size": path.stat().st_size,
        }
        if include_body:
            info["body"] = parsed["body"]
        return info

    def list(self) -> list[dict]:
        """Alias per list_all()."""
        return self.list_all()

    def rebuild_index(self):
        """Reindicizza tutte le note per la KB RAG."""
        from .knowledge import get_knowledge_base
        kb = get_knowledge_base()
        _ensure()
        for f in NOTES_DIR.glob("*.md"):
            try:
                text = f.read_text()
                parsed = parse_note(text, str(f))
                kb.add_file(str(f))
            except Exception:
                pass
        kb._save_index()


# ── Singolo globale ─────────────────────────────────────────────────────

_note_store = None
_ns_lock = threading.Lock()


def get_note_store():
    global _note_store
    with _ns_lock:
        if _note_store is None:
            _note_store = NoteStore()
        return _note_store


def reset_note_store():
    global _note_store
    with _ns_lock:
        _note_store = NoteStore()
