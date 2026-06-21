import os, re, json, time, hashlib, shutil, uuid
from pathlib import Path

SKILLS_DIR = Path.home() / ".config" / "hybrid-coder" / "skills"

KNOWN_PLATFORMS = {
    "opencode":     {"ext": ".md",   "fmt": "markdown"},
    "chatgpt":      {"ext": ".json", "fmt": "json"},
    "claude":       {"ext": ".json", "fmt": "json"},
    "custom":       {"ext": ".json", "fmt": "json"},
}

SKILL_SCHEMA = {
    "id": "",
    "name": "",
    "description": "",
    "source_platform": "custom",
    "source_format": "json",
    "instructions": "",
    "adapted_instructions": "",
    "system_prompt": "",
    "tools": [],
    "tags": [],
    "created_at": 0.0,
    "updated_at": 0.0,
    "version": 1,
    "metadata": {},
}


def _ensure_dir():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def _skill_path(skill_id):
    return SKILLS_DIR / f"{skill_id}.json"


def list_skills():
    _ensure_dir()
    skills = []
    for fp in sorted(SKILLS_DIR.iterdir(), reverse=True):
        if fp.suffix == ".json":
            try:
                data = json.loads(fp.read_text())
                skills.append(_summarize(data))
            except Exception:
                continue
    return skills


def _summarize(data):
    return {
        "id": data.get("id"),
        "name": data.get("name", "Untitled"),
        "description": data.get("description", ""),
        "source_platform": data.get("source_platform", "custom"),
        "tags": data.get("tags", []),
        "version": data.get("version", 1),
        "tools": data.get("tools", []),
        "created_at": data.get("created_at", 0),
        "updated_at": data.get("updated_at", 0),
    }


def get_skill(skill_id):
    fp = _skill_path(skill_id)
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def delete_skill(skill_id):
    fp = _skill_path(skill_id)
    if fp.exists():
        fp.unlink()
        return True
    return False


def create_skill(data):
    _ensure_dir()
    skill = dict(SKILL_SCHEMA)
    skill["id"] = uuid.uuid4().hex[:12]
    skill["name"] = data.get("name", "Untitled Skill")
    skill["description"] = data.get("description", "")
    skill["source_platform"] = data.get("source_platform", "custom")
    skill["tags"] = data.get("tags", [])
    skill["instructions"] = data.get("instructions", "")
    skill["adapted_instructions"] = data.get("adapted_instructions", "")
    skill["system_prompt"] = data.get("system_prompt", "")
    skill["tools"] = data.get("tools", [])
    now = time.time()
    skill["created_at"] = now
    skill["updated_at"] = now
    _write_skill(skill)
    return _summarize(skill)


def update_skill(skill_id, data):
    skill = get_skill(skill_id)
    if not skill:
        return None
    for key in ("name", "description", "instructions", "adapted_instructions",
                "system_prompt", "tags", "tools", "metadata"):
        if key in data:
            skill[key] = data[key]
    skill["updated_at"] = time.time()
    skill["version"] = skill.get("version", 1) + 1
    _write_skill(skill)
    return _summarize(skill)


def _write_skill(skill):
    _ensure_dir()
    fp = _skill_path(skill["id"])
    fp.write_text(json.dumps(skill, indent=2, ensure_ascii=False))


# ─── Adaptation engine ──────────────────────────────────────────────

def adapt_from_opencode(content):
    """Parse an opencode .md skill file → AI++ skill dict."""
    lines = content.split("\n")
    meta = {"name": "", "description": "", "tags": [], "tools": [], "system_prompt": ""}
    body_start = 0
    in_frontmatter = False
    fm_lines = []
    if lines and lines[0].strip() == "---":
        in_frontmatter = True
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                body_start = i + 1
                break
            fm_lines.append(line)
        for fl in fm_lines:
            m = re.match(r"^(\w+):\s*(.+)$", fl)
            if m:
                key = m.group(1).lower()
                val = m.group(2).strip()
                if key == "name":
                    meta["name"] = val
                elif key == "description":
                    meta["description"] = val
                elif key == "tags":
                    meta["tags"] = [t.strip() for t in val.split(",") if t.strip()]
                elif key == "tools":
                    meta["tools"] = [t.strip() for t in val.split(",") if t.strip()]
    body = "\n".join(lines[body_start:]).strip()
    meta["instructions"] = body

    system = body
    if system:
        meta["system_prompt"] = f"You are a specialized assistant. Follow these instructions precisely:\n\n{system}"
    return meta


def adapt_from_chatgpt(content):
    """Parse a ChatGPT GPT config JSON → AI++ skill dict."""
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return {"error": "Invalid ChatGPT JSON"}
    meta = {"name": "", "description": "", "tags": [], "tools": [], "system_prompt": "", "instructions": ""}
    meta["name"] = data.get("name", data.get("title", ""))
    meta["description"] = data.get("description", data.get("short_description", ""))
    meta["system_prompt"] = data.get("instructions", data.get("prompt", ""))
    meta["instructions"] = meta["system_prompt"]
    meta["tools"] = [t.get("type", t) if isinstance(t, dict) else t
                     for t in data.get("tools", [])]
    meta["tags"] = [c.get("name", c) if isinstance(c, dict) else c
                    for c in data.get("categories", [])]
    return meta


def adapt_from_claude(content):
    """Parse a Claude project JSON → AI++ skill dict."""
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return {"error": "Invalid Claude JSON"}
    meta = {"name": "", "description": "", "tags": [], "tools": [], "system_prompt": "", "instructions": ""}
    meta["name"] = data.get("name", data.get("project_name", ""))
    meta["description"] = data.get("description", data.get("purpose", ""))
    meta["system_prompt"] = data.get("system_prompt", data.get("instructions", ""))
    meta["instructions"] = meta["system_prompt"]
    meta["tags"] = data.get("tags", data.get("topics", []))
    return meta


def adapt_skill(content, source_platform="opencode"):
    """Main adaptation dispatcher."""
    adapters = {
        "opencode": adapt_from_opencode,
        "chatgpt":  adapt_from_chatgpt,
        "claude":   adapt_from_claude,
        "custom":   lambda c: {"instructions": c, "system_prompt": c if len(c) > 0 else ""},
    }
    adapter = adapters.get(source_platform, adapters["custom"])
    result = adapter(content)
    return result


def detect_platform(filename, content):
    """Auto-detect source platform from filename and content."""
    name = filename.lower()
    if name.endswith(".md"):
        if content.startswith("---"):
            return "opencode"
        return "custom"
    if name.endswith(".json"):
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except Exception:
            return "custom"
        if isinstance(data, dict):
            if "instructions" in data or "prompt" in data:
                return "chatgpt"
            if "system_prompt" in data or "project_name" in data:
                return "claude"
        return "custom"
    return "custom"
