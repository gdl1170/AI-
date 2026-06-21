"""
Knowledge base per AI+.
RAG: indicizza documenti (file, URL) e recupera contesto per le query.
TF-IDF + cosine similarity (zero dipendenze extra).
Opzionale: embedding Ollama per ricerca semantica.
"""

import os
import re
import json
import math
import time
import hashlib
import threading
from pathlib import Path
from collections import Counter, defaultdict

from .resources import get_resource_manager, KB_DIR

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
MAX_CONTEXT_CHARS = 4000
TOP_K_DEFAULT = 5

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_TIMEOUT = 2
HYBRID_WEIGHT_TFIDF = 0.3
HYBRID_WEIGHT_EMB = 0.7
SIMPLE_QUERY_MAX_CHARS = 30
RAG_CACHE_TTL = 600

_SIMPLE_QUERIES = frozenset({
    'hello', 'hi', 'hey', 'ciao', 'salve', 'buongiorno', 'buonasera',
    'help', 'aiuto', 'thanks', 'grazie', 'yes', 'no', 'si', 'ok', 'okay',
    'test', 'prova', 'come stai', 'how are you', 'who are you',
    'chi sei', 'cosa sai fare', 'what can you do',
})

_STRONG_RAG_KEYWORDS = frozenset({
    'cerca', 'search', 'trova', 'find', 'knowledge', 'wiki', 'kb',
    'documentazione', 'docs', 'documentation', 'libro', 'book',
    'riferimento', 'reference', 'guida', 'guide', 'manuale', 'manual',
    'cos\'è', 'what is', 'define', 'definisci', 'spiega', 'spieghi',
    'spiegami', 'spiegare', 'explain', 'riassumi', 'riassunto',
    'summarize', 'confronta', 'confronto', 'compare', 'dimmi',
    'descrivi', 'describe', 'analizza', 'analyze', 'approfondisci',
})

_EMBEDDING_SESSION = None
_EMBEDDING_LOCK = threading.Lock()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_file(path):
    path = Path(path)
    try:
        text = path.read_text("utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text("latin-1", errors="replace")
        except Exception:
            return []

    ext = path.suffix.lower()
    fname = path.name
    chunks = []
    for i, c in enumerate(chunk_text(text)):
        chunks.append({"text": c, "source": str(path), "file": fname, "type": ext, "chunk_id": i})
    return chunks


def _tokenize(text):
    return re.findall(r'\w+', text.lower())


def _extract_title(html):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


# ─── Embedding helper (Ollama) ──────────────────────────────────────────

_OLLAMA_CHECKED = False
_OLLAMA_AVAILABLE = False
_OLLAMA_LOCK = threading.Lock()
_OLLAMA_LAST_CHECK = 0

def _ollama_available(cfg=None):
    """Check rapido se Ollama è in esecuzione (caching del risultato, retry ogni 30s)."""
    global _OLLAMA_CHECKED, _OLLAMA_AVAILABLE, _OLLAMA_LAST_CHECK
    now = time.time()
    if _OLLAMA_CHECKED and _OLLAMA_AVAILABLE:
        return True
    if _OLLAMA_CHECKED and now - _OLLAMA_LAST_CHECK < 30:
        return False
    with _OLLAMA_LOCK:
        if _OLLAMA_CHECKED and _OLLAMA_AVAILABLE:
            return True
        if _OLLAMA_CHECKED and now - _OLLAMA_LAST_CHECK < 30:
            return False
        try:
            import requests
            base = "http://localhost:11434"
            if cfg:
                base = cfg.get("local", {}).get("ollama_base_url", base)
            r = requests.get(f"{base}/api/tags", timeout=2)
            _OLLAMA_AVAILABLE = r.status_code == 200
        except Exception:
            _OLLAMA_AVAILABLE = False
        _OLLAMA_CHECKED = True
        _OLLAMA_LAST_CHECK = now
        return _OLLAMA_AVAILABLE

def _get_embedding(text, base_url="http://localhost:11434"):
    """Embedding vettoriale via Ollama. Restituisce lista di float o None."""
    if not _ollama_available():
        return None
    global _EMBEDDING_SESSION
    with _EMBEDDING_LOCK:
        if _EMBEDDING_SESSION is None:
            import requests as r
            _EMBEDDING_SESSION = r.Session()
        session = _EMBEDDING_SESSION
    try:
        resp = session.post(
            f"{base_url}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text[:2048]},
            timeout=EMBEDDING_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        emb = data.get("embeddings", [])
        return emb[0] if emb else None
    except Exception:
        return None


def _cosine_sim(a, b):
    if not a or not b:
        return 0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(y * y for y in b)) or 1
    return dot / (na * nb)


def _get_cached_embedding(text):
    """Embedding con cache su disco."""
    mgr = get_resource_manager()
    key = hashlib.sha256(text[:1024].encode()).hexdigest()
    cached = mgr.embedding_cache.get_json(key)
    if cached is not None:
        return cached
    vec = _get_embedding(text)
    if vec is not None:
        mgr.embedding_cache.set(key, vec, ttl=86400)
    return vec


# ─── Smart RAG gating ──────────────────────────────────────────────────

def _needs_rag(query):
    """Decide rapidamente se serve RAG. Evita chiamate inutili su query banali."""
    q = query.strip().lower()
    # Query molto corte → solo se contengono keyword esplicite di ricerca
    if len(q) < SIMPLE_QUERY_MAX_CHARS:
        if q in _SIMPLE_QUERIES:
            return False
        if any(kw in q for kw in _STRONG_RAG_KEYWORDS):
            return True
        return False
    if q in _SIMPLE_QUERIES:
        return False
    # Keyword esplicite: forza RAG
    if any(kw in q for kw in _STRONG_RAG_KEYWORDS):
        return True
    # Query lunghe → probabilmente merita RAG
    if len(q) > 200:
        return True
    # Query con domande esplicite
    if q.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'che', 'cosa', 'come', 'perché', 'perche', 'dove', 'quando', 'chi')):
        return not any(q.startswith(g) for g in ('what can you', 'who are you', 'chi sei', 'cosa sai'))
    return False


# ─── TF-IDF Index (zero dipendenze) ────────────────────────────────────

class TFIDFIndex:
    def __init__(self):
        self.documents = []
        self.vocab = set()
        self.idf = {}
        self.doc_vectors = []
        self._lock = threading.Lock()

    def add(self, documents):
        with self._lock:
            start = len(self.documents)
            self.documents.extend(documents)

            doc_tfs = []
            for d in documents:
                tf = Counter(_tokenize(d["text"]))
                doc_tfs.append(tf)
                self.vocab.update(tf.keys())

            n = len(self.documents)
            df = defaultdict(int)
            for d in self.documents:
                for t in set(_tokenize(d["text"])):
                    df[t] += 1

            self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in self.vocab}

            for tf in doc_tfs:
                self.doc_vectors.append({t: f * self.idf.get(t, 1) for t, f in tf.items()})

    def search(self, query, top_k=TOP_K_DEFAULT):
        tokens = _tokenize(query)
        qt = Counter(tokens)
        qv = {t: f * self.idf.get(t, 1) for t, f in qt.items()}
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1

        scores = []
        for i, dv in enumerate(self.doc_vectors):
            dot = sum(qv.get(t, 0) * v for t, v in dv.items())
            dn = math.sqrt(sum(v * v for v in dv.values())) or 1
            sim = dot / (qn * dn)
            if sim > 0:
                scores.append((sim, i))

        scores.sort(key=lambda x: -x[0])
        return [(s, self.documents[i]) for s, i in scores[:top_k]]


# ─── Knowledge Base ─────────────────────────────────────────────────────

class KnowledgeBase:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.index = TFIDFIndex()
        self.total_chunks = 0
        self.sources = {}
        self._embeddings = []
        self._lock = threading.Lock()

    # ── Aggiunta file ────────────────────────────────────────────────

    def add_file(self, path):
        path = Path(path).expanduser()
        if not path.is_file():
            return {"error": f"File non trovato: {path}"}
        n = self._index_file(path)
        return {"chunks_added": n, "file": str(path)}

    def add_directory(self, path, pattern="*", recursive=True, progress_dict=None):
        path = Path(path).expanduser()
        if not path.is_dir():
            return {"error": f"Directory non trovata: {path}"}

        raw = list(path.rglob(pattern) if recursive else path.glob(pattern))
        files = []
        for f in raw:
            if not f.is_file():
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            if sz == 0:
                continue
            files.append(f)

        total = 0
        errors = []
        total_files = len(files)

        if progress_dict is not None:
            progress_dict["total_files"] = total_files

        for idx, f in enumerate(files):
            try:
                if progress_dict is not None:
                    progress_dict["current_file"] = str(f.name)
                    progress_dict["progress"] = int((idx / total_files) * 100) if total_files else 0
                    progress_dict["message"] = f"({idx+1}/{total_files}) {f.name}"
                n = self._index_file(f)
                if n > 0:
                    total += n
            except Exception:
                errors.append(str(f))

        if progress_dict is not None:
            progress_dict["progress"] = 100

        self._save_index()
        return {
            "files_processed": len(files),
            "chunks_added": total,
            "errors": errors,
        }

    def preview_directory(self, path, pattern="*", recursive=True):
        """Preview files that would be indexed from a directory."""
        path = Path(path).expanduser()
        if not path.is_dir():
            return {"error": f"Directory non trovata: {path}"}

        raw = list(path.rglob(pattern) if recursive else path.glob(pattern))
        files = []
        for f in raw:
            if not f.is_file():
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            if sz == 0:
                continue
            files.append(f)

        sample_names = [f.name for f in files[:100]]
        total_chars = sum(f.stat().st_size for f in files[:100] if f.stat().st_size > 0)
        chunk_est = max(1, total_chars // 1000) if files else 0

        return {
            "total_files": len(files),
            "files": sample_names,
            "total_chunks_est": chunk_est,
        }

    def _index_file(self, path):
        chunks = chunk_file(path)
        if not chunks:
            return 0
        # Embedding batch (fallback silenzioso se non disponibile)
        embeddings = []
        for c in chunks:
            emb = _get_cached_embedding(c["text"][:512])
            embeddings.append(emb or [])
        with self._lock:
            self.index.add(chunks)
            self._embeddings.extend(embeddings)
            self.total_chunks += len(chunks)
            self.sources[str(path)] = {
                "file": path.name, "chunks": len(chunks),
                "type": path.suffix.lower(),
            }
        return len(chunks)

    # ── Aggiunta URL ─────────────────────────────────────────────────

    def add_url(self, url):
        import requests as r
        try:
            resp = r.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ai-plus/1.0)"
            })
            resp.raise_for_status()
        except Exception as e:
            return self._add_url_fallback(url, str(e))

        raw = resp.text
        body = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
        body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
        body = re.sub(r'<[^>]+>', ' ', body)
        text = re.sub(r'\s+', ' ', body).strip()

        if len(text) < 50:
            return {"error": "Contenuto troppo corto o non estraibile"}

        chunks = []
        for i, c in enumerate(chunk_text(text)):
            chunks.append({
                "text": c, "source": url, "file": url,
                "type": ".url", "chunk_id": i,
            })

        embeddings = []
        for c in chunks:
            emb = _get_cached_embedding(c["text"][:512])
            embeddings.append(emb or [])

        title = _extract_title(raw) or url

        with self._lock:
            self.index.add(chunks)
            self._embeddings.extend(embeddings)
            self.total_chunks += len(chunks)
            self.sources[url] = {
                "file": url, "chunks": len(chunks),
                "type": "web", "title": title,
            }

        return {"chunks_added": len(chunks), "text_length": len(text), "title": title}

    def _add_url_fallback(self, url, error_msg):
        """Fallback: se non può scaricare l'URL, cerca su web e indicizza."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        query = parsed.netloc + " " + parsed.path.strip("/").replace("/", " ")
        query = re.sub(r'\s+', ' ', query).strip() or parsed.netloc

        import requests as r
        try:
            resp = r.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception:
            return {"error": f"Impossibile scaricare {url}: {error_msg}"}

        html = resp.text
        results = []
        for m in re.finditer(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]*)</a>',
            html, re.DOTALL,
        ):
            url_result = m.group(1)
            title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            snippet = re.sub(r'<[^>]+>', '', m.group(3)).strip()
            if title and url_result:
                results.append({"title": title, "url": url_result, "snippet": snippet})
            if len(results) >= 8:
                break

        if not results:
            return {"error": f"Impossibile scaricare {url}: {error_msg}"}

        chunks = []
        for i, res in enumerate(results):
            text = f"{res['title']}\n{res['snippet']}"
            if len(text) < 20:
                continue
            for j, c in enumerate(chunk_text(text)):
                chunks.append({
                    "text": c, "source": res["url"], "file": url,
                    "type": ".url", "chunk_id": j,
                })

        if not chunks:
            return {"error": f"Impossibile scaricare {url}: {error_msg}"}

        embeddings = []
        for c in chunks:
            emb = _get_cached_embedding(c["text"][:512])
            embeddings.append(emb or [])

        with self._lock:
            self.index.add(chunks)
            self._embeddings.extend(embeddings)
            self.total_chunks += len(chunks)
            sources_added = {}
            for res in results:
                if res["url"] not in self.sources:
                    sources_added[res["url"]] = {"file": url, "chunks": 0, "type": "web", "title": res["title"]}
            for k, v in sources_added.items():
                self.sources[k] = v

        title = f"Ricerca web: {query}"
        return {"chunks_added": len(chunks), "text_length": sum(len(c["text"]) for c in chunks), "title": title, "fallback": True}

    # ── Ricerca ibrida TF-IDF + Embedding ────────────────────────────

    def _hybrid_search(self, text, top_k=TOP_K_DEFAULT):
        """Fonde risultati TF-IDF e embedding per maggiore accuratezza."""
        tfidf_results = self.index.search(text, top_k=top_k * 2)

        emb_vec = _get_cached_embedding(text)
        emb_results = []
        if emb_vec is not None and self._embeddings:
            scores = []
            for i, doc_emb in enumerate(self._embeddings):
                sim = _cosine_sim(emb_vec, doc_emb)
                if sim > 0.3:
                    scores.append((sim, i))
            scores.sort(key=lambda x: -x[0])
            emb_results = [(s, self.index.documents[i]) for s, i in scores[:top_k * 2]]

        # Fonde: punteggio ibrido = weight_tfidf * tfidf_score + weight_emb * emb_score
        combined = {}
        for s, d in tfidf_results:
            doc_id = d.get("chunk_id", id(d))
            combined[doc_id] = {
                "score": s * HYBRID_WEIGHT_TFIDF,
                "doc": d,
            }

        if emb_results:
            for s, d in emb_results:
                doc_id = d.get("chunk_id", id(d))
                if doc_id in combined:
                    combined[doc_id]["score"] += s * HYBRID_WEIGHT_EMB
                else:
                    combined[doc_id] = {
                        "score": s * HYBRID_WEIGHT_EMB,
                        "doc": d,
                    }

        sorted_results = sorted(combined.values(), key=lambda x: -x["score"])
        return [(r["score"], r["doc"]) for r in sorted_results[:top_k]]

    def query(self, text, top_k=TOP_K_DEFAULT):
        results = self._hybrid_search(text, top_k=top_k)
        return [{"score": round(s, 3), **d} for s, d in results]

    def build_context(self, query, max_chars=MAX_CONTEXT_CHARS):
        results = self.query(query)
        if not results:
            return ""

        parts = []
        total = 0
        for r in results:
            snippet = f"[Fonte: {r['source']}]\n{r['text']}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)

        if not parts:
            return ""
        return (
            "Ecco del contesto rilevante dalla knowledge base:\n\n"
            + "\n\n---\n\n".join(parts)
        )

    # ── Persistenza su disco ────────────────────────────────────────

    def _save_index(self):
        """Salva indice TF-IDF e fonti su disco con compressione."""
        KB_DIR.mkdir(parents=True, exist_ok=True)
        import gzip
        import logging
        data = {
            "total_chunks": self.total_chunks,
            "sources": self.sources,
            "embeddings": self._embeddings,
            "index": {
                "documents": self.index.documents,
                "vocab": list(self.index.vocab),
                "idf": self.index.idf,
                "doc_vectors": self.index.doc_vectors,
            }
        }
        tmp = KB_DIR / "_index.tmp.gz"
        dst = KB_DIR / "_index.json.gz"
        try:
            raw = json.dumps(data, ensure_ascii=False, default=str).encode()
            tmp.write_bytes(gzip.compress(raw, compresslevel=6))
            tmp.rename(dst)
        except Exception as e:
            logging.getLogger(__name__).warning("KB save failed: %s", e)

    def _load_index(self):
        """Carica indice da disco (supporta .json e .json.gz)."""
        import gzip
        import logging
        path_gz = KB_DIR / "_index.json.gz"
        path_json = KB_DIR / "_index.json"
        if path_gz.exists():
            path = path_gz
            use_gz = True
        elif path_json.exists():
            path = path_json
            use_gz = False
        else:
            return
        try:
            raw = path.read_bytes()
            if use_gz:
                raw = gzip.decompress(raw)
            data = json.loads(raw)
            self.total_chunks = data.get("total_chunks", 0)
            self.sources = data.get("sources", {})
            self._embeddings = data.get("embeddings", [])
            idx = data.get("index", {})
            self.index.documents = idx.get("documents", [])
            self.index.vocab = set(idx.get("vocab", []))
            self.index.idf = idx.get("idf", {})
            self.index.doc_vectors = idx.get("doc_vectors", [])
        except Exception as e:
            logging.getLogger(__name__).warning("KB load error (starting fresh): %s", e)
            self.clear()

    # ── Stato / Gestione ─────────────────────────────────────────────

    def status(self):
        return {
            "total_chunks": self.total_chunks,
            "total_sources": len(self.sources),
            "sources": [
                {"source": k, "chunks": v["chunks"], "type": v.get("type")}
                for k, v in self.sources.items()
            ],
        }

    def remove_source(self, source_key):
        """Rimuove una fonte specifica e tutti i suoi chunk dall'indice."""
        if source_key not in self.sources:
            return False
        with self._lock:
            # Find all indices to remove
            keep_indices = [
                i for i, d in enumerate(self.index.documents)
                if d.get("source") != source_key
            ]
            removed_count = len(self.index.documents) - len(keep_indices)
            if removed_count == 0:
                return False
            # Rebuild everything without this source
            docs = [self.index.documents[i] for i in keep_indices]
            embs = [self._embeddings[i] for i in keep_indices] if self._embeddings else []
            new_index = TFIDFIndex()
            if docs:
                new_index.add(docs)
            self.index = new_index
            self._embeddings = embs
            self.total_chunks = len(docs)
            del self.sources[source_key]
        self._save_index()
        return True

    def get_source_chunks(self, source_key):
        """Restituisce tutti i chunk di una fonte."""
        return [
            d for d in self.index.documents
            if d.get("source") == source_key
        ]

    def clear(self):
        with self._lock:
            self.index = TFIDFIndex()
            self.total_chunks = 0
            self.sources = {}
            self._embeddings = []
        for name in ("_index.json", "_index.json.gz", "_index.tmp.gz"):
            path = KB_DIR / name
            try:
                path.unlink()
            except OSError:
                pass


# ── Singolo globale (condiviso CLI + Web) ──────────────────────────────

_knowledge_base = None
_kb_lock = threading.Lock()


def get_knowledge_base(cfg=None):
    global _knowledge_base
    with _kb_lock:
        if _knowledge_base is None:
            kb = KnowledgeBase(cfg)
            kb._load_index()
            _knowledge_base = kb
        return _knowledge_base


def reset_knowledge_base():
    global _knowledge_base
    with _kb_lock:
        _knowledge_base = KnowledgeBase()
    # Delete persisted index so next load starts fresh
    idx = KB_DIR / "_index.json.gz"
    if idx.exists():
        try:
            idx.unlink()
        except OSError:
            pass


def build_rag_context(query):
    """Contesto RAG con cache e integrazione wiki."""
    kb = get_knowledge_base()
    if kb.total_chunks == 0:
        return ""

    # Cache su disco per query identiche
    mgr = get_resource_manager()
    cache_key = "rag:" + hashlib.sha256(query.encode()).hexdigest()[:32]
    cached = mgr.response_cache.get(cache_key)
    if cached is not None:
        return cached.decode()

    ctx = kb.build_context(query)

    # Cerca anche nelle pagine wiki se il contesto KB è scarso
    if len(ctx) < 200:
        wiki_pages = list_wiki_pages()
        wiki_hits = []
        q_lower = query.lower()
        for wp in wiki_pages:
            title_lower = wp["title"].lower()
            if any(word in q_lower for word in title_lower.split()) or \
               any(word in title_lower for word in q_lower.split()):
                page = get_wiki_page(wp["slug"])
                if page:
                    wiki_hits.append(page["content"][:500])
        if wiki_hits:
            wiki_block = "\n\n---\n\n".join(
                f"[Wiki: {wp['title']}]\n{content}"
                for content in wiki_hits
            )
            ctx = ctx + "\n\n" + wiki_block if ctx else (
                "Ecco delle voci wiki correlate:\n\n" + wiki_block
            )

    # Cache del risultato
    if ctx:
        mgr.response_cache.set(cache_key, ctx, ttl=RAG_CACHE_TTL)

    return ctx


# ── Knowledge Graph (relazioni tra chunk) ─────────────────────────────

def build_knowledge_graph(kb=None):
    """Costruisce un grafo delle relazioni tra chunk della KB.
    Due chunk sono collegati se appartengono alla stessa fonte o
    condividono termini significativi (TF-IDF overlap)."""
    if kb is None:
        kb = get_knowledge_base()
    if kb.total_chunks == 0:
        return {"nodes": [], "edges": []}

    documents = kb.index.documents
    nodes = []
    edges = []
    edge_set = set()

    # Raggruppa per fonte per dare colori uniformi
    source_groups = defaultdict(list)
    for i, d in enumerate(documents):
        source_groups[d.get("source", "?").split("/")[-1].split("\\")[-1]].append(i)

    for i, d in enumerate(documents):
        label = d.get("file", d.get("source", f"chunk-{i}"))
        fname = label.split("/")[-1].split("\\")[-1]
        text_preview = d.get("text", "")[:120].replace("\n", " ")
        nodes.append({
            "id": f"chunk-{i}",
            "label": f"{fname} #{i}",
            "title": fname,
            "source": d.get("source", ""),
            "group": fname,
            "type": d.get("type", ""),
            "chunk_id": i,
            "text": text_preview,
        })

        # Collega chunk consecutivi della stessa fonte
        source = d.get("source", "")
        for j in range(i):
            if documents[j].get("source") == source and abs(i - j) <= 5:
                key = (min(i, j), max(i, j))
                if key not in edge_set and len(edges) < 300:
                    edge_set.add(key)
                    edges.append({
                        "source": f"chunk-{i}",
                        "target": f"chunk-{j}",
                        "label": "stessa fonte",
                        "weight": 2,
                    })

    # Collegamenti semantici tra chunk di fonti diverse
    if len(documents) <= 500:
        for i in range(len(documents)):
            if len(edges) >= 400:
                break
            ti = set(k for k, v in kb.index.doc_vectors[i].items() if v > 0.6)
            if not ti:
                continue
            for j in range(i + 1, len(documents)):
                if len(edges) >= 400:
                    break
                if documents[i].get("source") == documents[j].get("source"):
                    continue
                tj = set(k for k, v in kb.index.doc_vectors[j].items() if v > 0.6)
                overlap = ti & tj
                if len(overlap) >= 3:
                    key = (i, j)
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append({
                            "source": f"chunk-{i}",
                            "target": f"chunk-{j}",
                            "label": f"{len(overlap)} termini",
                            "weight": min(len(overlap), 5),
                        })

    return {"nodes": nodes, "edges": edges}


# ── LLM Wiki (genera voci wiki dalla KB) ──────────────────────────────

LLMWIKI_DIR = Path.home() / ".config" / "hybrid-coder" / "llmwiki"


def _wiki_path():
    LLMWIKI_DIR.mkdir(parents=True, exist_ok=True)
    return LLMWIKI_DIR


def list_wiki_pages():
    d = _wiki_path()
    pages = []
    for f in sorted(d.glob("*.md")):
        try:
            content = f.read_text()
            title = content.split("\n")[0].replace("# ", "").strip()
            pages.append({
                "slug": f.stem,
                "title": title,
                "path": str(f),
                "size": f.stat().st_size,
            })
        except Exception:
            pass
    return pages


def get_wiki_page(slug):
    d = _wiki_path()
    f = d / f"{slug}.md"
    if f.exists():
        return {"slug": slug, "content": f.read_text()}
    return None


def delete_wiki_page(slug):
    d = _wiki_path()
    f = d / f"{slug}.md"
    if f.exists():
        f.unlink()
        return True
    return False


def suggest_wiki_topics(kb=None):
    """Restituisce potenziali argomenti wiki dai chunk della KB."""
    if kb is None:
        kb = get_knowledge_base()
    if kb.total_chunks == 0:
        return []

    from collections import Counter
    word_freq = Counter()
    for d in kb.index.documents:
        for t in _tokenize(d["text"]):
            word_freq[t] += 1

    # Filtra: parole che appaiono in 2+ chunk ma non in tutti
    n = len(kb.index.documents)
    topics = []
    for word, freq in word_freq.most_common(50):
        if 2 <= freq <= n * 0.8 and len(word) > 3:
            rows = []
            for d in kb.index.documents:
                if word in _tokenize(d["text"]):
                    rows.append(d)
            if len(rows) >= 2:
                topics.append({
                    "word": word,
                    "frequency": freq,
                    "chunks": [d.get("source", "") for d in rows[:5]],
                })
    return topics[:30]


def auto_generate_wiki(response_text, sources=None):
    """Genera pagine wiki automaticamente da risposte significative (>300 token)."""
    if len(response_text) < 600:
        return []
    kb = get_knowledge_base()
    if kb.total_chunks == 0:
        return []
    topics = suggest_wiki_topics(kb)
    if not topics:
        return []
    created = []
    for t in topics[:3]:
        slug = t["word"].lower().replace(" ", "-").replace("/", "-")
        existing = get_wiki_page(slug)
        if existing:
            continue
        page = generate_wiki_page(t["word"], kb)
        if page:
            created.append(page)
    return created


def generate_wiki_page(topic, kb=None):
    """Genera una pagina wiki su un topic usando i chunk della KB."""
    if kb is None:
        kb = get_knowledge_base()
    if kb.total_chunks == 0:
        return None

    chunks = kb.query(topic, top_k=10)
    if not chunks:
        return None

    d = _wiki_path()
    slug = topic.lower().replace(" ", "-").replace("/", "-")
    f = d / f"{slug}.md"

    # Costruisci pagina markdown
    lines = [f"# {topic}", "", "## Knowledge Base", ""]
    seen = set()
    for c in chunks:
        source = c.get("source", "sconosciuto")
        text_preview = c.get("text", "")[:300]
        lines.append(f"- **Fonte:** {source}")
        lines.append(f"  {text_preview}")
        lines.append("")

    # Aggiungi relazioni
    related = []
    for c in chunks:
        for t in _tokenize(c.get("text", "")):
            if t.lower() != topic.lower() and len(t) > 3 and t not in seen:
                seen.add(t)
                related.append(t)
    if related:
        lines.append("## Collegamenti")
        lines.append("")
        for r in related[:10]:
            rs = r.lower().replace(" ", "-").replace("/", "-")
            lines.append(f"- [[{r}]]")

    content = "\n".join(lines)
    f.write_text(content)
    return {"slug": slug, "title": topic, "content": content}
