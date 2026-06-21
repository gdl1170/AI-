"""
Smart Router: decide se usare AI locale o online.
Versione consapevole della Knowledge Base: se la KB ha contesto
rilevante, preferisce il modello locale (più veloce e gratuito).
"""

from .knowledge import _needs_rag, get_knowledge_base


class SmartRouter:
    def __init__(self, cfg):
        self.cfg = cfg["router"]
        self.threshold = self.cfg["complexity_threshold"]
        self._online_kw = [k.lower() for k in self.cfg["keyword_boost_online"]]
        self._local_kw = [k.lower() for k in self.cfg["keyword_boost_local"]]
        self._online_cache = {}
        self._local_cache = {}

    def decide(self, prompt):
        c = self.cfg
        if c["always_local"] or c["mode"] == "local":
            return "local", 0
        if c["always_online"] or c["mode"] == "online":
            return "online", 0
        return self._auto_decision(prompt)

    def _kb_boost(self, prompt):
        """Se la KB ha contenuti rilevanti, abbassa la soglia per il locale."""
        if not _needs_rag(prompt):
            return 0
        kb = get_knowledge_base()
        if kb.total_chunks == 0:
            return 0
        # Quick check: quanti chunk hanno keyword in comune col prompt?
        q_words = set(prompt.lower().split())
        if len(q_words) < 2:
            return 0
        # Conta quanti chunk condividono almeno 2 parole col prompt
        matches = 0
        for d in kb.index.documents:
            d_words = set(d["text"].lower().split())
            if len(q_words & d_words) >= 2:
                matches += 1
                if matches >= 3:
                    break
        # Boost locale: -1 per ogni chunk rilevante (rende più facile routing locale)
        return -min(matches, 5)

    def _auto_decision(self, prompt):
        pl = prompt.lower()
        pw = prompt.split()
        word_count = len(pw)
        char_len = len(prompt)

        score = 0

        for kw in self._online_kw:
            if kw in pl:
                score += 2
                if score >= self.threshold + 4:
                    break

        if score < self.threshold:
            for kw in self._local_kw:
                if kw in pl:
                    score -= 1
                    if score < 0:
                        break

        if char_len > 500:
            score += 3
        elif char_len > 200:
            score += 1
        elif char_len < 30:
            score -= 1

        score += word_count // 10

        # KB-aware boost: se la KB ha contesto, preferisci locale
        score += self._kb_boost(prompt)

        return ("online" if score >= self.threshold else "local", score)
