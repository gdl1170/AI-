import json
import time
import threading
from pathlib import Path
from datetime import datetime


class SessionTracker:
    def __init__(self, cfg):
        self.cfg = cfg["tracking"]
        self.history_file = Path(self.cfg["history_file"]).expanduser()
        self.session_start = time.time()

        self.local_tokens_in = 0
        self.local_tokens_out = 0
        self.online_tokens_in = 0
        self.online_tokens_out = 0
        self.local_calls = 0
        self.online_calls = 0
        self.local_time = 0.0
        self.online_time = 0.0
        self.history = []
        self._dirty = False
        self._save_lock = threading.Lock()
        self._debounce_timer = None
        self._load_history()

    def _load_history(self):
        """Carica cronologia dal disco e ricostruisce i contatori."""
        try:
            if self.history_file.exists():
                with open(self.history_file) as f:
                    saved = json.load(f)
                self.history = saved[-500:]
                for entry in self.history:
                    src = entry.get("source", "")
                    t_in = entry.get("tokens_in", 0)
                    t_out = entry.get("tokens_out", 0)
                    ts = entry.get("time_s", 0)
                    if src == "local":
                        self.local_tokens_in += t_in
                        self.local_tokens_out += t_out
                        self.local_calls += 1
                        self.local_time += ts
                    else:
                        self.online_tokens_in += t_in
                        self.online_tokens_out += t_out
                        self.online_calls += 1
                        self.online_time += ts
        except Exception:
            pass

    def record(self, prompt, result, agent=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:200],
            "source": result.source,
            "model": result.model,
            "cached": result.cached,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "tokens_total": result.tokens_total,
            "time_s": round(result.time_s, 2),
            "response_preview": result.text[:150],
            "agent": agent,
        }

        if result.source == "local":
            self.local_tokens_in += result.tokens_in
            self.local_tokens_out += result.tokens_out
            self.local_calls += 1
            self.local_time += result.time_s
        else:
            self.online_tokens_in += result.tokens_in
            self.online_tokens_out += result.tokens_out
            self.online_calls += 1
            self.online_time += result.time_s

        self.history.append(entry)
        self._debounced_save()

    def _debounced_save(self):
        if not self.cfg["save_history"]:
            return
        self._dirty = True

        if self._debounce_timer and self._debounce_timer.is_alive():
            return

        def save():
            time.sleep(2)
            with self._save_lock:
                if self._dirty:
                    self._write_history()
                    self._dirty = False

        self._debounce_timer = threading.Thread(target=save, daemon=True)
        self._debounce_timer.start()

    def flush(self):
        """Forza scrittura su disco (chiamare in uscita)."""
        if self._dirty:
            with self._save_lock:
                if self._dirty:
                    self._write_history()
                    self._dirty = False

    def _write_history(self):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_file.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.history[-500:], f, indent=2, ensure_ascii=False)
            tmp.replace(self.history_file)
        except OSError:
            pass

    @property
    def total_tokens_in(self):
        return self.local_tokens_in + self.online_tokens_in

    @property
    def total_tokens_out(self):
        return self.local_tokens_out + self.online_tokens_out

    @property
    def total_tokens(self):
        return self.total_tokens_in + self.total_tokens_out

    @property
    def total_calls(self):
        return self.local_calls + self.online_calls

    @property
    def total_time(self):
        return self.local_time + self.online_time

    @property
    def session_duration(self):
        return time.time() - self.session_start

    def estimate_cost(self):
        c = self.cfg
        local_cost = (
            (self.local_tokens_in + self.local_tokens_out)
            * c["local_cost_per_1k_tokens"] / 1000
        )
        online_input_cost = self.online_tokens_in * c["online_cost_per_1k_input"] / 1000
        online_output_cost = self.online_tokens_out * c["online_cost_per_1k_output"] / 1000
        return local_cost + online_input_cost + online_output_cost

    def estimate_remaining(self):
        if self.total_calls == 0:
            return {"pct": 0, "remaining_s": 0, "total_estimate_s": 0}

        avg_time = self.total_time / self.total_calls
        calls_projected = max(20, self.total_calls * 3)
        projected_s = avg_time * calls_projected
        pct = min(100, (self.total_calls / calls_projected) * 100)
        remaining_s = max(0, projected_s - self.total_time)

        return {
            "pct": round(pct, 1),
            "remaining_s": round(remaining_s, 1),
            "total_estimate_s": round(projected_s, 1),
            "calls_projected": calls_projected,
        }

    def summary_dict(self):
        rem = self.estimate_remaining()
        cost = self.estimate_cost()
        return {
            "local": {
                "calls": self.local_calls,
                "tokens_in": self.local_tokens_in,
                "tokens_out": self.local_tokens_out,
                "tokens_total": self.local_tokens_in + self.local_tokens_out,
                "time_s": round(self.local_time, 2),
            },
            "online": {
                "calls": self.online_calls,
                "tokens_in": self.online_tokens_in,
                "tokens_out": self.online_tokens_out,
                "tokens_total": self.online_tokens_in + self.online_tokens_out,
                "time_s": round(self.online_time, 2),
            },
            "total": {
                "calls": self.total_calls,
                "tokens_in": self.total_tokens_in,
                "tokens_out": self.total_tokens_out,
                "tokens_total": self.total_tokens,
                "time_s": round(self.total_time, 2),
            },
            "cost": round(cost, 6),
            "progress_pct": rem["pct"],
            "remaining_s": rem["remaining_s"],
            "total_estimate_s": rem["total_estimate_s"],
            "session_duration_s": round(self.session_duration, 1),
        }
