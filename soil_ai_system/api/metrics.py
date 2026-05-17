import json
import os
import time
from pathlib import Path
from config.environment import LOGS_DIR
import threading

METRICS_FILE = LOGS_DIR / "metrics.json"

class SystemMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = self._load()
        
        # Initialize structure if empty
        if not self.data:
            self.data = {
                "total_requests": 0,
                "total_errors": 0,
                "latency_history_ms": [],
                "model_selections": {},
                "uptime_start": time.time(),
                "cache_hits": 0,
                "cache_misses": 0
            }

    def _load(self):
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def _save(self):
        with open(METRICS_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_request(self, latency_ms: float, error: bool = False):
        with self.lock:
            self.data["total_requests"] += 1
            if error:
                self.data["total_errors"] += 1
            
            self.data["latency_history_ms"].append(latency_ms)
            # Keep only last 100 requests for moving average
            if len(self.data["latency_history_ms"]) > 100:
                self.data["latency_history_ms"].pop(0)
                
            self._save()

    def record_prediction(self, crop: str):
        with self.lock:
            if crop not in self.data["model_selections"]:
                self.data["model_selections"][crop] = 0
            self.data["model_selections"][crop] += 1
            self._save()
            
    def record_cache_hit(self):
        with self.lock:
            self.data["cache_hits"] += 1
            self._save()
            
    def record_cache_miss(self):
        with self.lock:
            self.data["cache_misses"] += 1
            self._save()

    def get_snapshot(self):
        with self.lock:
            avg_latency = 0
            if self.data["latency_history_ms"]:
                avg_latency = sum(self.data["latency_history_ms"]) / len(self.data["latency_history_ms"])
                
            uptime_seconds = time.time() - self.data["uptime_start"]
            
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_requests": self.data["total_requests"],
                "total_errors": self.data["total_errors"],
                "error_rate_percent": round((self.data["total_errors"] / max(1, self.data["total_requests"])) * 100, 2),
                "average_latency_ms": round(avg_latency, 2),
                "cache_hit_ratio": round(self.data["cache_hits"] / max(1, (self.data["cache_hits"] + self.data["cache_misses"])), 2),
                "top_predictions": self.data["model_selections"]
            }

global_metrics = SystemMetrics()
