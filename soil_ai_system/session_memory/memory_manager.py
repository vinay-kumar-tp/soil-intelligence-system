"""Phase 7C - Session Memory Intelligence.

Maintains session-level field telemetry histories, recurring deficiencies,
and weather patterns to optimize crop and soil recommendation stability.
"""
from typing import Dict, Any, List, Optional
import time

class SessionMemoryManager:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.history: List[Dict[str, Any]] = []
        
    def add_entry(self, telemetry: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Stores standard, anonymized field parameters and AI outputs in history."""
        entry = {
            "timestamp": time.time(),
            "N": float(telemetry.get("N", 50)),
            "P": float(telemetry.get("P", 40)),
            "K": float(telemetry.get("K", 40)),
            "temperature": float(telemetry.get("temperature", 25.0)),
            "humidity": float(telemetry.get("humidity", 50.0)),
            "rainfall": float(telemetry.get("rainfall", 100.0)),
            "crop_prediction": results.get("predictions", {}).get("crop", {}).get("prediction", ""),
            "deficiency": results.get("predictions", {}).get("deficiency", {}).get("prediction", "")
        }
        
        self.history.append(entry)
        if len(self.history) > self.capacity:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
        
    def clear(self) -> None:
        self.history.clear()
        
    def detect_recurring_deficiencies(self) -> List[str]:
        """Analyzes past runs in session to isolate persistent nutrient gaps."""
        if len(self.history) < 2:
            return []
            
        deficiencies = [e["deficiency"].lower() for e in self.history if e["deficiency"]]
        recurring = []
        for item in set(deficiencies):
            if deficiencies.count(item) >= 2 and item != "healthy":
                recurring.append(item.capitalize())
        return recurring

    def detect_environmental_drift(self) -> Dict[str, Any]:
        """Calculates temperature and rainfall trends over session timelines."""
        if len(self.history) < 2:
            return {"drought_trend": False, "rainfall_trend": "stable", "temp_drift": 0.0}
            
        temps = [e["temperature"] for e in self.history]
        temp_drift = temps[-1] - temps[0]
        
        # Drought trend: rising temperature paired with decreasing humidity
        hums = [e["humidity"] for e in self.history]
        drought_trend = (temp_drift > 2.0) and (hums[-1] < hums[0])
        
        rains = [e["rainfall"] for e in self.history]
        if rains[-1] < rains[0] - 50.0:
            rain_trend = "drying"
        elif rains[-1] > rains[0] + 50.0:
            rain_trend = "wetting"
        else:
            rain_trend = "stable"
            
        return {
            "drought_trend": drought_trend,
            "rainfall_trend": rain_trend,
            "temp_drift": round(temp_drift, 1)
        }

# Global singleton representing active session memory
global_session_memory = SessionMemoryManager()
