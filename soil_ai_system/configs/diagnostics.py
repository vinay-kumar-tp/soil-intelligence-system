import sys
import os
import platform
import subprocess
from pathlib import Path

def get_system_diagnostics():
    """Gather diagnostic information about the host environment."""
    diagnostics = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "is_venv": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
        "cwd": os.getcwd()
    }
    
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
        deps = [line.strip() for line in pip_freeze.split('\n') if line.strip()]
        diagnostics["dependencies"] = deps
    except Exception as e:
        diagnostics["dependencies"] = [f"Error gathering dependencies: {e}"]
        
    return diagnostics

if __name__ == "__main__":
    import pprint
    pprint.pprint(get_system_diagnostics())
