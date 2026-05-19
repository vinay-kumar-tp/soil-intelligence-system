# Package init with automatic fallback to root config.py to resolve namespace collisions
import importlib.util
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
config_py_path = root_dir / "config.py"

if config_py_path.exists():
    spec = importlib.util.spec_from_file_location("root_config", str(config_py_path))
    if spec and spec.loader:
        root_config = importlib.util.module_from_spec(spec)
        sys.modules["root_config"] = root_config
        spec.loader.exec_module(root_config)
        # Expose all its public attributes in this package namespace
        for attr in dir(root_config):
            if not attr.startswith("_"):
                globals()[attr] = getattr(root_config, attr)
