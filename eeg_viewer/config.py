"""
Configuration loader.
All parameters come from config.yaml - nothing hardcoded in logic modules.
"""

import yaml
from pathlib import Path


def load_config(config_path="config.yaml") -> dict:
    """Load YAML config and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)