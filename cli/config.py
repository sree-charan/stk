"""CLI config management."""
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / '.stk' / 'config.json'


def _load() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _save(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get(key: str, default: Any = None) -> Any:
    """Get a config value by key."""
    return _load().get(key, default)


def set_key(key: str, value: str) -> None:
    """Set a config key to a value and persist."""
    cfg = _load()
    cfg[key] = value
    _save(cfg)


def reset() -> None:
    """Reset config to empty defaults."""
    _save({})
