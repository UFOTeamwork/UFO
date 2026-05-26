from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

Config = Dict[str, Any]

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _resolve_env_string(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        resolved = os.getenv(env_name, default)
        if resolved is None:
            raise ValueError(f"Missing required environment variable: {env_name}")
        return resolved

    return _ENV_PATTERN.sub(repl, value)


def _resolve_env_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_values(v) for v in obj]
    if isinstance(obj, str):
        resolved = _resolve_env_string(obj)
        lowered = resolved.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return resolved
    return obj


def _validate_config(cfg: Config) -> None:
    vlm_cfg = cfg.get("vlm")
    if isinstance(vlm_cfg, dict):
        provider = vlm_cfg.get("provider")
        if provider is not None and (not isinstance(provider, str) or not provider.strip()):
            raise ValueError("Config value vlm.provider must be a non-empty string")


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Invalid config root type: {type(cfg)}')
    cfg = _resolve_env_values(cfg)
    _validate_config(cfg)
    return cfg
