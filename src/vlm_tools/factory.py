from __future__ import annotations

import os

from .models import normalize_vlm_name


def _resolve_provider_class(vlm_name: str):
    key = normalize_vlm_name(vlm_name)
    if key == "gpt":
        from .gpt import GPT4o

        return key, GPT4o

    if key == "gemini":
        from .gemini import Gemini

        return key, Gemini

    if key == "doubao":
        from .doubao import Doubao

        return key, Doubao

    if key == "qwen":
        from .qwen import QwenVL

        return key, QwenVL

    if key == "claude":
        from .claude import Claude

        return key, Claude

    raise ValueError(f"Unsupported VLM after normalization: {vlm_name}")


def build_vlm(vlm_name: str):
    key, provider_cls = _resolve_provider_class(vlm_name)
    resolved_model = provider_cls.default_model
    if not resolved_model:
        raise ValueError(f"No default model configured for VLM: {key}")
    api_url = provider_cls.default_api_url or ""
    api_key = ""
    env_vars = provider_cls.api_key_env_vars
    if env_vars is None and provider_cls.api_key_env_var:
        env_vars = (provider_cls.api_key_env_var,)
    if env_vars:
        for env_var in env_vars:
            api_key = os.getenv(env_var, "").strip()
            if api_key:
                break
        if not api_key:
            raise ValueError(
                f"Missing required environment variable for {key}. "
                f"Tried: {', '.join(env_vars)}"
            )

    return provider_cls(api_url=api_url, api_key=api_key, model=resolved_model)
