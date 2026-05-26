from __future__ import annotations

VLM_NAME_ALIASES: dict[str, str] = {
    "gpt": "gpt",
    "gpt4o": "gpt",
    "gemini": "gemini",
    "doubao": "doubao",
    "qwen": "qwen",
    "qwenvl": "qwen",
    "claude": "claude",
}


def normalize_vlm_name(vlm_name: str) -> str:
    key = vlm_name.lower().strip()
    if key not in VLM_NAME_ALIASES:
        raise ValueError(f"Unsupported VLM: {vlm_name}")
    return VLM_NAME_ALIASES[key]

__all__ = ["VLM_NAME_ALIASES", "normalize_vlm_name"]
