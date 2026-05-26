from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def _read_prompt(name: str) -> str:
    path = PROMPT_ROOT / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt_text_only():
    return _read_prompt("system_prompt_text_only.txt")


def load_system_prompt_image_only():
    return _read_prompt("system_prompt_image_only.txt")


def load_system_prompt_image_text():
    return _read_prompt("system_prompt_image_text.txt")
