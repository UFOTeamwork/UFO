from __future__ import annotations

from pathlib import Path


PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def _read_prompt(filename: str) -> str:
    path = PROMPT_ROOT/ filename

    with open(path, "rb") as f:
        raw = f.read()

    # 优先 UTF-8；失败则兼容中文 Windows/ANSI 编码
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace").strip()


def load_system_prompt_tgt() -> str:
    # return _read_prompt("system_prompt_tgt.txt")
    return _read_prompt("system_prompt_tgt_human.txt")

def load_system_prompt_qgen() -> str:
    return _read_prompt("system_prompt_qgen.txt")


