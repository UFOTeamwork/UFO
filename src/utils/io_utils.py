from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def atomic_json_dump(obj: Any, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
