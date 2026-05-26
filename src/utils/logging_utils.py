from __future__ import annotations

import json
import traceback
from pathlib import Path


def log(msg: str, level: str = 'INFO') -> None:
    print(f'[{level}] {msg}', flush=True)


def log_failure(stage: str, uid: str, category: str, subtype: str, edit_type: str, error: Exception, log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tb_text = ''.join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    record = {
        'stage': stage,
        'uid': uid,
        'category': category,
        'subtype': subtype,
        'edit_type': edit_type,
        'error_type': type(error).__name__,
        'error_msg': str(error),
        'traceback': tb_text,
    }
    print(
        f"[ERROR] stage={stage} uid={uid} category={category} subtype={subtype} "
        f"edit_type={edit_type} error={type(error).__name__}: {error}",
        flush=True,
    )
    if tb_text:
        print(tb_text, flush=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
