from __future__ import annotations

from typing import Set, Tuple
from .io_utils import load_json


def load_sampled_keys(sampled_json_path: str) -> Set[Tuple[str, str, str, str]]:
    data = load_json(sampled_json_path)
    if not isinstance(data, list):
        raise ValueError(f'Sampled json must be a list, got: {type(data)}')
    keys = set()
    for x in data:
        if not isinstance(x, dict):
            continue
        keys.add((x['category'], x.get('subtype') or '-', x['edit_type'], x['uid']))
    return keys
