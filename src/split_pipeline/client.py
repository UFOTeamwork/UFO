from __future__ import annotations

from typing import Any, Dict
from src.vlm_tools.factory import build_vlm


def init_vlm_client(cfg: Dict[str, Any]):
    vlm_cfg = cfg['vlm']
    return build_vlm(
        vlm_cfg['provider'],
        # model=vlm_cfg.get('model'),
    )
