from .factory import build_vlm
from .safe_vlm import SafeVLM

_VLM_CACHE = None

def get_vlm(cfg):
    global _VLM_CACHE
    if _VLM_CACHE is not None:
        return _VLM_CACHE

    vlm = build_vlm(cfg["vlm"]["provider"], model=cfg["vlm"].get("model"))
    _VLM_CACHE = SafeVLM(vlm)
    return _VLM_CACHE