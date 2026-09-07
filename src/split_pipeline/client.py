# src/split_pipeline/client.py

import threading
import time
import random

from src.vlm_tools.factory import build_vlm

_VLM_CACHE = {}
_VLM_CACHE_LOCK = threading.Lock()


def init_vlm_client(cfg):
    """Build (and cache) the VLM backend used by the split pipeline.

    Mirrors the evaluator pipeline (see VLMMatcher), which constructs the
    backend via ``build_vlm(cfg['vlm']['provider'])``. The result is cached per
    provider so worker threads reuse one instance instead of rebuilding it for
    every pair (important for local backends such as Qwen).
    """
    vlm_cfg = cfg['vlm']
    provider = vlm_cfg['provider']
    model = vlm_cfg.get('model')
    cache_key = (provider, model)
    with _VLM_CACHE_LOCK:
        if cache_key not in _VLM_CACHE:
            _VLM_CACHE[cache_key] = build_vlm(provider, model=model)
        return _VLM_CACHE[cache_key]


class RobustVLMClient:
    def __init__(self, client, max_retry=3, timeout=20):
        self.client = client
        self.max_retry = max_retry
        self.timeout = timeout

    def call(self, messages):
        last_err = None

        for _ in range(self.max_retry):
            try:
                resp = self.client.chat.completions.create(
                    messages=messages,
                    timeout=self.timeout,
                )

                text = resp.choices[0].message.content

                if self._bad(text):
                    continue

                return text

            except Exception as e:
                last_err = str(e)
                time.sleep(0.5 + random.random())

        return None

    def _bad(self, text):
        if text is None:
            return True
        t = text.lower()
        return (
            len(text.strip()) == 0 or
            "error" in t or
            "rate limit" in t or
            "cannot" in t and "request" in t
        )