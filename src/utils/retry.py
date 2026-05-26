from __future__ import annotations

import random
import time
from typing import Any, Callable


def now() -> float:
    return time.perf_counter()


def call_with_backoff(fn: Callable[[], Any], *, max_retries: int, base_sleep: float, max_sleep: float, retry_on: tuple[type[BaseException], ...] = (Exception,)):
    total_sleep_sec = 0.0
    t0 = now()
    for attempt in range(max_retries):
        try:
            result = fn()
            return {'result': result, 'retry_count': attempt, 'sleep_sec': total_sleep_sec, 'wall_sec': now() - t0}
        except retry_on as e:
            if attempt == max_retries - 1:
                raise
            sleep_sec = min(max_sleep, base_sleep * (2 ** attempt))
            sleep_sec = sleep_sec * (0.5 + random.random() / 2)
            print(f'[WARN] API failed ({type(e).__name__}: {e}), retry {attempt + 1}/{max_retries}, sleep {sleep_sec:.2f}s', flush=True)
            time.sleep(sleep_sec)
            total_sleep_sec += sleep_sec
