from __future__ import annotations

import random
import time
from typing import Any, Callable


def now() -> float:
    return time.perf_counter()


# HTTP status codes that represent permanent client-side errors. Retrying these
# is pointless (e.g. a deprecated/unknown model returns 400/404, bad key 401),
# so we fail fast instead of burning the whole retry budget.
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 422})


def is_retryable_error(exc: BaseException) -> bool:
    """Return False for permanent client errors that should not be retried.

    Works with openai/httpx style exceptions that expose ``status_code`` (or
    ``status``); anything without a recognizable permanent status is treated as
    retryable (network blips, timeouts, 429, 5xx, etc.).
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and status in NON_RETRYABLE_STATUS_CODES:
        return False
    return True


def call_with_backoff(
    fn: Callable[[], Any],
    *,
    max_retries: int,
    base_sleep: float,
    max_sleep: float,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    retryable: Callable[[BaseException], bool] | None = None,
):
    retryable = retryable or is_retryable_error
    total_sleep_sec = 0.0
    t0 = now()
    for attempt in range(max_retries):
        try:
            result = fn()
            return {'result': result, 'retry_count': attempt, 'sleep_sec': total_sleep_sec, 'wall_sec': now() - t0}
        except retry_on as e:
            # Fail fast on the last attempt or on permanent (non-retryable) errors.
            if attempt == max_retries - 1 or not retryable(e):
                raise
            sleep_sec = min(max_sleep, base_sleep * (2 ** attempt))
            sleep_sec = sleep_sec * (0.5 + random.random() / 2)
            print(f'[WARN] API failed ({type(e).__name__}: {e}), retry {attempt + 1}/{max_retries}, sleep {sleep_sec:.2f}s', flush=True)
            time.sleep(sleep_sec)
            total_sleep_sec += sleep_sec
