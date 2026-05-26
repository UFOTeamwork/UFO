from __future__ import annotations

from typing import Iterable, Optional


def summarize_values_ms(values: Iterable[Optional[float]]):
    vals = [float(v) * 1000.0 for v in values if v is not None]
    if not vals:
        return {'count': 0, 'mean_ms': None, 'min_ms': None, 'p50_ms': None, 'p95_ms': None, 'max_ms': None}
    vals = sorted(vals)
    n = len(vals)
    def pct(p: float):
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return vals[idx]
    return {'count': n, 'mean_ms': sum(vals)/n, 'min_ms': vals[0], 'p50_ms': pct(0.5), 'p95_ms': pct(0.95), 'max_ms': vals[-1]}


def summarize_values_raw(values: Iterable[Optional[float]]):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {'count': 0, 'mean': None, 'min': None, 'p50': None, 'p95': None, 'max': None}
    vals = sorted(vals)
    n = len(vals)
    def pct(p: float):
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return vals[idx]
    return {'count': n, 'mean': sum(vals)/n, 'min': vals[0], 'p50': pct(0.5), 'p95': pct(0.95), 'max': vals[-1]}
