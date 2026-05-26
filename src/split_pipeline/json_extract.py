from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _iter_candidate_texts(raw: str) -> list[str]:
    text = raw.strip()
    candidates = [text]

    answer_match = _ANSWER_PATTERN.search(text)
    if answer_match:
        candidates.append(answer_match.group(1).strip())

    for match in _JSON_FENCE_PATTERN.finditer(text):
        candidates.append(match.group(1).strip())

    return [candidate for candidate in candidates if candidate]


def _extract_balanced_json_text(text: str) -> str | None:
    start_positions = [idx for idx, char in enumerate(text) if char in "{["]
    end_char = "}" if "{" in text else "]"

    for start in start_positions:
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        end = text.rfind(closing)
        if end == -1 or end <= start:
            continue
        clean = text[start : end + 1]
        clean = re.sub(r",\s*([}\]])", r"\1", clean)
        clean = clean.replace("\ufeff", "").replace("\u200b", "")
        return clean
    return None


def extract_json_value(raw: str) -> tuple[Any, str]:
    last_error: JSONDecodeError | None = None

    for candidate in _iter_candidate_texts(raw):
        clean = _extract_balanced_json_text(candidate)
        if not clean:
            continue
        try:
            return json.loads(clean), clean
        except JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise JSONDecodeError("No JSON object or array found in model output", raw, 0)


__all__ = ["extract_json_value"]
