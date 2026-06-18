import json
import re

ERROR_KEYWORDS = [
    "rate limit",
    "quota",
    "timeout",
    "error",
    "exception",
    "failed",
    "unavailable",
    "api"
]

def is_invalid_response(text: str) -> bool:
    if not text or not isinstance(text, str):
        return True
    t = text.lower()
    return any(k in t for k in ERROR_KEYWORDS)


def safe_parse_json(text: str):
    """
    Force-extract JSON (guards against VLM output mixed with explanatory text).
    """
    if is_invalid_response(text):
        return None

    # try raw
    try:
        return json.loads(text)
    except:
        pass

    # extract first json block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None

    return None