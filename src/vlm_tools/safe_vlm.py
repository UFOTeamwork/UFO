import time

class SafeVLM:
    def __init__(self, vlm):
        self.vlm = vlm

    def _check(self, text: str):
        if text is None:
            raise ValueError("Empty response")

        t = text.strip().lower()

        blacklist = [
            "error",
            "refusal",
            "rate limit",
            "timeout",
            "openai response did not contain",
            "null",
            "{}"
        ]

        for b in blacklist:
            if b in t:
                raise ValueError(f"Bad VLM output: {text}")

        return text

    def complete_text(self, *args, retry=3, **kwargs):
        last_err = None

        for _ in range(retry):
            try:
                out = self.vlm.complete_text(*args, **kwargs)
                return self._check(out)
            except Exception as e:
                last_err = e
                time.sleep(1)

        raise last_err