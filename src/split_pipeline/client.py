# src/split_pipeline/client.py

import time
import random

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