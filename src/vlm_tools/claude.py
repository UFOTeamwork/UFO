from __future__ import annotations

from typing import Any, Sequence

import requests

from src.utils.retry import call_with_backoff, now

from .base import BaseVLM, parse_yes_no


class Claude(BaseVLM):
    default_api_url = "https://api.zhizengzeng.com/anthropic"
    default_model = "claude-sonnet-4-6"
    api_key_env_var = "ANTHROPIC_API_KEY"
    api_key_env_vars = ("UFO_VLM_API_KEY", "ANTHROPIC_API_KEY")
    default_text_max_tokens = 2048
    default_multimodal_max_tokens = 2048
    default_retry_max_retries = 6
    default_retry_base_sleep = 2.0
    default_retry_max_sleep = 20.0

    def _build_image_content(self, image: Any) -> dict[str, Any]:
        image_b64, mime_type = self._image_to_base64(image)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_b64,
            },
        }

    def _extract_text_from_response(self, response: dict) -> str:
        if not isinstance(response, dict):
            raise TypeError(f"Claude response must be dict, got {type(response)}")
        content = response.get("content", [])
        if not isinstance(content, list):
            raise ValueError(f"Invalid Claude response content: {content}")
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        text = "\n".join(t for t in texts if t.strip()).strip()
        if not text:
            raise ValueError(f"No text found in Claude response: {response}")
        return text

    def _chat_completion(self, *, system_prompt, user_content, max_tokens):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }
        response = requests.post(
            f"{self.api_url.rstrip('/')}/v1/messages",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def ask_vlm(self, prompt: str, images=None, image_roles=None, system_prompt=None, debug: bool = True):
        t0_total = now()
        content = []
        encode_sec = 0.0
        if images:
            if image_roles is None:
                raise ValueError("image_roles must be provided when images are used")
            if len(images) != len(image_roles):
                raise ValueError("images and image_roles length mismatch")
            for role, img in zip(image_roles, images):
                content.append({"type": "text", "text": f"{role}:"})
                t0_enc = now()
                content.append(self._build_image_content(img))
                encode_sec += now() - t0_enc
        content.append({"type": "text", "text": prompt})

        def _api_call():
            return self._chat_completion(system_prompt=system_prompt, user_content=content, max_tokens=32)

        try:
            api_info = call_with_backoff(
                _api_call,
                max_retries=self.default_retry_max_retries,
                base_sleep=self.default_retry_base_sleep,
                max_sleep=self.default_retry_max_sleep,
            )
            answer = self._normalize_yes_no_text(self._extract_text_from_response(api_info["result"]))
            if debug:
                print("[DEBUG] Raw answer:", answer, flush=True)
        except Exception as e:
            if debug:
                print("[ERROR] ask_vlm failed after retries:", e, flush=True)
            return {
                "answer": "no",
                "likelist": 0,
                "latency": {
                    "encode_sec": encode_sec,
                    "api_wall_sec": 0.0,
                    "retry_count": 0,
                    "sleep_sec": 0.0,
                    "total_sec": now() - t0_total,
                },
            }

        answer_norm, likelist = parse_yes_no(answer)
        return {
            "answer": answer_norm,
            "likelist": likelist,
            "latency": {
                "encode_sec": encode_sec,
                "api_wall_sec": api_info["wall_sec"],
                "retry_count": api_info["retry_count"],
                "sleep_sec": api_info["sleep_sec"],
                "total_sec": now() - t0_total,
            },
        }

    def ask_no_reason(
        self,
        question: str,
        prompt: str,
        relevance: str,
        ref_img,
        gen_img,
        system_prompt: str,
        debug: bool = True,
    ):
        t0_total = now()
        explanation_prompt = (
            "The previous judgment for the following question was NO.\n\n"
            f"Question:\n{question}\n\n"
            f"Reference Text:\n{prompt}\n\n"
            "Your task:\n"
            "Explain briefly WHY the generated image does NOT satisfy the requirement.\n"
            "Do NOT re-evaluate the answer.\n"
            "Do NOT say Yes or No.\n"
            "Only explain the reason."
        )
        content = [{"type": "text", "text": explanation_prompt}]
        encode_sec = 0.0
        for role, img in zip(["Reference Image", "Generated Image"], [ref_img, gen_img]):
            content.append({"type": "text", "text": f"{role}:"})
            t0_enc = now()
            content.append(self._build_image_content(img))
            encode_sec += now() - t0_enc

        def _api_call():
            return self._chat_completion(system_prompt=system_prompt, user_content=content, max_tokens=256)

        try:
            api_info = call_with_backoff(
                _api_call,
                max_retries=self.default_retry_max_retries,
                base_sleep=self.default_retry_base_sleep,
                max_sleep=self.default_retry_max_sleep,
            )
            reason = self._extract_text_from_response(api_info["result"])
            if debug:
                print("[DEBUG] No-reason:", reason, flush=True)
            return {
                "reason": reason,
                "latency": {
                    "encode_sec": encode_sec,
                    "api_wall_sec": api_info["wall_sec"],
                    "retry_count": api_info["retry_count"],
                    "sleep_sec": api_info["sleep_sec"],
                    "total_sec": now() - t0_total,
                },
            }
        except Exception as e:
            return {
                "reason": f"failed to get reason after retries: {e}",
                "latency": {
                    "encode_sec": encode_sec,
                    "api_wall_sec": 0.0,
                    "retry_count": 0,
                    "sleep_sec": 0.0,
                    "total_sec": now() - t0_total,
                },
            }

    def complete_text(self, *, system_prompt: str, user_text: str, max_tokens: int | None = None) -> str:
        response = self._chat_completion(
            system_prompt=system_prompt,
            user_content=[{"type": "text", "text": user_text}],
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=False),
        )
        return self._extract_text_from_response(response)

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: Sequence[Any],
        image_roles: Sequence[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content = []
        if images:
            if image_roles is None:
                raise ValueError("image_roles must be provided when images are used")
            if len(images) != len(image_roles):
                raise ValueError("images and image_roles length mismatch")
            for role, image in zip(image_roles, images):
                content.append({"type": "text", "text": f"{role}:"})
                content.append(self._build_image_content(image))
        content.append({"type": "text", "text": user_text})
        response = self._chat_completion(
            system_prompt=system_prompt,
            user_content=content,
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=True),
        )
        return self._extract_text_from_response(response)


class ClaudeVLM(Claude):
    pass
