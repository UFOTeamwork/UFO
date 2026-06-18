from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from PIL import Image

from src.utils.retry import call_with_backoff, now

from .base import BaseVLM


class Gemini(BaseVLM):
    default_api_url = "https://api.zhizengzeng.com/v1/chat/completions"
    default_model = "gemini-3-pro-preview"
    api_key_env_var = "GEMINI_API_KEY"
    api_key_env_vars = ("UFO_VLM_API_KEY", "GEMINI_API_KEY")
    default_text_max_tokens = 2048
    default_multimodal_max_tokens = 2048
    default_retry_max_retries = 6
    default_retry_base_sleep = 2.0
    default_retry_max_sleep = 20.0

    def __init__(self, *, api_url: str, api_key: str, model: str):
        super().__init__(api_url=api_url, api_key=api_key, model=model)
        self.client = OpenAI(base_url=api_url, api_key=api_key)

    def _build_image_content(self, image: Any) -> dict[str, Any]:
        image_b64, mime_type = self._image_to_base64(image)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}"
            },
        }

    def _chat_completion(self, *, system_prompt: str, user_content, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def _to_debug_string(self, value: Any, limit: int = 1200) -> str:
        try:
            if hasattr(value, "model_dump"):
                text = json.dumps(value.model_dump(), ensure_ascii=False)
            elif hasattr(value, "dict"):
                text = json.dumps(value.dict(), ensure_ascii=False)
            elif isinstance(value, (dict, list, tuple)):
                text = json.dumps(value, ensure_ascii=False, default=str)
            else:
                text = str(value)
        except Exception:
            text = repr(value)
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text

    def _extract_part_text(self, part: Any) -> str:
        if part is None:
            return ""
        if isinstance(part, str):
            return part.strip()
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "text":
                return str(part.get("text", "")).strip()
            nested_text = part.get("text")
            if isinstance(nested_text, dict):
                return str(nested_text.get("value", "")).strip()
            if nested_text is not None:
                return str(nested_text).strip()
            if "content" in part:
                return self._extract_part_text(part.get("content"))
            return ""

        part_type = getattr(part, "type", None)
        if part_type == "text":
            text_value = getattr(part, "text", None)
            if isinstance(text_value, str):
                return text_value.strip()
            if text_value is not None:
                value = getattr(text_value, "value", None)
                if isinstance(value, str):
                    return value.strip()
                return str(text_value).strip()

        text_attr = getattr(part, "text", None)
        if isinstance(text_attr, str):
            return text_attr.strip()
        if text_attr is not None:
            value = getattr(text_attr, "value", None)
            if isinstance(value, str):
                return value.strip()
            return str(text_attr).strip()

        content_attr = getattr(part, "content", None)
        if content_attr is not None:
            return self._extract_part_text(content_attr)

        if hasattr(part, "model_dump"):
            dumped = part.model_dump()
            if isinstance(dumped, dict):
                return self._extract_part_text(dumped)
        return ""

    def _extract_text(self, response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            raise ValueError("Empty Gemini response or no choices")

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            refusal = getattr(message, "refusal", None)
            if refusal:
                return str(refusal).strip()
            raise ValueError("Gemini response content is None; message=" + self._to_debug_string(message))

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(filter(None, (self._extract_part_text(part) for part in content))).strip()
        else:
            text = self._extract_part_text(content) or str(content).strip()

        if not text:
            raise ValueError(
                "Gemini response did not contain text content; "
                f"content_type={type(content).__name__}; "
                "message=" + self._to_debug_string(message)
            )
        return text

    def _debug_response_summary(self, response: Any, *, tag: str) -> None:
        choice = response.choices[0] if response and getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            content_len = len(content)
        elif isinstance(content, list):
            content_len = len("\n".join(filter(None, (self._extract_part_text(part) for part in content))))
        else:
            content_len = len(self._extract_part_text(content)) if content is not None else 0

        print(
            f"[DEBUG] {tag} response summary: "
            f"finish_reason={finish_reason!r}, "
            f"usage={self._to_debug_string(usage, limit=400)}, "
            f"content_type={type(content).__name__}, "
            f"content_len={content_len}",
            flush=True,
        )
        print(
            f"[DEBUG] {tag} raw first message: {self._to_debug_string(message)}",
            flush=True,
        )

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
                image_content = self._build_image_content(img)
                t1_enc = now()
                encode_sec += t1_enc - t0_enc

                content.append(image_content)

        content.append({"type": "text", "text": prompt})

        def _api_call():
            response = self._chat_completion(
                system_prompt=system_prompt,
                user_content=content,
                max_tokens=32,
            )

            if not response or not getattr(response, "choices", None):
                raise ValueError("Empty API response or no choices")

            answer = self._extract_text(response).strip().lower()
            if not answer:
                raise ValueError("API returned empty content")

            bad_signals = [
                "余额不足",
                "insufficient",
                "quota",
                "rate limit",
                "rate_limit",
                "too many requests",
                "credit",
                "billing",
                "unauthorized",
                "forbidden",
                "html",
                "<html",
                "bad gateway",
                "service unavailable",
                "request failed",
                "error",
            ]
            if any(x in answer for x in bad_signals):
                raise ValueError(f"API returned non-score error text: {answer[:200]}")

            if answer.startswith("yes"):
                return {"answer": "yes", "likelist": 1, "raw_answer": answer}
            if answer.startswith("no"):
                return {"answer": "no", "likelist": 0, "raw_answer": answer}

            raise ValueError(f"Cannot parse yes/no from model output: {answer[:200]}")

        try:
            api_info = call_with_backoff(
                _api_call,
                max_retries=self.default_retry_max_retries,
                base_sleep=self.default_retry_base_sleep,
                max_sleep=self.default_retry_max_sleep,
            )
            parsed = api_info["result"]

            if debug:
                self._debug_response_summary(api_info["result"], tag="Gemini yes/no")
                print("[DEBUG] Raw answer:", parsed["raw_answer"], flush=True)

            return {
                "answer": parsed["answer"],
                "likelist": parsed["likelist"],
                "latency": {
                    "encode_sec": encode_sec,
                    "api_wall_sec": api_info["wall_sec"],
                    "retry_count": api_info["retry_count"],
                    "sleep_sec": api_info["sleep_sec"],
                    "total_sec": now() - t0_total,
                },
            }

        except Exception as e:
            if debug:
                print("[ERROR] ask_vlm failed after retries:", repr(e), flush=True)
            raise RuntimeError(f"Failed to obtain valid yes/no answer after retries: {e}")

    def ask_no_reason(
        self,
        question: str,
        prompt: str,
        relevance: str,
        ref_img: Image.Image,
        gen_img: Image.Image,
        system_prompt: str,
        debug: bool = True,
    ):
        t0_total = now()

        explanation_prompt = f"""
The previous judgment for the following question was NO.

Question:
{question}

Reference Text:
{prompt}

Your task:
Explain briefly WHY the generated image does NOT satisfy the requirement.
Do NOT re-evaluate the answer.
Do NOT say Yes or No.
Only explain the reason.
"""

        content = [{"type": "text", "text": explanation_prompt}]
        encode_sec = 0.0

        for role, img in zip(["Reference Image", "Generated Image"], [ref_img, gen_img]):
            content.append({"type": "text", "text": f"{role}:"})

            t0_enc = now()
            image_content = self._build_image_content(img)
            t1_enc = now()
            encode_sec += t1_enc - t0_enc

            content.append(image_content)

        def _api_call():
            return self._chat_completion(
                system_prompt=system_prompt,
                user_content=content,
                max_tokens=256,
            )

        try:
            api_info = call_with_backoff(
                _api_call,
                max_retries=self.default_retry_max_retries,
                base_sleep=self.default_retry_base_sleep,
                max_sleep=self.default_retry_max_sleep,
            )
            response = api_info["result"]
            reason = self._extract_text(response)
            if debug:
                self._debug_response_summary(response, tag="Gemini no-reason")
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
                "reason": f"[reason_generation_failed] {str(e)}",
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
            user_content=user_text,
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=False),
        )
        return self._extract_text(response)

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images,
        image_roles=None,
        max_tokens: int | None = None,
    ) -> str:
        content = []

        if images:
            if image_roles is None:
                raise ValueError("image_roles must be provided when images are used")
            if len(images) != len(image_roles):
                raise ValueError("images and image_roles length mismatch")

            for role, img in zip(image_roles, images):
                content.append({"type": "text", "text": f"{role}:"})
                content.append(self._build_image_content(img))

        content.append({"type": "text", "text": user_text})

        print(
            "[DEBUG] Gemini multimodal request summary: "
            f"model={self.model}, images={len(images) if images else 0}, "
            f"image_roles={list(image_roles) if image_roles is not None else None}, "
            f"user_text={user_text!r}, max_tokens={self._resolve_max_tokens(max_tokens, multimodal=True)}",
            flush=True,
        )

        response = self._chat_completion(
            system_prompt=system_prompt,
            user_content=content,
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=True),
        )
        self._debug_response_summary(response, tag="Gemini multimodal")
        return self._extract_text(response)


class GeminiVLM(Gemini):
    pass
