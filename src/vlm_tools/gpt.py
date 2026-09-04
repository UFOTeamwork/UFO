from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Sequence

from openai import OpenAI
from PIL import Image

from src.utils.retry import call_with_backoff, now

from .base import BaseVLM, parse_yes_no

class GPT4o(BaseVLM):
    # default_api_url = "https://api.zhizengzeng.com/v1"
    default_api_url = "http://api.yesapikey.com/v1" #hui chan API_URL = "http://66.206.9.230:4000/v1"

    # default_model = "gpt-4o-2024-05-13"#pre
    default_model = "gpt-4o"#"gpt-4o-2024-11-20"
    # default_model = "gpt-5.5"
    api_key_env_var = "OPENAI_API_KEY"
    api_key_env_vars = ("UFO_VLM_API_KEY", "OPENAI_API_KEY")
    default_text_max_tokens = 2048
    default_multimodal_max_tokens = 2048
    default_retry_max_retries = 6
    default_retry_base_sleep = 2.0
    default_retry_max_sleep = 20.0

    def __init__(self, *, api_url: str, api_key: str, model: str):
        super().__init__(api_url=api_url, api_key=api_key, model=model)
        self.client = OpenAI(base_url=api_url, api_key=api_key)

    def _encode_image(self, image: Image.Image) -> str:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image")
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return buf.getvalue().hex()

    def _build_image_content(self, image: Any) -> dict[str, Any]:
        image_b64, mime_type = self._image_to_base64(image)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
        }

    def _chat_completion(self, *, system_prompt: str, user_content, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # temperature=0.0,
            max_completion_tokens=max_tokens,
        )

    def _to_debug_string(self, value: Any, limit: int = 800) -> str:
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
            return str(nested_text or "").strip()

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

        if hasattr(part, "model_dump"):
            dumped = part.model_dump()
            if isinstance(dumped, dict):
                return self._extract_part_text(dumped)
        return ""

    # def _extract_text(self, response: Any) -> str:
    #     if not response or not getattr(response, "choices", None):
    #         raise ValueError("Empty OpenAI response or no choices")
    #     message = response.choices[0].message
    #     content = getattr(message, "content", None)
    #     if content is None:
    #         refusal = getattr(message, "refusal", None)
    #         if refusal:
    #             return str(refusal).strip()
    #         raise ValueError("OpenAI response content is None; message=" + self._to_debug_string(message))
    #     if isinstance(content, str):
    #         text = content.strip()
    #     elif isinstance(content, list):
    #         text = "\n".join(filter(None, (self._extract_part_text(part) for part in content))).strip()
    #     else:
    #         text = self._extract_part_text(content) or str(content).strip()
    #     if not text:
    #         raise ValueError(
    #             "OpenAI response did not contain text content; "
    #             f"content_type={type(content).__name__}; "
    #             "message="
    #             + self._to_debug_string(message)
    #         )
    #     return text
    def _extract_text(self, response: Any) -> str:

        if not response or not getattr(response, "choices", None):
            raise ValueError("Empty OpenAI response or no choices")

        message = response.choices[0].message

        # 标准 content
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        # content=list
        if isinstance(content, list):
            text = "\n".join(
                filter(None, (self._extract_part_text(p) for p in content))
            ).strip()

            if text:
                return text

        # 国产兼容接口常见字段
        for key in [
            "reasoning_content",
            "reasoning",
            "text",
            "output_text",
        ]:
            value = getattr(message, key, None)

            if isinstance(value, str) and value.strip():
                return value.strip()

        # model_dump兜底
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()

            for key in [
                "content",
                "reasoning_content",
                "reasoning",
                "text",
                "output_text",
            ]:
                value = dumped.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

        raise ValueError(
            "OpenAI response did not contain text content; "
            "message=" + self._to_debug_string(message, limit=3000)
        )

    def ask_vlm(self, prompt: str, images=None, image_roles=None, system_prompt=None, debug: bool = True):
        """
        Run the main binary VLM judgment.

        Important semantics:
        - Parse the RAW model response exactly once via parse_yes_no().
        - API / extraction / parsing failures are technical failures.
        - Technical failures MUST propagate to QuestionEvaluator.
        - Never convert a technical failure into a valid "no" judgment.
        """
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
            return self._chat_completion(
                system_prompt=system_prompt,
                user_content=content,
                max_tokens=2048,
            )

        # API call.
        try:
            api_info = call_with_backoff(
                _api_call,
                max_retries=self.default_retry_max_retries,
                base_sleep=self.default_retry_base_sleep,
                max_sleep=self.default_retry_max_sleep,
            )
        except Exception as e:
            if debug:
                print(
                    "[ERROR] ask_vlm API failed after retries:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            raise RuntimeError(
                "ask_vlm API failed after retries | "
                f"{type(e).__name__}: {e}"
            ) from e

        # Extract raw response text.
        try:
            response = api_info["result"]
            raw_answer = self._extract_text(response)
        except Exception as e:
            if debug:
                print(
                    "[ERROR] ask_vlm response extraction failed:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            raise RuntimeError(
                "ask_vlm response extraction failed | "
                f"{type(e).__name__}: {e}"
            ) from e

        if debug:
            print(
                "[DEBUG] Raw VLM response:",
                raw_answer,
                flush=True,
            )

        # Single yes/no parsing entry point.
        try:
            answer_norm, likelist = parse_yes_no(raw_answer)
        except Exception as e:
            if debug:
                print(
                    "[ERROR] Cannot parse VLM yes/no response:",
                    raw_answer[:500],
                    flush=True,
                )
            raise RuntimeError(
                "ask_vlm yes/no parsing failed | "
                f"{type(e).__name__}: {e}"
            ) from e

        if debug:
            print(
                "[DEBUG] Parsed answer:",
                answer_norm,
                "| likelist=",
                likelist,
                flush=True,
            )

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
        ref_img: Image.Image,
        gen_img: Image.Image,
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
            reason = self._extract_text(api_info["result"])
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
            if debug:
                print(
                    "[ERROR] ask_no_reason failed after retries:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            # Explanation is auxiliary. Propagate the technical failure and
            # let QuestionEvaluator log it and keep the already valid score.
            raise RuntimeError(
                "ask_no_reason failed after retries | "
                f"{type(e).__name__}: {e}"
            ) from e

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
        return self._extract_text(response)


class GptVLM(GPT4o):
    pass
