from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI

from src.utils.retry import call_with_backoff, now

from .base import BaseVLM, parse_yes_no


class Doubao(BaseVLM):
    default_api_url = "https://api.zhizengzeng.com/v1"
    default_model = "doubao-seed-1-6-vision-250815"
    api_key_env_var = "DOUBAO_API_KEY"
    api_key_env_vars = ("UFO_VLM_API_KEY", "DOUBAO_API_KEY")
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
            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
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

    def _extract_text(self, response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            raise ValueError("Empty Doubao response or no choices")
        content = response.choices[0].message.content
        if isinstance(content, str):
            text = content.strip()
        else:
            text = str(content).strip()
        if not text:
            raise ValueError("Doubao response content is empty")
        return text

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
            answer = self._normalize_yes_no_text(self._extract_text(api_info["result"]))
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
