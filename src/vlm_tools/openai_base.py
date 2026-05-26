from __future__ import annotations

import base64
from typing import Any, Sequence

from openai import OpenAI

from .base import BaseVLM


class OpenAICompatibleVLM(BaseVLM):
    def __init__(self, *, api_url: str, api_key: str, model: str):
        super().__init__(api_url=api_url, api_key=api_key, model=model)
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self._image_data_url_cache: dict[str, str] = {}

    def _img_to_data_url(self, image: Any) -> str:
        cache_key, raw_bytes, mime_type = self._image_to_bytes(image)
        cached = self._image_data_url_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{payload}"
        self._image_data_url_cache[cache_key] = data_url
        return data_url

    def _extract_text(self, resp: Any) -> str:
        if not getattr(resp, "choices", None):
            raise ValueError("Empty OpenAI-compatible response or no choices")
        message = resp.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                part.get("text", "").strip()
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        else:
            text = ""
        if not text:
            raise ValueError("OpenAI-compatible response did not contain text content")
        return text

    def _token_param_name(self) -> str:
        return "max_tokens"

    def _temperature(self) -> float | None:
        return 0.0

    def _extra_create_kwargs(self) -> dict[str, Any]:
        return {}

    def _chat_completion(self, *, messages: list[dict[str, Any]], max_tokens: int) -> Any:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        create_kwargs[self._token_param_name()] = max_tokens
        temperature = self._temperature()
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        create_kwargs.update(self._extra_create_kwargs())
        return self.client.chat.completions.create(**create_kwargs)

    def _build_image_content(self, image: Any) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self._img_to_data_url(image)},
        }

    def complete_text(self, *, system_prompt: str, user_text: str, max_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        resp = self._chat_completion(
            messages=messages,
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=False),
        )
        return self._extract_text(resp)

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list[Any],
        image_roles: Sequence[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content = self._build_multimodal_user_content(
            user_text=user_text,
            images=images,
            image_roles=image_roles,
            image_item_builder=self._build_image_content,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        resp = self._chat_completion(
            messages=messages,
            max_tokens=self._resolve_max_tokens(max_tokens, multimodal=True),
        )
        return self._extract_text(resp)
