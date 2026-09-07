from __future__ import annotations

import base64
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

from PIL import Image


def parse_yes_no(text: str) -> tuple[str, int]:
    """
    Parse the FINAL yes/no judgment from a VLM response.

    Supported examples:
        yes
        no
        YES
        **yes**
        `no`
        Answer: yes
        Final answer: no
        ### answer
        yes
        ### decision
        **no**

    Reasoning text is allowed, but only an explicit final answer is used.
    Arbitrary occurrences of "yes" / "no" inside the reasoning body are
    intentionally ignored to avoid false parsing.
    """
    if not isinstance(text, str):
        raise ValueError(
            "Cannot parse yes/no from non-string value: "
            f"{type(text).__name__}"
        )

    raw = text.strip()
    if not raw:
        raise ValueError("Cannot parse yes/no from empty response.")

    # 1) Exact answer, optionally wrapped in common Markdown punctuation.
    cleaned = re.sub(
        r"^[\s#*_`>\-:]+|[\s#*_`>.,;:\-!]+$",
        "",
        raw,
    ).strip().lower()

    if cleaned == "yes":
        return "yes", 1
    if cleaned == "no":
        return "no", 0

    # 2) Explicit answer/decision marker.
    #    If several markers exist, the last one is treated as final.
    marker_pattern = re.compile(
        r"(?:^|\n)"
        r"\s*"
        r"(?:#{1,6}\s*)?"
        r"(?:\*\*)?"
        r"(?:final\s+)?"
        r"(?:answer|decision)"
        r"(?:\*\*)?"
        r"\s*"
        r"[:\-]?"
        r"\s*"
        r"(?:\n\s*)?"
        r"[*_`]*"
        r"\b(yes|no)\b"
        r"[*_`]*",
        re.IGNORECASE,
    )

    marker_matches = list(marker_pattern.finditer(raw))
    if marker_matches:
        answer = marker_matches[-1].group(1).lower()
        return answer, 1 if answer == "yes" else 0

    # 3) Last non-empty line is exactly yes/no after light Markdown cleanup.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        last_clean = re.sub(
            r"^[\s#*_`>\-:]+|[\s#*_`>.,;:\-!]+$",
            "",
            lines[-1],
        ).strip().lower()

        if last_clean == "yes":
            return "yes", 1
        if last_clean == "no":
            return "no", 0

    # 4) Explicit final sentence at the end of the response.
    sentence_pattern = re.compile(
        r"(?:"
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)"
        r"|"
        r"(?:the\s+)?(?:final\s+)?decision\s*(?:is|:)"
        r")"
        r"\s*"
        r"[*_`]*"
        r"\b(yes|no)\b"
        r"[*_`]*"
        r"[.!]?"
        r"\s*$",
        re.IGNORECASE,
    )

    sentence_match = sentence_pattern.search(raw)
    if sentence_match:
        answer = sentence_match.group(1).lower()
        return answer, 1 if answer == "yes" else 0

    raise ValueError(
        "Cannot reliably parse final yes/no answer from VLM response: "
        f"{raw[:500]}"
    )


@dataclass(frozen=True)
class PreparedImage:
    data_base64: str
    mime_type: str


class BaseVLM(ABC):
    default_api_url: str | None = None
    default_model: str | None = None
    api_key_env_var: str | None = None
    api_key_env_vars: tuple[str, ...] | None = None
    default_text_max_tokens = 1024
    default_multimodal_max_tokens = 1024

    def __init__(self, *, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _resolve_max_tokens(self, requested: int | None, *, multimodal: bool) -> int:
        if requested is not None:
            return requested
        return self.default_multimodal_max_tokens if multimodal else self.default_text_max_tokens

    def _infer_mime_type(self, image_path: str | None = None, image_format: str | None = None) -> str:
        if image_format:
            normalized = image_format.strip().lower()
            if normalized in {"jpg", "jpeg"}:
                return "image/jpeg"
            if normalized == "png":
                return "image/png"
            if normalized == "webp":
                return "image/webp"
        if image_path:
            guessed, _ = mimetypes.guess_type(image_path)
            if guessed and guessed.startswith("image/"):
                return guessed
        return "image/jpeg"

    def prepare_image(self, image: Any) -> PreparedImage:
        image_b64, mime_type = self._image_to_base64(image)
        return PreparedImage(data_base64=image_b64, mime_type=mime_type)

    def prepare_images(self, images: Sequence[Any]) -> list[Any]:
        return [self.prepare_image(image) for image in images]

    def _image_to_base64(self, image: Any) -> tuple[str, str]:
        if isinstance(image, PreparedImage):
            return image.data_base64, image.mime_type

        if isinstance(image, str):
            with open(image, "rb") as f:
                payload = f.read()
            return base64.b64encode(payload).decode("utf-8"), self._infer_mime_type(image_path=image)

        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")

        save_format = image.format or "JPEG"
        mime_type = self._infer_mime_type(image_format=save_format)
        export_image = image
        if save_format.upper() == "JPEG" and image.mode not in {"RGB", "L"}:
            export_image = image.convert("RGB")
        buf = BytesIO()
        export_image.save(buf, format=save_format)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), mime_type

    def _normalize_yes_no_text(self, text: str) -> str:
        """
        Normalize a VLM response to exactly "yes" or "no".

        All yes/no parsing is delegated to parse_yes_no() so the project has
        a single parsing policy. A parsing failure is intentionally propagated
        instead of being silently converted into a valid "no" judgment.
        """
        answer, _ = parse_yes_no(text)
        return answer


    @abstractmethod
    def ask_vlm(self, prompt: str, images=None, image_roles=None, system_prompt=None, debug: bool = True):
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def complete_text(self, *, system_prompt: str, user_text: str, max_tokens: int | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: Sequence[Any],
        image_roles: Sequence[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError
