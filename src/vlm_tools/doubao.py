from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI

from src.utils.retry import call_with_backoff, now

from .base import BaseVLM, parse_yes_no


class Doubao(BaseVLM):
    default_api_url = "https://api.zhizengzeng.com/v1"
    default_model = "doubao-seed-1-6-vision-250815"

    api_key_env_var = "DOUBAO_API_KEY"
    api_key_env_vars = (
        "UFO_VLM_API_KEY",
        "DOUBAO_API_KEY",
    )

    default_text_max_tokens = 2048
    default_multimodal_max_tokens = 2048

    default_retry_max_retries = 6
    default_retry_base_sleep = 2.0
    default_retry_max_sleep = 20.0

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
    ):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            model=model,
        )

        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )

    # ========================================================
    # Image
    # ========================================================

    def _build_image_content(
        self,
        image: Any,
    ) -> dict[str, Any]:

        image_b64, mime_type = (
            self._image_to_base64(
                image
            )
        )

        return {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{mime_type};base64,"
                    f"{image_b64}"
                )
            },
        }

    # ========================================================
    # API
    # ========================================================

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        user_content,
        max_tokens: int,
    ):

        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

    # ========================================================
    # Response extraction
    # ========================================================

    def _extract_text(
        self,
        response: Any,
    ) -> str:

        if (
            not response
            or not getattr(
                response,
                "choices",
                None,
            )
        ):
            raise ValueError(
                "Empty Doubao response or no choices"
            )

        message = (
            response
            .choices[0]
            .message
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            text = content.strip()

        else:
            text = str(
                content
            ).strip()

        if not text:
            raise ValueError(
                "Doubao response content is empty"
            )

        return text

    # ========================================================
    # Main Yes / No evaluation
    # ========================================================

    def ask_vlm(
        self,
        prompt: str,
        images=None,
        image_roles=None,
        system_prompt=None,
        debug: bool = True,
    ):
        """
        Main binary VLM judgment.

        Important semantics:

        1. Raw VLM output is parsed exactly once by parse_yes_no().
        2. API / extraction / parsing failures are technical failures.
        3. Technical failures MUST propagate upward.
        4. Technical failures MUST NOT be converted into a valid "no".
        """

        t0_total = now()

        content = []

        encode_sec = 0.0

        # ====================================================
        # Images
        # ====================================================

        if images:

            if image_roles is None:
                raise ValueError(
                    "image_roles must be provided "
                    "when images are used"
                )

            if len(images) != len(
                image_roles
            ):
                raise ValueError(
                    "images and image_roles "
                    "length mismatch"
                )

            for role, img in zip(
                image_roles,
                images,
            ):

                content.append({
                    "type": "text",
                    "text": f"{role}:",
                })

                t0_enc = now()

                content.append(
                    self._build_image_content(
                        img
                    )
                )

                encode_sec += (
                    now()
                    - t0_enc
                )

        # ====================================================
        # Question
        # ====================================================

        content.append({
            "type": "text",
            "text": prompt,
        })

        # ====================================================
        # API call
        # ====================================================

        def _api_call():

            return self._chat_completion(
                system_prompt=system_prompt,
                user_content=content,

                # Binary classification.
                # Current prompt requires exactly yes/no.
                max_tokens=32,
            )

        try:

            api_info = call_with_backoff(
                _api_call,

                max_retries=(
                    self.default_retry_max_retries
                ),

                base_sleep=(
                    self.default_retry_base_sleep
                ),

                max_sleep=(
                    self.default_retry_max_sleep
                ),
            )

        except Exception as e:

            if debug:

                print(
                    "[ERROR] Doubao ask_vlm API "
                    "failed after retries:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            # Important:
            # Never convert API failure into "no".
            raise RuntimeError(
                "Doubao ask_vlm API failed "
                "after retries | "
                f"{type(e).__name__}: {e}"
            ) from e

        # ====================================================
        # Extract RAW model response
        # ====================================================

        try:

            response = api_info[
                "result"
            ]

            raw_answer = (
                self._extract_text(
                    response
                )
            )

        except Exception as e:

            if debug:

                print(
                    "[ERROR] Doubao response "
                    "extraction failed:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            raise RuntimeError(
                "Doubao response extraction "
                "failed | "
                f"{type(e).__name__}: {e}"
            ) from e

        if debug:

            print(
                "[DEBUG] Raw Doubao response:",
                raw_answer,
                flush=True,
            )

        # ====================================================
        # Parse exactly ONCE
        # ====================================================

        try:

            answer_norm, likelist = (
                parse_yes_no(
                    raw_answer
                )
            )

        except Exception as e:

            if debug:

                print(
                    "[ERROR] Cannot parse "
                    "Doubao yes/no response:",
                    raw_answer[:500],
                    flush=True,
                )

            # Important:
            # Parsing failure != No.
            raise RuntimeError(
                "Doubao yes/no parsing failed | "
                f"{type(e).__name__}: {e}"
            ) from e

        if debug:

            print(
                "[DEBUG] Parsed Doubao answer:",
                answer_norm,
                "| likelist=",
                likelist,
                flush=True,
            )

        # ====================================================
        # Success
        # ====================================================

        return {
            "answer":
                answer_norm,

            "likelist":
                likelist,

            "latency": {
                "encode_sec":
                    encode_sec,

                "api_wall_sec":
                    api_info[
                        "wall_sec"
                    ],

                "retry_count":
                    api_info[
                        "retry_count"
                    ],

                "sleep_sec":
                    api_info[
                        "sleep_sec"
                    ],

                "total_sec":
                    now()
                    - t0_total,
            },
        }

    # ========================================================
    # Explain a valid NO judgment
    # ========================================================

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
        """
        Auxiliary explanation only.

        This method does NOT participate in the numerical score.

        If it fails, the technical failure is propagated to
        QuestionEvaluator, where it should be logged as a warning
        and should NOT invalidate the already valid yes/no score.
        """

        t0_total = now()

        explanation_prompt = (
            "The previous judgment for the following "
            "question was NO.\n\n"

            f"Question:\n{question}\n\n"

            f"Reference Text:\n{prompt}\n\n"

            "Your task:\n"
            "Explain briefly WHY the generated image "
            "does NOT satisfy the requirement.\n"
            "Do NOT re-evaluate the answer.\n"
            "Do NOT say Yes or No.\n"
            "Only explain the reason."
        )

        content = [
            {
                "type": "text",
                "text": explanation_prompt,
            }
        ]

        encode_sec = 0.0

        for role, img in zip(
            [
                "Reference Image",
                "Generated Image",
            ],
            [
                ref_img,
                gen_img,
            ],
        ):

            content.append({
                "type": "text",
                "text": f"{role}:",
            })

            t0_enc = now()

            content.append(
                self._build_image_content(
                    img
                )
            )

            encode_sec += (
                now()
                - t0_enc
            )

        def _api_call():

            return self._chat_completion(
                system_prompt=system_prompt,
                user_content=content,
                max_tokens=256,
            )

        try:

            api_info = call_with_backoff(
                _api_call,

                max_retries=(
                    self.default_retry_max_retries
                ),

                base_sleep=(
                    self.default_retry_base_sleep
                ),

                max_sleep=(
                    self.default_retry_max_sleep
                ),
            )

            reason = self._extract_text(
                api_info[
                    "result"
                ]
            )

            if debug:

                print(
                    "[DEBUG] No-reason:",
                    reason,
                    flush=True,
                )

            return {
                "reason":
                    reason,

                "latency": {
                    "encode_sec":
                        encode_sec,

                    "api_wall_sec":
                        api_info[
                            "wall_sec"
                        ],

                    "retry_count":
                        api_info[
                            "retry_count"
                        ],

                    "sleep_sec":
                        api_info[
                            "sleep_sec"
                        ],

                    "total_sec":
                        now()
                        - t0_total,
                },
            }

        except Exception as e:

            if debug:

                print(
                    "[ERROR] Doubao "
                    "ask_no_reason failed "
                    "after retries:",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            # Auxiliary failure:
            # propagate to QuestionEvaluator.
            #
            # QuestionEvaluator should:
            #   logger.warning(...)
            #   reason = ""
            #
            # but keep the valid main score.
            raise RuntimeError(
                "Doubao ask_no_reason failed "
                "after retries | "
                f"{type(e).__name__}: {e}"
            ) from e

    # ========================================================
    # Generic text completion
    # ========================================================

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_text: str,
        max_tokens: int | None = None,
    ) -> str:

        response = (
            self._chat_completion(
                system_prompt=
                    system_prompt,

                user_content=
                    user_text,

                max_tokens=
                    self._resolve_max_tokens(
                        max_tokens,
                        multimodal=False,
                    ),
            )
        )

        return self._extract_text(
            response
        )

    # ========================================================
    # Generic multimodal completion
    # ========================================================

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
                raise ValueError(
                    "image_roles must be provided "
                    "when images are used"
                )

            if len(images) != len(
                image_roles
            ):
                raise ValueError(
                    "images and image_roles "
                    "length mismatch"
                )

            for role, image in zip(
                image_roles,
                images,
            ):

                content.append({
                    "type": "text",
                    "text": f"{role}:",
                })

                content.append(
                    self._build_image_content(
                        image
                    )
                )

        content.append({
            "type": "text",
            "text": user_text,
        })

        response = (
            self._chat_completion(
                system_prompt=
                    system_prompt,

                user_content=
                    content,

                max_tokens=
                    self._resolve_max_tokens(
                        max_tokens,
                        multimodal=True,
                    ),
            )
        )

        return self._extract_text(
            response
        )