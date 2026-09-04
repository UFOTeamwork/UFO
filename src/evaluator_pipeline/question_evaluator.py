# question_evaluator.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from .prompts import (
    load_system_prompt_image_only,
    load_system_prompt_image_text,
    load_system_prompt_text_only,
)
from .vlm_matcher import VLMMatcher


logger = logging.getLogger(__name__)


class QuestionEvaluator:
    def __init__(self, cfg: Dict[str, Any]):
        self.matcher = VLMMatcher(cfg)

        self.prompt_text_only = load_system_prompt_text_only()
        self.prompt_image_only = load_system_prompt_image_only()
        self.prompt_image_text = load_system_prompt_image_text()

        # ====================================================
        # ArcFace
        # ====================================================

        face_cfg = dict(cfg.get("face_similarity", {}))
        self.face = None

        if face_cfg.get("enabled", False):
            # Heavy dependencies are loaded lazily.
            import torch

            from .face_similarity import FaceSimilarityEvaluator

            use_cuda = (
                bool(face_cfg.get("use_cuda", False))
                and torch.cuda.is_available()
            )

            face_cfg["use_cuda"] = use_cuda
            face_cfg["remove_bg"] = bool(
                face_cfg.get("remove_bg", False)
            )

            self.face = FaceSimilarityEvaluator(
                **face_cfg
            )

    # ========================================================
    # Prompt selection
    # ========================================================

    def _system_prompt(
        self,
        relevance: str,
    ) -> str:

        if relevance == "text_only":
            return self.prompt_text_only

        if relevance == "image_only":
            return self.prompt_image_only

        if relevance == "text_and_image":
            return self.prompt_image_text

        raise ValueError(
            f"Unknown relevance: {relevance}"
        )

    # ========================================================
    # Evaluate one Question
    # ========================================================

    def evaluate_question(
        self,
        q: Dict[str, Any],
        *,
        prompt: str,
        ref_img,
        gen_img,
        vlm_ref_img=None,
        vlm_gen_img=None,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if not isinstance(q, dict):
            raise ValueError(
                "Question must be a dict."
            )

        relevance = q.get("relevance")

        if relevance not in {
            "text_only",
            "image_only",
            "text_and_image",
        }:
            raise ValueError(
                f"Invalid question relevance: {relevance}"
            )

        question_obj = q.get("question")

        if not isinstance(question_obj, dict):
            raise ValueError(
                "Question missing valid 'question' object."
            )

        question_text = question_obj.get("question")

        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(
                "Question text is empty."
            )

        # ----------------------------------------------------
        # Prepare VLM images
        # ----------------------------------------------------

        if vlm_ref_img is None or vlm_gen_img is None:
            vlm_ref_img, vlm_gen_img = (
                self.matcher.prepare_images(
                    ref_img,
                    gen_img,
                )
            )

        reward = float(
            q.get("reward", 1.0)
        )

        if reward <= 0:
            raise ValueError(
                f"Invalid reward={reward} "
                f"for question={question_text!r}"
            )

        face_sec = 0.0

        use_arcface = (
            q.get("part") == "face"
            and self.face is not None
        )

        # ====================================================
        # Face question -> ArcFace first
        # ====================================================

        if use_arcface:

            t0_face = time.perf_counter()
            arcface_ok = False
            face_score = None

            try:
                face_result = (
                    self.face.compute_similarity_with_debug(
                        ref_img,
                        gen_img,
                    )
                )

                face_score = float(
                    face_result["cosine_similarity"]
                )

                # Keep ArcFace score in the same [0, 1] protocol
                # used by VLM question scores.
                face_score = max(
                    0.0,
                    min(
                        1.0,
                        face_score,
                    ),
                )

                arcface_ok = True

            except Exception as e:
                logger.warning(
                    "Face similarity failed | "
                    "question=%s | error=%s. "
                    "Falling back to VLM.",
                    question_text,
                    e,
                )

            face_sec = (
                time.perf_counter()
                - t0_face
            )

            logger.debug(
                "Face similarity | "
                "score=%s | "
                "time=%.4fs | "
                "arcface_ok=%s",
                face_score
                if arcface_ok
                else "N/A",
                face_sec,
                arcface_ok,
            )

            # ------------------------------------------------
            # ArcFace succeeded
            # ------------------------------------------------

            if arcface_ok:

                score = float(face_score)

                answer = (
                    "yes"
                    if score >= 0.5
                    else "no"
                )

                return {
                    "subject":
                        q.get("subject"),

                    "part":
                        q.get("part"),

                    "level":
                        q.get("level"),

                    "relevance":
                        relevance,

                    "question":
                        question_text,

                    "answer":
                        answer,

                    "reason":
                        "",

                    "likelist":
                        score,

                    "reward":
                        reward,

                    "weighted_score":
                        score * reward,

                    "evaluation_source":
                        "arcface",

                    "latency": {
                        "question_total_sec":
                            face_sec,

                        "ask_vlm_api_sec":
                            0.0,

                        "ask_vlm_retry_count":
                            0,

                        "ask_vlm_sleep_sec":
                            0.0,

                        "face_similarity_sec":
                            face_sec,

                        "ask_no_reason_total_sec":
                            0.0,

                        "ask_no_reason_api_sec":
                            0.0,
                    },
                }

            # ArcFace failed:
            # continue into VLM path.

        # ====================================================
        # VLM path
        # ====================================================

        try:
            res = self.matcher.ask_yes_no(
                system_prompt=self._system_prompt(
                    relevance
                ),
                question_text=question_text,
                prompt=prompt,
                ref_img=vlm_ref_img,
                gen_img=vlm_gen_img,
            )

        except Exception as e:
            raise RuntimeError(
                "ask_yes_no failed | "
                f"question={question_text!r} | "
                f"error={type(e).__name__}: {e}"
            ) from e

        # ----------------------------------------------------
        # Validate VLM result
        # ----------------------------------------------------

        if not isinstance(res, dict):
            raise RuntimeError(
                "ask_yes_no returned invalid result: "
                f"type={type(res).__name__}"
            )

        answer = res.get("answer")

        if isinstance(answer, str):
            answer = answer.strip().lower()

        if answer not in {
            "yes",
            "no",
        }:
            raise RuntimeError(
                "ask_yes_no returned invalid answer: "
                f"{answer!r}"
            )

        if "likelist" not in res:
            raise RuntimeError(
                "ask_yes_no result missing 'likelist'."
            )

        try:
            score = float(
                res["likelist"]
            )

        except Exception as e:
            raise RuntimeError(
                "Invalid likelist score: "
                f"{res.get('likelist')!r}"
            ) from e

        if not 0.0 <= score <= 1.0:
            raise RuntimeError(
                "VLM score out of range: "
                f"{score}"
            )

        latency = res.get(
            "latency",
            {},
        )

        # ====================================================
        # Explain "No"
        # ====================================================

        reason = ""

        reason_total_sec = 0.0
        reason_api_sec = 0.0

        if (
            answer == "no"
            and q.get("part") != "face"
        ):

            try:
                reason_info = (
                    self.matcher.explain_no(
                        system_prompt=self._system_prompt(
                            relevance
                        ),
                        question_text=question_text,
                        prompt=prompt,
                        relevance=relevance,
                        ref_img=vlm_ref_img,
                        gen_img=vlm_gen_img,
                    )
                )

                if not isinstance(
                    reason_info,
                    dict,
                ):
                    raise RuntimeError(
                        "explain_no returned invalid result."
                    )

                reason = str(
                    reason_info.get(
                        "reason",
                        "",
                    )
                ).strip()

                reason_latency = (
                    reason_info.get(
                        "latency",
                        {},
                    )
                )

                reason_total_sec = float(
                    reason_latency.get(
                        "total_sec",
                        0.0,
                    )
                )

                reason_api_sec = float(
                    reason_latency.get(
                        "api_wall_sec",
                        0.0,
                    )
                )

            except Exception as e:

                # =================================================
                # AUXILIARY EXPLANATION FAILURE
                #
                # The main score has already been produced by
                # ask_yes_no(). explain_no() is diagnostic only and
                # does not participate in final_score, so its failure
                # must not invalidate an otherwise valid Question.
                # =================================================

                logger.warning(
                    "explain_no failed | "
                    "question=%s | "
                    "error=%s: %s",
                    question_text,
                    type(e).__name__,
                    e,
                )

                reason = ""
                reason_total_sec = 0.0
                reason_api_sec = 0.0

        # ====================================================
        # Successful VLM result
        # ====================================================

        return {
            "subject":
                q.get("subject"),

            "part":
                q.get("part"),

            "level":
                q.get("level"),

            "relevance":
                relevance,

            "question":
                question_text,

            "answer":
                answer,

            "reason":
                reason,

            "likelist":
                score,

            "reward":
                reward,

            "weighted_score":
                score * reward,

            "evaluation_source":
                "vlm",

            "latency": {
                "question_total_sec":
                    float(
                        latency.get(
                            "total_sec",
                            0.0,
                        )
                    )
                    + face_sec,

                "ask_vlm_api_sec":
                    float(
                        latency.get(
                            "api_wall_sec",
                            0.0,
                        )
                    ),

                "ask_vlm_retry_count":
                    int(
                        latency.get(
                            "retry_count",
                            0,
                        )
                    ),

                "ask_vlm_sleep_sec":
                    float(
                        latency.get(
                            "sleep_sec",
                            0.0,
                        )
                    ),

                "face_similarity_sec":
                    face_sec,

                "ask_no_reason_total_sec":
                    reason_total_sec,

                "ask_no_reason_api_sec":
                    reason_api_sec,
            },
        }

    # ========================================================
    # Evaluate Question List
    # ========================================================

    def evaluate_question_list(
        self,
        question_list: List[Dict[str, Any]],
        *,
        prompt: str,
        ref_img,
        gen_img,
    ) -> Dict[str, Any]:
        """
        STRICT evaluation policy.

        A generated image is considered successfully evaluated
        only when ALL questions complete successfully.

        Any scoring-path API / VLM / parsing error in any Question
        aborts the whole generated-image evaluation. ArcFace failure
        may fall back to VLM. Auxiliary explain_no() failure is logged
        but does not invalidate an already valid yes/no score.

        Partial scoring is forbidden.
        """

        # ====================================================
        # Validate Question List
        # ====================================================

        if not isinstance(
            question_list,
            list,
        ):
            raise ValueError(
                "question_list must be a list."
            )

        if len(question_list) == 0:
            raise RuntimeError(
                "Question list is empty; "
                "cannot evaluate generated image."
            )

        expected_questions = len(
            question_list
        )

        # ====================================================
        # Prepare images once
        # ====================================================

        vlm_ref_img, vlm_gen_img = (
            self.matcher.prepare_images(
                ref_img,
                gen_img,
            )
        )

        details: List[
            Dict[str, Any]
        ] = []

        # ====================================================
        # STRICT:
        # one failure -> entire image fails
        # ====================================================

        for idx, q in enumerate(
            question_list
        ):

            question_text = ""

            try:
                if isinstance(q, dict):
                    question_text = str(
                        q.get(
                            "question",
                            {},
                        ).get(
                            "question",
                            "",
                        )
                    )

                result = self.evaluate_question(
                    q,
                    prompt=prompt,
                    ref_img=ref_img,
                    gen_img=gen_img,
                    vlm_ref_img=vlm_ref_img,
                    vlm_gen_img=vlm_gen_img,
                )

                if not isinstance(
                    result,
                    dict,
                ):
                    raise RuntimeError(
                        "evaluate_question "
                        "returned invalid result."
                    )

                details.append(
                    result
                )

            except Exception as e:

                logger.error(
                    "Question evaluation failed | "
                    "index=%d/%d | "
                    "question=%s | "
                    "error=%s",
                    idx + 1,
                    expected_questions,
                    question_text,
                    e,
                )

                # =============================================
                # Important:
                # DO NOT skip the failed Question.
                # DO NOT continue.
                # DO NOT compute a partial final_score.
                # =============================================

                raise RuntimeError(
                    "Generated-image evaluation failed "
                    "because one Question could not be evaluated. "
                    f"question_index="
                    f"{idx + 1}/{expected_questions} | "
                    f"question={question_text!r} | "
                    f"error={type(e).__name__}: {e}"
                ) from e

        # ====================================================
        # Completeness check
        # ====================================================

        successful_questions = len(
            details
        )

        if (
            successful_questions
            != expected_questions
        ):
            raise RuntimeError(
                "Incomplete Question evaluation: "
                f"expected={expected_questions}, "
                f"successful={successful_questions}. "
                "Partial scoring is forbidden."
            )

        # ====================================================
        # Reward validation
        # ====================================================

        rewards = [
            float(x["reward"])
            for x in details
        ]

        weighted = [
            float(x["weighted_score"])
            for x in details
        ]

        reward_sum = sum(
            rewards
        )

        if reward_sum <= 0:
            raise RuntimeError(
                "Total reward must be > 0; "
                f"reward_sum={reward_sum}"
            )

        # ====================================================
        # Final score
        # ====================================================

        final_score = (
            sum(weighted)
            / reward_sum
        )

        # ====================================================
        # Latency
        # ====================================================

        total_sec = sum(
            x["latency"].get(
                "question_total_sec",
                0.0,
            )
            for x in details
        )

        api_sec = sum(
            x["latency"].get(
                "ask_vlm_api_sec",
                0.0,
            )
            for x in details
        )

        no_reason_sec = sum(
            x["latency"].get(
                "ask_no_reason_total_sec",
                0.0,
            )
            for x in details
        )

        face_sec = sum(
            x["latency"].get(
                "face_similarity_sec",
                0.0,
            )
            for x in details
        )

        no_count = sum(
            1
            for x in details
            if x.get("answer") == "no"
        )

        face_count = sum(
            1
            for x in details
            if x.get("part") == "face"
        )

        arcface_count = sum(
            1
            for x in details
            if x.get(
                "evaluation_source"
            ) == "arcface"
        )

        vlm_count = sum(
            1
            for x in details
            if x.get(
                "evaluation_source"
            ) == "vlm"
        )

        # ====================================================
        # Success
        #
        # Reaching here means ALL Questions succeeded.
        # ====================================================

        return {
            "details":
                details,

            # Kept for compatibility with the previous schema.
            # In strict mode this must always be empty.
            "skipped":
                [],

            "final_score":
                final_score,

            "evaluation_complete":
                True,

            "num_expected_questions":
                expected_questions,

            "num_successful_questions":
                successful_questions,

            "latency_summary": {
                "num_questions":
                    successful_questions,

                "num_expected_questions":
                    expected_questions,

                "num_successful_questions":
                    successful_questions,

                "num_skipped":
                    0,

                "image_eval_total_ms":
                    total_sec * 1000.0,

                "mean_question_ms":
                    (
                        total_sec
                        / successful_questions
                        * 1000.0
                    ),

                "sum_question_ms":
                    total_sec * 1000.0,

                "sum_main_api_ms":
                    api_sec * 1000.0,

                "sum_no_reason_ms":
                    no_reason_sec
                    * 1000.0,

                "sum_face_ms":
                    face_sec
                    * 1000.0,

                "num_no_answers":
                    no_count,

                "num_face_questions":
                    face_count,

                "num_arcface_questions":
                    arcface_count,

                "num_vlm_questions":
                    vlm_count,
            },
        }