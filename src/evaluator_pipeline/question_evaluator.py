# question_evaluator.py
from __future__ import annotations

import logging
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

        face_cfg = dict(cfg.get("face_similarity", {}))
        self.face = None

        if face_cfg.get("enabled", False):
            # Heavy deps (torch/opencv/torchvision) are only required when face
            # similarity is enabled, so import them lazily here.
            import torch

            from .face_similarity import FaceSimilarityEvaluator

            use_cuda = bool(face_cfg.get("use_cuda", False)) and torch.cuda.is_available()
            face_cfg["use_cuda"] = use_cuda
            face_cfg["remove_bg"] = True
            self.face = FaceSimilarityEvaluator(**face_cfg)

    def _system_prompt(self, relevance: str) -> str:
        if relevance == "text_only":
            return self.prompt_text_only
        if relevance == "image_only":
            return self.prompt_image_only
        return self.prompt_image_text

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
        if vlm_ref_img is None or vlm_gen_img is None:
            vlm_ref_img, vlm_gen_img = self.matcher.prepare_images(ref_img, gen_img)

        reward = float(q.get("reward", 1.0))
        face_sec = 0.0
        use_arcface = q.get("part") == "face" and self.face is not None

        # Face questions: prefer ArcFace, skip the VLM when it succeeds.
        if use_arcface:
            import time
            t0_face = time.perf_counter()
            arcface_ok = False

            try:
                result = self.face.compute_similarity_with_debug(ref_img, gen_img)
                face_score = result["cosine_similarity"]
                face_score = max(0.0, face_score)
                arcface_ok = True
            except RuntimeError as e:
                logger.warning(f"Face similarity failed: {e}. Falling back to VLM.")

            face_sec = time.perf_counter() - t0_face

            import sys
            sys.stderr.write(
                f"[DEBUG] Face similarity: "
                f"score={face_score if arcface_ok else 'N/A'} "
                f"time={face_sec:.2f}s "
                f"arcface_ok={arcface_ok}\n"
            )
            sys.stderr.flush()

            if arcface_ok:
                score = face_score
                vlm_answer = "yes" if face_score >= 0.5 else "no"
                return {
                    "subject":        q["subject"],
                    "part":           q["part"],
                    "level":          q["level"],
                    "relevance":      q["relevance"],
                    "question":       q["question"]["question"],
                    "answer":         vlm_answer,
                    "reason":         "",
                    "likelist":       score,
                    "reward":         reward,
                    "weighted_score": score * reward,
                    "latency": {
                        "question_total_sec":      face_sec,
                        "ask_vlm_api_sec":         0.0,
                        "ask_vlm_retry_count":     0,
                        "ask_vlm_sleep_sec":       0.0,
                        "face_similarity_sec":     face_sec,
                        "ask_no_reason_total_sec": 0.0,
                        "ask_no_reason_api_sec":   0.0,
                    },
                }
            # ArcFace failed: fall through to the VLM path.

        # VLM path (non-face questions, or face questions where ArcFace failed).
        res = self.matcher.ask_yes_no(
            system_prompt=self._system_prompt(q["relevance"]),
            question_text=q["question"]["question"],
            prompt=prompt,
            ref_img=vlm_ref_img,
            gen_img=vlm_gen_img,
        )

        score = res["likelist"]

        reason = ""
        reason_total_sec = 0.0
        reason_api_sec = 0.0
        if res["answer"] == "no" and q.get("part") != "face":
            try:
                reason_info = self.matcher.explain_no(
                    system_prompt=self._system_prompt(q["relevance"]),
                    question_text=q["question"]["question"],
                    prompt=prompt,
                    relevance=q["relevance"],
                    ref_img=vlm_ref_img,
                    gen_img=vlm_gen_img,
                )
                reason = reason_info["reason"]
                reason_total_sec = reason_info["latency"]["total_sec"]
                reason_api_sec = reason_info["latency"]["api_wall_sec"]
            except Exception as e:
                # When explain_no exhausts retries or returns empty content, do
                # not pollute the reason field; raise so evaluate_question_list
                # can decide whether to skip or fail the whole batch.
                raise RuntimeError(
                    f"explain_no failed for question '{q['question']['question']}': {e}"
                ) from e

        return {
            "subject":        q["subject"],
            "part":           q["part"],
            "level":          q["level"],
            "relevance":      q["relevance"],
            "question":       q["question"]["question"],
            "answer":         res["answer"],
            "reason":         reason,
            "likelist":       score,
            "reward":         reward,
            "weighted_score": score * reward,
            "latency": {
                "question_total_sec":      res["latency"]["total_sec"] + face_sec,
                "ask_vlm_api_sec":         res["latency"]["api_wall_sec"],
                "ask_vlm_retry_count":     res["latency"]["retry_count"],
                "ask_vlm_sleep_sec":       res["latency"]["sleep_sec"],
                "face_similarity_sec":     face_sec,
                "ask_no_reason_total_sec": reason_total_sec,
                "ask_no_reason_api_sec":   reason_api_sec,
            },
        }

    def evaluate_question_list(
        self,
        question_list: List[Dict[str, Any]],
        *,
        prompt: str,
        ref_img,
        gen_img,
    ) -> Dict[str, Any]:
        vlm_ref_img, vlm_gen_img = self.matcher.prepare_images(ref_img, gen_img)

        details = []
        skipped = []
        for q in question_list:
            try:
                result = self.evaluate_question(
                    q,
                    prompt=prompt,
                    ref_img=ref_img,
                    gen_img=gen_img,
                    vlm_ref_img=vlm_ref_img,
                    vlm_gen_img=vlm_gen_img,
                )
                details.append(result)
            except RuntimeError as e:
                logger.error("Skipping question due to evaluation failure: %s", e)
                skipped.append({
                    "question": q["question"]["question"],
                    "error":    str(e),
                })

        if not details:
            raise RuntimeError("All questions failed evaluation; cannot compute final_score.")

        rewards = [x["reward"] for x in details]
        weighted = [x["weighted_score"] for x in details]

        final_score = sum(weighted) / sum(rewards) if rewards else 0.0
        total_sec = sum((x["latency"]["question_total_sec"] for x in details), 0.0)
        api_sec = sum((x["latency"]["ask_vlm_api_sec"] for x in details), 0.0)
        no_reason_sec = sum((x["latency"].get("ask_no_reason_total_sec", 0.0) for x in details), 0.0)
        face_sec = sum((x["latency"].get("face_similarity_sec", 0.0) for x in details), 0.0)
        no_count = sum(1 for x in details if x["answer"] == "no")
        face_count = sum(1 for x in details if x["part"] == "face")

        return {
            "details":     details,
            "skipped":     skipped,
            "final_score": final_score,
            "latency_summary": {
                "num_questions":       len(details),
                "num_skipped":         len(skipped),
                "image_eval_total_ms": total_sec * 1000.0,
                "mean_question_ms":    (total_sec / len(details) * 1000.0) if details else 0.0,
                "sum_question_ms":     total_sec * 1000.0,
                "sum_main_api_ms":     api_sec * 1000.0,
                "sum_no_reason_ms":    no_reason_sec * 1000.0,
                "sum_face_ms":         face_sec * 1000.0,
                "num_no_answers":      no_count,
                "num_face_questions":  face_count,
            },
        }