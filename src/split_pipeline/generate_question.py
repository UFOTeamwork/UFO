from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .json_extract import extract_json_value
from .prompts import load_system_prompt_qgen


def _generate_three_questions(vlm, subject: str, aspect: str) -> Dict[str, str]:
    user_text = f"""
Subject: {subject}
Aspect: {aspect}

Generate exactly three yes/no questions in JSON:
{{
  "Q1": "...",
  "Q2": "...",
  "Q3": "..."
}}

Q1 = text_only consistency
Q2 = image_only consistency
Q3 = text_and_image edit-result correctness
""".strip()
    raw = vlm.complete_text(
        system_prompt=load_system_prompt_qgen(),
        user_text=user_text,
        max_tokens=512,
    )
    try:
        parsed, _ = extract_json_value(raw)
        return parsed
    except Exception:
        print("[ERROR] Failed to parse question JSON from raw model output:", flush=True)
        print(raw, flush=True)
        raise


def _select_question(relevance: str, generated: Dict[str, str]) -> Tuple[str, str, str]:
    if relevance == "text_only":
        return generated["Q1"], "text_alignment", "reference_text + generated_image"
    if relevance == "image_only":
        return generated["Q2"], "image_alignment", "reference_image + generated_image"
    return generated["Q3"], "multimodal_alignment", "reference_text + reference_image + generated_image"


def _build_item(
    vlm,
    *,
    subject_name: str,
    aspect_name: str,
    level: str,
    relevance: str,
    reward: Any,
) -> Dict[str, Any]:
    generated = _generate_three_questions(vlm, subject_name, aspect_name)
    final_q, question_type, question_input = _select_question(relevance, generated)
    return {
        "subject": subject_name,
        "part": aspect_name,
        "level": level,
        "relevance": relevance,
        "reward": reward if reward is not None else 1,
        "question": {
            "question_type": question_type,
            "input": question_input,
            "question": final_q,
            "answer_space": ["Yes", "No"],
        },
    }


def generate_question(vlm, tgt) -> List[Dict[str, Any]]:
    subject = tgt["subject"]
    subject_name = subject.get("name", "object")
    items: List[Dict[str, Any]] = []

    for attr in subject.get("global_attributes", []):
        items.append(
            _build_item(
                vlm,
                subject_name=subject_name,
                aspect_name=attr["type"],
                level="global_attribute",
                relevance=attr["relevance"],
                reward=attr.get("reward", 1),
            )
        )

    for part in subject.get("parts", []):
        items.append(
            _build_item(
                vlm,
                subject_name=subject_name,
                aspect_name=part["name"],
                level="part",
                relevance=part["relevance"],
                reward=part.get("reward", 1),
            )
        )

    return items
