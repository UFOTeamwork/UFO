from __future__ import annotations


def valid_tgt_json(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    subject = obj.get("subject")
    if not isinstance(subject, dict):
        return False
    if not isinstance(subject.get("global_attributes"), list):
        return False
    if not isinstance(subject.get("parts"), list):
        return False
    return True


def valid_question_json(obj) -> bool:
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    for item in obj:
        if not isinstance(item, dict):
            return False
        question = item.get("question")
        if not isinstance(question, dict):
            return False
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            return False
        if question.get("answer_space") != ["Yes", "No"]:
            return False
        if item.get("relevance") not in {"text_only", "image_only", "text_and_image"}:
            return False
    return True
