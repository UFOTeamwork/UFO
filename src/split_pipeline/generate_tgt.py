from __future__ import annotations

from .prompts import load_system_prompt_tgt
from .json_extract import extract_json_value


def generate_tgt(vlm, *, prompt: str, ref_image_path: str):
    system_prompt = load_system_prompt_tgt()
    raw = vlm.complete_multimodal(
        system_prompt=system_prompt,
        user_text=prompt,
        images=[ref_image_path],
        image_roles=["Reference Image"],
        max_tokens=4096,
    )
    try:
        parsed, _ = extract_json_value(raw)
        return parsed, raw
    except Exception:
        print("[ERROR] T_gt input system prompt:", flush=True)
        print(system_prompt, flush=True)
        print("[ERROR] T_gt input user prompt:", flush=True)
        print(prompt, flush=True)
        print("[ERROR] T_gt input reference image:", flush=True)
        print(ref_image_path, flush=True)
        print("[ERROR] Failed to parse T_gt JSON from raw model output:", flush=True)
        print(raw, flush=True)
        raise
