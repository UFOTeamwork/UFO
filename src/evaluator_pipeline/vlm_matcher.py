from __future__ import annotations

from typing import Any, Dict

from src.vlm_tools.factory import build_vlm


class VLMMatcher:
    def __init__(self, cfg: Dict[str, Any]):
        vlm_cfg = cfg['vlm']
        self.vlm = build_vlm(vlm_cfg['provider'])

    def prepare_images(self, *images):
        prepared = self.vlm.prepare_images(list(images))
        return tuple(prepared)

    def ask_yes_no(self, *, system_prompt: str, question_text: str, prompt: str, ref_img, gen_img):
        full_text = f'Reference Text:\n{prompt}\n\nQuestion:\n{question_text}'
        result = self.vlm.ask_vlm(
            prompt=full_text,
            images=[ref_img, gen_img],
            image_roles=["Reference Image", "Generated Image"],
            system_prompt=system_prompt,
        )
        return {
            'answer': result['answer'],
            'likelist': result['likelist'],
            'latency': result['latency'],
        }

    def explain_no(self, *, system_prompt: str, question_text: str, prompt: str, relevance: str, ref_img, gen_img):
        result = self.vlm.ask_no_reason(
            question=question_text,
            prompt=prompt,
            relevance=relevance,
            ref_img=ref_img,
            gen_img=gen_img,
            system_prompt=system_prompt,
        )
        return {
            'reason': result['reason'],
            'latency': result['latency'],
        }
