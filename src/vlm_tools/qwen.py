from __future__ import annotations

import os
from typing import Any, Sequence

from src.utils.retry import now

from .base import BaseVLM


class QwenVL(BaseVLM):
    default_api_url = ""
    default_model = "Qwen/Qwen2.5-VL-8B-Instruct"
    local_model_env_vars = ("UFO_QWEN_MODEL_PATH", "QWEN_MODEL_PATH")
    api_key_env_var = None
    api_key_env_vars = None
    default_text_max_tokens = 1024
    default_multimodal_max_tokens = 1024

    def __init__(self, *, api_url: str, api_key: str, model: str):
        resolved_model = self._resolve_model_path(model)
        super().__init__(api_url=api_url, api_key=api_key, model=resolved_model)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
        except Exception as e:
            tried_paths = ", ".join(self.local_model_env_vars)
            raise RuntimeError(
                f"Failed to load Qwen processor from '{resolved_model}'. "
                f"Set a local model path via {tried_paths} or ensure the Hugging Face repo is accessible. "
                f"Original error: {e}"
            ) from e
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model_instance = AutoModelForImageTextToText.from_pretrained(resolved_model, **model_kwargs).to(
                self.device
            ).eval()
        except Exception as e:
            tried_paths = ", ".join(self.local_model_env_vars)
            raise RuntimeError(
                f"Failed to load Qwen model from '{resolved_model}'. "
                f"Set a local model path via {tried_paths} or ensure the Hugging Face repo is accessible. "
                f"Original error: {e}"
            ) from e
        self.model_instance.config.use_cache = True
        if getattr(self.model_instance, "generation_config", None) is not None:
            self.model_instance.generation_config.use_cache = True
        self._prepare_token_ids()

    def _resolve_model_path(self, model: str) -> str:
        for env_var in self.local_model_env_vars:
            candidate = os.getenv(env_var, "").strip()
            if candidate:
                return candidate
        return model

    def prepare_images(self, images: Sequence[Any]) -> list[Any]:
        return list(images)

    def _prepare_token_ids(self) -> None:
        tokenizer = self.processor.tokenizer

        def first_token(words: list[str]) -> list[int]:
            ids = set()
            for word in words:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if token_ids:
                    ids.add(token_ids[0])
            return sorted(ids)

        self.yes_ids = first_token([" yes", "yes", "Yes", " Yes", "\nYes", "\nyes"])
        self.no_ids = first_token([" no", "no", "No", " No", "\nNo", "\nno"])

    def build_messages(self, system_prompt, question, images=None, image_roles=None):
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        user_content = []
        if images:
            if image_roles is None:
                raise ValueError("image_roles must be provided when images are used")
            if len(images) != len(image_roles):
                raise ValueError("images and image_roles length mismatch")
            for role, img in zip(image_roles, images):
                user_content.append({"type": "text", "text": f"{role}:"})
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _move_inputs_to_device(self, inputs):
        return {
            key: (value.to(self.device, non_blocking=True) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

    def _forward_logits(self, inputs):
        outputs = self.model_instance(**inputs, use_cache=False, return_dict=True)
        return outputs.logits

    def _yes_no_prob_from_logits(self, logits):
        torch = self._torch
        last = torch.nan_to_num(logits[0, -1].float(), neginf=-1e4, posinf=1e4)
        probs = torch.softmax(last, dim=-1)
        p_yes = probs[self.yes_ids].sum().item()
        p_no = probs[self.no_ids].sum().item()
        answer = "yes" if p_yes >= p_no else "no"
        return {
            "answer": answer,
            "likelist": float(p_yes),
            "score": float(p_yes),
            "p_yes": float(p_yes),
            "p_no": float(p_no),
            "raw_output": "Yes" if answer == "yes" else "No",
        }

    def _generate(self, inputs, max_new_tokens=10):
        return self.model_instance.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=[
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    def _generate_text(self, messages, max_new_tokens: int = 80) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._move_inputs_to_device(inputs)
        with self._torch.inference_mode():
            generated = self._generate(inputs, max_new_tokens=max_new_tokens)
        input_length = inputs["input_ids"].shape[-1]
        gen_ids = generated[0][input_length:]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def ask_vlm(self, prompt: str, images=None, image_roles=None, system_prompt=None, debug: bool = True):
        t0_total = now()
        messages = self.build_messages(system_prompt=system_prompt, question=prompt, images=images, image_roles=image_roles)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._move_inputs_to_device(inputs)
        with self._torch.inference_mode():
            logits = self._forward_logits(inputs)
        parsed = self._yes_no_prob_from_logits(logits)
        if debug:
            print("[DEBUG] Raw answer:", parsed["raw_output"], flush=True)
        return {
            "answer": parsed["answer"],
            "likelist": parsed["likelist"],
            "latency": {
                "encode_sec": 0.0,
                "api_wall_sec": 0.0,
                "retry_count": 0,
                "sleep_sec": 0.0,
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
            "Explain briefly WHY the generated image does NOT satisfy the requirement.\n"
            "Do NOT re-evaluate the answer.\n"
            "Do NOT say Yes or No.\n"
            "Only explain the reason."
        )
        images = []
        image_roles = []
        if relevance == "image_only":
            images = [ref_img, gen_img]
            image_roles = ["Reference Image", "Generated Image"]
        elif relevance == "text_only":
            images = [gen_img]
            image_roles = ["Generated Image"]
        elif relevance == "text_and_image":
            images = [ref_img, gen_img]
            image_roles = ["Reference Image", "Generated Image"]
        messages = self.build_messages(system_prompt=system_prompt, question=explanation_prompt, images=images, image_roles=image_roles)
        reason = self._generate_text(messages, max_new_tokens=80)
        if debug:
            print("[DEBUG] No-reason:", reason, flush=True)
        return {
            "reason": reason,
            "latency": {
                "encode_sec": 0.0,
                "api_wall_sec": 0.0,
                "retry_count": 0,
                "sleep_sec": 0.0,
                "total_sec": now() - t0_total,
            },
        }

    def complete_text(self, *, system_prompt: str, user_text: str, max_tokens: int | None = None) -> str:
        messages = self.build_messages(system_prompt=system_prompt, question=user_text, images=None, image_roles=None)
        return self._generate_text(messages, max_new_tokens=self._resolve_max_tokens(max_tokens, multimodal=False))

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: Sequence[Any],
        image_roles: Sequence[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = self.build_messages(
            system_prompt=system_prompt,
            question=user_text,
            images=list(images),
            image_roles=image_roles,
        )
        return self._generate_text(messages, max_new_tokens=self._resolve_max_tokens(max_tokens, multimodal=True))


class QwenVLM(QwenVL):
    pass
