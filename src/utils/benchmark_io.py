# -*- coding: utf-8 -*-
import os
import json
import re
from typing import Dict, Iterator, List, Any, Optional, Tuple
from PIL import Image
from pathlib import Path

class BenchmarkIO:
    """
    Benchmark-style IO helper

    功能：
    1. 从 image_root + metadata json/jsonl 读取 (ref_image, prompt) 对
    2. 按 benchmark 目录规范（无 model 层）存储生成结果
    3. 从 benchmark 输出图像目录读取并解析生成图像
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    GEN_FILENAME_PATTERN = re.compile(
        r"""
        ^(?P<img_id>\d+)
        _template(?P<template_idx>\d+)
        _sample(?P<sample_idx>\d+)
        \.(png|jpg|jpeg|bmp)$
        """,
        re.VERBOSE,
    )

    # 类别名称到目录名的映射
    CATEGORY_TO_DIR = {
        "Rigid object": "Rigid_object",
        "Soft object": "Soft_object",
        "Human": "Human",
        "Animal": "Animal",
        "Character-Full body": "Character_Full_body",
        "Logo": "Logo",
        "Scene": "Scene"
    }
    def __init__(self, image_root: str, metadata_jsonl: str, output_root: str):
        self.image_root = image_root
        self.metadata_jsonl = metadata_jsonl
        self.output_root = output_root
        self._metadata = self.load_templates(metadata_jsonl)

    # -----------------------------------------------------
    # utils
    # -----------------------------------------------------
    @staticmethod
    def safe_filename(s: str) -> str:
        """
        将字符串转换为安全文件名：
        - 空格和 '-' 替换为 '_'
        - 非字母数字下划线点保留，其他字符替换为 '_'
        """
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"[^\w_.]+", "_", s)
        return s.strip("_")
    
    # @staticmethod
    # def safe_filename_sub(s: str) -> str:
    #     s = s.replace(" ", "_")
    #     return re.sub(r"[^\w\-_. ]+", "_", s).strip()

    @staticmethod
    def _is_image(fn: str) -> bool:
        return fn.lower().endswith(BenchmarkIO.IMAGE_EXTS)

    # -----------------------------------------------------
    # metadata loader (FIXED)
    # -----------------------------------------------------
    @staticmethod
    def load_templates(path: str) -> List[Dict[str, Any]]:
        """
        支持三种格式：
        - 一个 JSON 数组文件（最常见）
        - 一个单独的 JSON 对象（只有一个 category）
        - 多个 JSON 对象连着写在文件里（会尝试逐个解析）
        返回 list of category objects
        """
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
        except Exception:
            # 尝试按行或按多个对象解析
            objs = []
            # 用正则寻找所有大括号包裹的 JSON 对象（简单策略）
            matches = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", txt, flags=re.DOTALL)
            for m in matches:
                try:
                    objs.append(json.loads(m))
                except Exception:
                    continue
            if objs:
                return objs
        raise ValueError(f"无法解析 templates 文件: {path}")

    # -----------------------------------------------------
    # reader: input side (ref_image + prompt)
    # -----------------------------------------------------
    def iter_pairs(self) -> Iterator[Dict]:
        """
        Yield dicts with:
        uid, category, subtype, edit_type, template_idx, image_id, prompt, ref_image
        """

        for spec in self._metadata:
            category = spec["category"]
            subtypes: List[str] = spec.get("subtypes")  # 可以为空列表
            templates: Dict[str, List[str]] = spec["templates"]

            # category 级目录
            category_dir = os.path.join(self.image_root, self.safe_filename(category)) 
            
            if not os.path.isdir(category_dir):
                print(f"[WARN] Category '{category}' directory doesn't exist: {category_dir}")
                continue

            # 如果没有 subtypes，就把 category 自己作为唯一 subtype
            effective_subtypes = subtypes if subtypes else [category]

            for subtype in effective_subtypes:
                # subtype 目录存在就用它，否则 fallback 到 category 目录
                img_dir = os.path.join(category_dir, subtype)
                if not os.path.isdir(img_dir):
                    # fallback 目录
                    img_dir = os.path.join(category_dir, category)

                images = sorted(fn for fn in os.listdir(img_dir) if self._is_image(fn))
                if not images:
                    print(f"[WARN] No images found in {img_dir}")
                    continue

                for edit_type, tmpl_list in templates.items():
                    for t_idx, tmpl in enumerate(tmpl_list):
                        for img_id, img_name in enumerate(images):
                            img_stem = os.path.splitext(img_name)[0]

                            uid = f"{img_stem}_template{t_idx}"

                            yield {
                                "uid": uid,
                                "category": category,
                                "subtype": subtype,
                                "edit_type": edit_type,
                                "template_idx": t_idx,
                                "image_id": img_stem,
                                "prompt": tmpl.replace("[CATEGORY]", subtype or category),
                                "ref_image": os.path.join(img_dir, img_name),
                            }
    
    # -----------------------------------------------------
    # image loaders
    # -----------------------------------------------------
    @staticmethod
    def load_image(path: str) -> Image.Image | None:
        if not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")
    
    # -----------------------------------------------------
    # writer: json result (non-image)
    # -----------------------------------------------------
    def save_result(self, pair: Dict, result_json: Dict) -> str:
        category = pair["category"]
        subtype = pair["subtype"]
        edit_type = pair["edit_type"]
        uid = pair["uid"]

        if subtype:
            save_dir = os.path.join(self.safe_filename(category), subtype, edit_type)
        else:
            save_dir = os.path.join(self.safe_filename(category), category, edit_type)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{uid}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        return save_path

    # -----------------------------------------------------
    # reader: generated images
    # -----------------------------------------------------
    def iter_generated_images(self, model_name: str) -> Iterator[Dict]:
        """
        遍历生成图像：
        output_root / model_name / category / subtype / edit_type / *.png
        """
        model_dir = os.path.join(self.output_root, self.safe_filename(model_name))
        if not os.path.isdir(model_dir):
            return

        for category in sorted(os.listdir(model_dir)):
            cat_dir = os.path.join(model_dir, category)
            if not os.path.isdir(cat_dir):
                continue

            for subtype in sorted(os.listdir(cat_dir)):
                sub_dir = os.path.join(cat_dir, subtype)
                if not os.path.isdir(sub_dir):
                    continue

                for edit_type in sorted(os.listdir(sub_dir)):
                    edit_dir = os.path.join(sub_dir, edit_type)
                    if not os.path.isdir(edit_dir):
                        continue

                    for fname in sorted(os.listdir(edit_dir)):
                        if not fname.lower().endswith(self.IMAGE_EXTS):
                            continue

                        m = self.GEN_FILENAME_PATTERN.match(fname)
                        if not m:
                            continue

                        image_id = int(m.group("img_id"))
                        template_idx = int(m.group("template_idx"))
                        sample_idx = int(m.group("sample_idx"))

                        uid = f"{image_id}_template{template_idx}"

                        yield {
                            "uid": uid,                     
                            "category": category,
                            "subtype": subtype,
                            "edit_type": edit_type,
                            "image_id": image_id,
                            "template_idx": template_idx,
                            "sample_idx": sample_idx,
                            "image_path": os.path.join(edit_dir, fname),
                        }

    def iter_samples(self, model_name: str):
        """
        Yield:
        uid, category, subtype, edit_type,
        ref_image(PIL), gen_image(PIL), prompt
        """

        model_dir = os.path.join(self.output_root, self.safe_filename(model_name))
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir not found: {model_dir}")

        for pair in self.iter_pairs():
            uid = pair["uid"]
            category = self.safe_filename(pair["category"])
            subtype = pair["subtype"]
            edit_type = pair["edit_type"]

            # ---------------------------
            # reference image
            # ---------------------------
            ref_path = pair["ref_image"]
            if not os.path.exists(ref_path):
                yield {"uid": uid, "error": "missing_ref_image"}
                continue

            # ---------------------------
            # generated image
            # ---------------------------
            gen_dir = os.path.join(
                model_dir, category, subtype, edit_type
            )
            if not os.path.isdir(gen_dir):
                yield {"uid": uid, "error": "missing_gen_dir"}
                continue

            # 默认 sample0
            gen_name = f"{pair['image_id']}_template{pair['template_idx']}_sample0.png"
            gen_path = os.path.join(gen_dir, gen_name)

            if not os.path.exists(gen_path):
                yield {"uid": uid, "error": "missing_gen_image"}
                continue

            yield {
                "uid": uid,
                "category": category,
                "subtype": subtype,
                "edit_type": edit_type,
                "prompt": pair["prompt"],
                "ref_image": Image.open(ref_path).convert("RGB"),
                "gen_image": Image.open(gen_path).convert("RGB"),
            }

    def make_out_path(
        self,
        root: str,
        category: str,
        subtype: str,
        edit_type: str,
        uid: str,
        ext: str = ".json",
    ):
        out_dir = os.path.join(
            root,
            self.safe_filename(category),
            subtype,
            edit_type,
        )
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, uid + ext)

    # new

    def find_original_image(
        self, 
        category: str, 
        subtype: str, 
        image_id: str
    ) -> Optional[str]:
        """
        从 image_root 中查找原始图片
        
        Args:
            category: 类别名称（如 "Rigid object"）
            subtype: 子类型名称
            image_id: 基础文件名（可能包含 _template 后缀）
        
        Returns:
            原始图片的完整路径，如果找不到返回 None
        """
        # 获取类别目录名
        category_dir_name = self.CATEGORY_TO_DIR.get(category, self.safe_filename(category))
        
        # 提取原始文件名（去除 _template 后缀）
        orig_key = image_id
        if "_template" in image_id:
            orig_key = image_id.split('_template', 1)[0]
        
        # 尝试多个可能的路径
        candidate_dirs = [
            os.path.join(self.image_root, category_dir_name, subtype),
            os.path.join(self.image_root, category_dir_name, self.safe_filename(subtype)),
            os.path.join(self.image_root, category_dir_name),
            os.path.join(self.image_root, subtype),
        ]
        
        for candidate_dir in candidate_dirs:
            if not os.path.isdir(candidate_dir):
                continue
            
            # 精确匹配：orig_key + 扩展名
            for ext in self.IMAGE_EXTS:
                img_path = os.path.join(candidate_dir, orig_key + ext)
                if os.path.exists(img_path):
                    return img_path
            
            # 模糊匹配：文件名包含 orig_key
            try:
                for fname in os.listdir(candidate_dir):
                    if orig_key in fname and self._is_image(fname):
                        return os.path.join(candidate_dir, fname)
            except (OSError, PermissionError):
                continue
        
        return None

    def find_generated_samples(
        self,
        model_name: str,
        category: str,
        subtype: str,
        edit_type: str,
        image_id: str
    ) -> List[str]:
        """
        查找特定条目的所有生成样本图片
        
        Args:
            model_name: 模型名称
            category: 类别
            subtype: 子类型
            edit_type: 编辑类型
            image_id: 基础文件名（不含 _sample 后缀）
        
        Returns:
            样本图片路径列表，按 sample_idx 排序
        """
        gen_dir = os.path.join(
            self.output_root,
            self.safe_filename(model_name),
            self.safe_filename(category),
            subtype,
            edit_type
        )
        
        if not os.path.isdir(gen_dir):
            return []
        
        # 提取 image_id 和 template_idx
        pattern = re.compile(
            rf"^{re.escape(image_id)}_sample(\d+)\.(png|jpg|jpeg|bmp)$",
            re.IGNORECASE
        )
        
        samples = []
        try:
            for fname in os.listdir(gen_dir):
                match = pattern.match(fname)
                if match:
                    sample_idx = int(match.group(1))
                    samples.append({
                        'idx': sample_idx,
                        'path': os.path.join(gen_dir, fname)
                    })
        except (OSError, PermissionError):
            return []
        
        # 按 sample_idx 排序
        samples.sort(key=lambda x: x['idx'])
        return [s['path'] for s in samples]

    def find_prompt(
        self,
        model_name: str,
        category: str,
        subtype: str,
        edit_type: str,
        image_id: str
    ) -> Optional[str]:
        """
        查找对应的 prompt 文本
        
        Returns:
            prompt 文本内容，如果找不到返回 None
        """
        prompt_file = os.path.join(
            self.output_root,
            self.safe_filename(model_name),
            self.safe_filename(category),
            subtype,
            edit_type,
            f"{image_id}.prompt.txt"
        )
        
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"[WARN] Failed to read prompt file {prompt_file}: {e}")
                return None
        
        return None

    def get_all_base_files(
        self,
        model_name: str,
        category: str,
        subtype: str,
        edit_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取指定目录下的所有 image_id 及其对应的文件信息
        
        Returns:
            {
                image_id: {
                    'dir': 目录路径,
                    'prompt': prompt 文件路径或 None,
                    'samples': [sample 图片路径列表]
                }
            }
        """
        gen_dir = os.path.join(
            self.output_root,
            self.safe_filename(model_name),
            self.safe_filename(category),
            subtype,
            edit_type
        )
        
        if not os.path.isdir(gen_dir):
            return {}
        
        base_map = {}
        
        try:
            for fname in os.listdir(gen_dir):
                fpath = os.path.join(gen_dir, fname)
                
                # Prompt 文件
                if fname.endswith('.prompt.txt'):
                    base = fname.replace('.prompt.txt', '')
                    base_map.setdefault(base, {
                        'dir': gen_dir,
                        'prompt': None,
                        'samples': []
                    })
                    base_map[base]['prompt'] = fpath
                
                # 样本图片
                elif self._is_image(fname):
                    match = self.GEN_FILENAME_PATTERN.match(fname)
                    if match:
                        img_id = match.group('img_id')
                        template_idx = match.group('template_idx')
                        base = f"{img_id}_template{template_idx}"
                        
                        base_map.setdefault(base, {
                            'dir': gen_dir,
                            'prompt': None,
                            'samples': []
                        })
                        base_map[base]['samples'].append(fpath)
        
        except (OSError, PermissionError) as e:
            print(f"[WARN] Failed to list directory {gen_dir}: {e}")
            return {}
        
        # 对每个 base 的 samples 排序
        for base_info in base_map.values():
            base_info['samples'].sort()
        
        return base_map

    def get_item_info(
        self,
        model_name: str,
        category: str,
        subtype: str,
        edit_type: str,
        image_id: str
    ) -> Dict[str, Any]:
        """
        获取单个条目的完整信息
        
        Returns:
            {
                'image_id': str,
                'original_image': str | None,
                'prompt': str | None,
                'samples': [str],  # 样本图片路径列表
                'category': str,
                'subtype': str,
                'edit_type': str
            }
        """
        return {
            'image_id': image_id,
            'original_image': self.find_original_image(category, subtype, image_id),
            'prompt': self.find_prompt(model_name, category, subtype, edit_type, image_id),
            'samples': self.find_generated_samples(model_name, category, subtype, edit_type, image_id),
            'category': category,
            'subtype': subtype,
            'edit_type': edit_type,
            'model_name': model_name
        }

    def list_models(self) -> List[str]:
        """列出 output_root 下的所有模型目录"""
        if not os.path.isdir(self.output_root):
            return []
        return sorted([
            d for d in os.listdir(self.output_root)
            if os.path.isdir(os.path.join(self.output_root, d))
            and not d.startswith('.')
        ])

    def list_categories(self, model_name: str) -> List[str]:
        """列出指定模型下的所有类别"""
        model_dir = os.path.join(self.output_root, self.safe_filename(model_name))
        if not os.path.isdir(model_dir):
            return []
        return sorted([
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
            and not d.startswith('.')
        ])

    def list_subtypes(self, model_name: str, category: str) -> List[str]:
        """列出指定模型和类别下的所有子类型"""
        cat_dir = os.path.join(
            self.output_root,
            self.safe_filename(model_name),
            self.safe_filename(category)
        )
        if not os.path.isdir(cat_dir):
            return []
        return sorted([
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
            and not d.startswith('.')
        ])

    def list_edit_types(self, model_name: str, category: str, subtype: str) -> List[str]:
        """列出指定路径下的所有编辑类型"""
        subtype_dir = os.path.join(
            self.output_root,
            self.safe_filename(model_name),
            self.safe_filename(category),
            subtype
        )
        if not os.path.isdir(subtype_dir):
            return []
        return sorted([
            d for d in os.listdir(subtype_dir)
            if os.path.isdir(os.path.join(subtype_dir, d))
            and not d.startswith('.')
        ])