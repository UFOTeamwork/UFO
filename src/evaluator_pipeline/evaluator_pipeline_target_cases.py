# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Set, Tuple, List

from PIL import Image
from tqdm import tqdm

from src.utils.benchmark_io import BenchmarkIO
from src.utils.io_utils import atomic_json_dump, load_json
from src.utils.logging_utils import log, log_failure
from src.utils.paths import eval_paths
from src.utils.retry import now
from src.utils.sampling import load_sampled_keys
from src.utils.stats import summarize_values_ms, summarize_values_raw

from .analysis import write_all_sample_scores, write_debug_reports, write_failure_summary
from .question_evaluator import QuestionEvaluator

_THREAD_LOCAL = threading.local()

GeneratedKey = Tuple[str, str, str, str, str, int]
PairKey = Tuple[str, str, str, str]


def get_thread_evaluator(cfg):
    if not hasattr(_THREAD_LOCAL, "evaluator"):
        _THREAD_LOCAL.evaluator = QuestionEvaluator(cfg)
    return _THREAD_LOCAL.evaluator


def _get_question_path(question_root: str, gen_path: str, gen_root: str) -> str:
    """
    根据生成图路径反推对应 Question_list 路径。

    生成图一般形如：
      generated_root/model/category/subtype/edit_type/uid_sample0.png

    输出：
      question_root/category/subtype/edit_type/uid.json
    """
    rel = os.path.relpath(gen_path, gen_root)
    parts = rel.split(os.sep)[1:]  # 去掉 model_name
    parts[-1] = parts[-1].split("_sample")[0] + ".json"
    return os.path.join(question_root, *parts)


def _canonical_key_component(bio: BenchmarkIO, value: str | None) -> str:
    if value in (None, ""):
        return "-"
    if value == "-":
        return value
    return bio.safe_filename(str(value).strip())


def _canonical_pair_key(
    bio: BenchmarkIO,
    *,
    category: str,
    subtype: str | None,
    edit_type: str,
    uid: str,
) -> PairKey:
    return (
        _canonical_key_component(bio, category),
        _canonical_key_component(bio, subtype),
        _canonical_key_component(bio, edit_type),
        str(uid).strip(),
    )


def _canonical_generated_key(
    bio: BenchmarkIO,
    *,
    model: str,
    category: str,
    subtype: str | None,
    edit_type: str,
    uid: str,
    sample_idx: int,
) -> GeneratedKey:
    category, subtype, edit_type, uid = _canonical_pair_key(
        bio,
        category=category,
        subtype=subtype,
        edit_type=edit_type,
        uid=uid,
    )
    return (str(model).strip(), category, subtype, edit_type, uid, int(sample_idx))


def _format_generated_key(key: GeneratedKey) -> str:
    model, category, subtype, edit_type, uid, sample_idx = key
    return f"{model}/{category}/{subtype}/{edit_type}/{uid}_sample{sample_idx}"


def _format_case_key(
    *,
    model: str,
    uid: str,
    category: str,
    subtype: str,
    edit_type: str,
    sample_idx: int | None = None,
) -> str:
    parts = [
        f"model={model}",
        f"uid={uid}",
        f"category={category}",
        f"subtype={subtype}",
        f"edit_type={edit_type}",
    ]
    if sample_idx is not None:
        parts.append(f"sample_idx={sample_idx}")
    return " | ".join(parts)


def _parse_uid_sample(name: str) -> tuple[str, int]:
    """解析 11_template0_sample1 / 11_template0_sample1.png。"""
    stem = os.path.splitext(str(name).strip())[0]
    m = re.match(r"^(?P<uid>.+)_sample(?P<sample>\d+)$", stem)
    if not m:
        raise ValueError(
            f"Cannot parse uid/sample from {name!r}. Expected format like 11_template0_sample1."
        )
    return m.group("uid"), int(m.group("sample"))


def _parse_target_generated_case_item(item: Any, bio: BenchmarkIO) -> GeneratedKey:
    """
    支持两种写法：

    1) 字符串：
       uno/Human/Face/Local-Editing/11_template0_sample1

    2) 字典：
       model: uno
       category: Human
       subtype: Face
       edit_type: Local-Editing
       uid: 11_template0
       sample_idx: 1
    """
    if isinstance(item, str):
        parts = [p for p in item.strip().split("/") if p != ""]
        if len(parts) != 5:
            raise ValueError(
                "target_generated_cases string must be: "
                "model/category/subtype/edit_type/uid_sampleN, "
                f"but got {item!r}"
            )
        model, category, subtype, edit_type, uid_sample = parts
        uid, sample_idx = _parse_uid_sample(uid_sample)
        return _canonical_generated_key(
            bio,
            model=model,
            category=category,
            subtype=subtype,
            edit_type=edit_type,
            uid=uid,
            sample_idx=sample_idx,
        )

    if isinstance(item, dict):
        required = ["model", "category", "edit_type", "uid", "sample_idx"]
        missing = [k for k in required if item.get(k) in (None, "")]
        if missing:
            raise ValueError(f"target_generated_cases item missing required keys: {missing}")
        return _canonical_generated_key(
            bio,
            model=item["model"],
            category=item["category"],
            subtype=item.get("subtype") or "-",
            edit_type=item["edit_type"],
            uid=item["uid"],
            sample_idx=int(item["sample_idx"]),
        )

    raise ValueError(f"Unsupported target_generated_cases item type: {type(item)}")


def _load_target_generated_cases(cfg: Dict[str, Any], bio: BenchmarkIO) -> Optional[Set[GeneratedKey]]:
    """
    精确筛选生成图，包含 model/category/subtype/edit_type/uid/sample_idx。

    推荐配置：
      sampling:
        target_generated_cases:
          - uno/Human/Face/Local-Editing/11_template0_sample1
          - qwen/Human/Face/Complex-Editing/15_template0_sample3

    或：
      sampling:
        target_generated_cases:
          - model: uno
            category: Human
            subtype: Face
            edit_type: Local-Editing
            uid: 11_template0
            sample_idx: 1
    """
    sampling_cfg = cfg.get("sampling") or {}
    items = sampling_cfg.get("target_generated_cases")
    if not items:
        return None
    if isinstance(items, (str, dict)):
        items = [items]
    if not isinstance(items, list):
        raise ValueError("sampling.target_generated_cases must be a list, string, or dict.")
    return {_parse_target_generated_case_item(item, bio) for item in items}


def _parse_target_case(cfg: Dict[str, Any], bio: BenchmarkIO) -> Optional[PairKey]:
    sampling_cfg = cfg.get("sampling") or {}
    target_case = sampling_cfg.get("target_case")

    if not target_case:
        return None

    if isinstance(target_case, dict) and target_case.get("enable") is False:
        return None

    if not isinstance(target_case, dict):
        raise ValueError("sampling.target_case must be a dict.")

    required = ["category", "edit_type", "uid"]
    missing = [k for k in required if not target_case.get(k)]
    if missing:
        raise ValueError(f"sampling.target_case missing required keys: {missing}")

    return _canonical_pair_key(
        bio,
        category=target_case["category"],
        subtype=target_case.get("subtype") or "-",
        edit_type=target_case["edit_type"],
        uid=target_case["uid"],
    )


def _load_sampled_case_set(cfg: Dict[str, Any], bio: BenchmarkIO) -> Optional[Set[PairKey]]:
    """
    兼容旧筛选方式：
    1) sampling.target_case: 一个 case，不含 model/sample
    2) sampling.use_sample_list + sample_list_json: 多个 case，不含 model/sample

    如果使用 sampling.target_generated_cases，则这个函数仍可用于 prompt_index 的二级过滤，
    但真正的生成图筛选以 target_generated_cases 为准。
    """
    target_case_key = _parse_target_case(cfg, bio)
    if target_case_key is not None:
        return {target_case_key}

    sampling_cfg = cfg.get("sampling") or {}
    if not sampling_cfg.get("use_sample_list"):
        return None

    sampled_raw = load_sampled_keys(sampling_cfg["sample_list_json"])
    return {
        _canonical_pair_key(
            bio,
            category=category,
            subtype=subtype,
            edit_type=edit_type,
            uid=uid,
        )
        for category, subtype, edit_type, uid in sampled_raw
    }


def _target_sample_indices(cfg: Dict[str, Any]) -> Optional[Set[int]]:
    """
    兼容原来的 target_sample_idx，并支持多 sample。
    注意：如果 sampling.target_generated_cases 已设置，则优先使用其中的 sample_idx，
    这里的 target_sample_idx/target_sample_indices 不再额外过滤。
    """
    sampling_cfg = cfg.get("sampling") or {}

    if "target_sample_indices" in sampling_cfg and sampling_cfg["target_sample_indices"] is not None:
        values = sampling_cfg["target_sample_indices"]
        if not isinstance(values, (list, tuple, set)):
            raise ValueError("sampling.target_sample_indices must be a list.")
        return {int(x) for x in values}

    if "target_sample_idx" in sampling_cfg and sampling_cfg["target_sample_idx"] is not None:
        return {int(sampling_cfg["target_sample_idx"])}

    return None


def _target_models(cfg: Dict[str, Any], target_generated: Optional[Set[GeneratedKey]]) -> List[str]:
    """
    支持 target_generated_cases 一次跑多个 model。
    如果没设置 target_generated_cases，则使用原来的 cfg['model_name']。
    """
    if target_generated:
        return sorted({key[0] for key in target_generated})
    return [cfg.get("model_name", "bagel")]


def run_evaluator_pipeline(cfg: Dict[str, Any]) -> None:
    paths = eval_paths(cfg)

    bio = BenchmarkIO(
        image_root=cfg["benchmark"]["image_root"],
        metadata_jsonl=cfg["benchmark"]["metadata_jsonl"],
        output_root=cfg["benchmark"]["generated_root"],
    )

    target_generated = _load_target_generated_cases(cfg, bio)
    sampled = _load_sampled_case_set(cfg, bio)

    if target_generated:
        # 从精确生成图筛选项反推出 pair 级筛选，用于加速 prompt_index。
        sampled_from_generated = {(c, s, e, u) for _, c, s, e, u, _ in target_generated}
        sampled = sampled_from_generated if sampled is None else sampled & sampled_from_generated

    target_sample_set = None if target_generated else _target_sample_indices(cfg)

    question_root = cfg["benchmark"]["question_root"]
    selected_categories = cfg.get("sampling", {}).get("selected_categories")
    runtime = cfg["runtime"]
    models_to_scan = _target_models(cfg, target_generated)

    log(
        f"Evaluator pipeline started | models={models_to_scan} | "
        f"resume={'on' if runtime['resume'] else 'off'} | "
        f"workers={runtime['num_workers']} | "
        f"target_generated_cases={len(target_generated) if target_generated else 0} | "
        f"case_filter={'on' if sampled is not None else 'off'} | "
        f"target_samples={sorted(target_sample_set) if target_sample_set is not None else 'from_target_generated_or_all'}"
    )

    # 1. 建立 prompt_index，只保留目标 case 对应 prompt。
    prompt_index: Dict[PairKey, str] = {}
    prompt_keys: Set[PairKey] = set()

    for pair in bio.iter_pairs():
        key = _canonical_pair_key(
            bio,
            category=pair["category"],
            subtype=pair.get("subtype") or "-",
            edit_type=pair["edit_type"],
            uid=pair["uid"],
        )

        if selected_categories and pair["category"] not in selected_categories:
            continue

        if sampled is not None and key not in sampled:
            continue

        prompt_index[key] = pair["prompt"]
        prompt_keys.add(key)

    # 2. 扫描生成图，只保留目标 model + case + sample。
    generated: List[Dict[str, Any]] = []
    available_generated_keys: Set[GeneratedKey] = set()
    available_pair_keys: Set[PairKey] = set()
    scanned_total = 0

    max_eval_images = int(cfg.get("sampling", {}).get("max_eval_images", 10**18))

    for model_name in models_to_scan:
        for gen in bio.iter_generated_images(model_name):
            scanned_total += 1

            gen_pair_key = _canonical_pair_key(
                bio,
                category=gen["category"],
                subtype=gen.get("subtype") or "-",
                edit_type=gen["edit_type"],
                uid=gen["uid"],
            )
            gen_full_key = _canonical_generated_key(
                bio,
                model=model_name,
                category=gen["category"],
                subtype=gen.get("subtype") or "-",
                edit_type=gen["edit_type"],
                uid=gen["uid"],
                sample_idx=int(gen["sample_idx"]),
            )

            available_pair_keys.add(gen_pair_key)
            available_generated_keys.add(gen_full_key)

            if selected_categories and gen["category"] not in selected_categories:
                continue

            if target_generated is not None:
                if gen_full_key not in target_generated:
                    continue
            else:
                if sampled is not None and gen_pair_key not in sampled:
                    continue
                if target_sample_set is not None and int(gen["sample_idx"]) not in target_sample_set:
                    continue

            gen = dict(gen)
            gen["_model_name"] = model_name
            gen["_generated_key"] = gen_full_key
            generated.append(gen)

            if len(generated) >= max_eval_images:
                break
        if len(generated) >= max_eval_images:
            break

    # 3. 筛选诊断。
    if target_generated is not None:
        missing_generated = sorted(target_generated - available_generated_keys)
        missing_prompts = sorted({(c, s, e, u) for _, c, s, e, u, _ in target_generated} - prompt_keys)

        log(
            f"Target generated summary | requested={len(target_generated)} | "
            f"selected_generated_images={len(generated)} | "
            f"missing_generated={len(missing_generated)} | missing_prompts={len(missing_prompts)}"
        )
        for key in missing_generated[:30]:
            log(f"Missing generated item | {_format_generated_key(key)}")
        for category, subtype, edit_type, uid in missing_prompts[:30]:
            log(
                "Missing prompt case | "
                f"category={category} | subtype={subtype} | edit_type={edit_type} | uid={uid}"
            )
    elif sampled is not None:
        missing_generated_cases = sorted(sampled - available_pair_keys)
        missing_prompts = sorted(sampled - prompt_keys)

        log(
            f"Sampling summary | requested_cases={len(sampled)} | "
            f"selected_generated_images={len(generated)} | "
            f"missing_generated_cases={len(missing_generated_cases)} | "
            f"missing_prompt_cases={len(missing_prompts)}"
        )

        for category, subtype, edit_type, uid in missing_generated_cases[:20]:
            log(
                "Missing generated case | "
                f"category={category} | subtype={subtype} | edit_type={edit_type} | uid={uid}"
            )

        for category, subtype, edit_type, uid in missing_prompts[:20]:
            log(
                "Missing prompt case | "
                f"category={category} | subtype={subtype} | edit_type={edit_type} | uid={uid}"
            )

    log(
        f"Generated scan summary | scanned={scanned_total} | selected_for_eval={len(generated)}"
    )

    if len(generated) == 0:
        log(
            "No generated images selected. Please check target_generated_cases, model_name, "
            "generated_root, category/subtype/edit_type/uid, and sample_idx."
        )
        return

    failures = Counter()
    results: List[Dict[str, Any]] = []
    latency_records: List[Dict[str, Any]] = []

    def process(gen):
        evaluator = get_thread_evaluator(cfg)

        model_name = gen.get("_model_name") or cfg.get("model_name", "bagel")
        category = gen["category"]
        subtype = gen.get("subtype") or "-"
        edit_type = gen["edit_type"]
        uid = gen["uid"]
        sample_idx = int(gen["sample_idx"])

        case_key = _format_case_key(
            model=model_name,
            uid=uid,
            category=category,
            subtype=subtype,
            edit_type=edit_type,
            sample_idx=sample_idx,
        )

        out_json = (
            paths["result_dir"]
            / model_name
            / category
            / subtype
            / edit_type
            / f"{uid}_sample{sample_idx}.json"
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if runtime["resume"] and out_json.exists():
            log(f"Eval skip existing | {case_key}")
            return load_json(out_json), "skipped"

        try:
            log(f"Eval start | {case_key}")

            ref_path = bio.find_original_image(category, subtype, str(gen["image_id"]))
            if not ref_path:
                log(f"Ref missing | {case_key} | image_id={gen.get('image_id')}")
                return None, "ref_missing"

            qpath = _get_question_path(question_root, gen["image_path"], bio.output_root)
            if not os.path.exists(qpath):
                log(f"Question missing | {case_key} | qpath={qpath}")
                return None, "question_missing"

            prompt_key = _canonical_pair_key(
                bio,
                category=category,
                subtype=subtype,
                edit_type=edit_type,
                uid=uid,
            )
            prompt = prompt_index.get(prompt_key)
            if prompt is None:
                log(f"Prompt missing | {case_key}")
                return None, "prompt_missing"

            ref_img = Image.open(ref_path).convert("RGB").resize((1024, 1024))
            gen_img = Image.open(gen["image_path"]).convert("RGB").resize((1024, 1024))

            raw_q = load_json(qpath)
            questions = raw_q["questions"] if isinstance(raw_q, dict) else raw_q

            result = evaluator.evaluate_question_list(
                questions,
                prompt=prompt,
                ref_img=ref_img,
                gen_img=gen_img,
            )

            payload = {
                "uid": uid,
                "sample_idx": sample_idx,
                "model": model_name,
                "category": category,
                "subtype": subtype,
                "edit_type": edit_type,
                "image_id": gen.get("image_id"),
                "image_name": os.path.basename(gen["image_path"]),
                "gen_image_path": gen["image_path"],
                "ref_image_path": ref_path,
                "question_path": qpath,
                **result,
            }

            atomic_json_dump(payload, out_json)
            log(f"Eval done | {case_key} | final_score={payload.get('final_score')}")
            return payload, "ok"

        except Exception as e:
            log_failure(
                "evaluator_pipeline",
                uid,
                category,
                subtype,
                edit_type,
                e,
                paths["failure_log_jsonl"],
            )
            return None, f"eval_fail:{type(e).__name__}"

    # 4. 开始评分。
    t0 = now()

    with ThreadPoolExecutor(max_workers=runtime["num_workers"]) as ex:
        futures = [ex.submit(process, gen) for gen in generated]

        with tqdm(total=len(futures), desc="Scoring", unit="img") as pbar:
            for fut in as_completed(futures):
                payload, status = fut.result()

                if status not in ("ok", "skipped"):
                    failures[status] += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        ok=len(results),
                        skipped=failures.get("skipped", 0),
                        failed=sum(v for k, v in failures.items() if k != "skipped"),
                    )
                    continue

                if payload:
                    results.append(payload)

                    if "latency_summary" in payload:
                        latency_records.append(
                            {
                                "uid": payload["uid"],
                                "model": payload.get("model"),
                                "category": payload.get("category"),
                                "subtype": payload.get("subtype"),
                                "edit_type": payload.get("edit_type"),
                                "sample_idx": payload.get("sample_idx"),
                                **payload["latency_summary"],
                            }
                        )

                if status == "skipped":
                    failures["skipped"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    ok=len(results),
                    skipped=failures.get("skipped", 0),
                    failed=sum(v for k, v in failures.items() if k != "skipped"),
                )

    # 5. 汇总输出。支持同一批里混合多个 model / case / sample。
    group_to_sample_scores = defaultdict(list)
    group_to_items = defaultdict(list)

    for item in results:
        group_key = (
            item.get("model"),
            item.get("category"),
            item.get("subtype"),
            item.get("edit_type"),
            item.get("uid"),
        )
        group_to_sample_scores[group_key].append(item["final_score"])
        group_to_items[group_key].append(item)

    write_all_sample_scores(
        results=results,
        json_path=paths["all_sample_scores_json"],
        csv_path=paths["all_sample_scores_csv"],
    )

    for group_key, sample_scores in group_to_sample_scores.items():
        if not sample_scores:
            continue

        model_name, category, subtype, edit_type, uid = group_key
        mean_score = sum(sample_scores) / len(sample_scores)
        variance = sum((x - mean_score) ** 2 for x in sample_scores) / len(sample_scores)

        summary_payload = {
            "uid": uid,
            "model": model_name,
            "category": category,
            "subtype": subtype,
            "edit_type": edit_type,
            "num_samples": len(sample_scores),
            "final_score_mean": mean_score,
            "final_score_std": variance ** 0.5,
            "sample_scores": sample_scores,
            "sample_indices": [x.get("sample_idx") for x in group_to_items[group_key]],
        }

        summary_path = (
            paths["result_dir"]
            / str(model_name)
            / str(category)
            / str(subtype)
            / str(edit_type)
            / f"{uid}_summary.json"
        )
        atomic_json_dump(summary_payload, summary_path)

    total_wall_sec = now() - t0
    latency_summary = {
        "num_images": len(results),
        "total_wall_time_ms": total_wall_sec * 1000.0,
        "throughput_img_per_sec": (len(results) / total_wall_sec if total_wall_sec > 0 else 0.0),
        "failure_stats": dict(failures),
        "num_questions": summarize_values_raw([x.get("num_questions") for x in latency_records]),
        "num_no_answers": summarize_values_raw([x.get("num_no_answers") for x in latency_records]),
        "num_face_questions": summarize_values_raw([x.get("num_face_questions") for x in latency_records]),
        "image_eval_latency_ms": summarize_values_ms(
            [(x.get("image_eval_total_ms", 0.0) / 1000.0) for x in latency_records]
        ),
        "question_latency_ms": summarize_values_ms(
            [(x.get("mean_question_ms", 0.0) / 1000.0) for x in latency_records]
        ),
        "question_sum_latency_ms": summarize_values_ms(
            [(x.get("sum_question_ms", 0.0) / 1000.0) for x in latency_records]
        ),
        "main_api_latency_ms": summarize_values_ms(
            [(x.get("sum_main_api_ms", 0.0) / 1000.0) for x in latency_records]
        ),
        "no_reason_latency_ms": summarize_values_ms(
            [(x.get("sum_no_reason_ms", 0.0) / 1000.0) for x in latency_records]
        ),
        "face_latency_ms": summarize_values_ms(
            [(x.get("sum_face_ms", 0.0) / 1000.0) for x in latency_records]
        ),
    }

    atomic_json_dump(latency_summary, paths["latency_summary_json"])
    atomic_json_dump(latency_records, paths["latency_detail_json"])

    analysis_cfg = cfg.get("analysis") or {}

    if analysis_cfg.get("failure_summary", True):
        failure_only = {k: v for k, v in failures.items() if k != "skipped"}
        write_failure_summary(
            output_path=paths["failure_summary_json"],
            model_name="mixed" if len(models_to_scan) > 1 else models_to_scan[0],
            failure_stats=failure_only,
        )

    # 指定一批 case 时建议设 analysis.debug_reports=false，避免额外扫描全量数据。
    if analysis_cfg.get("debug_reports", True):
        for model_name in models_to_scan:
            cfg_for_debug = dict(cfg)
            cfg_for_debug["model_name"] = model_name
            write_debug_reports(
                cfg=cfg_for_debug,
                bio=bio,
                paths=paths,
                model_name=model_name,
            )

    log("Evaluator pipeline finished")
