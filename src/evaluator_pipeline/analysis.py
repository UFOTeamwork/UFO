from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from src.utils.io_utils import atomic_json_dump
from src.utils.logging_utils import log


def collect_benchmark_statistics(
    bio,
    *,
    model_name: str,
) -> Dict[str, Any]:
    total_pairs = 0
    pairs_with_samples = 0
    total_generated_images = 0
    gen_index = {}

    for gen_meta in bio.iter_generated_images(model_name):
        key = (
            gen_meta["uid"],
            gen_meta["category"],
            gen_meta["subtype"],
            gen_meta["edit_type"],
            gen_meta["template_idx"],
        )
        gen_index[key] = gen_index.get(key, 0) + 1
        total_generated_images += 1

    for pair in bio.iter_pairs():
        total_pairs += 1
        key = (
            pair["uid"],
            pair["category"],
            pair["subtype"],
            pair["edit_type"],
            pair["template_idx"],
        )
        if key in gen_index:
            pairs_with_samples += 1

    return {
        "model_name": model_name,
        "total_pairs": total_pairs,
        "pairs_with_samples": pairs_with_samples,
        "pairs_without_samples": total_pairs - pairs_with_samples,
        "total_generated_samples": total_generated_images,
    }


def collect_folder_level_statistics(result_root: str | Path, *, stage_name: str) -> Dict[str, Any]:
    result_root = Path(result_root)
    folder_counter: dict[Tuple[str, str, str], int] = defaultdict(int)
    total_files = 0

    if not result_root.exists():
        return {
            "stage_name": stage_name,
            "result_root": str(result_root),
            "per_folder": [],
            "total_files": 0,
        }

    for folder in result_root.rglob("*"):
        if not folder.is_dir():
            continue
        json_files = [p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"]
        if not json_files:
            continue

        rel_parts = folder.relative_to(result_root).parts
        if len(rel_parts) == 2:
            category, edit_type = rel_parts
            subtype = "-"
        elif len(rel_parts) >= 3:
            category, subtype, edit_type = rel_parts[:3]
        else:
            continue

        count = len(json_files)
        folder_counter[(category, subtype, edit_type)] += count
        total_files += count

    records = [
        {
            "category": category,
            "subtype": subtype,
            "edit_type": edit_type,
            "count": count,
        }
        for (category, subtype, edit_type), count in sorted(folder_counter.items())
    ]
    return {
        "stage_name": stage_name,
        "result_root": str(result_root),
        "per_folder": records,
        "total_files": total_files,
    }


def write_failure_summary(
    *,
    output_path: str | Path,
    model_name: str,
    failure_stats: Dict[str, int],
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "failure_stats": dict(failure_stats),
    }
    atomic_json_dump(payload, output_path)
    return payload


def write_all_sample_scores(
    *,
    results: Iterable[Dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_scores: Dict[str, float] = {}
    rows = []

    for item in results:
        category = item.get("category")
        subtype = item.get("subtype") or "-"
        edit_type = item.get("edit_type")
        rel_path = f"{category}/{subtype}/{edit_type}/{item['uid']}_sample{item['sample_idx']}.json"
        full_key = f"{item['model']}/{rel_path}"
        final_score = item["final_score"]
        all_scores[full_key] = final_score
        rows.append(
            {
                "model": item["model"],
                "category": category,
                "subtype": subtype,
                "edit_type": edit_type,
                "uid": item["uid"],
                "sample_idx": item["sample_idx"],
                "final_score": final_score,
                "rel_path": rel_path,
            }
        )

    atomic_json_dump(all_scores, json_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "category",
                "subtype",
                "edit_type",
                "uid",
                "sample_idx",
                "final_score",
                "rel_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "num_samples": len(all_scores),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def write_debug_reports(
    *,
    cfg: Dict[str, Any],
    bio,
    paths: Dict[str, Path],
    model_name: str,
) -> None:
    debug_cfg = cfg.get("debug") or {}
    if not debug_cfg.get("enabled", False):
        return

    analysis_cfg = cfg.get("analysis") or {}
    debug_dir = paths["debug_dir"]
    debug_dir.mkdir(parents=True, exist_ok=True)

    if analysis_cfg.get("benchmark_stats", True):
        benchmark_stats = collect_benchmark_statistics(bio, model_name=model_name)
        out_path = debug_dir / analysis_cfg.get("benchmark_stats_json", "benchmark_stats_eval.json")
        atomic_json_dump(benchmark_stats, out_path)
        log(f"[DEBUG] benchmark stats saved to {out_path}")

    if analysis_cfg.get("folder_stats", True):
        question_root = Path(cfg["benchmark"]["question_root"])
        question_stats = collect_folder_level_statistics(question_root, stage_name="Question")
        question_out = debug_dir / analysis_cfg.get("question_folder_stats_json", "folder_stats_question.json")
        atomic_json_dump(question_stats, question_out)
        log(f"[DEBUG] question folder stats saved to {question_out}")

        result_root = paths["result_dir"] / model_name
        result_stats = collect_folder_level_statistics(result_root, stage_name="EvalResult")
        result_out = debug_dir / analysis_cfg.get("result_folder_stats_json", "folder_stats_eval_result.json")
        atomic_json_dump(result_stats, result_out)
        log(f"[DEBUG] eval result folder stats saved to {result_out}")
