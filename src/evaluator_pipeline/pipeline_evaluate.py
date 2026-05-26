from __future__ import annotations

import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

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


def get_thread_evaluator(cfg):
    if not hasattr(_THREAD_LOCAL, 'evaluator'):
        _THREAD_LOCAL.evaluator = QuestionEvaluator(cfg)
    return _THREAD_LOCAL.evaluator


def _get_question_path(question_root: str, gen_path: str, gen_root: str) -> str:
    rel = os.path.relpath(gen_path, gen_root)
    parts = rel.split(os.sep)[1:]
    parts[-1] = parts[-1].split('_sample')[0] + '.json'
    return os.path.join(question_root, *parts)


def _canonical_key_component(bio: BenchmarkIO, value: str | None) -> str:
    if value in (None, ""):
        return "-"
    if value == "-":
        return value
    return bio.safe_filename(str(value))


def _canonical_pair_key(
    bio: BenchmarkIO,
    *,
    category: str,
    subtype: str | None,
    edit_type: str,
    uid: str,
) -> tuple[str, str, str, str]:
    return (
        _canonical_key_component(bio, category),
        _canonical_key_component(bio, subtype),
        _canonical_key_component(bio, edit_type),
        uid,
    )


def _format_case_key(*, uid: str, category: str, subtype: str, edit_type: str, sample_idx: int | None = None) -> str:
    parts = [
        f"uid={uid}",
        f"category={category}",
        f"subtype={subtype}",
        f"edit_type={edit_type}",
    ]
    if sample_idx is not None:
        parts.append(f"sample_idx={sample_idx}")
    return " | ".join(parts)


def run_evaluator_pipeline(cfg: Dict[str, Any]) -> None:
    paths = eval_paths(cfg)
    bio = BenchmarkIO(
        image_root=cfg['benchmark']['image_root'],
        metadata_jsonl=cfg['benchmark']['metadata_jsonl'],
        output_root=cfg['benchmark']['generated_root'],
    )
    sampled_raw = load_sampled_keys(cfg['sampling']['sample_list_json']) if cfg['sampling']['use_sample_list'] else None
    sampled = None
    if sampled_raw is not None:
        sampled = {
            _canonical_pair_key(
                bio,
                category=category,
                subtype=subtype,
                edit_type=edit_type,
                uid=uid,
            )
            for category, subtype, edit_type, uid in sampled_raw
        }
    target_sample_idx = cfg['sampling']['target_sample_idx']
    question_root = cfg['benchmark']['question_root']
    model_name = cfg.get('model_name', 'bagel')
    selected_categories = cfg['sampling'].get('selected_categories')
    runtime = cfg['runtime']
    log(
        f"Evaluator pipeline started | model_name={model_name} | "
        f"resume={'on' if runtime['resume'] else 'off'} | workers={runtime['num_workers']}"
    )

    prompt_index = {}
    prompt_keys = set()
    for pair in bio.iter_pairs():
        key = _canonical_pair_key(
            bio,
            category=pair['category'],
            subtype=pair.get('subtype') or '-',
            edit_type=pair['edit_type'],
            uid=pair['uid'],
        )
        if selected_categories and pair['category'] not in selected_categories:
            continue
        if sampled is not None and key not in sampled:
            continue
        prompt_index[key] = pair['prompt']
        prompt_keys.add(key)

    generated = []
    generated_all = []
    for gen in bio.iter_generated_images(model_name):
        gen_key = _canonical_pair_key(
            bio,
            category=gen['category'],
            subtype=gen['subtype'] or '-',
            edit_type=gen['edit_type'],
            uid=gen['uid'],
        )
        generated_all.append((gen, gen_key))
        if gen['sample_idx'] != target_sample_idx:
            continue
        if sampled is not None and gen_key not in sampled:
            continue
        generated.append(gen)
        if len(generated) >= cfg['sampling']['max_eval_images']:
            break

    available_generated_keys = {key for _, key in generated_all}
    if sampled is not None:
        missing_generated = sorted(sampled - available_generated_keys)
        missing_prompts = sorted(sampled - prompt_keys)
        log(
            f"Sampling summary | requested={len(sampled)} | "
            f"generated_matches={len(generated)} | "
            f"missing_generated={len(missing_generated)} | missing_prompts={len(missing_prompts)}"
        )
        for category, subtype, edit_type, uid in missing_generated[:20]:
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
        f"Generated scan summary | scanned={len(generated_all)} | "
        f"target_sample_idx={target_sample_idx} | selected_for_eval={len(generated)}"
    )

    failures = Counter()
    results = []
    latency_records = []

    def process(gen):
        evaluator = get_thread_evaluator(cfg)
        category = gen['category']
        subtype = gen['subtype']
        edit_type = gen['edit_type']
        uid = gen['uid']
        case_key = _format_case_key(
            uid=uid,
            category=category,
            subtype=subtype,
            edit_type=edit_type,
            sample_idx=gen['sample_idx'],
        )
        out_json = paths['result_dir'] / model_name / category / subtype / edit_type / f"{uid}_sample{gen['sample_idx']}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if runtime['resume'] and out_json.exists():
            log(f"Eval skip existing | {case_key}")
            return load_json(out_json), 'skipped'
        try:
            log(f"Eval start | {case_key}")
            ref_path = bio.find_original_image(category, subtype, str(gen['image_id']))
            if not ref_path:
                return None, 'ref_missing'
            ref_img = Image.open(ref_path).convert('RGB').resize((1024, 1024))
            gen_img = Image.open(gen['image_path']).convert('RGB').resize((1024, 1024))
            qpath = _get_question_path(question_root, gen['image_path'], bio.output_root)
            if not os.path.exists(qpath):
                return None, 'question_missing'
            raw_q = load_json(qpath)
            questions = raw_q['questions'] if isinstance(raw_q, dict) else raw_q
            prompt_key = _canonical_pair_key(
                bio,
                category=category,
                subtype=subtype or '-',
                edit_type=edit_type,
                uid=uid,
            )
            prompt = prompt_index.get(prompt_key)
            if prompt is None:
                return None, 'prompt_missing'
            result = evaluator.evaluate_question_list(questions, prompt=prompt, ref_img=ref_img, gen_img=gen_img)
            payload = {
                'uid': uid,
                'sample_idx': gen['sample_idx'],
                'model': model_name,
                'category': category,
                'subtype': subtype,
                'edit_type': edit_type,
                'image_name': os.path.basename(gen['image_path']),
                **result,
            }
            atomic_json_dump(payload, out_json)
            log(f"Eval done | {case_key}")
            return payload, 'ok'
        except Exception as e:
            log_failure('evaluator_pipeline', uid, category, subtype, edit_type, e, paths['failure_log_jsonl'])
            return None, f'eval_fail:{type(e).__name__}'

    t0 = now()
    with ThreadPoolExecutor(max_workers=runtime['num_workers']) as ex:
        futures = [ex.submit(process, gen) for gen in generated]
        with tqdm(total=len(futures), desc='Scoring', unit='img') as pbar:
            for fut in as_completed(futures):
                payload, status = fut.result()
                if status not in ('ok', 'skipped'):
                    failures[status] += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        ok=len(results),
                        skipped=failures.get('skipped', 0),
                        failed=sum(failures.values()),
                    )
                    continue
                if payload:
                    results.append(payload)
                    if 'latency_summary' in payload:
                        latency_records.append(
                            {
                                'uid': payload['uid'],
                                'category': payload.get('category'),
                                'subtype': payload.get('subtype'),
                                'edit_type': payload.get('edit_type'),
                                **payload['latency_summary'],
                            }
                        )
                if status == 'skipped':
                    failures['skipped'] += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=len(results),
                    skipped=failures.get('skipped', 0),
                    failed=sum(v for k, v in failures.items() if k != 'skipped'),
                )

    uid_to_sample_scores = defaultdict(list)
    for item in results:
        uid_to_sample_scores[item['uid']].append(item['final_score'])

    write_all_sample_scores(
        results=results,
        json_path=paths['all_sample_scores_json'],
        csv_path=paths['all_sample_scores_csv'],
    )

    for uid, sample_scores in uid_to_sample_scores.items():
        if not sample_scores:
            continue
        mean_score = sum(sample_scores) / len(sample_scores)
        variance = sum((x - mean_score) ** 2 for x in sample_scores) / len(sample_scores)
        summary_payload = {
            'uid': uid,
            'model': model_name,
            'num_samples': len(sample_scores),
            'final_score_mean': mean_score,
            'final_score_std': variance ** 0.5,
            'sample_scores': sample_scores,
        }
        matching = next((item for item in results if item['uid'] == uid), None)
        if matching is not None:
            summary_path = (
                paths['result_dir']
                / model_name
                / matching['category']
                / matching['subtype']
                / matching['edit_type']
                / f'{uid}_summary.json'
            )
        else:
            summary_path = paths['result_dir'] / model_name / f'{uid}_summary.json'
        atomic_json_dump(summary_payload, summary_path)

    total_wall_sec = now() - t0
    latency_summary = {
        'num_images': len(results),
        'total_wall_time_ms': total_wall_sec * 1000.0,
        'throughput_img_per_sec': (len(results) / total_wall_sec if total_wall_sec > 0 else 0.0),
        'failure_stats': dict(failures),
        'num_questions': summarize_values_raw([x.get('num_questions') for x in latency_records]),
        'num_no_answers': summarize_values_raw([x.get('num_no_answers') for x in latency_records]),
        'num_face_questions': summarize_values_raw([x.get('num_face_questions') for x in latency_records]),
        'image_eval_latency_ms': summarize_values_ms([(x.get('image_eval_total_ms', 0.0) / 1000.0) for x in latency_records]),
        'question_latency_ms': summarize_values_ms([(x.get('mean_question_ms', 0.0) / 1000.0) for x in latency_records]),
        'question_sum_latency_ms': summarize_values_ms([(x.get('sum_question_ms', 0.0) / 1000.0) for x in latency_records]),
        'main_api_latency_ms': summarize_values_ms([(x.get('sum_main_api_ms', 0.0) / 1000.0) for x in latency_records]),
        'no_reason_latency_ms': summarize_values_ms([(x.get('sum_no_reason_ms', 0.0) / 1000.0) for x in latency_records]),
        'face_latency_ms': summarize_values_ms([(x.get('sum_face_ms', 0.0) / 1000.0) for x in latency_records]),
    }
    atomic_json_dump(latency_summary, paths['latency_summary_json'])
    atomic_json_dump(latency_records, paths['latency_detail_json'])
    analysis_cfg = cfg.get('analysis') or {}
    if analysis_cfg.get('failure_summary', True):
        failure_only = {k: v for k, v in failures.items() if k != 'skipped'}
        write_failure_summary(
            output_path=paths['failure_summary_json'],
            model_name=model_name,
            failure_stats=failure_only,
        )
    write_debug_reports(
        cfg=cfg,
        bio=bio,
        paths=paths,
        model_name=model_name,
    )
    log('Evaluator pipeline finished')
