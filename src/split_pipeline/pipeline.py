from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from src.utils.benchmark_io import BenchmarkIO
from src.utils.io_utils import atomic_json_dump, load_json, write_text
from src.utils.logging_utils import log, log_failure
from src.utils.paths import split_paths
from src.utils.retry import call_with_backoff, now
from src.utils.sampling import load_sampled_keys
from .client import init_vlm_client
from .generate_tgt import generate_tgt
from .prompts import load_system_prompt_tgt
from .generate_question import generate_question
from .validators import valid_question_json, valid_tgt_json
from .latency_dump import dump_latency_summary


def _make_out_path(root, category, subtype, edit_type, uid):
    path = root / category / subtype / edit_type / f'{uid}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _make_prompt_path(root, category, subtype, edit_type, uid):
    path = root / category / subtype / edit_type / f'{uid}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _format_case_key(*, uid: str, category: str, subtype: str, edit_type: str) -> str:
    return f"uid={uid} | category={category} | subtype={subtype} | edit_type={edit_type}"


def run_split_pipeline(cfg: Dict[str, Any]) -> None:
    paths = split_paths(cfg)
    bio = BenchmarkIO(
        image_root=cfg['benchmark']['image_root'],
        metadata_jsonl=cfg['benchmark']['metadata_jsonl'],
        output_root=cfg['benchmark']['generated_root'],
    )
    sampled_keys = None
    if cfg['sampling']['use_sample_list']:
        sampled_keys = load_sampled_keys(cfg['sampling']['sample_list_json'])
    selected_categories = cfg['sampling'].get('selected_categories')
    pairs = []
    available_keys = set()
    for pair in bio.iter_pairs():
        key = (pair['category'], pair.get('subtype') or '-', pair['edit_type'], pair['uid'])
        available_keys.add(key)
        if selected_categories and pair['category'] not in selected_categories:
            continue
        if sampled_keys is not None and key not in sampled_keys:
            continue
        pairs.append(pair)

    if sampled_keys is not None:
        missing_sampled_keys = sorted(sampled_keys - available_keys)
        log(
            f"Sampling summary | requested={len(sampled_keys)} | "
            f"matched={len(pairs)} | missing={len(missing_sampled_keys)}"
        )
        for category, subtype, edit_type, uid in missing_sampled_keys[:20]:
            log(
                "Missing sampled case | "
                f"category={category} | subtype={subtype} | edit_type={edit_type} | uid={uid}"
            )

    runtime = cfg['runtime']
    log(
        f"Split pipeline started | {len(pairs)} pairs found | "
        f"resume={'on' if runtime['resume'] else 'off'} | workers={runtime['num_workers']}"
    )

    def process_tgt(pair):
        uid = pair['uid']
        category = pair['category']
        subtype = pair['subtype']
        edit_type = pair['edit_type']
        case_key = _format_case_key(uid=uid, category=category, subtype=subtype, edit_type=edit_type)
        stat = {
            'uid': uid,
            'category': category,
            'subtype': subtype,
            'edit_type': edit_type,
            'success': False,
            'stage': 'tgt',
            'pair_total_sec': 0.0,
            'tgt_retry_count': 0,
            'skip_tgt': False,
        }
        t0 = now()
        try:
            log(f"Split T_gt start | {case_key}")
            vlm = init_vlm_client(cfg)
            tgt_path = _make_out_path(paths['tgt_dir'], category, subtype, edit_type, uid)
            tgt_prompt_txt = _make_prompt_path(paths['tgt_prompt_dir'], category, subtype, edit_type, uid)

            def is_valid_file(path):
                try:
                    if not os.path.exists(path):
                        return False
                    data = load_json(path)
                    return data is not None and "uid" in data
                except:
                    return False

            if not (runtime['resume'] and is_valid_file(tgt_path)):
                tgt_info = call_with_backoff(
                    lambda: generate_tgt(
                        vlm,
                        prompt=pair['prompt'],
                        ref_image_path=pair['ref_image'],
                    ),
                    max_retries=runtime['max_retries'],
                    base_sleep=runtime['base_sleep_sec'],
                    max_sleep=runtime['max_sleep_sec'],
                )
                tgt, _ = tgt_info['result']
                stat['tgt_retry_count'] = tgt_info['retry_count']

                if not valid_tgt_json(tgt):
                    print("[ERROR] Invalid T_gt JSON after parsing.", flush=True)
                    print("[ERROR] T_gt input system prompt:", flush=True)
                    print(load_system_prompt_tgt(), flush=True)
                    print("[ERROR] T_gt input user prompt:", flush=True)
                    print(pair['prompt'], flush=True)
                    print("[ERROR] T_gt input reference image:", flush=True)
                    print(pair['ref_image'], flush=True)
                    print("[ERROR] Parsed T_gt JSON:", flush=True)
                    print(json.dumps(tgt, ensure_ascii=False, indent=2), flush=True)
                    raise ValueError('Invalid T_gt JSON')
                tgt['uid'] = uid
                atomic_json_dump(tgt, tgt_path)
                write_text(pair['prompt'], tgt_prompt_txt)
            else:
                stat['skip_tgt'] = True

            stat['success'] = True
            log(f"Split T_gt done | {case_key} | skipped={'yes' if stat['skip_tgt'] else 'no'}")
        except Exception as e:
            log_failure('split_pipeline_tgt', uid, category, subtype, edit_type, e, paths['failure_log_jsonl'])
            stat['error'] = f'{type(e).__name__}: {e}'
        stat['pair_total_sec'] = now() - t0
        return stat

    def process_question(pair):
        uid = pair['uid']
        category = pair['category']
        subtype = pair['subtype']
        edit_type = pair['edit_type']
        case_key = _format_case_key(uid=uid, category=category, subtype=subtype, edit_type=edit_type)
        stat = {
            'uid': uid,
            'category': category,
            'subtype': subtype,
            'edit_type': edit_type,
            'success': False,
            'stage': 'question',
            'pair_total_sec': 0.0,
            'question_retry_count': 0,
            'skip_question': False,
        }
        t0 = now()
        try:
            log(f"Split Question start | {case_key}")
            vlm = init_vlm_client(cfg)
            tgt_path = _make_out_path(paths['tgt_dir'], category, subtype, edit_type, uid)
            q_path = _make_out_path(paths['question_dir'], category, subtype, edit_type, uid)
            q_prompt_txt = _make_prompt_path(paths['question_prompt_dir'], category, subtype, edit_type, uid)

            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f'Missing T_gt file: {tgt_path}')
            tgt = load_json(tgt_path)
            if not valid_tgt_json(tgt):
                raise ValueError('Invalid T_gt JSON on disk')

            if not (runtime['resume'] and os.path.exists(q_path)):
                q_info = call_with_backoff(
                    lambda: generate_question(vlm, tgt),
                    max_retries=runtime['max_retries'],
                    base_sleep=runtime['base_sleep_sec'],
                    max_sleep=runtime['max_sleep_sec'],
                )
                questions = q_info['result']
                stat['question_retry_count'] = q_info['retry_count']
                if not valid_question_json(questions):
                    raise ValueError('Invalid question JSON')
                atomic_json_dump({'source': uid, 'questions': questions}, q_path)
                write_text(f'prompt: {pair["prompt"]}\nuid: {uid}\n', q_prompt_txt)
            else:
                stat['skip_question'] = True

            stat['success'] = True
            log(f"Split Question done | {case_key} | skipped={'yes' if stat['skip_question'] else 'no'}")
        except Exception as e:
            log_failure('split_pipeline_question', uid, category, subtype, edit_type, e, paths['failure_log_jsonl'])
            stat['error'] = f'{type(e).__name__}: {e}'
        stat['pair_total_sec'] = now() - t0
        return stat

    def _run_stage(stage_name: str, stage_desc: str, stage_items: List[Dict[str, Any]], worker_fn):
        stage_results: List[Dict[str, Any]] = []
        progress_stats = {
            'ok': 0,
            'fail': 0,
            'skipped': 0,
        }
        with ThreadPoolExecutor(max_workers=runtime['num_workers']) as ex:
            futures = [ex.submit(worker_fn, item) for item in stage_items]
            with tqdm(total=len(futures), desc=stage_desc, unit='pair') as pbar:
                for fut in as_completed(futures):
                    stat = fut.result()
                    stage_results.append(stat)
                    if stat.get('success'):
                        progress_stats['ok'] += 1
                    else:
                        progress_stats['fail'] += 1
                    if stat.get('skip_tgt') or stat.get('skip_question'):
                        progress_stats['skipped'] += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        ok=progress_stats['ok'],
                        fail=progress_stats['fail'],
                        skipped=progress_stats['skipped'],
                    )
        log(
            f"{stage_name} finished | ok={progress_stats['ok']} | "
            f"fail={progress_stats['fail']} | skipped={progress_stats['skipped']}"
        )
        return stage_results

    t0_all = now()
    results: List[Dict[str, Any]] = []
    log('Stage 1/2 | Running T_gt decomposition for all pairs')
    tgt_results = _run_stage('T_gt stage', 'T_gt', pairs, process_tgt)
    results.extend(tgt_results)

    succeeded_uids = {
        item['uid']
        for item in tgt_results
        if item.get('success')
    }
    question_pairs = [pair for pair in pairs if pair['uid'] in succeeded_uids]
    log(
        f"Stage 2/2 | Running question generation after T_gt completion | "
        f"eligible_pairs={len(question_pairs)}"
    )
    question_results = _run_stage('Question stage', 'Question', question_pairs, process_question)
    results.extend(question_results)
    dump_latency_summary(results, paths['latency_detail_json'], paths['latency_summary_json'], now() - t0_all)
    log('Split pipeline finished')
