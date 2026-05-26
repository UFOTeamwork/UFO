from __future__ import annotations

from src.utils.io_utils import atomic_json_dump
from src.utils.stats import summarize_values_ms, summarize_values_raw


def dump_latency_summary(results, detail_path, summary_path, total_wall_sec):
    atomic_json_dump(results, detail_path)
    ok = [x for x in results if x and x.get('success')]
    tgt_results = [x for x in results if x and x.get('stage') == 'tgt']
    question_results = [x for x in results if x and x.get('stage') == 'question']
    tgt_ok = [x for x in tgt_results if x.get('success')]
    question_ok = [x for x in question_results if x.get('success')]
    unique_uids = sorted({x.get('uid') for x in results if x and x.get('uid')})
    summary = {
        'num_pairs': len(unique_uids),
        'num_stage_records': len(results),
        'num_success': len(ok),
        'num_failed': len(results) - len(ok),
        'tgt_num_success': len(tgt_ok),
        'tgt_num_failed': len(tgt_results) - len(tgt_ok),
        'question_num_success': len(question_ok),
        'question_num_failed': len(question_results) - len(question_ok),
        'total_wall_time_ms': total_wall_sec * 1000.0,
        'throughput_pair_per_sec': (len(question_ok) / total_wall_sec if total_wall_sec > 0 else 0.0),
        'pair_latency_ms': summarize_values_ms([x.get('pair_total_sec') for x in ok]),
        'tgt_latency_ms': summarize_values_ms([x.get('pair_total_sec') for x in tgt_ok]),
        'question_latency_ms': summarize_values_ms([x.get('pair_total_sec') for x in question_ok]),
        'tgt_retry_count': summarize_values_raw([x.get('tgt_retry_count') for x in tgt_ok]),
        'question_retry_count': summarize_values_raw([x.get('question_retry_count') for x in question_ok]),
    }
    atomic_json_dump(summary, summary_path)
