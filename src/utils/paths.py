from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def build_project_output_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg['project']['root_dir']) / cfg['project']['output_dir']


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = build_project_output_root(cfg)
    paths_cfg = cfg['paths']
    return {
        'output_root': ensure_dir(root),
        'tgt_dir': ensure_dir(root / paths_cfg['tgt_dir']),
        'question_dir': ensure_dir(root / paths_cfg['question_dir']),
        'prompt_root': ensure_dir(root / paths_cfg['prompt_root']),
        'tgt_prompt_dir': ensure_dir(root / paths_cfg['tgt_prompt_dir']),
        'question_prompt_dir': ensure_dir(root / paths_cfg['question_prompt_dir']),
        'latency_detail_json': root / paths_cfg['latency_detail_json'],
        'latency_summary_json': root / paths_cfg['latency_summary_json'],
        'failure_log_jsonl': root / paths_cfg['failure_log_jsonl'],
    }


def eval_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = build_project_output_root(cfg)
    paths_cfg = cfg['paths']
    model_name = cfg.get('model_name', 'bagel')
    result_dir = ensure_dir(root / paths_cfg['result_dir'])
    model_dir = ensure_dir(result_dir / model_name)
    return {
        'output_root': ensure_dir(root),
        'result_dir': result_dir,
        'model_dir': model_dir,
        'debug_dir': ensure_dir(model_dir / paths_cfg.get('debug_dir', 'debug')),
        'latency_detail_json': model_dir / paths_cfg['latency_detail_json'],
        'latency_summary_json': model_dir / paths_cfg['latency_summary_json'],
        'all_sample_scores_json': model_dir / paths_cfg['all_sample_scores_json'],
        'all_sample_scores_csv': model_dir / paths_cfg['all_sample_scores_csv'],
        'failure_log_jsonl': model_dir / paths_cfg['failure_log_jsonl'],
        'failure_summary_json': model_dir / paths_cfg.get('failure_summary_json', 'failure_summary.json'),
    }
