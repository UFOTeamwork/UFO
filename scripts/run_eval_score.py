from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.evaluator_pipeline.pipeline_evaluate import run_evaluator_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name', default='bagel')
    parser.add_argument('--vlm', required=True)
    parser.add_argument('--vlm_model', default=None,
                        help='Override the judge VLM model name (e.g. a non-deprecated model). '
                             'Falls back to config vlm.model, then the backend default.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['model_name'] = args.model_name
    vlm_cfg = dict(cfg.get('vlm') or {})
    vlm_cfg['provider'] = args.vlm
    if args.vlm_model:
        vlm_cfg['model'] = args.vlm_model
    cfg['vlm'] = vlm_cfg
    debug_cfg = dict(cfg.get('debug') or {})
    if args.debug:
        debug_cfg['enabled'] = True
    cfg['debug'] = debug_cfg
    run_evaluator_pipeline(cfg)


if __name__ == '__main__':
    main()
