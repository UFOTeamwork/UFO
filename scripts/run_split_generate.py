from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.split_pipeline.pipeline import run_split_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--vlm', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    vlm_cfg = dict(cfg.get('vlm') or {})
    vlm_cfg['provider'] = args.vlm
    cfg['vlm'] = vlm_cfg
    run_split_pipeline(cfg)


if __name__ == '__main__':
    main()
