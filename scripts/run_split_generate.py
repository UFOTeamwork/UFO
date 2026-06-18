from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.split_pipeline.pipeline import run_split_pipeline
from src.vlm_tools.models import normalize_vlm_name

SUPPORTED_VLMS = [
    "gpt",
    "gemini",
    "doubao",
    "qwen",
    "claude"
]


def validate_vlm(vlm_name: str) -> str:
    try:
        provider = normalize_vlm_name(vlm_name)
        return provider
    except Exception:
        print("\n? [UFO ERROR] Unsupported VLM:", vlm_name)
        print("\n? Available VLM options:")
        for v in SUPPORTED_VLMS:
            print("  -", v)

        print("\n? Suggestion:")
        print("  Please choose one of the above VLM names, e.g.:")
        print("  python run_split.py --vlm gpt-4o\n")

        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--vlm', required=True, help='VLM backend name')

    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # ===== VLM validation + normalization =====
    provider = validate_vlm(args.vlm)

    vlm_cfg = dict(cfg.get('vlm') or {})
    vlm_cfg['provider'] = provider      # internal key (gpt/qwen/...)
    vlm_cfg['model'] = args.vlm         # raw model name
    cfg['vlm'] = vlm_cfg

    # run pipeline
    run_split_pipeline(cfg)


if __name__ == '__main__':
    main()