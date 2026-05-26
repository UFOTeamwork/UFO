# Multimodal Evaluation Research Project

A modular research-grade project layout for:

1. Split pipeline: generate `T_gt` and question files from reference image + prompt.
2. Evaluation pipeline: read generated questions and score model outputs with a selected VLM.

## Run

```bash
pip install -r requirements.txt
python scripts/run_split_generate.py --config config/split_config.yaml --vlm gpt
python scripts/run_eval_score.py --config config/eval_config.yaml --model_name doubao --vlm gpt
```

Debug split generation for one case:

```bash
python scripts/debug_split_tgt.py --config config/split_config.yaml --vlm gpt --uid 15_template1
```

## VLM Selection

Choose the VLM provider at startup with `--vlm`.

Difference between `name` and `model`:

- `name` or `provider` means which VLM backend implementation we use, such as `gpt`, `gemini`, `claude`, `doubao`, or `qwen`.
- `model` means the concrete model identifier sent to that backend, such as `gpt-4o` or `gemini-3-pro-preview`.

Now the project uses only `provider` externally. Each VLM file defines its own internal default model, so the command line decides only which VLM to use.

This applies independently to both pipelines:

- Split pipeline: choose the provider with `scripts/run_split_generate.py --vlm ...`
- Eval pipeline: choose the provider with `scripts/run_eval_score.py --vlm ...`
- Debug split inspection: choose the provider with `scripts/debug_split_tgt.py --vlm ...`

The generated-image model being evaluated is a different concept:

- `--model_name` in `run_eval_score.py` means the image generation model folder under `generated_root`, such as `bagel`
- `--vlm` means the judge model family used by the split or evaluation pipeline

The config now looks like:

```yaml
vlm:
  provider: gpt
```

Supported `--vlm` values:

- `gpt`
- `gemini`
- `doubao`
- `claude`
- `qwen`

Each provider reads its own API base URL from its Python implementation and loads credentials from environment variables:

- Unified key for remote providers: `UFO_VLM_API_KEY`
- Backward-compatible fallbacks:
  - `gpt`: `OPENAI_API_KEY`
  - `gemini`: `GEMINI_API_KEY`
  - `doubao`: `DOUBAO_API_KEY`
  - `claude`: `ANTHROPIC_API_KEY`
- `qwen`: no API key required for local model loading
- `qwen`: prefers a local model path from `UFO_QWEN_MODEL_PATH` or `QWEN_MODEL_PATH`; if neither is set, it falls back to the built-in Hugging Face model id

Current default API endpoints in provider files:

- `gpt`: `https://api.zhizengzeng.com/v1`
- `gemini`: `https://api.zhizengzeng.com/v1`
- `doubao`: `https://api.zhizengzeng.com/v1`
- `claude`: `https://api.zhizengzeng.com/anthropic`

Default models are currently defined in:

- `gpt`: [src/vlm_tools/gpt.py](e:/1_image_generate/tasks/UFO/src/vlm_tools/gpt.py)
- `gemini`: [src/vlm_tools/gemini.py](e:/1_image_generate/tasks/UFO/src/vlm_tools/gemini.py)
- `doubao`: [src/vlm_tools/doubao.py](e:/1_image_generate/tasks/UFO/src/vlm_tools/doubao.py)
- `claude`: [src/vlm_tools/claude.py](e:/1_image_generate/tasks/UFO/src/vlm_tools/claude.py)
- `qwen`: [src/vlm_tools/qwen.py](e:/1_image_generate/tasks/UFO/src/vlm_tools/qwen.py)

Examples:

```bash
export UFO_VLM_API_KEY=...
python scripts/run_split_generate.py --config config/split_config.yaml --vlm gpt

export UFO_VLM_API_KEY=...
python scripts/run_eval_score.py --config config/eval_config.yaml --model_name bagel --vlm gpt

export UFO_VLM_API_KEY=...
python scripts/debug_split_tgt.py --config config/split_config.yaml --vlm claude --uid 15_template1
```

PowerShell examples on Windows:

```powershell
$env:UFO_VLM_API_KEY="your-api-key"
python scripts/run_split_generate.py --config config/split_config.yaml --vlm gpt

$env:UFO_VLM_API_KEY="your-api-key"
python scripts/run_eval_score.py --config config/eval_config.yaml --model_name bagel --vlm gemini
```

Qwen local model examples:

```bash
export UFO_QWEN_MODEL_PATH=/data/zhangdanning/models/Qwen2.5-VL-8B-Instruct
python scripts/test_vlm.py --name qwen --mode text --text "Reply with exactly: OK"
```

```powershell
$env:UFO_QWEN_MODEL_PATH="E:\\models\\Qwen2.5-VL-8B-Instruct"
python scripts/test_vlm.py --name qwen --mode text --text "Reply with exactly: OK"
```

If Qwen hits CUDA memory fragmentation or out-of-memory, try setting:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UFO_QWEN_MODEL_PATH=/data/zhangdanning/models/Qwen2.5-VL-8B-Instruct
python scripts/test_vlm.py --name qwen --mode multimodal --image /path/to/test.png --text "What is in this image?"
```

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:UFO_QWEN_MODEL_PATH="E:\\models\\Qwen2.5-VL-8B-Instruct"
python scripts/test_vlm.py --name qwen --mode multimodal --image E:\\path\\to\\test.png --text "What is in this image?"
```

Use a different GPU by changing the device id, for example:

```bash
export CUDA_VISIBLE_DEVICES=1
python scripts/test_vlm.py --name qwen --mode text --text "Reply with exactly: OK"
```

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python scripts/test_vlm.py --name qwen --mode text --text "Reply with exactly: OK"
```

Common provider examples:

```powershell
$env:UFO_VLM_API_KEY="your-api-key"
python scripts/run_split_generate.py --config config/split_config.yaml --vlm gpt

$env:UFO_VLM_API_KEY="your-api-key"
python scripts/run_eval_score.py --config config/eval_config.yaml --model_name bagel --vlm claude

$env:UFO_VLM_API_KEY="your-api-key"
python scripts/run_eval_score.py --config config/eval_config.yaml --model_name bagel --vlm doubao

python scripts/run_eval_score.py --config config/eval_config.yaml --model_name bagel --vlm qwen

$env:UFO_VLM_API_KEY="your-api-key"
python scripts/debug_split_tgt.py --config config/split_config.yaml --vlm gemini --uid 15_template1
```

Notes:

- `--vlm` decides which provider implementation is used at runtime.
- The concrete model name is fixed inside each provider implementation.
- Split and eval can use different providers; they are selected independently per command.
- `--model_name` in eval is the target generated-image model to be scored, not the judging VLM.
- `qwen` is treated as a local model backend, so it does not read an API key.
- Remote providers first read `UFO_VLM_API_KEY`; older provider-specific env vars are still accepted as fallback.
