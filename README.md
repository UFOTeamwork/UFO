# UFO: Unified Fine-grained Omni-conditional Evaluation for Image Generation

UFO is a fine-grained multimodal evaluation framework for text-and-reference-image conditioned image generation tasks.

The framework decomposes complex generation requirements into atomic evaluation units and performs structured consistency evaluation across:

- text instructions,
- reference images,
- and generated images.

UFO is designed for evaluating modern multimodal image generation and editing systems under complex conditional settings.

---

# Features

- Fine-grained atomic evaluation
- Joint text-image consistency analysis
- Structured question decomposition
- Weighted score aggregation
- Support for multiple VLM backends
- Human-aligned multimodal evaluation
- Reference-image-aware evaluation
- Extensible evaluation pipeline

---

# Project Structure

```text
UFO/
|-- config/                     # Example configuration files
|-- scripts/                    # Entry scripts
|-- src/
|   |-- evaluator_pipeline/     # Evaluation pipeline
|   |-- split_pipeline/         # Atomic decomposition pipeline
|   |-- prompts/                # Prompt templates
|   |-- utils/                  # Utility functions
|   `-- vlm_tools/              # VLM API wrappers
|-- requirements.txt
`-- README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/UFOTeamwork/UFO.git
cd UFO
```

## Create Environment

```bash
conda create -n ufo python=3.10
conda activate ufo
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Supported VLM Backends

UFO currently supports:

- GPT
- Qwen
- Gemini
- Claude
- Doubao

Backend implementations are located in:

```text
src/vlm_tools/
```

---

# API Keys

Except for `qwen` (which runs locally), every VLM backend calls a remote API and
**requires an API key exported as an environment variable before running**.

Each backend first looks for the universal `UFO_VLM_API_KEY`; if it is not set,
it falls back to the provider-specific variable below.

| `--vlm` | Universal variable | Provider-specific variable |
|---|---|---|
| `gpt`    | `UFO_VLM_API_KEY` | `OPENAI_API_KEY` |
| `gemini` | `UFO_VLM_API_KEY` | `GEMINI_API_KEY` |
| `claude` | `UFO_VLM_API_KEY` | `ANTHROPIC_API_KEY` |
| `doubao` | `UFO_VLM_API_KEY` | `DOUBAO_API_KEY` |
| `qwen`   | — (local model, no key needed) | — |

Export the key in your shell, for example:

```bash
# Option A: one universal key for whichever backend you use
export UFO_VLM_API_KEY="your-api-key"

# Option B: provider-specific key (example for GPT)
export OPENAI_API_KEY="your-openai-key"
```

If no matching key is found, the run fails fast with an error such as:

```text
Missing required environment variable for gpt. Tried: UFO_VLM_API_KEY, OPENAI_API_KEY
```

> `qwen` loads a local model (default `Qwen/Qwen2.5-VL-8B-Instruct`). Point it at
> a local checkpoint with `UFO_QWEN_MODEL_PATH` if needed, and install the extra
> dependencies it requires (e.g. `torch`, `transformers`).

---

# Configuration

Example configuration files are provided in:

```text
config/
```

Create your own runtime configuration files:

```bash
cp config/eval_config.example.yaml config/eval_config.yaml
cp config/split_config.example.yaml config/split_config.yaml
```

Then modify:

- API keys
- model names
- dataset paths
- output paths

---

# Split Pipeline

The split pipeline decomposes multimodal generation tasks into atomic evaluation units.

Run (remember to export the API key first, see [API Keys](#api-keys)):

```bash
export UFO_VLM_API_KEY="your-api-key"

python scripts/run_split_generate.py \
    --config config/split_config.yaml \
    --vlm gpt
```

The generated outputs include:

- target descriptions
- structured question lists
- atomic semantic units

---

# Evaluation Pipeline

The evaluation pipeline evaluates generated images using multimodal reasoning.

Run (remember to export the API key first, see [API Keys](#api-keys)):

```bash
export UFO_VLM_API_KEY="your-api-key"

python scripts/run_eval_score.py \
    --config config/eval_config.yaml \
    --vlm gpt \
    --model_name bagel
```

- `--vlm` selects the VLM judge (`gpt` / `gemini` / `claude` / `doubao` / `qwen`).
- `--model_name` selects the generated-image model to evaluate; it must match the
  directory name under `<generated_root>/<model_name>/...`.

Evaluation includes:

- text consistency
- image consistency
- joint multimodal consistency
- weighted aggregation

## Evaluate All Models at Once

To evaluate several generated-image models in one batch, use the helper script,
which loops the evaluation over each model and writes a separate log per model:

```bash
export UFO_VLM_API_KEY="your-api-key"

chmod +x run_all_models_eval.sh
./run_all_models_eval.sh

# or run in the background
nohup ./run_all_models_eval.sh > run_all_models.log 2>&1 &
```

Defaults (model list, VLM judge, config, log dir) can be overridden via
environment variables:

```bash
VLM=gemini CONFIG=config/eval_config.yaml MODELS="bagel uno" ./run_all_models_eval.sh
```

---

# Evaluation Design

UFO evaluates generation quality from three aspects:

| Aspect | Description |
|---|---|
| Text Consistency | Whether the generated image follows text instructions |
| Image Consistency | Whether the generated image preserves reference-image characteristics |
| Joint Consistency | Whether both modalities are simultaneously satisfied |

The framework decomposes complex conditions into fine-grained semantic units for improved interpretability and robustness.

---

# Example Workflow

```text
Reference Image + Text Prompt
            |
            v
Atomic Task Decomposition
            |
            v
Question Generation
            |
            v
VLM-based Evaluation
            |
            v
Weighted Aggregation
            |
            v
Final UFO Score
```

# License

This project is released for academic research purposes.