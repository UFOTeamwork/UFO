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

Run:

```bash
python scripts/run_split_generate.py \
    --config config/split_config.yaml
```

The generated outputs include:

- target descriptions
- structured question lists
- atomic semantic units

---

# Evaluation Pipeline

The evaluation pipeline evaluates generated images using multimodal reasoning.

Run:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config.yaml
```

Evaluation includes:

- text consistency
- image consistency
- joint multimodal consistency
- weighted aggregation

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

---

# Citation

If you find this project useful, please consider citing:

```bibtex
@article{ufo2026,
  title={UFO: Unified Fine-grained Omni-conditional Evaluation for Image Generation},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}
```

---

# License

This project is released for academic research purposes.