# UFO: Unified Fine-grained Omni-conditional Evaluation for Image Generation

**UFO** is a fine-grained multimodal evaluation framework for **text-and-reference-image conditioned image generation and editing**.

Modern image generation systems increasingly support complex multimodal conditions, where a model must simultaneously understand:

* textual instructions,
* one or more reference images,
* subject identity and attributes,
* and the relationships between textual and visual conditions.

However, existing evaluation metrics often rely on global image-text or image-image similarity, making it difficult to determine **which specific requirements are satisfied or violated**.

UFO addresses this problem by decomposing complex multimodal generation requirements into **fine-grained atomic evaluation units**, followed by structured VLM-based consistency evaluation.

---

# ✨ Features

* **Fine-grained atomic evaluation**
  Decomposes complex generation requirements into interpretable semantic units.

* **Joint text-image consistency analysis**
  Evaluates the interaction between textual instructions and reference-image information.

* **Reference-image-aware evaluation**
  Explicitly considers subject identity, parts, attributes, and visual characteristics from reference images.

* **Structured question decomposition**
  Converts complex multimodal requirements into atomic evaluation questions.

* **VLM-based evaluation**
  Supports multiple vision-language model backends for multimodal reasoning.

* **Weighted score aggregation**
  Aggregates atomic evaluation results into an overall evaluation score.

* **Human-aligned evaluation**
  Provides fine-grained evaluation signals designed to better reflect human judgments.

* **Extensible evaluation pipeline**
  Supports modular integration of additional VLM backends and evaluation modules.

---

# 🏗️ Overview

The UFO framework consists of two major stages: **atomic task decomposition** and **fine-grained multimodal evaluation**.

```text
             Text Prompt
                  +
          Reference Image(s)
                  |
                  v
      ┌─────────────────────────┐
      │   Atomic Task           │
      │   Decomposition         │
      │   / Split Pipeline      │
      └─────────────────────────┘
                  |
                  v
      Fine-grained Evaluation
             Questions
                  |
                  v
      ┌─────────────────────────┐
      │     Generated Image     │
      │           +             │
      │  Reference Image(s)     │
      │           +             │
      │    Text Instructions    │
      └─────────────────────────┘
                  |
                  v
      ┌─────────────────────────┐
      │    VLM-based Evaluator  │
      └─────────────────────────┘
                  |
                  v
      ┌─────────────────────────┐
      │   Atomic Evaluation     │
      │        Scores           │
      └─────────────────────────┘
                  |
                  v
      ┌─────────────────────────┐
      │ Weighted Score          │
      │ Aggregation             │
      └─────────────────────────┘
                  |
                  v
            Final UFO Score
```

The key idea is to transform a complex multimodal generation task into a collection of **atomic and interpretable evaluation units**.

---

# 🔍 Evaluation Design

UFO evaluates multimodal image generation from three complementary perspectives:

| Dimension             | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| **Text Consistency**  | Whether the generated image follows the textual instructions                   |
| **Image Consistency** | Whether the generated image preserves relevant reference-image characteristics |
| **Joint Consistency** | Whether textual and visual conditions are simultaneously satisfied             |

Instead of treating an image as a single undifferentiated evaluation target, UFO performs **fine-grained semantic evaluation**.

A complex subject can be represented hierarchically as:

```text
Subject
   │
   ├── Part
   │      │
   │      └── Attribute
   │             │
   │             └── Value
   │
   ├── Part
   │      │
   │      └── Attribute
   │             │
   │             └── Value
   │
   └── ...
```

For example:

```text
Subject
 ├── Head
 │    ├── Hair Color → Black
 │    └── Hair Style → Short
 │
 └── Body
      ├── Clothing Color → Red
      └── Clothing Type → Jacket
```

This representation enables UFO to evaluate individual semantic requirements rather than relying solely on global similarity.

---

# 📊 UFO-Bench

UFO is accompanied by **UFO-Bench**, a benchmark designed for evaluating subject-driven and personalized image generation under complex text-and-reference-image conditions.

## Dataset

The complete UFO-Bench dataset is publicly available on Hugging Face:

**[UFO-Bench Dataset — UFOTeamwork/UFO-Bench](https://huggingface.co/datasets/UFOTeamwork/UFO-Bench/tree/main)**

UFO-Bench provides the benchmark data required by the UFO evaluation pipeline, including reference images, benchmark metadata, and structured evaluation annotations.

## Benchmark Statistics

| Property                      | UFO-Bench |
| ----------------------------- | --------: |
| **Total Cases**               |   **660** |
| **Subject Categories**        |     **7** |
| **Editing Paradigms**         |     **4** |
| **Unique Reference Subjects** |    **86** |

## Subject Categories

UFO-Bench covers seven representative subject categories:

* **Rigid Object**
* **Soft Object**
* **Human**
* **Full-body Character**
* **Animal**
* **Logo**
* **Scene**

## Editing Paradigms

The benchmark contains four types of generation and editing scenarios:

1. **Non-Editing**
2. **Local Editing**
3. **Global Editing**
4. **Complex Editing**

These settings progressively evaluate the ability of image generation systems to preserve relevant subject characteristics while following increasingly complex instructions.

---

# 🔀 Text-Image Condition Interaction

A key characteristic of UFO-Bench is its emphasis on **text-image condition interaction**.

Some benchmark cases contain conflicting information between the reference image and the textual instruction.

For example:

```text
Reference Image
      │
      └── Original attribute: Red

Text Instruction
      │
      └── Requested attribute: Blue

            ↓

     Model must reason about
     the requested modification
```

This setting evaluates whether a model can distinguish between:

* information that should be preserved from the reference image;
* information that should be modified according to the text;
* and information that should remain unchanged.

Such cases are particularly important for evaluating modern image editing and subject-driven generation systems.

---

# 📁 UFO-Bench Data

The benchmark contains the data required for both task decomposition and evaluation.

Typical components include:

```text
UFO-Bench/
├── reference/
│   └── Reference images
│
├── T_gt/
│   └── Ground-truth structured annotations
│
├── Question_list/
│   └── Fine-grained evaluation questions
│
├── generated/
│   └── Generated images from evaluated models
│
└── metadata*.jsonl
    └── Benchmark metadata
```

### `reference/`

Reference images used as visual conditions for subject-driven generation and editing.

### `T_gt/`

Ground-truth structured semantic annotations describing relevant subject properties and editing requirements.

### `Question_list/`

Fine-grained evaluation questions used by the evaluation pipeline.

### `generated/`

Generated images produced by different image generation and editing systems.

### `metadata*.jsonl`

Metadata describing individual benchmark cases and their corresponding conditions.

> The exact file organization may vary depending on the dataset version. Please refer to the Hugging Face repository for the latest released dataset structure.

---

# 🤖 Supported VLM Backends

UFO currently supports the following VLM backends:

* **GPT**
* **Qwen**
* **Gemini**
* **Claude**
* **Doubao**

Backend implementations are located in:

```text
src/vlm_tools/
```

The VLM interface is designed to be modular, allowing additional vision-language models to be integrated into the evaluation pipeline.

---

# 📂 Project Structure

```text
UFO/
│
├── config/
│   ├── eval_config_660.yaml
│   ├── eval_config_example.yaml
│   └── split_config_example.yaml
│
├── scripts/
│   ├── run_eval_score.py
│   ├── run_split_generate.py
│   └── ...
│
├── src/
│   ├── evaluator_pipeline/
│   │   └── Evaluation pipeline
│   │
│   ├── split_pipeline/
│   │   └── Atomic task decomposition
│   │
│   ├── prompts/
│   │   └── VLM prompt templates
│   │
│   ├── utils/
│   │   └── Utility functions
│   │
│   └── vlm_tools/
│       └── VLM API wrappers
│
├── requirements.txt
├── run_all_models_eval.sh
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/UFOTeamwork/UFO.git
cd UFO
```

## 2. Create Environment

```bash
conda create -n ufo python=3.10
conda activate ufo
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Example configuration files are provided under:

```text
config/
```

The repository provides:

```text
config/
├── eval_config_660.yaml
├── eval_config_example.yaml
└── split_config_example.yaml
```

## Evaluation Configuration

`config/eval_config_660.yaml` is the standard configuration for the **660-case UFO-Bench evaluation**.

Before running the evaluation, please check the paths and parameters in:

```bash
config/eval_config_660.yaml
```

In particular, configure the paths corresponding to:

* UFO-Bench metadata;
* reference images;
* ground-truth annotations;
* generated images;
* evaluation output directory.

For a customized evaluation configuration, you can start from:

```bash
cp config/eval_config_example.yaml config/eval_config.yaml
```

Then modify the required paths and evaluation settings.

> **Important:** Do not commit API keys, passwords, or other credentials to the repository.

---

# 🔑 API Keys

Except for **Qwen**, which can run with a local model, the supported VLM backends call remote APIs and require an API key.

Each backend first checks the universal environment variable:

```bash
UFO_VLM_API_KEY
```

If it is not set, the backend falls back to its provider-specific environment variable.

| `--vlm`  | Universal Variable | Provider-specific Variable |
| -------- | ------------------ | -------------------------- |
| `gpt`    | `UFO_VLM_API_KEY`  | `OPENAI_API_KEY`           |
| `gemini` | `UFO_VLM_API_KEY`  | `GEMINI_API_KEY`           |
| `claude` | `UFO_VLM_API_KEY`  | `ANTHROPIC_API_KEY`        |
| `doubao` | `UFO_VLM_API_KEY`  | `DOUBAO_API_KEY`           |
| `qwen`   | —                  | —                          |

For example:

```bash
export UFO_VLM_API_KEY="your-api-key"
```

Alternatively, for GPT:

```bash
export OPENAI_API_KEY="your-openai-key"
```

If no matching API key is found, the evaluation fails fast with an error similar to:

```text
Missing required environment variable for gpt.
Tried: UFO_VLM_API_KEY, OPENAI_API_KEY
```

For Qwen, a local checkpoint can be specified through:

```bash
export UFO_QWEN_MODEL_PATH="/path/to/qwen/checkpoint"
```

---

# 🧩 Split Pipeline

The split pipeline decomposes complex multimodal generation requirements into atomic evaluation units.

Run:

```bash
export UFO_VLM_API_KEY="your-api-key"

python scripts/run_split_generate.py \
    --config config/split_config.yaml \
    --vlm gpt
```

The generated outputs include:

* structured target descriptions;
* semantic components;
* atomic evaluation questions;
* fine-grained evaluation requirements.

---

# 📈 Evaluation Pipeline

After generated images are available, the evaluation pipeline evaluates their consistency with the corresponding text instructions and reference images.

The evaluation process is:

```text
Generated Image
      +
Reference Image
      +
Text Instruction
      |
      v
Fine-grained VLM Evaluation
      |
      v
Atomic Evaluation Scores
      |
      v
Weighted Aggregation
      |
      v
Final UFO Score
```

---

# 🧪 Evaluate UFO-Bench (660 Cases)

The standard UFO-Bench benchmark contains **660 cases**.

The recommended configuration is:

```text
config/eval_config_660.yaml
```

## Evaluate a Single Model

First export the required API key:

```bash
export UFO_VLM_API_KEY="your-api-key"
```

Then run:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --vlm gpt \
    --model_name bagel
```

Here:

* `--config` specifies the evaluation configuration.
* `--vlm` specifies the VLM judge.
* `--model_name` specifies the generated-image model to evaluate.

The `--model_name` value should correspond to the generated-image model directory configured under the generated-image root.

For example:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --vlm gpt \
    --model_name uno
```

---

# 🤖 VLM Model Selection

The supported VLM judges are:

```text
gpt
gemini
claude
doubao
qwen
```

The `--vlm` argument is optional.

If omitted, the evaluation pipeline uses:

```yaml
vlm:
  provider: ...
```

from the configuration file.

For example:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --model_name bagel
```

will use the VLM provider specified by `eval_config_660.yaml`.

---

# 🔧 Override the VLM Model

The `--vlm_model` argument can be used to override the default judge model.

For example:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --vlm gemini \
    --vlm_model gemini-2.5-pro \
    --model_name bagel
```

Alternatively, specify the model in the configuration:

```yaml
vlm:
  provider: gemini
  model: gemini-2.5-pro
```

If a provider has deprecated or removed the configured model, specify a valid model using `--vlm_model` or `vlm.model`.

Permanent client errors such as deprecated or unavailable models fail fast instead of consuming the full retry budget.

---

# 🚀 Evaluate Multiple Models

UFO provides:

```text
run_all_models_eval.sh
```

for evaluating multiple generated-image models in one run.

First make the script executable:

```bash
chmod +x run_all_models_eval.sh
```

Then run the complete 660-case evaluation:

```bash
export UFO_VLM_API_KEY="your-api-key"

CONFIG=config/eval_config_660.yaml \
./run_all_models_eval.sh
```

To specify a particular VLM and model list:

```bash
VLM=gpt \
CONFIG=config/eval_config_660.yaml \
MODELS="bagel uno omnigen2" \
./run_all_models_eval.sh
```

The script evaluates each model separately and writes separate logs for the corresponding model.

---

# 💤 Run Evaluation in the Background

For long-running evaluations, use:

```bash
export UFO_VLM_API_KEY="your-api-key"

CONFIG=config/eval_config_660.yaml \
nohup ./run_all_models_eval.sh > run_all_models.log 2>&1 &
```

Monitor the evaluation log with:

```bash
tail -f run_all_models.log
```

You can also override the VLM:

```bash
VLM=gemini \
CONFIG=config/eval_config_660.yaml \
nohup ./run_all_models_eval.sh > run_all_models.log 2>&1 &
```

---

# 📊 Evaluation Outputs

The evaluation pipeline produces structured results based on the fine-grained semantic evaluation units.

The evaluation includes:

* **Text Consistency**
* **Image Consistency**
* **Joint Multimodal Consistency**
* **Weighted Aggregation**

The final UFO score is obtained by aggregating fine-grained evaluation results according to the configured weighting scheme.

---

# 📐 Complementary Metrics

UFO can be combined with conventional image-generation metrics and VLM-based evaluation.

| Metric               | Evaluation Target                       |
| -------------------- | --------------------------------------- |
| **CLIP**             | Text-image semantic alignment           |
| **DINO**             | Reference-generated image similarity    |
| **VLM**              | Fine-grained multimodal consistency     |
| **Human Evaluation** | Human-perceived correctness and quality |

Global similarity metrics and fine-grained semantic evaluation provide complementary signals for evaluating multimodal image generation systems.

---

# 🎯 Supported Tasks

UFO is designed for evaluating:

* Subject-driven image generation
* Personalized image generation
* Reference-image-conditioned generation
* Text-and-image conditioned generation
* Image editing
* Local attribute editing
* Global semantic editing
* Complex multimodal editing
* Multimodal instruction following

---

# 📚 Resources

| Resource                  | Link                                                            |
| ------------------------- | --------------------------------------------------------------- |
| **UFO GitHub Repository** | https://github.com/UFOTeamwork/UFO                              |
| **UFO-Bench Dataset**     | https://huggingface.co/datasets/UFOTeamwork/UFO-Bench/tree/main |

The complete UFO-Bench dataset can be downloaded from:

**[UFOTeamwork/UFO-Bench](https://huggingface.co/datasets/UFOTeamwork/UFO-Bench/tree/main)**

---

# 📝 Citation

If you find UFO or UFO-Bench useful for your research, please cite the corresponding paper:

```bibtex
@article{ufo2026,
  title={UFO: Unified Fine-grained Omni-conditional Evaluation for Image Generation},
  author={UFO Teamwork},
  year={2026}
}
```

For the benchmark:

```bibtex
@misc{ufo_bench2026,
  title={UFO-Bench},
  author={UFO Teamwork},
  year={2026}
}
```

> Please replace the BibTeX entries above with the official citation information of the corresponding publication once finalized.

---

# 📄 License

This project is released for **academic research purposes**.

Please also refer to the licenses and usage restrictions of the corresponding foundation models, VLMs, datasets, reference images, and generated content before using this project for commercial purposes.

---

# ⭐ Acknowledgements

We thank the open-source community for providing the foundation models, vision-language models, datasets, and evaluation tools that make this project possible.
