# UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation (ICML 2026)

> Danning Zhang, Yijing Lin, Shuhan Zhuang, Mengqi Huang, Shaojin Wu, Shancheng Fang, Zhendong Mao

💻 [GitHub](https://github.com/UFOTeamwork/UFO) | 🤗 [Hugging Face](https://huggingface.co/datasets/UFOTeamwork/UFO-Bench) | 📰 [Paper]()

**Overall framework of UFO.** UFO performs fine-grained multi-modal evaluation through a four-stage **Chain-of-Evaluation** pipeline. First, **Atomic Evaluation Units (AEUs)** are generated conditioned on the visual and textual inputs. Each AEU is then assigned a **modality-relevance class**. Subsequently, UFO qualifies the alignment of each AEU using either **Visual Question Answering (VQA)** or **function calls**. Finally, **adaptive importance weights** aggregate AEU-level scores into a holistic **UFO Score**.

UFO provides a fine-grained evaluation framework for text-and-reference-image conditioned image generation.

- We decompose complex multi-modal requirements into interpretable **Atomic Evaluation Units (AEUs)**.
- We explicitly model the modality relevance of each AEU to distinguish **image**, **text**, and **joint multi-modal** requirements.
- We combine **VQA and function calls** to evaluate different types of semantic requirements.
- We apply **adaptive importance weighting** to aggregate AEU-level scores.
- We release **UFO-Bench**, containing **660 evaluation cases**, **86 unique reference subjects**, **7 subject categories**, and **4 editing paradigms**.

------

## 📝 TODO

- [🔜] Release official ICML 2026 paper.
- [✅ 2026] Release UFO-Bench.
- [✅ 2026] Upload UFO-Bench to Hugging Face.
- [✅ 2026] Release evaluation code.
- [✅ 2026] Release the standard `eval_config_660.yaml`.
- [✅ 2026] Release multi-model evaluation scripts.

------

## 📦 Dataset

![bench_00](./img/bench_00-1788975832145-2.png)

**UFO-Bench** is designed for evaluating subject-driven, personalized, and reference-image-conditioned image generation under complex visual and textual conditions.

The benchmark contains **660 evaluation cases** constructed from **86 unique reference subjects**.

### Benchmark Statistics

| Property                  | UFO-Bench |
| ------------------------- | --------- |
| Evaluation Cases          | **660**   |
| Unique Reference Subjects | **86**    |
| Subject Categories        | **7**     |
| Editing Paradigms         | **4**     |

### Subject Categories

UFO-Bench contains seven representative categories:

- Logo
- Animal
- Full-body Character
- Scene
- Human
- Soft Object
- Rigid Object

### Editing Paradigms

Each benchmark case belongs to one of four settings:

- **Non-Editing**
- **Global Editing**
- **Local Editing**
- **Complex Editing**

These settings evaluate whether a generation model can preserve relevant reference-image characteristics while correctly following textual instructions.

A key property of UFO-Bench is **text-image condition interaction**. For example, if the reference image contains a red object while the text explicitly requests a blue object, a correct generated image should preserve unrelated subject characteristics while modifying the requested attribute.

You can download the complete dataset from:

🤗 [**UFO-Bench on Hugging Face**](https://huggingface.co/datasets/UFOTeamwork/UFO-Bench)

### Data Structure

The released UFO-Bench contains the following components:

```text
UFO-Bench/
├── reference/
│   └── Reference images
│
├── T_gt/
│   └── Structured semantic annotations
│
├── Question_list/
│   └── Fine-grained evaluation questions
│
├── generated/
│   ├── bagel/
│   ├── uno/
│   ├── omnigen2/
│   └── ...
│
└── metadata*.jsonl
    └── Prompt annotations for benchmark cases
```

- **`reference/`**: reference images used as visual conditions.
- **`T_gt/`**: structured semantic annotations describing subject characteristics and generation/editing requirements.
- **`Question_list/`**: fine-grained evaluation questions derived from Atomic Evaluation Units.
- **`generated/`**: generated images from evaluated models.
- **`metadata\*.jsonl`**: prompt annotations for benchmark samples across different subject categories and editing settings.

Each entry in `metadata*.jsonl` provides the textual prompt used as the generation or editing condition for the corresponding benchmark case.

------

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/UFOTeamwork/UFO.git
cd UFO

conda create -n ufo python=3.10
conda activate ufo
pip install -r requirements.txt
```

Configure the VLM API key if required:

```bash
export UFO_VLM_API_KEY="your-api-key"
```

------

### 2. Generate AEUs for New Cases

For new text-reference-image cases, first create a split configuration:

```bash
cp config/split_config_example.yaml config/split_config.yaml
```

Then run the split pipeline:

```bash
python scripts/run_split_generate.py \
    --config config/split_config.yaml \
    --vlm gpt
```

The pipeline generates:

```text
T_gt/
Question_list/
```

which are used by the downstream UFO evaluator.

> The released UFO-Bench already provides `T_gt/` and `Question_list/`, so this step is not required when reproducing the official benchmark.

------

### 3. Evaluate Your Model

For a customized evaluation, first create a local evaluation configuration:

```bash
cp config/eval_config_example.yaml config/eval_config.yaml
```

Modify the required paths in `config/eval_config.yaml`, including:

- metadata / prompts,
- reference images,
- `T_gt`,
- `Question_list`,
- generated-image root,
- evaluation output directory.

Then run:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config.yaml \
    --vlm gpt \
    --model_name your_model
```

where:

- `--config` specifies the evaluation configuration;
- `--vlm` specifies the VLM evaluator;
- `--model_name` specifies the generation model to evaluate.

The value of `--model_name` should correspond to the generated-image directory configured under the generated-image root.

------

### 4. Evaluate the Official UFO-Bench (660 Cases)

The released UFO-Bench contains **660 official evaluation cases** with prepared `T_gt/` and `Question_list/` annotations.

Use the provided standard configuration:

```text
config/eval_config_660.yaml
```

Update the generated-image root and output path in the configuration if necessary.

Then evaluate a model:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --vlm gpt \
    --model_name bagel
```

To evaluate your own model:

```bash
python scripts/run_eval_score.py \
    --config config/eval_config_660.yaml \
    --vlm gpt \
    --model_name your_model
```

To evaluate multiple models:

```bash
VLM=gpt \
CONFIG=config/eval_config_660.yaml \
MODELS="bagel uno omnigen2" \
./run_all_models_eval.sh
```

The evaluation produces:

- **AEU-level alignment scores**, and
- an aggregated **UFO Score**.

------

## 🤖 Supported VLM

UFO currently supports multiple VLM evaluators:

- GPT
- Gemini
- Claude
- Doubao
- Qwen

Backend implementations are located in:

```text
src/vlm_tools/
```

A universal API key can be configured using:

```bash
export UFO_VLM_API_KEY="your-api-key"
```

Provider-specific environment variables can also be used when supported.

For example:

```bash
export OPENAI_API_KEY="your-openai-key"
```

For local Qwen evaluation:

```bash
export UFO_QWEN_MODEL_PATH="/path/to/qwen/checkpoint"
```

> Never commit API keys, access tokens, passwords, or other credentials to the repository.

------

## 📂 Project Structure

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
│   ├── split_pipeline/
│   ├── prompts/
│   ├── utils/
│   └── vlm_tools/
│
├── assets/
│   ├── ufo_pipeline.png
│   └── bench_00.png
│
├── requirements.txt
├── run_all_models_eval.sh
└── README.md
```

------

## 📚 Citation

If you use **UFO** or **UFO-Bench** in your research, please cite our paper:

```bibtex
@inproceedings{zhang2026ufo,
  title={UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation},
  author={Zhang, Danning and Lin, Yijing and Zhuang, Shuhan and Huang, Mengqi and Wu, Shaojin and Fang, Shancheng and Mao, Zhendong},
  booktitle={Proceedings of the Forty-Third International Conference on Machine Learning},
  year={2026}
}
```

> The citation will be updated with the complete official ICML 2026 publication metadata.

------

## ❤️ Acknowledgement

We thank the authors and developers of the open-source image generation, personalized generation, vision-language modeling, and evaluation methods used in this project.

We also thank the broader research community for making their models, datasets, and evaluation tools publicly available.

------

## 📄 License

This repository is released for **academic research purposes**.

Please also follow the licenses and usage restrictions of the corresponding foundation models, VLM providers, datasets, reference images, and generated content.
