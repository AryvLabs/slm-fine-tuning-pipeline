# SLM Fine-Tuning Pipeline

A generic, model-agnostic supervised fine-tuning pipeline for small language models. Supports CUDA (NVIDIA), Apple Silicon (MPS), and CPU out of the box.

## Project Structure

```
.
├── slm.py              # entry point
├── config.yaml         # all hyperparameters and paths
├── requirements.txt
├── data/
│   └── dataset.json    # training data
└── src/
    ├── setup.py        # environment setup, GPU detection, HF login
    ├── data.py         # dataset loading and formatting
    ├── model.py        # model and tokenizer loading
    ├── train.py        # LoRA config, training loop, artifact saving
    └── evaluate.py     # post-training inference
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

If the model you want to fine-tune is gated (e.g. Gemma, Llama), accept its license on the Hugging Face model page before running.

## Usage

```bash
python slm.py
```

To use a different config file:

```bash
python slm.py --config my_config.yaml
```

## Configuration

All settings live in `config.yaml`.

**Swap the model** — set any public or gated Hugging Face model ID:

```yaml
model:
  model_id: "meta-llama/Llama-3.2-3B-Instruct"
  max_length: 512
```

**Bring your own data** — point to any JSON file containing a list of objects:

```yaml
dataset:
  path: "data/dataset.json"
  instruction_key: "instruction"
  response_key: "response"
```

The JSON file should look like:

```json
[
  { "instruction": "What is X?", "response": "X is ..." },
  ...
]
```

**Key training parameters:**

| Parameter | Description |
|---|---|
| `model.model_id` | Any HuggingFace model ID |
| `lora.r` | LoRA rank — higher = more capacity, more memory |
| `lora.alpha` | LoRA scaling factor (typically 2x rank) |
| `training.max_steps` | Cap steps for quick runs; set to `-1` to use `num_train_epochs` |
| `training.per_device_train_batch_size` | Lower this first if you hit OOM |
| `quantization.load_in_4bit` | 4-bit quantization (CUDA only; ignored on MPS/CPU) |

## Device Support

| Device | Quantization | Precision |
|---|---|---|
| NVIDIA GPU (CUDA) | 4-bit via bitsandbytes | bfloat16 |
| Apple Silicon (MPS) | Full precision | float16 |
| CPU | Full precision | bfloat16 |

## Output

After training, two artifacts are saved:

- `output/adapter/` — the raw LoRA adapter weights (small, reloadable with PEFT)
- `output/merged/` — the adapter merged into the base model (ready for inference)

Both paths are configurable under `output` in `config.yaml`.

To push the merged model to the Hugging Face Hub:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="./output/merged", repo_id="your-username/your-model-name")
```
---

## For Enterprise

If you're looking to build a custom SLM or other AI feature for your enterprise, check us out [here](https://aryvlabs.com). Our team can help you build your next production-ready AI workflow.
