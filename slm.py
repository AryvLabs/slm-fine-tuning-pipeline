import argparse
import sys
from pathlib import Path

from src.setup import (
    check_gpu,
    check_torchvision,
    configure_cuda_memory,
    huggingface_login,
    verify_model_access,
)

configure_cuda_memory()

import yaml

from src.data import build_datasets
from src.evaluate import load_inference_model, run_evaluation
from src.model import (
    clear_gpu_memory,
    get_model_id,
    load_model,
    load_tokenizer,
    prepare_for_lora_training,
)
from src.train import run_training, save_artifacts


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{path}'.\n"
            f"Make sure config.yaml exists in the project root or pass --config <path>."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from '{path}'.")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generic SLM fine-tuning pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file (default: config.yaml)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("STEP 1 — Environment setup")
    print("═" * 60)

    if not check_gpu():
        sys.exit(1)

    cfg = load_config(args.config)
    model_id = get_model_id(cfg)

    hf_token = huggingface_login()
    verify_model_access(model_id, hf_token)
    check_torchvision()

    print("\n" + "═" * 60)
    print("STEP 2 — Loading dataset")
    print("═" * 60)

    train_dataset, eval_dataset = build_datasets(cfg)

    print("\n" + "═" * 60)
    print("STEP 3 — Loading model")
    print("═" * 60)

    tokenizer = load_tokenizer(model_id, cfg["model"]["max_length"])
    model = load_model(model_id, cfg)
    model = prepare_for_lora_training(model)

    print("\n" + "═" * 60)
    print("STEP 4 — Fine-tuning")
    print("═" * 60)

    trainer = run_training(model, tokenizer, train_dataset, eval_dataset, cfg)

    print("\n" + "═" * 60)
    print("STEP 5 — Saving artifacts")
    print("═" * 60)

    save_artifacts(trainer, tokenizer, cfg)
    clear_gpu_memory("model", "trainer", global_ns=globals())

    print("\n" + "═" * 60)
    print("STEP 6 — Evaluation")
    print("═" * 60)

    inference_model = load_inference_model(cfg, tokenizer)
    run_evaluation(inference_model, tokenizer, cfg)

    print("\nPipeline complete.")
    print(f"   Adapter  : {cfg['output']['adapter_dir']}")
    print(f"   Merged   : {cfg['output']['merged_dir']}")


if __name__ == "__main__":
    main()
