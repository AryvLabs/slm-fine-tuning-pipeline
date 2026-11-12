import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

from src.setup import detect_device


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type=lora["task_type"],
    )


def build_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    device = detect_device()

    # bf16 is only supported on CUDA; use fp16 on MPS, fp32 on CPU
    use_bf16 = t["bf16"] and device == "cuda"
    use_fp16 = not use_bf16 and device == "mps"

    return SFTConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_steps=t.get("max_steps", -1),
        num_train_epochs=t.get("num_train_epochs", 3),
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        report_to=t["report_to"],
        packing=t["packing"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def run_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: dict,
) -> SFTTrainer:
    from pathlib import Path
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["adapter_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["merged_dir"]).mkdir(parents=True, exist_ok=True)

    lora_config = build_lora_config(cfg)
    training_args = build_training_args(cfg)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print(f"Trainer ready. Steps: {training_args.max_steps} | LR: {training_args.learning_rate} | LoRA r={cfg['lora']['r']}")

    try:
        result = trainer.train()
        print(f"Training complete. Loss: {result.training_loss:.4f}")
        return trainer

    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" in str(exc).lower():
            print("OOM during training. Lower per_device_train_batch_size in config.yaml.")
        else:
            print(f"Training failed: {exc}")
        raise


def save_artifacts(trainer: SFTTrainer, tokenizer: AutoTokenizer, cfg: dict):
    adapter_dir = cfg["output"]["adapter_dir"]
    merged_dir = cfg["output"]["merged_dir"]

    trainer.model.save_pretrained(adapter_dir)
    print(f"Adapter saved to '{adapter_dir}'")

    merged_model = trainer.model.merge_and_unload()
    merged_model.eval()

    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to '{merged_dir}'")

    return merged_model
