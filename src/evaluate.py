import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.setup import detect_device


def load_inference_model(cfg: dict, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    merged_dir = cfg["output"]["merged_dir"]
    adapter_dir = cfg["output"]["adapter_dir"]
    model_id = cfg["model"]["model_id"]
    device = detect_device()

    is_merged = os.path.exists(os.path.join(merged_dir, "config.json"))
    is_adapter = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

    if not is_merged and not is_adapter:
        raise RuntimeError(f"No saved weights found at '{merged_dir}' or '{adapter_dir}'. Run training first.")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    dtype = torch.float16 if device == "mps" else torch.bfloat16
    device_map = "auto" if device == "cuda" else {"": device}

    if is_merged:
        model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=dtype, device_map=device_map)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base, adapter_dir)

    model.eval()
    print(f"Inference model loaded [{device}].")
    return model


def ask(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 80) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_evaluation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, cfg: dict):
    eval_cfg = cfg["evaluation"]
    prompts = eval_cfg.get("test_prompts", [])
    max_new_tokens = eval_cfg.get("max_new_tokens", 80)

    if not prompts:
        print("No evaluation prompts in config.yaml.")
        return

    print("\n" + "─" * 60)
    for prompt in prompts:
        print(f"\nPrompt  : {prompt}")
        try:
            print(f"Response: {ask(prompt, model, tokenizer, max_new_tokens)}")
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}")
    print("─" * 60)
