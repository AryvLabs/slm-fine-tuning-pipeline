import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.setup import detect_device


def _compute_dtype(cfg: dict) -> torch.dtype:
    device = detect_device()
    requested = cfg["quantization"]["compute_dtype"]

    if device == "mps" and requested == "bfloat16":
        return torch.float16

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if requested not in mapping:
        raise ValueError(f"Unsupported compute_dtype '{requested}'. Choose from: {list(mapping)}.")
    return mapping[requested]


def get_model_id(cfg: dict) -> str:
    model_id = cfg["model"].get("model_id")
    if not model_id:
        raise ValueError("model.model_id is not set in config.yaml.")
    print(f"Model: {model_id}")
    return model_id


def load_tokenizer(model_id: str, max_length: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=max_length)
    print(f"Tokenizer loaded (max_length={max_length}).")
    return tokenizer


def load_model(model_id: str, cfg: dict) -> AutoModelForCausalLM:
    device = detect_device()
    compute_dtype = _compute_dtype(cfg)
    q_cfg = cfg["quantization"]

    use_4bit = q_cfg["load_in_4bit"] and device == "cuda"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q_cfg["quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=q_cfg["double_quant"],
        )
        load_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    elif device == "mps":
        load_kwargs = {"device_map": {"": "mps"}}
    else:
        load_kwargs = {"device_map": {"": "cpu"}}

    if not use_4bit:
        print(f"4-bit quantization unavailable on {device}. Loading in {compute_dtype} full precision.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            **load_kwargs,
        )
        print(f"Model loaded: {model_id} [{device}, {compute_dtype}]")
        return model

    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" in str(exc).lower():
            print(f"OOM loading model on {device}. Try a smaller model or reduce max_length in config.yaml.")
        else:
            print(f"Model load failed: {exc}")
        raise


def prepare_for_lora_training(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    device = detect_device()

    for param in model.parameters():
        param.requires_grad = False

    upcast_count = 0
    for _, module in model.named_modules():
        if "norm" in module.__class__.__name__.lower():
            if device != "mps":
                module.to(torch.float32)
            upcast_count += 1

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print(f"{upcast_count} norm layers prepared for training.")

    _flush_memory(device)
    _print_memory(device)

    return model


def _flush_memory(device: str):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _print_memory(device: str, label: str = ""):
    tag = f" ({label})" if label else ""
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU{tag}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    elif device == "mps":
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"MPS{tag}: {allocated:.1f} GB allocated")


def clear_gpu_memory(*var_names, global_ns: dict | None = None):
    device = detect_device()

    if global_ns:
        for name in var_names:
            if name in global_ns:
                del global_ns[name]

    _flush_memory(device)
    _print_memory(device, "cleared")
