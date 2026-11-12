import os
import subprocess


def configure_cuda_memory():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_gpu() -> bool:
    device = detect_device()

    if device == "cuda":
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("CUDA GPU detected:")
        for line in result.stdout.splitlines():
            if any(tag in line for tag in ("MiB", "Tesla", "T4", "A100", "L4")):
                print(line)
        print()

    elif device == "mps":
        print("Apple Silicon GPU (MPS) detected.")

    else:
        print("No GPU detected. Training will run on CPU and will be very slow.")

    return device != "cpu"


def check_torchvision():
    import torch
    print(f"torch {torch.__version__} | device: {detect_device()}")

    try:
        import torchvision
        print("torchvision installed — potential ABI mismatch on CUDA. Run: pip uninstall -y torchvision torchaudio")
    except ImportError:
        print("torchvision not installed.")


def huggingface_login(hf_token: str | None = None):
    from huggingface_hub import login

    token = hf_token or os.environ.get("HF_TOKEN")

    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        raise RuntimeError("No HF_TOKEN found. Set the HF_TOKEN environment variable or add it as a Colab secret.")

    login(token=token)
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("Hugging Face login successful.")
    return token


def verify_model_access(model_id: str, token: str):
    from huggingface_hub import HfApi

    try:
        HfApi().model_info(model_id, token=token)
        print(f"Access confirmed for {model_id}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access {model_id}. Accept the license at https://huggingface.co/{model_id} then re-run."
        ) from exc
