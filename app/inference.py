"""Sana inference, independent of the web UI and safe to import without loading weights."""

import json
import math
import os
import random
import threading
from pathlib import Path

import torch
from diffusers import SanaPipeline

MODEL_ID = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"


def get_device():
    requested = os.environ.get("SANA_DEVICE")
    config = Path(__file__).resolve().parent.parent / "env" / "sana-device.json"
    if requested is None and config.exists():
        requested = json.loads(config.read_text(encoding="utf-8"))["device"]
    if requested == "cpu":
        return "cpu"
    if requested not in (None, "gpu", "auto"):
        raise ValueError("SANA_DEVICE must be cpu, gpu, or auto")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
# Sana's text encoder and VAE require BF16 or FP32. Use FP32 on MPS
# and older CUDA cards; casting FP16 weights back up cannot recover overflow.
DTYPE = (torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
         else torch.float32)
pipe = None
_pipeline_lock = threading.RLock()


def load_model():
    global pipe
    with _pipeline_lock:
        if pipe is None:
            print(f"Loading Sana model on {DEVICE} ({DTYPE})...")
            candidate = SanaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
            candidate.to(DEVICE)
            # Publish only after setup succeeds, so a failed transfer can be retried.
            pipe = candidate
        return pipe


def _integer(value, name, minimum, maximum, multiple=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum or value % multiple:
        raise ValueError(f"{name} must be {minimum}–{maximum}, in increments of {multiple}")
    return value


def generate_image(prompt, steps=20, guidance=4.5, seed=42, use_random_seed=True,
                   width=1024, height=1024):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty prompt")
    steps = _integer(steps, "Steps", 5, 30 if DEVICE == "cpu" else 50)
    max_size = 1024 if DEVICE == "cpu" else 2048
    width = _integer(width, "Width", 256, max_size, 64)
    height = _integer(height, "Height", 256, max_size, 64)
    if (isinstance(guidance, bool) or not isinstance(guidance, (int, float))
            or not math.isfinite(guidance) or not 1 <= guidance <= 10):
        raise ValueError("Guidance must be between 1 and 10")
    if not isinstance(use_random_seed, bool):
        raise ValueError("Random seed mode must be true or false")
    actual_seed = (random.randint(0, 2**32 - 1) if use_random_seed
                   else _integer(seed, "Seed", 0, 2**32 - 1))
    # Diffusers transfers CPU noise to the inference device; MPS generators
    # are unsupported. CPU generators also keep seed handling consistent.
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
    # Schedulers mutate during inference: serialize UI and API callers alike.
    with _pipeline_lock:
        result = load_model()(
            prompt=prompt, height=height, width=width, guidance_scale=guidance,
            num_inference_steps=steps, generator=generator,
        )
    return result.images[0], actual_seed
