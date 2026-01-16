# ai/model/llm_loader.py
"""
Auto-detect CUDA → load llama-cpp with GPU if available, else CPU.

Usage:
    from ai.model.llm_loader import load_llm
    llm = load_llm()
    resp = llm.create_completion(prompt="hello", max_tokens=50)
"""

import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_FILE = Path(__file__).parent / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"


def has_cuda():
    """
    Detect if CUDA is available on the system.
    Returns True only if:
      1. 'nvidia-smi' exists
      2. the command executes successfully
    """
    try:
        if shutil.which("nvidia-smi") is None:
            return False
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def load_llm(model_path: str | Path = None):
    """
    Load the Llama model using llama-cpp-python.
    If CUDA is available, use GPU layers.
    If not, fall back to CPU mode.

    GPU mode uses:
       n_gpu_layers = -1 (all layers)
    CPU mode uses:
       n_gpu_layers = 0
    """

    try:
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError("llama_cpp not installed. Install llama-cpp-python.") from e

    model_path = str(model_path or MODEL_FILE)

    if has_cuda():
        logger.info("CUDA detected → loading model with GPU acceleration…")
        print("\n[INFO] CUDA available → Using GPU acceleration\n")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,          # send all layers to GPU
            gpu_split_mode="auto",    # let llama-cpp decide VRAM split
        )
    else:
        logger.info("CUDA NOT detected → loading model on CPU…")
        print("\n[INFO] CUDA NOT available → Running on CPU\n")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=0            # CPU only
        )

    return llm
