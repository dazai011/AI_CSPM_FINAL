"""
LocalLLM - small wrapper around llama-cpp-python's Llama class.

Patched version: sets a sensible default n_ctx and improves context-window error handling
so that the runtime's default 512 window does not cause the "Requested tokens ... exceed
context window of 512" error when avoidable.

Notes:
- DEFAULT_N_CTX is conservative (8192). Change to 32768 only if you have enough memory/VRAM.
- You can override n_ctx by passing extra_llama_kwargs={'n_ctx': <value>} when constructing LocalLLM.
"""

from __future__ import annotations
import os
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_FILENAME = "mistral-7b-instruct-v0.3-q4_k_m.gguf"


def _run_cmd(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def detect_cuda() -> Dict[str, Any]:
    """Attempt multiple ways to detect a CUDA-capable GPU environment."""
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return {"available": False, "reason": "FORCE_CPU set"}

    try:
        import torch

        if torch.cuda.is_available():
            return {"available": True, "reason": "torch.cuda available", "torch_device_count": torch.cuda.device_count()}
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        if count and count > 0:
            return {"available": True, "reason": "pynvml detected GPU", "gpu_count": count}
    except Exception:
        pass

    code, out, err = _run_cmd("nvidia-smi -L")
    if code == 0 and out:
        return {"available": True, "reason": "nvidia-smi found GPUs", "nvidia_smi": out.splitlines()}

    if os.environ.get("GGML_CUDA", "").lower() in ("1", "on", "true") or os.environ.get("USE_CUDA", "").lower() in ("1", "true", "on"):
        return {"available": True, "reason": "env hint GGML_CUDA/USE_CUDA present"}

    return {"available": False, "reason": "no CUDA detected by checks"}


class LocalLLM:
    """
    Thin wrapper for llama-cpp-python Llama model.
    - model_path: path to gguf/ggml model file
    - prefer_cuda: attempt GPU load when detected
    - n_gpu_layers: number of top-most transformer layers to place on GPU (if supported).
    - extra_llama_kwargs: passed directly to Llama(...)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        prefer_cuda: bool = True,
        n_gpu_layers: int = 16,
        extra_llama_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = Path(model_path) if model_path else (Path(__file__).parent / DEFAULT_MODEL_FILENAME)
        self.prefer_cuda = bool(prefer_cuda)
        self.n_gpu_layers = int(n_gpu_layers or 0)
        self.extra_llama_kwargs = extra_llama_kwargs or {}
        self.llm = None
        self._loaded_on = "none"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        detect = detect_cuda()
        self._cuda_detect = detect
        log.info("CUDA detection: %s", detect)

        self._load_model()

    def _load_model(self):
        """Try to create a llama_cpp.Llama instance with GPU if possible, otherwise CPU."""
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError("llama_cpp (llama-cpp-python) not importable. Install it before using LocalLLM.") from e

        # Determine requested context window: prefer extra_llama_kwargs if provided, otherwise default to DEFAULT_N_CTX
        DEFAULT_N_CTX = 8192  # conservative default; change to 32768 only if you have enough memory
        requested_n_ctx = None
        if "n_ctx" in self.extra_llama_kwargs:
            try:
                requested_n_ctx = int(self.extra_llama_kwargs["n_ctx"])
            except Exception:
                requested_n_ctx = None

        if requested_n_ctx is None:
            requested_n_ctx = DEFAULT_N_CTX

        # include n_ctx in base kwargs (user can still override via extra_llama_kwargs)
        base_kwargs = dict(model_path=str(self.model_path), n_ctx=requested_n_ctx, **self.extra_llama_kwargs)

        # If user prefers GPU and detection reported available, attempt GPU load
        if self.prefer_cuda and self._cuda_detect.get("available", False):
            gpu_kwargs = dict(base_kwargs)
            try:
                gpu_kwargs["n_gpu_layers"] = max(0, int(self.n_gpu_layers))
            except Exception:
                gpu_kwargs["n_gpu_layers"] = 16

            log.info("Attempting to load model on GPU with n_gpu_layers=%s and n_ctx=%s ...", gpu_kwargs.get("n_gpu_layers"), gpu_kwargs.get("n_ctx"))
            try:
                self.llm = Llama(**gpu_kwargs)
                self._loaded_on = "cuda"
                log.info("Loaded model on GPU successfully.")
                return
            except Exception as e:
                log.warning("GPU load failed (falling back to CPU). Exception: %s", e)

        # CPU fallback
        log.info("Loading model on CPU with n_ctx=%s...", base_kwargs.get("n_ctx"))
        try:
            self.llm = Llama(**base_kwargs)
            self._loaded_on = "cpu"
            log.info("Loaded model on CPU.")
        except Exception as e:
            log.exception("Failed to load model on CPU: %s", e)
            raise

    @property
    def loaded_on(self) -> str:
        return self._loaded_on

    def create_completion(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0, **kwargs) -> str:
        """
        Create a simple completion using Llama.create_completion. If the underlying Llama API differs,
        this wrapper tries a few known call signatures. Returns the generated text (string).
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        call_kwargs = dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=seed)
        call_kwargs.update(kwargs)

        attempt = 0
        max_attempts = 4
        requested_max = int(call_kwargs.get("max_tokens", 0) or 0)
        while attempt < max_attempts:
            try:
                resp = self.llm.create_completion(**call_kwargs)
                if isinstance(resp, dict):
                    if "choices" in resp and resp["choices"]:
                        # llama-cpp-python sometimes returns choices[0].text
                        text = resp["choices"][0].get("text") or resp["choices"][0].get("message") or ""
                    else:
                        text = resp.get("text", "")
                else:
                    text = str(resp)
                return (text or "").strip()

            except Exception as e:
                msg = str(e)
                # handle context-window/token request errors from llama backend
                if ("Requested tokens" in msg and "context window" in msg) or ("exceed context window" in msg):
                    import re

                    m = re.search(r"context window (?:of )?(\d+)", msg)
                    if m:
                        allowed = int(m.group(1))
                        orig_prompt = str(call_kwargs.get("prompt", prompt) or "")
                        # conservative token estimate: ~4 chars per token
                        est_prompt_tokens = max(1, len(orig_prompt) // 4)
                        margin = 8

                        allowed_output = max(1, allowed - est_prompt_tokens - margin)

                        if allowed_output < requested_max:
                            log.warning(
                                "Requested tokens (%s) exceed runtime context (%s). Retrying with max_tokens=%s",
                                requested_max, allowed, allowed_output,
                            )
                            call_kwargs["max_tokens"] = allowed_output
                            attempt += 1
                            continue

                        desired_prompt_tokens = max(1, allowed - requested_max - margin)
                        if est_prompt_tokens > desired_prompt_tokens:
                            def _trim_prompt(text: str, target_tokens: int) -> str:
                                target_chars = max(40, target_tokens * 4)
                                if len(text) <= target_chars:
                                    return text
                                keep_each = target_chars // 2
                                return text[:keep_each] + "\n...TRUNCATED...\n" + text[-keep_each:]

                            new_prompt = _trim_prompt(orig_prompt, desired_prompt_tokens)
                            log.info(
                                "Trimming prompt from %s chars to %s chars to fit context window",
                                len(orig_prompt), len(new_prompt)
                            )
                            call_kwargs["prompt"] = new_prompt
                            attempt += 1
                            continue
                    else:
                        # fallback conservative shrink
                        new_max = max(1, requested_max // 4)
                        if new_max == call_kwargs.get("max_tokens", None):
                            log.exception("create_completion failed due to context window and cannot reduce further: %s", msg)
                            raise
                        log.warning("Could not parse context window size from error; retrying with max_tokens=%s", new_max)
                        call_kwargs["max_tokens"] = new_max
                        attempt += 1
                        continue

                    log.exception("create_completion failed due to context window and cannot reduce further: %s", msg)
                    raise

                # If signature mismatch (older/newer llama-cpp-python) try alternative call once
                if isinstance(e, TypeError) and attempt == 0:
                    try:
                        resp = self.llm.create_completion(prompt=prompt, max_tokens=call_kwargs.get("max_tokens", None))
                        if isinstance(resp, dict):
                            if "choices" in resp and resp["choices"]:
                                text = resp["choices"][0].get("text", "")
                            else:
                                text = resp.get("text", "")
                        else:
                            text = str(resp)
                        return (text or "").strip()
                    except Exception:
                        pass

                log.exception("create_completion failed: %s", e)
                raise

        raise RuntimeError("create_completion failed after retries")

    def generate_structured(self, prompt: str, schema_desc: Optional[str] = None, max_tokens: int = 128) -> Any:
        """
        Generate output that is expected to be a JSON-like structure.
        Attempts to parse JSON from the model output. Returns parsed object or {"raw_text": "..."} if parse fails.
        """
        full_prompt = prompt
        if schema_desc:
            full_prompt = f"{prompt}\n\n{schema_desc}"

        raw = self.create_completion(full_prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            start = raw.find("{")
            if start >= 0:
                candidate = raw[start:]
                return json.loads(candidate)
        except Exception:
            log.debug("JSON parse failed, returning raw_text")

        return {"raw_text": raw}

    def close(self):
        """Try to free resources."""
        try:
            if self.llm is not None:
                if hasattr(self.llm, "close"):
                    self.llm.close()
                del self.llm
                self.llm = None
                self._loaded_on = "none"
                log.info("LocalLLM closed and freed.")
        except Exception as e:
            log.warning("Error closing LocalLLM: %s", e)

    def info(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "loaded_on": self._loaded_on,
            "cuda_detect": self._cuda_detect,
            "n_gpu_layers": self.n_gpu_layers,
        }


if __name__ == "__main__":
    import time

    MODEL = Path(__file__).parent / DEFAULT_MODEL_FILENAME
    print("Model exists:", MODEL.exists())
    print("Detecting CUDA ...")
    print(detect_cuda())
    print("Initializing LocalLLM (may take a while)...")
    t0 = time.time()
    try:
        llm = LocalLLM(model_path=MODEL, prefer_cuda=True, n_gpu_layers=8)
        print("Loaded on:", llm.loaded_on)
        info = llm.info()
        print("Info:", json.dumps(info, indent=2))
        print("Trying a short prompt ...")
        out = llm.create_completion("Write a one-sentence description: 'This test aims to verify the functionality of loading local files for the AI_CSPM project using LLM.'", max_tokens=64)
        print("Output:", out)
    except Exception as e:
        print("Failed to initialize LocalLLM:", e)
    finally:
        try:
            llm.close()
        except Exception:
            pass
        print("Elapsed: %.2fs" % (time.time() - t0))
