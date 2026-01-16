# ai/model/llm_runner.py
"""
Small wrapper for local gguf model using llama-cpp-python.
Auto-detects a usable call method and exposes a simple `generate` function.

Place your .gguf file in ai/model/ and set MODEL_PATH accordingly.
"""

import os
import threading
from typing import List, Dict, Any

# path relative to project root
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")

# tweak these defaults as needed
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STOP = None  # or e.g. ["</s>"]

_lock = threading.Lock()
_llm = None
_llm_call = None  # str describing chosen call method


def _load_llm():
    global _llm, _llm_call
    if _llm is not None:
        return _llm

    # lazy import so module can be imported even if llama-cpp-python not installed yet
    from llama_cpp import Llama

    # instantiate
    print(f"[llm_runner] loading model from {MODEL_PATH} ...")
    _llm = Llama(model_path=MODEL_PATH)
    print("[llm_runner] model loaded, detecting call method...")

    # detect best call form
    # we prefer create_completion (works in your logs), then create_completion-like, then __call__
    if hasattr(_llm, "create_completion"):
        _llm_call = "create_completion"
    elif hasattr(_llm, "complete"):
        _llm_call = "complete"
    elif callable(_llm):
        # some bindings implement __call__
        _llm_call = "__call__"
    else:
        # fallback to generate if present
        _llm_call = "generate" if hasattr(_llm, "generate") else None

    print(f"[llm_runner] chosen llm call: {_llm_call}")
    return _llm


def _invoke_llm(prompt: str, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE, stop=None) -> Dict[str, Any]:
    """
    Unified call. Returns a dict with 'text' and optionally raw 'response'.
    """
    global _llm_call
    with _lock:
        llm = _load_llm()

        stop = stop or DEFAULT_STOP

        # many local models produce different shaped outputs; handle common ones
        if _llm_call == "create_completion":
            resp = llm.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
            # older create_completion returned a dict like {choices:[{text:...}]}
            text = ""
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if len(choices):
                    # prefer "text" legacy field
                    text = choices[0].get("text") or choices[0].get("delta") or ""
                else:
                    # some versions return 'text' directly
                    text = resp.get("text", "")
            else:
                text = str(resp)
            return {"text": text, "raw": resp}

        elif _llm_call == "complete":
            resp = llm.complete(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            # llama-cpp's 'complete' often returns a string or an object
            if isinstance(resp, dict):
                txt = resp.get("text") or resp.get("choices", [{}])[0].get("text", "")
            else:
                txt = str(resp)
            return {"text": txt, "raw": resp}

        elif _llm_call == "__call__":  # llm(prompt, max_new_tokens=..)
            # try common keywords
            try:
                resp = llm(prompt, max_new_tokens=max_tokens, temperature=temperature)
            except TypeError:
                resp = llm(prompt, max_tokens=max_tokens, temperature=temperature)
            # unify
            if isinstance(resp, dict):
                text = resp.get("choices", [{}])[0].get("text") or resp.get("text", "")
            else:
                text = str(resp)
            return {"text": text, "raw": resp}

        elif _llm_call == "generate":
            # generate may want keyword args like n_predict etc.
            try:
                resp = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            except TypeError:
                resp = llm.generate(prompt, n_predict=max_tokens, temperature=temperature)
            # generate usually returns dict with 'choices'
            if isinstance(resp, dict):
                txt = resp.get("choices", [{}])[0].get("text", "")
            else:
                txt = str(resp)
            return {"text": txt, "raw": resp}

        else:
            raise RuntimeError("No compatible llama-cpp-python call method detected.")


def chat_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convert a message list ({"role": "user"/"assistant", "content": "..."} ...)
    to the model's chat template. Use the gguf model chat template format:
      [INST] user message [/INST] assistant message
    We'll produce a single prompt string.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"[INST] {content} [/INST]")
        else:
            # assistant content appended raw (gguf chat template expects it)
            parts.append(content)
    return " ".join(parts)


def generate_from_messages(messages: List[Dict[str, str]],
                           max_tokens=DEFAULT_MAX_TOKENS,
                           temperature=DEFAULT_TEMPERATURE) -> Dict[str, Any]:
    """
    High-level call: pass in messages (list of role/content) and get back the assistant text.
    """
    prompt = chat_from_messages(messages)
    return _invoke_llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
