# ai/ai_engine.py
"""
High-level AI interface used by the analyzer and other modules.

Provides:
- AIEngine: manages a LocalLLM instance and exposes convenience methods.
- generate_from_messages(messages, ...): lightweight chat-style wrapper that returns a dict
  with 'text' and optionally 'raw' and 'json' (if parsing success).
- analyze_findings_and_report(findings): example function that builds a prompt and returns LLM result.

This version adds:
- Automatic enforcement of model context window (truncate prompt if needed).
- Conservative token estimation to avoid requesting more tokens than allowed.
- Graceful retry on context-window errors.
"""

from __future__ import annotations
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Import LocalLLM from the model module
try:
    from ai.model.local_llm import LocalLLM
except Exception as e:
    raise ImportError("LocalLLM not found. Ensure ai/model/local_llm.py exists and is importable.") from e

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default model filename in your repo (adjust if different)
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "mistral-7b-instruct-v0.3-q4_k_m.gguf"

# Model context window (tokens). Update if your model has a different context.
MODEL_CONTEXT_WINDOW = 512
SAFETY_MARGIN_TOKENS = 16  # small buffer for EOS/meta tokens


def estimate_tokens_from_text(text: str) -> int:
    """
    Conservative token estimator.
    Uses average ~3.5 characters per token approximation (conservative).
    Returns estimated token count (>=1 for non-empty text).
    """
    if not text:
        return 0
    chars = len(text)
    # tokens_est = ceil(chars / 3.5) -> implement as integer arithmetic
    tokens_est = (chars * 10 + 34) // 35
    return max(1, tokens_est)


class AIEngine:
    """
    High level engine that owns a LocalLLM instance and exposes helper methods.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        prefer_cuda: bool = True,
        n_gpu_layers: int = 8,
        extra_llama_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = None,
    ):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.context_window = context_window or MODEL_CONTEXT_WINDOW
        log.info(
            "Initializing AIEngine with model: %s (prefer_cuda=%s, n_gpu_layers=%s, context_window=%s)",
            model_path,
            prefer_cuda,
            n_gpu_layers,
            self.context_window,
        )
        self.llm = LocalLLM(
            model_path=model_path,
            prefer_cuda=prefer_cuda,
            n_gpu_layers=n_gpu_layers,
            extra_llama_kwargs=extra_llama_kwargs or {},
        )
        # LocalLLM is expected to expose .loaded_on and a create_completion(...) method
        log.info("AIEngine loaded on: %s", getattr(self.llm, "loaded_on", "unknown"))

    def _enforce_context_and_cap(self, prompt: str, requested_max_tokens: int) -> (str, int):
        """
        Ensure prompt + completion fits within context window.
        Returns possibly-truncated prompt and an adjusted max_tokens value.
        """
        prompt_tokens = estimate_tokens_from_text(prompt)
        allowed_for_completion = self.context_window - prompt_tokens - SAFETY_MARGIN_TOKENS

        if allowed_for_completion < 1:
            # Prompt alone exceeds or nearly equals context. Truncate oldest content.
            warnings.warn("Prompt too long for model context window; truncating older content.")
            # Pick a reasonable target for prompt tokens to keep (leave some room for completion)
            target_prompt_tokens = max(32, self.context_window - SAFETY_MARGIN_TOKENS - 64)
            # Convert tokens back to chars approximating 3.5 chars per token: chars = tokens * 3.5
            chars_needed = (target_prompt_tokens * 35) // 10
            # Keep the last `chars_needed` characters of the prompt (assumes end of prompt more relevant)
            prompt = prompt[-chars_needed:]
            prompt_tokens = estimate_tokens_from_text(prompt)
            allowed_for_completion = self.context_window - prompt_tokens - SAFETY_MARGIN_TOKENS

        # Cap requested max_tokens to allowed_for_completion
        if requested_max_tokens > allowed_for_completion:
            old = requested_max_tokens
            requested_max_tokens = max(1, allowed_for_completion)
            warnings.warn(
                f"Requested max_tokens reduced from {old} to {requested_max_tokens} to fit context window ({self.context_window})."
            )

        return prompt, int(max(1, requested_max_tokens))

    def create_completion(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0, **kwargs) -> str:
        """
        Wrapper that enforces model context limits and reduces/truncates as needed,
        then calls self.llm.create_completion(...) which should return generated text.
        Handles ValueError from the model (e.g., requested tokens exceed context) with a retry.
        """
        # Sanity defaults
        if max_tokens is None:
            max_tokens = 256

        # Enforce context constraints and cap tokens
        tried_prompt, tried_max = self._enforce_context_and_cap(prompt, int(max_tokens))

        # Try call; if model raises context-related ValueError, attempt one retry with reduced tokens.
        try:
            text = self.llm.create_completion(prompt=tried_prompt, max_tokens=tried_max, temperature=temperature, seed=seed, **kwargs)
            return text
        except ValueError as e:
            msg = str(e)
            log.warning("LLM raised ValueError during create_completion: %s", msg)
            # heuristically detect context-window error messages
            if "exceed context window" in msg or "Requested tokens" in msg or "context" in msg:
                # compute allowed tokens more conservatively and retry once
                prompt_tokens = estimate_tokens_from_text(tried_prompt)
                allowed = self.context_window - prompt_tokens - SAFETY_MARGIN_TOKENS
                allowed = max(1, allowed)
                log.warning("Retrying create_completion with reduced max_tokens=%s", allowed)
                try:
                    text = self.llm.create_completion(prompt=tried_prompt, max_tokens=allowed, temperature=temperature, seed=seed, **kwargs)
                    return text
                except Exception as e2:
                    log.exception("Retry after context-limit failed: %s", e2)
                    raise
            # not a context error — re-raise
            raise

    def generate_structured(self, prompt: str, schema_desc: Optional[str] = None, max_tokens: int = 512) -> Any:
        """
        Ask the model to return JSON-like structure. Will attempt to parse JSON and return object or raw_text dict.
        """
        if schema_desc:
            prompt = prompt + "\n\nSchema Description:\n" + schema_desc
        text = self.create_completion(prompt=prompt, max_tokens=max_tokens)
        # try to parse JSON from output
        try:
            start = text.find("{")
            if start >= 0:
                candidate = text[start:]
                parsed = json.loads(candidate)
                return parsed
        except Exception:
            log.debug("generate_structured: JSON parse failed; returning raw text.")
        return {"raw": text}

    def generate_from_messages(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        role_format: str = "[{role}] {content}\n",
    ) -> Dict[str, Any]:
        """
        Convert simple chat-style messages into a single prompt and call the LLM.
        - messages: sequence of {"role": "user"|"assistant"|"system", "content": "..."}
        Returns dict with keys:
            - text: the raw generated text
            - raw: raw model output (same as text)
            - json: parsed JSON if possible (otherwise absent)
        """
        # Build a concise prompt combining the messages. You can change formatting as needed.
        built = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content", "")
            if role == "system":
                built.append(f"System: {content}\n")
            else:
                built.append(role_format.format(role=role, content=content))

        prompt = "\n".join(built).strip()
        log.debug("Generated prompt for LLM (len=%d chars).", len(prompt))

        # Ensure we don't exceed context window
        prompt_for_call, capped_max_tokens = self._enforce_context_and_cap(prompt, int(max_tokens))

        text = self.create_completion(prompt=prompt_for_call, max_tokens=capped_max_tokens, temperature=temperature)
        result: Dict[str, Any] = {"text": text, "raw": text}

        # Try parse JSON from returned text
        try:
            start = text.find("{")
            if start >= 0:
                candidate = text[start:]
                parsed = json.loads(candidate)
                result["json"] = parsed
        except Exception:
            log.debug("generate_from_messages: JSON parse failed; returning raw text.")
        return result

    def analyze_findings_and_report(self, findings: List[Union[str, Dict[str, Any]]], max_tokens: int = 512) -> Dict[str, Any]:
        """
        Example helper — given a list of findings, create a short JSON-style report.
        Returns the parsed output if LLM returns JSON, else raw text.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a security auditor assistant. Analyze the following findings and produce a JSON array named 'report'. "
                "Each element should be an object with keys: 'id' (unique short id), 'severity' (Low/Medium/High/Critical), "
                "'summary' (one-sentence), and 'remediation' (concise actionable steps). Return ONLY valid JSON."
            ),
        }

        lines = []
        for idx, f in enumerate(findings, start=1):
            if isinstance(f, dict):
                # keep dict compact
                lines.append(f"{idx}. {json.dumps(f, separators=(',', ':'))}")
            else:
                lines.append(f"{idx}. {str(f)}")

        user_msg = {"role": "user", "content": "Findings:\n" + "\n".join(lines)}

        resp = self.generate_from_messages([system_msg, user_msg], max_tokens=max_tokens, temperature=0.0)
        if "json" in resp:
            parsed = resp["json"]
            if isinstance(parsed, dict) and "report" in parsed:
                return {"ok": True, "report": parsed["report"], "raw": resp["raw"]}
            if isinstance(parsed, list):
                return {"ok": True, "report": parsed, "raw": resp["raw"]}
            return {"ok": True, "report": parsed, "raw": resp["raw"]}

        return {"ok": False, "raw": resp["raw"], "text": resp["text"]}

    def info(self) -> Dict[str, Any]:
        """Return some information about the engine and underlying model."""
        info = {"loaded_on": getattr(self.llm, "loaded_on", None)}
        try:
            llm_info = getattr(self.llm, "info", lambda: {})()
            if isinstance(llm_info, dict):
                info.update(llm_info)
        except Exception:
            pass
        # include configured context window
        info["context_window"] = self.context_window
        return info

    def close(self):
        """Close and free LLM resources."""
        try:
            self.llm.close()
        except Exception as e:
            log.debug("Error closing LLM: %s", e)


# Convenience top-level functions for compatibility with older code that called generate_from_messages directly.
_engine_singleton: Optional[AIEngine] = None


def get_engine(
    model_path: Optional[Union[str, Path]] = None,
    prefer_cuda: bool = True,
    n_gpu_layers: int = 8,
    context_window: Optional[int] = None,
) -> AIEngine:
    """
    Return a singleton engine configured on first call. Useful for scripts/tests.
    """
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = AIEngine(
            model_path=model_path, prefer_cuda=prefer_cuda, n_gpu_layers=n_gpu_layers, context_window=context_window
        )
    return _engine_singleton


def generate_from_messages(
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    model_path: Optional[Union[str, Path]] = None,
    prefer_cuda: bool = True,
    n_gpu_layers: int = 8,
    context_window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lightweight wrapper to quickly call the default engine.
    Returns dict with keys 'text', 'raw', and optionally 'json'.
    """
    engine = get_engine(model_path=model_path, prefer_cuda=prefer_cuda, n_gpu_layers=n_gpu_layers, context_window=context_window)
    return engine.generate_from_messages(messages, max_tokens=max_tokens, temperature=temperature)


# Example: keep a helper that replicates your earlier analyze_findings_and_report behavior
def analyze_findings_and_report(findings: List[Union[str, Dict[str, Any]]]) -> str:
    """
    Return plain text (string) compatible with older code: returns resp.get('text','') or JSON-serialized report.
    """
    engine = get_engine()
    result = engine.analyze_findings_and_report(findings)
    if result.get("ok"):
        return json.dumps(result["report"], indent=2)
    return result.get("raw", result.get("text", ""))


if __name__ == "__main__":
    # Quick smoke test when run directly.
    import time

    logging.getLogger().setLevel(logging.INFO)
    print("AI Engine quick test. Model path:", DEFAULT_MODEL_PATH)
    try:
        eng = AIEngine()
        print("Engine info:", eng.info())
        t0 = time.time()
        r = eng.generate_from_messages([{"role": "user", "content": "Write a short one-line description of this system: AI_CSPM project."}], max_tokens=64)
        print("Result text:\n", r.get("text"))
        print("Elapsed: %.2fs" % (time.time() - t0))
    finally:
        try:
            eng.close()
        except Exception:
            pass
