"""
Compatibility layer: provide analyze_with_llm and LocalLLM for older imports.
This file intentionally lazily instantiates the local LLM to avoid heavy startup
when the module is imported.
"""
from __future__ import annotations
import json
from typing import Any, Dict

from ai.model.local_llm import LocalLLM


def analyze_with_llm(evidence: list, redacted_sample: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    """Build a deterministic prompt from evidence and redacted_sample and return parsed JSON if possible.

    This lazily creates a `LocalLLM` instance so importing this module doesn't try to load the model.
    """
    instruction = (
        "You are a cloud security analyst. Return ONLY valid JSON with the schema:\n"
        "{\n"
        '  "summary": "<one-paragraph summary>",\n'
        '  "findings": [ { "severity":"high|medium|low", "issue":"", "resource":"", "evidence":"", "remediation":"" } ],\n'
        '  "recommendations": [ "..." ]\n'
        "}\n\n"
        "Ground your output ONLY on the provided evidence and redacted sample. Do NOT invent resources or ARNs. Output must be strict JSON with no extra text."
    )

    body = {"evidence": evidence, "redacted_sample": redacted_sample}
    prompt = instruction + "\n\n" + json.dumps(body, indent=2)

    # instantiate LLM when needed
    llm = LocalLLM()
    try:
        result = llm.generate_structured(prompt, schema_desc=None)
        if isinstance(result, dict):
            return result
        # if the LLM returned raw text, try to parse it
        try:
            start = result.find("{")
            end = result.rfind("}")
            if start != -1 and end != -1:
                return json.loads(result[start:end+1])
        except Exception:
            pass
        return {"raw_text": str(result)}
    finally:
        try:
            llm.close()
        except Exception:
            pass


# Re-export LocalLLM for compatibility
LocalLLM = LocalLLM
