"""
Utilities for parsing and normalizing LLM JSON responses.
Keeps agents resilient to minor formatting issues.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json_from_text(text: str) -> str:
    """Extract the JSON object from a raw LLM response string."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    """Best-effort parse of an LLM response to JSON."""
    cleaned = extract_json_from_text(text)
    if not cleaned:
        return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned2 = cleaned.replace("'", "\"")
        cleaned2 = cleaned2.replace("None", "null").replace("True", "true").replace("False", "false")
        try:
            return json.loads(cleaned2)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(cleaned)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert common numeric string formats to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d\.\-]", "", value)
        if cleaned in ("", ".", "-", "-."):
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def normalize_confidence(value: Any, default: float = 0.0) -> float:
    """Clamp confidence to 0-1 and normalize percentages."""
    conf = coerce_float(value, default)
    if conf > 1.0 and conf <= 100.0:
        conf = conf / 100.0
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return conf


def ensure_list(value: Any) -> List[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


def normalize_string(value: Any) -> str:
    """Normalize a value into a clean string."""
    if value is None:
        return ""
    return str(value).strip()
