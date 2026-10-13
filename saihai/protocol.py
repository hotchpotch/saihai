"""Shared native-compatible prompt serialization; no training dependencies."""

import json
import math


def canonical_json(value):
    """Match minja's six-significant-digit float JSON representation explicitly."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite input number")
        return format(value, ".6g")
    if isinstance(value, list):
        return "[" + ", ".join(canonical_json(x) for x in value) + "]"
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return (
            "{"
            + ", ".join(canonical_json(k) + ": " + canonical_json(v) for k, v in value.items())
            + "}"
        )
    raise ValueError("Input must contain JSON values with string object keys")


def render_input(inp, contract):
    kind = inp["type"]
    count = len(inp["options"])
    if kind not in ("choice", "noul", "score"):
        raise ValueError("Unsupported decision type")
    minimum = 1 if kind == "choice" else 2
    if count < minimum or (kind == "noul" and count != 2) or (kind == "score" and count > 10):
        raise ValueError("Invalid native candidate count")
    if kind == "noul" and [o["label"] for o in inp["options"]] != ["false", "true"]:
        raise ValueError("Native Noul requires false then true")
    options = []
    if len(inp["options"]) > len(contract["codes"]):
        raise ValueError("Native candidate capacity exceeded")
    for code, o in zip(contract["codes"], inp["options"]):
        item = {"code": code, "label": o["label"], "definition": o["description"]}
        if inp["type"] == "score":
            value = o["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Score values must be finite numbers")
            item["numeric_value"] = value
        options.append(item)
    return (
        contract["prompt_prefix"]
        + canonical_json(
            {
                "instruction": inp["instructions"],
                "state": inp["state"],
                "type": inp["type"],
                "options": options,
            }
        )
        + contract["prompt_suffix"]
    )
