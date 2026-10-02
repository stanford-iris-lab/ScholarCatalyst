"""The first JSON array in a model reply, with one repair for a missing closing bracket."""
from __future__ import annotations

import json
import re

_DECODER = json.JSONDecoder(strict=False)  # raw newlines inside strings are common in long items


def first_json_array(text: str) -> list | None:
    text = text or ""
    for m in re.finditer(r"\[", text):
        try:
            value, _ = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    start = text.find("[")
    if start < 0:
        return None
    body = text[start:].rstrip().rstrip(",")
    for tail in ("]", "\"]"):  # missing bracket; missing quote and bracket
        try:
            value = json.loads(body + tail, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None
