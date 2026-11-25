"""A subset of JSON Schema compiled to a regex over JSON text.

By default the regex describes compact JSON (``separators=(",", ":")``). With ``ws`` set
(for example ``WS``, spaces and newlines), that pattern may appear around every ``{ } [ ] , :``,
so the model can also write the pretty-printed layout it saw in training.

Supported:
- ``type``: ``string`` (with ``minLength``/``maxLength`` and ``pattern`` as a raw sub-regex),
  ``integer`` (with optional ``minimum ≥ 0`` to forbid the sign), ``number``, ``boolean``,
  ``null``, ``array`` (``items``, ``minItems``/``maxItems``) and ``object``;
- ``properties`` with ``required``: required keys appear in ``properties`` order, and
  optional ones may be omitted;
- ``enum`` and ``const`` of JSON scalars.

Objects are closed, as with ``additionalProperties: false``, and key order is fixed. That is
what generation needs: a canonical serialisation the model is steered into, not a validator
for arbitrary documents. (``validate`` below checks a parsed value against the same subset.)
"""

from __future__ import annotations

import json
import re

# One JSON string character as *valid UTF-8 bytes* (RFC 3629): printable ASCII except '"'
# and '\', an escape, or a well-formed 2-, 3- or 4-byte sequence (no overlongs, no surrogates).
_C = r"[\x80-\xbf]"
UTF8_MULTI = (
    rf"[\xc2-\xdf]{_C}"
    rf"|\xe0[\xa0-\xbf]{_C}|[\xe1-\xec\xee\xef]{_C}{{2}}|\xed[\x80-\x9f]{_C}"
    rf"|\xf0[\x90-\xbf]{_C}{{2}}|[\xf1-\xf3]{_C}{{3}}|\xf4[\x80-\x8f]{_C}{{2}}"
)
STRING_CHAR = r'(?:[\x20\x21\x23-\x5b\x5d-\x7f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|' + UTF8_MULTI + ")"
INTEGER = r"-?(?:0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"


def _escape(s: str) -> str:
    """Regex-escape a literal for our engine."""
    return re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", s)


def _lit(value) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


WS = r"[ \n]*"  # JSON whitespace (spaces and newlines) between structural characters


def to_regex(schema: dict, ws: str = "") -> str:
    """Regex for ``schema``; ``ws`` is the whitespace pattern allowed between tokens."""
    if "const" in schema:
        return _lit(schema["const"])
    if "enum" in schema:
        return "(?:" + "|".join(_lit(v) for v in schema["enum"]) + ")"
    t = schema.get("type")
    if t == "string":
        if "pattern" in schema:
            return '"' + schema["pattern"] + '"'
        lo, hi = schema.get("minLength", 0), schema.get("maxLength")
        q = "*" if (lo == 0 and hi is None) else f"{{{lo},{'' if hi is None else hi}}}"
        return '"' + STRING_CHAR + q + '"'
    if t == "integer":
        return r"(?:0|[1-9][0-9]*)" if schema.get("minimum", -1) >= 0 else INTEGER
    if t == "number":
        return NUMBER
    if t == "boolean":
        return "(?:true|false)"
    if t == "null":
        return "null"
    if t == "array":
        item = to_regex(schema.get("items", {"type": "string"}), ws)
        lo, hi = schema.get("minItems", 0), schema.get("maxItems")
        if hi == 0:
            return rf"\[{ws}\]"
        rest_lo = max(lo - 1, 0)
        rest_hi = "" if hi is None else str(hi - 1)
        body = f"{item}(?:{ws},{ws}{item}){{{rest_lo},{rest_hi}}}"
        if lo >= 1:
            return rf"\[{ws}{body}{ws}\]"
        return rf"\[{ws}(?:{body}{ws})?\]"
    if t == "object":
        return _object(schema, ws)
    raise ValueError(f"unsupported schema: {schema}")


def _object(schema: dict, ws: str = "") -> str:
    props = schema.get("properties", {})
    required = set(schema.get("required", list(props)))
    parts = [(_lit(k) + f"{ws}:{ws}" + to_regex(s, ws), k in required) for k, s in props.items()]
    if not parts:
        return rf"\{{{ws}\}}"
    # Build "first present member, then ',member' for each later one", with optional members.
    # States: whether something has been emitted yet. Enumerate prefixes left to right.
    # regex for members i.. given that a member was already emitted (needs a leading comma):
    tail = [""] * (len(parts) + 1)
    for i in range(len(parts) - 1, -1, -1):
        m, req = parts[i]
        c = f"{ws},{ws}"
        tail[i] = f"{c}{m}{tail[i + 1]}" if req else f"(?:{c}{m})?{tail[i + 1]}"
    # regex for members i.. when nothing has been emitted yet:
    head = [""] * (len(parts) + 1)
    for i in range(len(parts) - 1, -1, -1):
        m, req = parts[i]
        if req:
            head[i] = f"{m}{tail[i + 1]}"
        else:
            head[i] = f"(?:{m}{tail[i + 1]}|{head[i + 1]})" if head[i + 1] else f"(?:{m})?"
    return r"\{" + ws + head[0] + ws + r"\}"


def dumps(value) -> str:
    """The serialisation the regex describes."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate(value, schema: dict, ordered: bool = True) -> bool:
    """Does a parsed JSON value satisfy the subset? ``ordered`` also checks object key order."""
    if "const" in schema:
        return value == schema["const"] and type(value) is type(schema["const"])
    if "enum" in schema:
        return any(value == v and type(value) is type(v) for v in schema["enum"])
    t = schema.get("type")
    if t == "string":
        if not isinstance(value, str):
            return False
        if "pattern" in schema:
            return re.fullmatch(schema["pattern"], value) is not None
        hi = schema.get("maxLength")
        return len(value) >= schema.get("minLength", 0) and (hi is None or len(value) <= hi)
    if t == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
        return ok and value >= schema.get("minimum", value)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "array":
        if not isinstance(value, list):
            return False
        hi = schema.get("maxItems")
        n_ok = len(value) >= schema.get("minItems", 0) and (hi is None or len(value) <= hi)
        item = schema.get("items", {"type": "string"})
        return n_ok and all(validate(v, item, ordered) for v in value)
    if t == "object":
        if not isinstance(value, dict):
            return False
        props = schema.get("properties", {})
        required = set(schema.get("required", list(props)))
        keys = list(value)
        if not set(keys) <= set(props) or not required <= set(keys):
            return False
        if ordered and keys != [k for k in props if k in value]:
            return False
        return all(validate(value[k], props[k], ordered) for k in keys)
    raise ValueError(f"unsupported schema: {schema}")
