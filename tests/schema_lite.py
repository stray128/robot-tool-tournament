"""A small JSON Schema (draft 2020-12) validator for the repository's own schemas.

It implements only the keywords those schemas use (:data:`SUPPORTED`), so tests need no
extra dependency. :func:`unsupported_keywords` lets a test prove a schema stays inside
that subset; when the ``jsonschema`` package is installed, tests cross-check with it too.
"""

from __future__ import annotations

import re
from typing import Any

ANNOTATIONS = frozenset({"$schema", "$id", "$comment", "title", "description", "default", "examples"})
SUPPORTED = frozenset(
    {
        "$ref",
        "$defs",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "propertyNames",
        "minProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "anyOf",
        "if",
        "then",
        "else",
    }
)
# keywords whose value is a map of name -> subschema, or a list of subschemas
_SCHEMA_MAPS = frozenset({"properties", "$defs"})
_SCHEMA_LISTS = frozenset({"anyOf"})
_SCHEMA_ONE = frozenset({"additionalProperties", "propertyNames", "items", "if", "then", "else"})


def unsupported_keywords(schema: Any, path: str = "#") -> list[str]:
    """Every keyword in ``schema`` (recursively) that this validator does not implement."""
    bad: list[str] = []
    if isinstance(schema, bool):
        return bad
    if not isinstance(schema, dict):
        return [f"{path}: not a schema"]
    for key, value in schema.items():
        if key in ANNOTATIONS:
            continue
        if key not in SUPPORTED:
            bad.append(f"{path}/{key}")
            continue
        if key in _SCHEMA_MAPS:
            for name, sub in value.items():
                bad += unsupported_keywords(sub, f"{path}/{key}/{name}")
        elif key in _SCHEMA_LISTS:
            for i, sub in enumerate(value):
                bad += unsupported_keywords(sub, f"{path}/{key}/{i}")
        elif key in _SCHEMA_ONE:
            bad += unsupported_keywords(value, f"{path}/{key}")
    return bad


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unknown type {name!r}")


def _resolve(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"only local references are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


class Validator:
    """Validate instances against one root schema; :meth:`errors` lists every failure."""

    def __init__(self, schema: dict[str, Any]) -> None:
        bad = unsupported_keywords(schema)
        if bad:
            raise ValueError(f"schema uses keywords this validator does not implement: {bad}")
        self.root = schema

    def errors(self, instance: Any, schema: Any = None, path: str = "$") -> list[str]:
        return self._check(instance, self.root if schema is None else schema, path)

    def is_valid(self, instance: Any, schema: Any = None) -> bool:
        return not self.errors(instance, schema)

    def _check(self, inst: Any, schema: Any, path: str) -> list[str]:
        if schema is True:
            return []
        if schema is False:
            return [f"{path}: not allowed"]
        errs: list[str] = []
        if "$ref" in schema:
            errs += self._check(inst, _resolve(self.root, schema["$ref"]), path)
        if "type" in schema:
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(_is_type(inst, t) for t in types):
                return errs + [f"{path}: expected type {types}, got {type(inst).__name__}"]
        if "const" in schema and not _json_equal(inst, schema["const"]):
            errs.append(f"{path}: expected {schema['const']!r}")
        if "enum" in schema and not any(_json_equal(inst, e) for e in schema["enum"]):
            errs.append(f"{path}: {inst!r} not in {schema['enum']}")
        if isinstance(inst, str):
            if len(inst) < schema.get("minLength", 0):
                errs.append(f"{path}: shorter than {schema['minLength']}")
            if "maxLength" in schema and len(inst) > schema["maxLength"]:
                errs.append(f"{path}: longer than {schema['maxLength']}")
            if "pattern" in schema and not re.search(schema["pattern"], inst):
                errs.append(f"{path}: {inst!r} does not match {schema['pattern']}")
        if _is_type(inst, "number"):
            if "minimum" in schema and inst < schema["minimum"]:
                errs.append(f"{path}: {inst} < {schema['minimum']}")
            if "maximum" in schema and inst > schema["maximum"]:
                errs.append(f"{path}: {inst} > {schema['maximum']}")
            if "exclusiveMinimum" in schema and inst <= schema["exclusiveMinimum"]:
                errs.append(f"{path}: {inst} <= {schema['exclusiveMinimum']}")
        if isinstance(inst, list):
            errs += self._check_array(inst, schema, path)
        if isinstance(inst, dict):
            errs += self._check_object(inst, schema, path)
        if "anyOf" in schema and not any(not self._check(inst, s, path) for s in schema["anyOf"]):
            errs.append(f"{path}: matches none of anyOf")
        if "if" in schema:
            branch = "then" if not self._check(inst, schema["if"], path) else "else"
            if branch in schema:
                errs += self._check(inst, schema[branch], path)
        return errs

    def _check_array(self, inst: list[Any], schema: dict[str, Any], path: str) -> list[str]:
        errs: list[str] = []
        if len(inst) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            for i, a in enumerate(inst):
                if any(_json_equal(a, b) for b in inst[i + 1 :]):
                    errs.append(f"{path}: duplicate item {a!r}")
        if "items" in schema:
            for i, item in enumerate(inst):
                errs += self._check(item, schema["items"], f"{path}[{i}]")
        return errs

    def _check_object(self, inst: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
        errs: list[str] = []
        for key in schema.get("required", ()):
            if key not in inst:
                errs.append(f"{path}: missing {key!r}")
        if len(inst) < schema.get("minProperties", 0):
            errs.append(f"{path}: fewer than {schema['minProperties']} properties")
        props = schema.get("properties", {})
        for key, value in inst.items():
            if "propertyNames" in schema:
                errs += self._check(key, schema["propertyNames"], f"{path}<key {key!r}>")
            if key in props:
                errs += self._check(value, props[key], f"{path}.{key}")
            elif "additionalProperties" in schema:
                errs += self._check(value, schema["additionalProperties"], f"{path}.{key}")
        return errs
