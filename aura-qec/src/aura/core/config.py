"""
Strict dataclass <-> dict conversion for configuration files (TOML / JSON).

Unknown keys are rejected (typos in lab configs are a classic source of
silent mis-configuration), nested dataclasses are built recursively and basic
types are checked.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import typing
from pathlib import Path
from typing import Any, Type, TypeVar

from .errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = None

T = TypeVar("T")


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if origin is typing.Union:
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(f"{path}: {value!r} matches none of {args}: {errors}")
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected table for {tp.__name__}, got {type(value).__name__}")
        return from_dict(tp, value, path)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ConfigError(f"{path}: expected {len(args)} items")
            return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        inner = args[0] if args else Any
        out = [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(out) if origin is tuple else out
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected table")
        kt, vt = args if args else (Any, Any)
        return {_coerce(kt, k, path): _coerce(vt, v, f"{path}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{path}: expected integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected string, got {value!r}")
        return value
    if isinstance(tp, type) and not isinstance(value, tp):
        try:
            return tp(value)
        except Exception as e:  # pragma: no cover
            raise ConfigError(f"{path}: cannot convert {value!r} to {tp}") from e
    return value


def from_dict(cls: Type[T], data: dict, path: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: required key missing")
    obj = cls(**kwargs)
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
    return obj


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return obj


def load_file(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    if p.suffix == ".toml":
        with p.open("rb") as fh:
            return tomllib.load(fh)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ConfigError(f"unsupported config format {p.suffix} (use .toml or .json)")
