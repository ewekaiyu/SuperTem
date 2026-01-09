"""
supertem.structure.base

Dataclass-based structures for TEM automation, covering:
  - settings (beam / detector / stage / acquisition outputs)
  - microscope state snapshots
  - image metadata containers
  - executable requests (control-plane objects)

This module supports two distinct contexts:
  (1) data-plane: ingestion and storage of imperfect data (metadata/state/logs)
  (2) control-plane: strict, safe execution of microscope commands

===============================================================================
I. Object Lifecycle (End-to-End)
===============================================================================

The intended lifecycle for objects in this module is:

  1) Ingest (untrusted input)
     - Source: vendor SDK returns, JSON logs, user configs, network payloads
     - Entry:  Class.from_dict(payload, mode=LENIENT or STRICT)
     - Goal:   interpret the payload without losing information

  2) Normalize (structural correctness)
     - Happens during __post_init__ (and helper parsers)
     - Goal: produce a well-typed internal representation
     - Examples:
         * dict -> dataclass
         * "128" -> 128
         * unknown keys -> Extras.unknown_keys
         * unparseable values -> Extras.raw + Extras.notes

  3) Validate (semantic correctness)
     - Happens in validate(mode=...)
     - Goal: enforce domain constraints and invariants
     - Examples:
         * ROI width/height must be >= 0
         * required fields for executable requests must exist
         * cross-field consistency (e.g., detector_id alignment)
     - Mode behavior:
         * LENIENT: record issue and repair-to-safe / disable unsafe fields
         * STRICT: raise via note_or_raise(...)

  4) Serialize (JSON-capable representation)
     - Happens in to_dict()
     - Goal: produce JSON-serializable output for logging/storage/transport

  5) Execute (control boundary)
     - Only applicable to control-plane objects (requests/settings)
     - Policy: validate STRICTLY immediately before hardware interaction
     - Goal: ensure commands applied to the microscope are safe and consistent

A key rule:
  Objects created under LENIENT mode may be stored and inspected, but MUST NOT be
  executed unless re-validated under STRICT mode at the control boundary.

===============================================================================
II. ParseMode and Context
===============================================================================

ParseMode.LENIENT (data-plane default)
  Intended for metadata/state/log ingestion where completeness is not guaranteed.
  Guarantees:
    - construction should not fail due to malformed/missing fields
    - issues are recorded in Extras.notes; raw inputs may be preserved in Extras.raw
    - unsafe subfields may be set to None or replaced by safe defaults

ParseMode.STRICT (control-plane default)
  Intended for objects that will be applied to hardware (requests/settings).
  Guarantees:
    - semantic constraints are enforced
    - invalid values raise via note_or_raise(...)
    - objects leaving STRICT validation are safe to execute

===============================================================================
III. The Normalization vs. Validation Rulebook
===============================================================================

To maintain safety without sacrificing robustness, this module enforces a strict
separation of concerns between `__post_init__` and `validate()`.

1. Normalization (__post_init__)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Structure (Type & Shape Safety)
   INPUT:   "Dirty" data (Strings, Nones, Dicts, Missing Keys)
   OUTPUT:  "Clean" data (Correct Python Types, Structurally Complete)

   Rules:
   A. Coercion is Normalization.
      Convert inputs to their target types.
      (e.g., "128" -> 128, "10 nm" -> Quantity(10, 'nm'))

   B. Structural Defaults are Normalization.
      If a field is `None` but the system requires a value to function (to avoid
      AttributeError/TypeError later), set a safe default here.
      (e.g., `width=None` -> `width=512`)

   C. Structural Patching is Normalization.
      If a required value exists elsewhere in the object graph (e.g., copying
      an ID from an inner object to a missing outer field), perform the copy
      here to complete the structure.

   D. DO NOT Check Logic.
      Do not check if a number is positive, finite, or consistent with other
      fields. If the type is right, let it pass.
      (e.g., `width=-100` is a valid integer. Leave it for validation.)

2. Validation (validate)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Meaning (Domain Safety & Logic)
   INPUT:   "Clean" data (guaranteed types from step 1)
   OUTPUT:  Boolean success flag (and populated Extras.notes)

   Rules:
   A. Domain Constraints are Validation.
      Check physical and logical bounds.
      (e.g., `width > 0`, `voltage < max_limit`, `isfinite(position)`)

   B. Cross-Field Consistency is Validation.
      Check if two fields contradict each other.
      (e.g., `outer_id != inner_id`)

   C. Healing is Validation (Lenient Mode Only).
      If a value is structurally sound (correct type) but logically invalid
      (e.g., `width=-50`):
        - STRICT Mode: Raise an Exception.
        - LENIENT Mode: "Heal" it to a safe value or disable the feature.

   Summary Table:
   +---------------------+-----------------------+------------------+
   | Scenario            | Action                | Responsibility   |
   +=====================+=======================+==================+
   | Input is None       | Set Default (512)     | __post_init__    |
   | Input is "128"      | Convert (int)         | __post_init__    |
   | Missing Outer ID    | Copy Inner ID         | __post_init__    |
   +---------------------+-----------------------+------------------+
   | Input is -100       | Check > 0             | validate()       |
   | Input is NaN        | Check isfinite()      | validate()       |
   | ID Mismatch         | Check A == B          | validate()       |
   +---------------------+-----------------------+------------------+

===============================================================================
IV. Extras: Preservation and Diagnostics
===============================================================================

`Extras` is the structured container for non-canonical information:
  - vendor: vendor-specific extension payloads (namespaced by vendor key)
  - unknown: unknown top-level keys swept during from_dict (forward compatibility)
  - raw: raw values replaced/rejected during normalization
  - notes: structured diagnostics produced by normalization/validation

Conventions:
  - Use namespaced note keys, e.g. "DetectorSettings.roi_invalid_shape".
  - LENIENT mode should preserve information rather than discard it.

===============================================================================
V. Serialization Contract: to_dict Must Be JSON-Capable
===============================================================================

All to_dict() methods must return JSON-serializable output:
  - dataclasses -> dict
  - enums -> str
  - tuples -> lists
  - quantities/units -> plain numbers (and/or unit annotations per project convention)
  - any non-JSON-native objects must be converted via jsonable helpers

If a value cannot be expressed safely as JSON, preserve a safe representation in
Extras.raw and record a diagnostic note.

===============================================================================
VI. Implementation Conventions
===============================================================================

- If a class stores `_mode`, it SHOULD provide validate() (even if minimal).
- __post_init__ should:
    1) normalize types and nested objects
    2) normalize Extras
    3) defer validation to the boundary (do not call validate() here)
- from_dict(...) should be thin:
    - construct with _mode set
    - rely on __post_init__ for parsing and validate() for logic

===============================================================================
Rationale
===============================================================================

TEM automation consumes heterogeneous and imperfect inputs (vendor SDKs, partial
configs, historical logs). Strict parsing everywhere makes metadata pipelines
brittle; lenient parsing everywhere makes control paths unsafe. This module
provides a consistent lifecycle to be both robust (LENIENT ingestion/storage)
and safe (STRICT validation/execution).

"""

import datetime
import json
import math
import os
import ipaddress
from dataclasses import dataclass, field, fields, replace, is_dataclass
from pathlib import Path
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Set, TypeVar, Type
import numpy as np
from PIL import Image

import tifffile as tff

try:
    from supertem.config import METADATA_VERSION  # type: ignore
except Exception:
    METADATA_VERSION = "1"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("supertem")
except PackageNotFoundError:
    try:
        __version__ = version("SuperTem")
    except PackageNotFoundError:
        __version__ = "unknown"


# =============================================================================
# Parsing & Validation Helpers
# =============================================================================

class ParseMode(str, Enum):
    """Defines the strictness level for data ingestion."""
    STRICT = "strict"   # Raise errors immediately (Control Plane / Execution)
    LENIENT = "lenient" # Log errors to Extras and continue (Data Plane / Logging)


def as_parse_mode(mode: Union["ParseMode", str, None]) -> "ParseMode":
    """Normalize a string or enum into a valid ParseMode. Defaults to STRICT."""
    if isinstance(mode, ParseMode):
        return mode
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m == "lenient":
            return ParseMode.LENIENT
        if m == "strict":
            return ParseMode.STRICT
    return ParseMode.STRICT


def is_strict(mode: Union["ParseMode", str, None]) -> bool:
    """Helper to check if the effective mode is STRICT."""
    return as_parse_mode(mode) == ParseMode.STRICT


def note_or_raise(extra: Optional["Extras"], key: str, exc: Exception, *, mode: Union["ParseMode", str, None] = ParseMode.STRICT, raw: Any = None) -> None:
    """Handle a validation error according to the ParseMode.

    In STRICT mode: Raises the exception immediately to prevent unsafe execution.
    In LENIENT mode: Catches the exception, records it in `extra.notes`,
                     and optionally saves the `raw` value in `extra.raw` for debugging.
    """
    if is_strict(mode):
        raise exc
    if extra is None:
        return
    try:
        if raw is not None:
            extra.raw[key] = _jsonable(raw)
        extra.notes[key] = {"error": repr(exc)}
    except Exception:
        pass


# =============================================================================
# Unit Handling (Pint Integration)
# =============================================================================

try:
    from pint import UnitRegistry
except ImportError as e:
    raise ImportError(
        "Dependency missing: 'pint' is required. Install it with `pip install pint`."
    ) from e

# Initialize central registry
ureg = UnitRegistry()
Q_ = ureg.Quantity

# Quantity type import is version-dependent across Pint releases.
try:  # Pint >= 0.20 often exposes Quantity at top-level
    from pint import Quantity  # type: ignore
except Exception:
    try:
        from pint.facets.plain.quantity import Quantity  # type: ignore
    except ImportError:
        # Fallback for very old/new structures if facets path changes
        Quantity = type(Q_(1, "nm"))

def ensure_quantity(value: Any, unit: str) -> Optional["Quantity"]:
    """Coerce arbitrary input into a Pint Quantity with the target unit.

    This function acts as a firewall against ambiguous units.
    It handles:
    - Pint Objects: Converts them to the target unit (e.g. 1000V -> 1kV).
    - Dicts: Parses {"value": 1, "unit": "nm"} structures.
    - Strings: Parses "10 nm" or "5 degree".
    - Numbers: Assumes the target unit (legacy behavior).

    Returns None if parsing fails, allowing the caller to decide whether to raise
    an error (Strict) or ignore it (Lenient).
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None

    # OPTIMIZATION: Fast path for plain numbers
    # Avoids the overhead of the Pint string parser for standard inputs.
    if isinstance(value, (int, float, np.number)):
        # Trust that raw numbers are already in the target base unit
        return Q_(float(value), unit)

    try:
        # 1. Handle existing Pint Quantities (safe conversion)
        if isinstance(value, Quantity):
            q = Q_(value.magnitude, str(value.units))
            return q.to(unit)

        # 2. Handle Dictionary representations (e.g. from JSON)
        if isinstance(value, dict):
            mag = value.get("magnitude", value.get("value", None))
            u = value.get("unit", value.get("units", None))
            if mag is None or isinstance(mag, (bool, np.bool_)):
                return None
            q = Q_(mag, u) if u else Q_(mag, unit)
            return q.to(unit)

        # 3. Handle Strings (parse unit if present)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            # Try numeric parsing first (e.g. "100")
            try:
                q = Q_(float(s), unit)
                return q.to(unit)
            except Exception:
                pass
            # Slow path: full string parsing (e.g., "5.2 nm")
            q = Q_(s)
            return q.to(unit)

        # Fallback
        q = Q_(float(value), unit)
        return q.to(unit)

    except Exception:
        return None

def serialize_quantity(q: Optional["Quantity"], target_unit: str) -> Optional[float]:
    """Convert a Quantity to a plain float magnitude in the target unit.

    This strips the unit information for safe JSON serialization.
    Example: serialize_quantity(Q_(300, 'kV'), 'V') -> 300000.0
    """
    if q is None:
        return None
    try:
        if not isinstance(q, Quantity):
            # Fallback if a float crept in somehow
            return float(q)
        return float(q.to(target_unit).magnitude)
    except Exception:
        return None

def _check_data_format(data: np.ndarray) -> bool:
    """Validate if numpy array is a valid 2D image (uint8/uint16)."""
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[2] == 1:
            data = data[:, :, 0]
    if data.ndim != 2:
        return False
    return (data.dtype.kind == "u") and (data.dtype.itemsize in (1, 2))


# =============================================================================
# Extras Container
# =============================================================================

@dataclass
class Extras:
    """Structured container for non-standard data.

    This class supports the 'Lenient Parsing' philosophy. Any data that doesn't
    fit the strict schema ends up here for later inspection instead of causing a crash.

    Attributes:
        vendor: Namespaced storage for vendor-specific extensions.
        unknown: Storage for JSON keys not recognized by the schema.
        raw: Original raw values that failed type coercion/validation.
        notes: Error messages or warnings generated during parsing.
    """
    vendor: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unknown: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.vendor or self.unknown or self.raw or self.notes)

    def to_native_dict(self) -> Dict[str, Any]:
        """Return a deep-copied native-python representation."""
        out: Dict[str, Any] = {}
        if self.vendor:
            out["vendor"] = deepcopy(self.vendor)
        if self.unknown:
            out["unknown"] = deepcopy(self.unknown)
        if self.raw:
            out["raw"] = deepcopy(self.raw)
        if self.notes:
            out["notes"] = deepcopy(self.notes)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe payload representation."""
        return _jsonable(self.to_native_dict())

    @staticmethod
    def from_any(value: Any, *, owner: str = "unknown") -> "Extras":
        """Intelligently parse 'extra' fields from various inputs."""
        if value is None:
            return Extras()
        if isinstance(value, Extras):
            return value
        if isinstance(value, dict):
            # Check if this is already a structured Extras dict (has keys like 'vendor', 'notes')
            known_buckets = {"vendor", "unknown", "raw", "notes"}
            keys = set(value.keys())
            if keys and keys.issubset(known_buckets):
                ex = Extras()
                if "vendor" in value:
                    v = value["vendor"]
                    if isinstance(v, dict):
                        for vend, payload in v.items():
                            if isinstance(payload, dict): ex.vendor[str(vend)] = deepcopy(payload)
                            else: ex.vendor[str(vend)] = {"_value": deepcopy(payload)}
                    elif v is not None: ex.raw[f"{owner}.extra.vendor"] = deepcopy(v)
                if "unknown" in value: ex.unknown = deepcopy(value["unknown"]) if isinstance(value["unknown"], dict) else {}
                if "raw" in value: ex.raw = deepcopy(value["raw"]) if isinstance(value["raw"], dict) else {}
                if "notes" in value: ex.notes = deepcopy(value["notes"]) if isinstance(value["notes"], dict) else {}
                return ex
            elif "vendor" in keys or "unknown" in keys:
                # Partial match logic
                ex = Extras()
                ex.vendor = deepcopy(value.get("vendor", {}))
                ex.unknown = deepcopy(value.get("unknown", {}))
                ex.raw = deepcopy(value.get("raw", {}))
                ex.notes = deepcopy(value.get("notes", {}))
                return ex

            # If it's just a flat dict, treat the whole thing as 'unknown' properties
            ex = Extras()
            try: ex.unknown = deepcopy(value)
            except Exception: ex.raw[f"{owner}.extra"] = repr(value)
            return ex

        # Fallback: treat scalar values as raw garbage
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(value)
        return ex


def _extra_put_raw(extra: Any, key: str, value: Any) -> None:
    if extra is None:
        return
    if isinstance(extra, Extras):
        extra.raw[key] = value
        return
    if isinstance(extra, dict):
        extra[f"{key}_raw"] = value

def collect_extra(d: Optional[Dict[str, Any]], known: Iterable[str], *, owner: str = "unknown") -> Extras:
    """Harvest unknown keys from a source dict into an Extras object.

    This ensures forward compatibility: if the hardware sends new fields we don't
    recognize yet, we preserve them in 'unknown' rather than discarding them.
    """
    if not isinstance(d, dict):
        return Extras()
    known_set = set(known)
    ex = normalize_extra_lenient(d.get("extra", None), owner)
    for k, v in d.items():
        if k == "extra":
            continue
        if k not in known_set:
            try:
                ex.unknown[str(k)] = deepcopy(v)
            except Exception:
                ex.unknown[str(k)] = repr(v)
    return ex


def add_extra_if_any(out: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    """Append serialized extras to the output dict if not empty."""
    if extra is None:
        return out
    if isinstance(extra, Extras):
        payload = extra.to_dict()
    else:
        payload = Extras.from_any(extra).to_dict()
    payload = {k: v for k, v in payload.items() if v}
    if payload:
        out["extra"] = payload
    return out


def drop_none_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in out.items() if v is not None}


def normalize_extra(extra: Any) -> Extras:
    """Strictly convert dict/None to Extras."""
    if extra is None:
        return Extras()
    if isinstance(extra, Extras):
        return extra
    if isinstance(extra, dict):
        return Extras.from_any(extra)
    raise TypeError(f"extra must be Extras, dict, or None, got {type(extra)}")


def normalize_extra_lenient(extra: Any, owner: str) -> Extras:
    """Safely convert anything to Extras, capturing garbage in 'raw'."""
    try:
        return normalize_extra(extra)
    except Exception:
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(extra)
        return ex

def _deep_merge_dict_inplace(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if k in dst and isinstance(dst.get(k), dict) and isinstance(v, dict):
            _deep_merge_dict_inplace(dst[k], v)  # type: ignore[arg-type]
        else:
            dst[k] = deepcopy(v)
    return dst

def merge_extras(dst: Extras, src: Any, *, owner: str) -> Extras:
    """Merge src into dst, preserving dst's existing data where possible."""
    s = Extras.from_any(src, owner=owner)
    for vend, payload in s.vendor.items():
        if vend in dst.vendor and isinstance(dst.vendor.get(vend), dict) and isinstance(payload, dict):
            _deep_merge_dict_inplace(dst.vendor[vend], payload)
        else:
            dst.vendor[vend] = deepcopy(payload)
    dst.unknown.update(deepcopy(s.unknown))
    dst.raw.update(deepcopy(s.raw))
    dst.notes.update(deepcopy(s.notes))
    return dst


# =============================================================================
# Type Parsers
# =============================================================================

def parse_bool_like(value: Any, default: bool = False, *, strict: bool = False) -> bool:
    """Strictly or leniently parse a boolean-like value.

    Handles strings like 'on', 'yes', '1', 'true'.
    Strict mode raises TypeError/ValueError on invalid input.
    Lenient mode returns the default.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        if strict:
            raise ValueError(f"Invalid boolean string: {value!r}")
        return default
    if strict:
        raise TypeError(f"Invalid boolean type: {type(value)}")
    return bool(value)

def parse_optional_int_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[int]:
    """Parse a value into an integer, or return None.

    Rejects bools (True != 1). Accepts integer-floats (1.0 -> 1).
    Captures raw value in extra if lenient parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise TypeError(f"{name} must be int-like, got bool")
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            f = float(value)
            if f.is_integer():
                return int(f)
            raise ValueError(f"{name} must be an integer value, got {value!r}")
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            f = float(s)
            if f.is_integer():
                return int(f)
            raise ValueError(f"{name} must be an integer value, got {value!r}")
        raise TypeError(f"{name} must be int/float/str, got {type(value)}")
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None


def parse_optional_float_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[float]:
    """Parse a value into a float, or return None."""
    if value is None:
        return None
    if isinstance(value, bool):
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise TypeError(f"{name} must be float-like, got bool")
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            return float(s)
        raise TypeError(f"{name} must be float/int/str, got {type(value)}")
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None


def parse_optional_bool_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[bool]:
    """Parse a value into a bool or None (tristate logic).

    Used for capabilities where 'None' implies "Unknown/Not Reported".
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return parse_bool_like(value, default=False, strict=True)
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None


def parse_optional_str_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[str]:
    """Parse a value into a non-empty string or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if strict:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        raise TypeError(f"{name} must be str-like, got {type(value)}")
    if isinstance(value, bool):
        if extra is not None:
            _extra_put_raw(extra, name, value)
        return None
    try:
        s = str(value).strip()
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        return None
    return s or None

def parse_optional_pair_int_like(value: Any, *, name: str, sort: bool = False, strict: bool = False, extra: Any = None) -> Optional[Tuple[int, int]]:
    """Parse a 2-element sequence into a tuple of ints."""
    if value is None:
        return None
    try:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(f"{name} must be a 2-tuple/list, got {value!r}")
        a = parse_optional_int_like(value[0], name=f"{name}[0]", strict=True)
        b = parse_optional_int_like(value[1], name=f"{name}[1]", strict=True)
        if a is None or b is None:
            raise ValueError(f"{name} contains None: {value!r}")
        if sort and a > b:
            a, b = b, a
        return (int(a), int(b))
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None

def parse_optional_pair_float_like(value: Any, *, name: str, sort: bool = False, strict: bool = False, extra: Any = None,) -> Optional[Tuple[float, float]]:
    """Parse a 2-element sequence into a tuple of floats."""
    if value is None:
        return None
    try:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(f"{name} must be a 2-tuple/list, got {value!r}")
        a = parse_optional_float_like(value[0], name=f"{name}[0]", strict=True)
        b = parse_optional_float_like(value[1], name=f"{name}[1]", strict=True)
        if a is None or b is None:
            raise ValueError(f"{name} contains None: {value!r}")
        if sort and a > b:
            a, b = b, a
        return (float(a), float(b))
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None

def parse_optional_id_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[str]:
    """Parse a value into a safe string ID.

    Logs a warning note if the resulting ID is an empty string, as this usually
    indicates a misconfiguration in the source.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        if extra is not None:
            _extra_put_raw(extra, name, value)
            try:
                extra.notes.setdefault("empty_id_fields", []).append(name)
            except Exception:
                pass
        return None
    return parse_optional_str_like(value, name=name, strict=strict, extra=extra)

T = TypeVar("T")

def maybe_from_dict(
    cls: Type[T],
    raw: Any,
    *,
    mode: Union[ParseMode, str, None] = ParseMode.LENIENT,
    extra: Optional["Extras"] = None,
    key: str = "",
    allow_empty_dict: bool = False,
) -> Optional[T]:
    """Generic helper to instantiate a Dataclass from a dict (or list/tuple) safely.

    This function handles the 'Maybe' pattern common in parsing:
    - If input is None -> return None.
    - If input is wrong type -> Raise (Strict) or Log (Lenient).
    - If input is valid -> Recursively parse.
    """
    mode = as_parse_mode(mode)
    if raw is None:
        return None
    if isinstance(raw, dict) and (not raw) and (not allow_empty_dict):
        return None
    try:
        if isinstance(raw, cls):
            try:
                if is_dataclass(raw):
                    return replace(raw, _mode=mode)  # type: ignore[call-arg]
            except Exception:
                pass
            return raw
    except TypeError:
        pass

    # Expanded type check to allow lists/tuples if the target class can handle them
    if not isinstance(raw, (dict, list, tuple)):
         note_or_raise(
            extra,
            key or f"{getattr(cls, '__name__', 'object')}",
            TypeError(f"expected dict/list/tuple for {getattr(cls, '__name__', 'object')}, got {type(raw)}"),
            mode=mode,
            raw=raw,
        )
         return None

    from_dict = getattr(cls, "from_dict", None)
    if not callable(from_dict):
        # Fallback: if it's a dict and the class is a basic dataclass without from_dict
        if is_dataclass(cls) and isinstance(raw, dict):
             try:
                 return cls(**raw) # type: ignore
             except Exception as e:
                 note_or_raise(extra, key or f"{getattr(cls, '__name__', 'object')}", e, mode=mode, raw=raw)
                 return None
        return None

    try:
        try:
            return from_dict(raw, mode=mode)  # type: ignore[misc]
        except TypeError:
            return from_dict(raw)  # type: ignore[misc]
    except Exception as e:
        note_or_raise(extra, key or f"{getattr(cls, '__name__', 'object')}", e, mode=mode, raw=raw)
        return None


def _jsonable(obj: Any) -> Any:
    """Recursively convert object to JSON-safe primitives (dicts/lists/floats)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:
        pass
    try:
        if isinstance(obj, Quantity):
            # Pint objects become explicit dicts for serialization
            return {"magnitude": float(obj.magnitude), "unit": str(obj.units)}
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _setup_init(obj: Any, mode_input: Any, owner_name: str) -> Tuple[ParseMode, bool, Extras]:
    """Reduce boilerplate in __post_init__ methods.

    Returns:
        mode: The resolved ParseMode (Strict/Lenient).
        strict: Boolean flag for convenience (True if mode is Strict).
        extra: The normalized Extras container.
    """
    mode = as_parse_mode(mode_input)
    strict = is_strict(mode)
    extra = normalize_extra(obj.extra) if strict else normalize_extra_lenient(obj.extra, owner_name)
    return mode, strict, extra


# =============================================================================
# Structures (Dataclasses)
# =============================================================================

@dataclass
class Point:
    """
    A 3D coordinate vector with an optional label.

    Used to represent beam shifts, stigmation vectors, and logical coordinates.

    Attributes:
        x: X-axis component.
        y: Y-axis component.
        z: Z-axis component (defaults to 0.0 for 2D vectors).
        name: Optional label (e.g., "center", "stigmator_a").

    Notes:
        This class is lightweight and does not include the `Extras` container.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    name: Optional[str] = None
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        # NORMALIZATION: Strict type coercion (float). Defaults None -> 0.0.
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        def _norm(v, name):
            out = parse_optional_float_like(v, name=name, strict=strict)
            return float(out) if out is not None else 0.0

        self.x = _norm(self.x, "Point.x")
        self.y = _norm(self.y, "Point.y")
        self.z = _norm(self.z, "Point.z")
        self.name = parse_optional_str_like(self.name, name="Point.name", strict=False)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Semantic check (Finiteness).
        mode = as_parse_mode(self._mode if mode is None else mode)
        if not (math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)):
            note_or_raise(
                None, "Point.coordinates",
                ValueError(f"Coordinates must be finite: x={self.x}, y={self.y}, z={self.z}"),
                mode=mode, raw={"x": self.x, "y": self.y, "z": self.z}
            )
            return False
        return True

    def to_dict(self) -> dict:
        return _jsonable(drop_none_keys({"x": self.x, "y": self.y, "z": self.z, "name": self.name}))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Point":
        mode = as_parse_mode(mode)
        if isinstance(d, Point):
            return replace(d, _mode=mode)
        if isinstance(d, dict):
            return Point(
                x=d.get("x"), y=d.get("y"), z=d.get("z"),
                name=d.get("name"), _mode=mode
            )
        if isinstance(d, (list, tuple)) and len(d) in (2, 3):
            return Point(x=d[0], y=d[1], z=d[2] if len(d) == 3 else 0.0, _mode=mode)
        return Point(_mode=mode)

    def to_list(self) -> list:
        return [self.x, self.y, self.z]


@dataclass
class ROI:
    """
    Defines a rectangular Region of Interest on a detector.

    Specifies the offset and dimensions for image acquisition relative to the full sensor.

    Attributes:
        x: Horizontal offset from the left edge (0-indexed).
        y: Vertical offset from the top edge (0-indexed).
        width: Width of the region in pixels.
        height: Height of the region in pixels.

    Notes:
        In `STRICT` mode, non-positive dimensions raise specific validation errors.
        In `LENIENT` mode, invalid dimensions are auto-corrected to defaults to ensure continuity.
    """
    x: int = 0
    y: int = 0
    width: int = 512
    height: int = 512
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        # NORMALIZATION: Type coercion and Default Injection.
        # CRITICAL: We must distinguish between None (missing) and 0 (invalid value).
        mode, strict, self.extra = _setup_init(self, self._mode, "ROI")

        def _norm(val, name, default):
            v = parse_optional_int_like(val, name=name, strict=strict, extra=self.extra)
            return v if v is not None else default

        self.x = _norm(self.x, "ROI.x", 0)
        self.y = _norm(self.y, "ROI.y", 0)
        self.width = _norm(self.width, "ROI.width", 512)
        self.height = _norm(self.height, "ROI.height", 512)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Semantic bounds check and Healing.
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        was_valid = True

        # Rule 1: Origin must be non-negative
        if self.x < 0 or self.y < 0:
            was_valid = False
            note_or_raise(
                self.extra, "ROI.xy", ValueError(f"ROI.x/ROI.y must be >= 0, got x={self.x}, y={self.y}"),
                mode=mode, raw={"x": self.x, "y": self.y},
            )
            if not strict:
                self.x = max(self.x, 0)
                self.y = max(self.y, 0)

        # Rule 2: Dimensions must be positive
        if (self.width <= 0) or (self.height <= 0):
            was_valid = False
            note_or_raise(
                self.extra, "ROI.size", ValueError(f"ROI.width/ROI.height must be > 0, got width={self.width}, height={self.height}"),
                mode=mode, raw={"width": self.width, "height": self.height},
            )
            if not strict:
                # Heal to safe defaults
                self.width = 512 if self.width <= 0 else self.width
                self.height = 512 if self.height <= 0 else self.height
        return was_valid

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ROI":
        mode = as_parse_mode(mode)
        if isinstance(d, ROI):
            try:
                if is_dataclass(d): return replace(d, _mode=mode)
            except Exception: pass
            return d
        ex = Extras() if is_strict(mode) else normalize_extra_lenient(None, "ROI")
        if d is None:
            return ROI(extra=ex, _mode=mode)
        if isinstance(d, (list, tuple)):
            if len(d) != 4:
                note_or_raise(ex, "ROI", ValueError(f"ROI list/tuple must have len 4, got {len(d)}"), mode=mode, raw=deepcopy(d))
                return ROI(extra=ex, _mode=mode)
            return ROI(x=d[0], y=d[1], width=d[2], height=d[3], extra=ex, _mode=mode)
        if not isinstance(d, dict):
            note_or_raise(ex, "ROI", TypeError(f"ROI must be dict/list/tuple/ROI, got {type(d)}"), mode=mode, raw=deepcopy(d))
            return ROI(extra=ex, _mode=mode)
        ex = collect_extra(d, known=("x", "y", "width", "height", "w", "h"), owner="ROI")
        return ROI(
            x=d.get("x"), y=d.get("y"),
            width=d.get("width", d.get("w")),
            height=d.get("height", d.get("h")),
            extra=ex, _mode=mode,
        )

@dataclass
class StagePosition:
    """
    Represents a 5-axis microscope stage position with physical units.

    Stores coordinates as Pint Quantities to ensure unit safety (e.g., meters vs nanometers).
    Supports vector arithmetic for calculating relative movements.

    Attributes:
        name: Optional label for this position (e.g., "Sample Center").
        x: Physical X-axis position (Length).
        y: Physical Y-axis position (Length).
        z: Physical Z-axis height (Length).
        r: Stage rotation (Angle).
        tilt_x: Alpha tilt (Angle).
        tilt_y: Beta tilt (Angle).
        coordinate_system: Label for the reference frame (e.g., "Raw", "Cartesian").
    """
    name: Optional[str] = None
    x: Optional["Quantity"] = None
    y: Optional["Quantity"] = None
    z: Optional["Quantity"] = None
    r: Optional["Quantity"] = None
    tilt_x: Optional["Quantity"] = None
    tilt_y: Optional["Quantity"] = None
    coordinate_system: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        # NORMALIZATION: Parse strings/dicts into Pint Quantities.
        mode, strict, self.extra = _setup_init(self, self._mode, "StagePosition")
        self.name = parse_optional_str_like(self.name, name="StagePosition.name", strict=strict, extra=self.extra)
        self.coordinate_system = parse_optional_str_like(self.coordinate_system, name="StagePosition.coordinate_system", strict=strict, extra=self.extra)

        def coerce_axis(raw: Any, unit: str, field_name: str) -> Optional["Quantity"]:
            q = ensure_quantity(raw, unit)
            if (raw is not None) and (q is None):
                note_or_raise(self.extra, field_name, ValueError(f"{field_name} must be convertible to {unit}, got {raw!r}"), mode=mode, raw=raw)
                return None
            return q

        self.x = coerce_axis(self.x, "nanometer", "StagePosition.x")
        self.y = coerce_axis(self.y, "nanometer", "StagePosition.y")
        self.z = coerce_axis(self.z, "nanometer", "StagePosition.z")
        self.r = coerce_axis(self.r, "degree", "StagePosition.r")
        self.tilt_x = coerce_axis(self.tilt_x, "degree", "StagePosition.tilt_x")
        self.tilt_y = coerce_axis(self.tilt_y, "degree", "StagePosition.tilt_y")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Check finiteness. Trust types from post_init.
        mode = as_parse_mode(self._mode if mode is None else mode)
        is_valid = True

        def check_axis(q: Any, field_name: str) -> bool:
            if q is None: return True
            mag = float(q.magnitude)
            if not math.isfinite(mag):
                note_or_raise(self.extra, field_name, ValueError(f"{field_name} magnitude must be finite"), mode=mode,
                              raw=mag)
                return False
            return True

        is_valid = check_axis(self.x, "StagePosition.x") and is_valid
        is_valid = check_axis(self.y, "StagePosition.y") and is_valid
        is_valid = check_axis(self.z, "StagePosition.z") and is_valid
        is_valid = check_axis(self.r, "StagePosition.r") and is_valid
        is_valid = check_axis(self.tilt_x, "StagePosition.tilt_x") and is_valid
        is_valid = check_axis(self.tilt_y, "StagePosition.tilt_y") and is_valid

        return is_valid

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "x_nm": serialize_quantity(self.x, "nm"),
            "y_nm": serialize_quantity(self.y, "nm"),
            "z_nm": serialize_quantity(self.z, "nm"),
            "r_deg": serialize_quantity(self.r, "degree"),
            "tilt_x_deg": serialize_quantity(self.tilt_x, "degree"),
            "tilt_y_deg": serialize_quantity(self.tilt_y, "degree"),
            "coordinate_system": self.coordinate_system,
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> 'StagePosition':
        mode = as_parse_mode(mode)
        strict = is_strict(mode)
        if isinstance(d, StagePosition):
            try:
                if is_dataclass(d): return replace(d, _mode=mode)
            except Exception: pass
            return d
        if d is None: return StagePosition(_mode=mode)
        if not isinstance(d, dict):
            if strict: raise TypeError(f"StagePosition.from_dict expects a dict, got {type(d)}")
            return StagePosition(_mode=mode)
        ex = collect_extra(d, known=("name", "x", "y", "z", "r", "tilt_x", "tilt_y", "x_nm", "y_nm", "z_nm", "r_deg", "tilt_x_deg", "tilt_y_deg", "coordinate_system", "coord_system", "cs", "extra"), owner="StagePosition")
        return StagePosition(
            name=d.get("name", None),
            x=d.get("x", d.get("x_nm")),
            y=d.get("y", d.get("y_nm")),
            z=d.get("z", d.get("z_nm")),
            r=d.get("r", d.get("r_deg")),
            tilt_x=d.get("tilt_x", d.get("tilt_x_deg")),
            tilt_y=d.get("tilt_y", d.get("tilt_y_deg")),
            coordinate_system=d.get("coordinate_system", d.get("coord_system", d.get("cs", None))),
            extra=ex, _mode=mode,
        )

    def __add__(self, other: 'StagePosition') -> 'StagePosition':
        """Enable vector addition for relative movements."""
        if not isinstance(other, StagePosition): return NotImplemented
        def add_axis(a, b, unit: str):
            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)
            if qa is None and qb is None: return None
            if qa is None: return qb
            if qb is None: return qa
            return qa + qb
        return StagePosition(
            name=self.name,
            x=add_axis(self.x, other.x, "nanometer"),
            y=add_axis(self.y, other.y, "nanometer"),
            z=add_axis(self.z, other.z, "nanometer"),
            r=add_axis(self.r, other.r, "degree"),
            tilt_x=add_axis(self.tilt_x, other.tilt_x, "degree"),
            tilt_y=add_axis(self.tilt_y, other.tilt_y, "degree"),
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: 'StagePosition') -> 'StagePosition':
        if not isinstance(other, StagePosition): return NotImplemented
        def sub_axis(a, b, unit: str):
            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)
            if qa is None and qb is None: return None
            if qa is None: return -qb if qb is not None else None
            if qb is None: return qa
            return qa - qb
        return StagePosition(
            name=self.name,
            x=sub_axis(self.x, other.x, "nanometer"),
            y=sub_axis(self.y, other.y, "nanometer"),
            z=sub_axis(self.z, other.z, "nanometer"),
            r=sub_axis(self.r, other.r, "degree"),
            tilt_x=sub_axis(self.tilt_x, other.tilt_x, "degree"),
            tilt_y=sub_axis(self.tilt_y, other.tilt_y, "degree"),
            coordinate_system=self.coordinate_system,
        )

    def is_close(self, other: 'StagePosition', tol_nm: float = 1.0, tol_deg: float = 1e-3, *, compare_only_specified: bool = True) -> bool:
        def close_axis(a, b, unit: str, tol: float) -> bool:
            if compare_only_specified and (a is None or b is None): return True
            if a is None or b is None: return False
            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)
            if qa is None or qb is None: return False
            da = abs(qa - qb)
            return float(da.m_as(unit)) <= float(tol)
        return (close_axis(self.x, other.x, "nanometer", tol_nm) and
                close_axis(self.y, other.y, "nanometer", tol_nm) and
                close_axis(self.z, other.z, "nanometer", tol_nm) and
                close_axis(self.r, other.r, "degree", tol_deg) and
                close_axis(self.tilt_x, other.tilt_x, "degree", tol_deg) and
                close_axis(self.tilt_y, other.tilt_y, "degree", tol_deg))

@dataclass
class StageSystemSettings:
    """
    Configuration and safety limits for the microscope stage.

    Defines enabled axes, movement boundaries, and step size limits to ensure hardware safety.

    Attributes:
        enabled: Master switch to enable/disable stage control.
        can_*: Capability flags for specific axes (x, y, z, r, tilt).
        *_limits: Tuple of (min, max) Quantities defining the allowable range for each axis.
        max_step_distance: Safety limit for the largest single lateral move allowed.
        max_step_angle: Safety limit for the largest single tilt/rotation move allowed.
        eucentric_z: The calibrated Z-height where the sample is at the eucentric plane.
        settle_time_s: Time to wait for stabilization after movement.
        timeout_s: Maximum duration to wait for a movement command.
    """
    enabled: bool = True
    can_x: bool = True
    can_y: bool = True
    can_z: bool = True
    can_r: bool = False
    can_tilt_x: bool = False
    can_tilt_y: bool = False

    x_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    y_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    z_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    r_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    tilt_x_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    tilt_y_limits: Optional[Tuple["Quantity", "Quantity"]] = None

    max_step_distance: "Quantity" = field(default_factory=lambda: Q_(50000.0, "nm"))
    max_step_angle: "Quantity" = field(default_factory=lambda: Q_(1.0, "degree"))
    eucentric_z: Optional["Quantity"] = None
    settle_time_s: float = 0.2
    timeout_s: float = 10.0
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "StageSystemSettings")

        self.enabled = parse_bool_like(self.enabled, default=True)
        self.can_x = parse_bool_like(self.can_x, default=True)
        self.can_y = parse_bool_like(self.can_y, default=True)
        self.can_z = parse_bool_like(self.can_z, default=True)
        self.can_r = parse_bool_like(self.can_r, default=False)
        self.can_tilt_x = parse_bool_like(self.can_tilt_x, default=False)
        self.can_tilt_y = parse_bool_like(self.can_tilt_y, default=False)

        def _lim(val, unit):
            if val is None: return None
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (ensure_quantity(val[0], unit), ensure_quantity(val[1], unit))
            return None

        self.x_limits = _lim(self.x_limits, "nm")
        self.y_limits = _lim(self.y_limits, "nm")
        self.z_limits = _lim(self.z_limits, "nm")
        self.r_limits = _lim(self.r_limits, "degree")
        self.tilt_x_limits = _lim(self.tilt_x_limits, "degree")
        self.tilt_y_limits = _lim(self.tilt_y_limits, "degree")

        self.max_step_distance = ensure_quantity(self.max_step_distance, "nm") or Q_(50000.0, "nm")
        self.max_step_angle = ensure_quantity(self.max_step_angle, "degree") or Q_(1.0, "degree")
        self.eucentric_z = ensure_quantity(self.eucentric_z, "nm")

        self.settle_time_s = parse_optional_float_like(self.settle_time_s, name="settle_time_s", strict=strict, extra=self.extra) or 0.2
        self.timeout_s = parse_optional_float_like(self.timeout_s, name="timeout_s", strict=strict, extra=self.extra) or 10.0

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        ok = True

        def _check_range(lims, name):
            if lims:
                mn, mx = lims
                if mn > mx:
                    note_or_raise(self.extra, f"StageSystemSettings.{name}_limits", ValueError(f"{name} limits invalid: min > max ({mn} > {mx})"), mode=mode)
                    if not strict:
                        try: setattr(self, f"{name}_limits", (mx, mn))
                        except Exception: pass
                    return False
            return True

        ok = _check_range(self.x_limits, "x") and ok
        ok = _check_range(self.y_limits, "y") and ok
        ok = _check_range(self.z_limits, "z") and ok
        ok = _check_range(self.r_limits, "r") and ok
        ok = _check_range(self.tilt_x_limits, "tilt_x") and ok
        ok = _check_range(self.tilt_y_limits, "tilt_y") and ok

        if self.max_step_distance.magnitude <= 0:
            note_or_raise(self.extra, "StageSystemSettings.max_step_distance", ValueError("max_step_distance must be > 0"), mode=mode)
            ok = False

        if self.eucentric_z is not None and self.z_limits:
            z_min, z_max = self.z_limits
            if not (z_min <= self.eucentric_z <= z_max):
                note_or_raise(self.extra, "StageSystemSettings.eucentric_z", ValueError(f"eucentric_z ({self.eucentric_z}) outside z_limits"), mode=mode)
                ok = False

        if self.settle_time_s < 0:
            note_or_raise(self.extra, "StageSystemSettings.settle_time_s",
                          ValueError("Settle time must be >= 0"), mode=mode)
            ok = False

        if self.timeout_s <= 0:
            note_or_raise(self.extra, "StageSystemSettings.timeout_s",
                          ValueError("Timeout must be > 0"), mode=mode)
            ok = False

        return ok

    def to_dict(self) -> dict:
        def _s_lim(val, u): return [serialize_quantity(v, u) for v in val] if val else None
        d = {
            "enabled": self.enabled,
            "can_x": self.can_x,
            "can_y": self.can_y,
            "can_z": self.can_z,
            "can_r": self.can_r,
            "can_tilt_x": self.can_tilt_x,
            "can_tilt_y": self.can_tilt_y,
            "x_limits_nm": _s_lim(self.x_limits, "nm"),
            "y_limits_nm": _s_lim(self.y_limits, "nm"),
            "z_limits_nm": _s_lim(self.z_limits, "nm"),
            "r_limits_deg": _s_lim(self.r_limits, "degree"),
            "tilt_x_limits_deg": _s_lim(self.tilt_x_limits, "degree"),
            "tilt_y_limits_deg": _s_lim(self.tilt_y_limits, "degree"),
            "max_step_nm": serialize_quantity(self.max_step_distance, "nm"),
            "max_step_deg": serialize_quantity(self.max_step_angle, "degree"),
            "eucentric_z_nm": serialize_quantity(self.eucentric_z, "nm"),
            "settle_time_s": self.settle_time_s,
            "timeout_s": self.timeout_s
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageSystemSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, StageSystemSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return StageSystemSettings(_mode=mode)
        def _get(key, suffix_key, default=None): return d.get(key, d.get(suffix_key, default))
        return StageSystemSettings(
            enabled=d.get("enabled", True),
            can_x=d.get("can_x", True),
            can_y=d.get("can_y", True),
            can_z=d.get("can_z", True),
            can_r=d.get("can_r", False),
            can_tilt_x=d.get("can_tilt_x", False),
            can_tilt_y=d.get("can_tilt_y", False),
            x_limits=_get("x_limits", "x_limits_nm"),
            y_limits=_get("y_limits", "y_limits_nm"),
            z_limits=_get("z_limits", "z_limits_nm"),
            r_limits=_get("r_limits", "r_limits_deg"),
            tilt_x_limits=_get("tilt_x_limits", "tilt_x_limits_deg"),
            tilt_y_limits=_get("tilt_y_limits", "tilt_y_limits_deg"),
            max_step_distance=_get("max_step_distance", "max_step_nm"),
            max_step_angle=_get("max_step_angle", "max_step_deg"),
            eucentric_z=_get("eucentric_z", "eucentric_z_nm"),
            settle_time_s=d.get("settle_time_s"),
            timeout_s=d.get("timeout_s"),
            extra=collect_extra(d, (), owner="StageSystemSettings"),
            _mode=mode
        )

@dataclass
class BeamSettings:
    """
    Parameters controlling the electron beam and electromagnetic lenses.

    Encapsulates optical settings including accelerating voltage, current, and lens deflections.

    Attributes:
        voltage: Accelerating voltage (High Tension).
        beam_current: Probe current measured at the specimen or screen.
        spot_size: Discrete index representing the condenser lens combination.
        convergence_angle: Semi-convergence angle of the probe in STEM mode.
        defocus: Deviation from the focal plane (positive usually implies overfocus).
        stigmation: 2D vector controlling stigmator coils.
        beam_shift: 2D vector controlling beam tilt/shift coils.
        image_shift: 2D vector controlling image shift coils.
        scan_rotation: Rotation of the scanning raster.
    """
    voltage: Optional["Quantity"] = None
    beam_current: Optional["Quantity"] = None
    spot_size: Optional[int] = None
    convergence_angle: Optional["Quantity"] = None
    defocus: Optional["Quantity"] = None
    stigmation: Optional[Point] = None
    beam_shift: Optional[Point] = None
    image_shift: Optional[Point] = None
    scan_rotation: Optional["Quantity"] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "BeamSettings")

        def _q(val, unit, name):
            q = ensure_quantity(val, unit)
            if val is not None and q is None:
                note_or_raise(self.extra, name, ValueError(f"Invalid {name}: {val!r}"), mode=mode, raw=val)
            return q

        self.voltage = _q(self.voltage, "kV", "BeamSettings.voltage")
        self.beam_current = _q(self.beam_current, "nA", "BeamSettings.beam_current")
        self.convergence_angle = _q(self.convergence_angle, "mrad", "BeamSettings.convergence_angle")
        self.defocus = _q(self.defocus, "nanometer", "BeamSettings.defocus")
        self.scan_rotation = _q(self.scan_rotation, "degree", "BeamSettings.scan_rotation")
        self.spot_size = parse_optional_int_like(self.spot_size, name="BeamSettings.spot_size", strict=strict,
                                                 extra=self.extra)

        self.stigmation = maybe_from_dict(Point, self.stigmation, extra=self.extra, key="BeamSettings.stigmation", mode=mode)
        self.beam_shift = maybe_from_dict(Point, self.beam_shift, extra=self.extra, key="BeamSettings.beam_shift", mode=mode)
        self.image_shift = maybe_from_dict(Point, self.image_shift, extra=self.extra, key="BeamSettings.image_shift", mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        ok = True
        if self.stigmation: ok = self.stigmation.validate(mode=mode) and ok
        if self.beam_shift: ok = self.beam_shift.validate(mode=mode) and ok
        if self.image_shift: ok = self.image_shift.validate(mode=mode) and ok
        if self.convergence_angle is not None and self.convergence_angle.magnitude < 0:
            note_or_raise(self.extra, "BeamSettings.convergence_angle",
                          ValueError("Convergence angle must be >= 0"), mode=mode)
            ok = False
        if self.voltage is not None and self.voltage.magnitude <= 0:
            note_or_raise(self.extra, "BeamSettings.voltage", ValueError("Voltage must be > 0"), mode=mode)
            ok = False
        if self.beam_current is not None and self.beam_current.magnitude < 0:
            note_or_raise(self.extra, "BeamSettings.beam_current", ValueError("Beam current must be >= 0"), mode=mode)
            ok = False
        if self.spot_size is not None and self.spot_size < 0:
            note_or_raise(self.extra, "BeamSettings.spot_size", ValueError("Spot size must be >= 0"), mode=mode)
            ok = False
        return ok

    def to_dict(self) -> dict:
        d = {
            "voltage_kv": serialize_quantity(self.voltage, "kV"),
            "beam_current_na": serialize_quantity(self.beam_current, "nA"),
            "convergence_angle_mrad": serialize_quantity(self.convergence_angle, "mrad"),
            "defocus_nm": serialize_quantity(self.defocus, "nm"),
            "scan_rotation_deg": serialize_quantity(self.scan_rotation, "degree"),
            "spot_size": self.spot_size,
            "stigmation": self.stigmation.to_dict() if self.stigmation else None,
            "beam_shift": self.beam_shift.to_dict() if self.beam_shift else None,
            "image_shift": self.image_shift.to_dict() if self.image_shift else None,
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, BeamSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return BeamSettings(_mode=mode)

        field_names = {f.name for f in fields(BeamSettings)}
        aliases = {
            "voltage_kv", "accelerating_voltage_kv",
            "beam_current_na", "current",
            "spot",
            "convergence_mrad", "convergence_angle_mrad",
            "defocus_nm",
            "scan_rotation_deg",
            "extra"
        }
        known = field_names | aliases

        extra = collect_extra(d, known, owner="BeamSettings")

        return BeamSettings(
            voltage=d.get("voltage", d.get("voltage_kv", d.get("accelerating_voltage_kv"))),
            beam_current=d.get("beam_current", d.get("beam_current_na", d.get("current"))),
            spot_size=d.get("spot_size", d.get("spot")),
            convergence_angle=d.get("convergence_angle", d.get("convergence_angle_mrad", d.get("convergence_mrad"))),
            defocus=d.get("defocus", d.get("defocus_nm")),
            stigmation=d.get("stigmation"),
            beam_shift=d.get("beam_shift"),
            image_shift=d.get("image_shift"),
            scan_rotation=d.get("scan_rotation", d.get("scan_rotation_deg")),
            extra=extra, _mode=mode
        )

@dataclass
class BeamSystemSettings:
    """
    Operational constraints and defaults for the electron beam.

    Defines safe operating ranges for voltage and current to prevent invalid hardware states.

    Attributes:
        enabled: Master switch to enable/disable beam control.
        default_beam: A safe, default configuration to fallback to.
        voltage_limits: Allowable range (min, max) for accelerating voltage.
        beam_current_limits: Allowable range (min, max) for beam current.
        spot_size_limits: Min/Max valid indices for spot size.
        convergence_angle_limits: Allowable range (min, max) for convergence angle.
    """
    enabled: bool = True
    default_beam: BeamSettings = field(default_factory=BeamSettings)
    voltage_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    beam_current_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    spot_size_limits: Optional[Tuple[int, int]] = None
    convergence_angle_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "BeamSystemSettings")
        self.enabled = parse_bool_like(self.enabled, default=True)
        self.default_beam = maybe_from_dict(BeamSettings, self.default_beam, mode=mode) or BeamSettings(_mode=mode)

        def _lim(val, unit):
            if val is None: return None
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (ensure_quantity(val[0], unit), ensure_quantity(val[1], unit))
            return None

        self.voltage_limits = _lim(self.voltage_limits, "kV")
        self.beam_current_limits = _lim(self.beam_current_limits, "nA")
        self.convergence_angle_limits = _lim(self.convergence_angle_limits, "mrad")
        self.spot_size_limits = parse_optional_pair_int_like(self.spot_size_limits, name="BeamSystemSettings.spot_size_limits", sort=False, strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        ok = self.default_beam.validate(mode=mode)

        def _check(rng, name):
            if rng:
                mn, mx = rng
                if mn > mx:
                    note_or_raise(self.extra, name, ValueError(f"{name} invalid: min > max"), mode=mode)
                    if not strict:
                        try: setattr(self, name, (mx, mn))
                        except Exception: pass
                    return False

                if mn.magnitude < 0:
                    note_or_raise(self.extra, f"BeamSystemSettings.{name}",
                                  ValueError(f"{name} invalid: min < 0"), mode=mode)
                    return False
            return True

        ok = _check(self.voltage_limits, "voltage_limits") and ok
        ok = _check(self.beam_current_limits, "beam_current_limits") and ok
        ok = _check(self.convergence_angle_limits, "convergence_angle_limits") and ok
        ok = _check(self.spot_size_limits, "spot_size_limits") and ok

        return ok

    def to_dict(self) -> dict:
        def _s_lim(val, u): return [serialize_quantity(v, u) for v in val] if val else None
        d = {
            "enabled": self.enabled,
            "default_beam": self.default_beam.to_dict(),
            "voltage_limits_kv": _s_lim(self.voltage_limits, "kV"),
            "beam_current_limits_na": _s_lim(self.beam_current_limits, "nA"),
            "spot_size_limits": self.spot_size_limits,
            "convergence_angle_limits_mrad": _s_lim(self.convergence_angle_limits, "mrad"),
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSystemSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, BeamSystemSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return BeamSystemSettings(_mode=mode)

        known = {"enabled", "default_beam", "voltage_limits", "voltage_limits_kv", "voltage_range", "voltage_range_kv",
                 "beam_current_limits", "beam_current_limits_na", "beam_current_range", "beam_current_range_na",
                 "spot_size_limits", "spot_size_range",
                 "convergence_angle_limits", "convergence_angle_limits_mrad", "convergence_angle_range", "extra"}
        extra = collect_extra(d, known, owner="BeamSystemSettings")

        return BeamSystemSettings(
            enabled=d.get("enabled", True),
            default_beam=d.get("default_beam"),
            voltage_limits=d.get("voltage_limits"),
            beam_current_limits=d.get("beam_current_limits"),
            spot_size_limits=d.get("spot_size_limits"),
            convergence_angle_limits=d.get("convergence_angle_limits"),
            extra=extra, _mode=mode
        )

@dataclass
class DetectorSettings:
    """
    Configuration for a single image acquisition.

    Specifies which detector to use and how the image should be captured (exposure, binning, ROI).

    Attributes:
        detector_id: Unique identifier for the camera.
        exposure: Integration time (Time quantity).
        binning_index: Discrete binning level index.
        binning_xy: Explicit (x, y) binning factors.
        roi: Region of Interest to read from the sensor.
        frame_integration: Number of internal frames to accumulate.
        gain_index: Index for hardware gain setting.
        offset_index: Index for hardware offset/black-level setting.
        digital_rotation_deg: Rotation applied to the image.

    Notes:
        Strictly validates that exposure is positive and ROI dimensions are safe.
    """
    detector_id: Optional[str] = None
    exposure: Optional["Quantity"] = None  # ms
    binning_index: Optional[int] = None
    binning_xy: Optional[Tuple[int, int]] = None
    roi: Optional[ROI] = None
    frame_integration: Optional[int] = None
    gain_index: Optional[int] = None
    offset_index: Optional[int] = None
    digital_rotation_deg: Optional[float] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        # NORMALIZATION
        mode, strict, self.extra = _setup_init(self, self._mode, "DetectorSettings")

        def _q(val, unit, name):
            q = ensure_quantity(val, unit)
            if val is not None and q is None:
                note_or_raise(self.extra, name, ValueError(f"Invalid {name}: {val!r}"), mode=mode, raw=val)
            return q

        self.exposure = _q(self.exposure, "ms", "DetectorSettings.exposure")
        self.detector_id = parse_optional_id_like(self.detector_id, name="DetectorSettings.detector_id", strict=False, extra=self.extra)

        # Structure normalization: Dict -> Dataclass
        self.roi = maybe_from_dict(ROI, self.roi, mode=mode, extra=self.extra, key="DetectorSettings.roi")

        self.binning_index = parse_optional_int_like(self.binning_index, name="DetectorSettings.binning_index", strict=strict, extra=self.extra)
        self.binning_xy = parse_optional_pair_int_like(self.binning_xy, name="DetectorSettings.binning_xy", sort=False, strict=strict, extra=self.extra)
        self.frame_integration = parse_optional_int_like(self.frame_integration, name="DetectorSettings.frame_integration", strict=strict, extra=self.extra)
        self.gain_index = parse_optional_int_like(self.gain_index, name="DetectorSettings.gain_index", strict=strict, extra=self.extra)
        self.offset_index = parse_optional_int_like(self.offset_index, name="DetectorSettings.offset_index", strict=strict, extra=self.extra)
        self.digital_rotation_deg = parse_optional_float_like(self.digital_rotation_deg, name="DetectorSettings.digital_rotation_deg", strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Semantic logic
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        ok = True

        if self.exposure is not None:
             if self.exposure.magnitude <= 0:
                 note_or_raise(self.extra, "DetectorSettings.exposure", ValueError("Exposure must be > 0"), mode=mode)
                 ok = False

        if self.binning_xy:
             # Semantic check: Binning must be positive integers
             if self.binning_xy[0] <= 0 or self.binning_xy[1] <= 0:
                 note_or_raise(self.extra, "DetectorSettings.binning_xy", ValueError("binning_xy must be >= 0"), mode=mode, raw=self.binning_xy)
                 ok = False
                 if not strict:
                     # Repair: Disable explicit binning if invalid
                     self.binning_xy = None

        if self.frame_integration is not None and self.frame_integration < 1:
            note_or_raise(self.extra, "DetectorSettings.frame_integration",
                          ValueError(f"Frame integration must be >= 1, got {self.frame_integration}"), mode=mode)
            ok = False

        if self.gain_index is not None and self.gain_index < 0:
            note_or_raise(self.extra, "DetectorSettings.gain_index",
                          ValueError("Gain index must be >= 0"), mode=mode)
            ok = False

        if self.roi:
            if not self.roi.validate(mode=mode):
                ok = False
        return ok

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "exposure_ms": serialize_quantity(self.exposure, "ms"),
            "binning_index": self.binning_index,
            "binning_xy": list(self.binning_xy) if self.binning_xy else None,
            "frame_integration": self.frame_integration,
            "gain_index": self.gain_index,
            "offset_index": self.offset_index,
            "digital_rotation_deg": self.digital_rotation_deg,
            "roi": self.roi.to_dict() if self.roi else None,
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, DetectorSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return DetectorSettings(_mode=mode)

        field_names = {f.name for f in fields(DetectorSettings)}
        aliases = {"exposure_ms", "detector_roi", "imaging_area", "extra"}
        known = field_names | aliases

        extra = collect_extra(d, known, owner="DetectorSettings")

        roi_raw = d.get("roi")
        if roi_raw is None:
            for alias in ("detector_roi", "imaging_area"):
                if alias in d:
                    roi_raw = d.get(alias)
                    break

        return DetectorSettings(
            detector_id=d.get("detector_id"),
            exposure=d.get("exposure", d.get("exposure_ms")),
            binning_index=d.get("binning_index"),
            binning_xy=d.get("binning_xy"),
            frame_integration=d.get("frame_integration"),
            roi=roi_raw,
            gain_index=d.get("gain_index"),
            offset_index=d.get("offset_index"),
            digital_rotation_deg=d.get("digital_rotation_deg"),
            extra=extra, _mode=mode
        )

@dataclass
class DetectorCapabilities:
    """
    Read-only hardware capabilities of a specific detector.

    Describes supported ranges (e.g., min/max exposure) and features (e.g., binning) reported by drivers.

    Attributes:
        can_*: Capability flags (binning, gain, offset, rotation).
        *_min/max: Supported ranges for binning, exposure, ROI size, gain, and offset.

    Notes:
        This class is permanently `LENIENT` to safely ingest driver reports without validation errors.
    """
    can_binning: Optional[bool] = None
    binning_index_min: Optional[int] = None
    binning_index_max: Optional[int] = None
    binning_xy_min: Optional[Tuple[int, int]] = None
    binning_xy_max: Optional[Tuple[int, int]] = None
    exposure_ms_min: Optional[float] = None
    exposure_ms_max: Optional[float] = None
    frame_integration_min: Optional[int] = None
    frame_integration_max: Optional[int] = None
    roi_size_min: Optional[Tuple[int, int]] = None
    roi_size_max: Optional[Tuple[int, int]] = None
    can_gain: Optional[bool] = None
    gain_index_min: Optional[int] = None
    gain_index_max: Optional[int] = None
    can_offset: Optional[bool] = None
    offset_index_min: Optional[int] = None
    offset_index_max: Optional[int] = None
    can_digital_rotation: Optional[bool] = None
    digital_rotation_deg_min: Optional[float] = None
    digital_rotation_deg_max: Optional[float] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "DetectorCapabilities")

        # Pairs
        self.binning_xy_min = parse_optional_pair_int_like(self.binning_xy_min, name="DetectorCapabilities.binning_xy_min", strict=False, extra=self.extra)
        self.binning_xy_max = parse_optional_pair_int_like(self.binning_xy_max, name="DetectorCapabilities.binning_xy_max", strict=False, extra=self.extra)
        self.roi_size_min = parse_optional_pair_int_like(self.roi_size_min, name="DetectorCapabilities.roi_size_min", strict=False, extra=self.extra)
        self.roi_size_max = parse_optional_pair_int_like(self.roi_size_max, name="DetectorCapabilities.roi_size_max", strict=False, extra=self.extra)

        # Ints
        self.binning_index_min = parse_optional_int_like(self.binning_index_min, name="DetectorCapabilities.binning_index_min", strict=False, extra=self.extra)
        self.binning_index_max = parse_optional_int_like(self.binning_index_max, name="DetectorCapabilities.binning_index_max", strict=False, extra=self.extra)
        self.frame_integration_min = parse_optional_int_like(self.frame_integration_min, name="DetectorCapabilities.frame_integration_min", strict=False, extra=self.extra)
        self.frame_integration_max = parse_optional_int_like(self.frame_integration_max, name="DetectorCapabilities.frame_integration_max", strict=False, extra=self.extra)
        self.gain_index_min = parse_optional_int_like(self.gain_index_min, name="DetectorCapabilities.gain_index_min", strict=False, extra=self.extra)
        self.gain_index_max = parse_optional_int_like(self.gain_index_max, name="DetectorCapabilities.gain_index_max", strict=False, extra=self.extra)
        self.offset_index_min = parse_optional_int_like(self.offset_index_min, name="DetectorCapabilities.offset_index_min", strict=False, extra=self.extra)
        self.offset_index_max = parse_optional_int_like(self.offset_index_max, name="DetectorCapabilities.offset_index_max", strict=False, extra=self.extra)

        # Floats
        self.exposure_ms_min = parse_optional_float_like(self.exposure_ms_min, name="DetectorCapabilities.exposure_ms_min", strict=False, extra=self.extra)
        self.exposure_ms_max = parse_optional_float_like(self.exposure_ms_max, name="DetectorCapabilities.exposure_ms_max", strict=False, extra=self.extra)
        self.digital_rotation_deg_min = parse_optional_float_like(self.digital_rotation_deg_min, name="DetectorCapabilities.digital_rotation_deg_min", strict=False, extra=self.extra)
        self.digital_rotation_deg_max = parse_optional_float_like(self.digital_rotation_deg_max, name="DetectorCapabilities.digital_rotation_deg_max", strict=False, extra=self.extra)

        # Bools
        self.can_binning = parse_optional_bool_like(self.can_binning, name="DetectorCapabilities.can_binning", strict=False, extra=self.extra)
        self.can_gain = parse_optional_bool_like(self.can_gain, name="DetectorCapabilities.can_gain", strict=False, extra=self.extra)
        self.can_offset = parse_optional_bool_like(self.can_offset, name="DetectorCapabilities.can_offset", strict=False, extra=self.extra)
        self.can_digital_rotation = parse_optional_bool_like(self.can_digital_rotation, name="DetectorCapabilities.can_digital_rotation", strict=False, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Check consistency (min <= max) and non-negativity.
        # We trust types are correct because __post_init__ guaranteed it.
        mode = as_parse_mode(self._mode if mode is None else mode)
        ok = True

        def _check_range(min_val, max_val, name):
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    note_or_raise(self.extra, name, ValueError(f"{name} invalid: min > max ({min_val} > {max_val})"), mode=mode)
                    return False
            return True

        def _check_pos(val, name):
            if val is not None and val < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode, raw=val)
                return False
            return True

        # 1. Range Consistency
        ok = _check_range(self.binning_index_min, self.binning_index_max, "DetectorCapabilities.binning_index") and ok
        ok = _check_range(self.frame_integration_min, self.frame_integration_max, "DetectorCapabilities.frame_integration") and ok
        ok = _check_range(self.gain_index_min, self.gain_index_max, "DetectorCapabilities.gain_index") and ok
        ok = _check_range(self.offset_index_min, self.offset_index_max, "DetectorCapabilities.offset_index") and ok
        ok = _check_range(self.exposure_ms_min, self.exposure_ms_max, "DetectorCapabilities.exposure_ms") and ok

        # 2. Physical Non-negativity
        ok = _check_pos(self.exposure_ms_min, "DetectorCapabilities.exposure_ms_min") and ok

        # 3. Tuple consistency (ROI/Binning)
        if self.roi_size_min and self.roi_size_max:
             if self.roi_size_min[0] > self.roi_size_max[0] or self.roi_size_min[1] > self.roi_size_max[1]:
                 note_or_raise(self.extra, "DetectorCapabilities.roi_size", ValueError("ROI size min > max"), mode=mode)
                 ok = False

        return ok

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {}
        for f in fields(DetectorCapabilities):
            if f.name == "extra": continue
            v = getattr(self, f.name)
            if v is not None: d[f.name] = v
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "DetectorCapabilities":
        mode = as_parse_mode(mode)
        if isinstance(d, DetectorCapabilities): return replace(d, _mode=mode)
        if not isinstance(d, dict): return DetectorCapabilities(_mode=mode)

        known = {f.name for f in fields(DetectorCapabilities)} | {"roi_min", "roi_max", "extra"}
        extra = collect_extra(d, known, owner="DetectorCapabilities")

        # Helper to extract kwargs manually since this is a complex mix of types
        kwargs: Dict[str, Any] = {}
        pair_fields = {"binning_xy_min", "binning_xy_max", "roi_size_min", "roi_size_max"}
        int_fields = {"binning_index_min", "binning_index_max", "frame_integration_min", "frame_integration_max", "gain_index_min", "gain_index_max", "offset_index_min", "offset_index_max"}
        float_fields = {"exposure_ms_min", "exposure_ms_max", "digital_rotation_deg_min", "digital_rotation_deg_max"}
        bool_fields = {"can_binning", "can_gain", "can_offset", "can_digital_rotation"}

        for f in fields(DetectorCapabilities):
            if f.name == "extra" or f.name not in d: continue
            v = d.get(f.name)
            if f.name in pair_fields: kwargs[f.name] = parse_optional_pair_int_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in int_fields: kwargs[f.name] = parse_optional_int_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in float_fields: kwargs[f.name] = parse_optional_float_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in bool_fields: kwargs[f.name] = parse_optional_bool_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            else: kwargs[f.name] = v

        return DetectorCapabilities(**kwargs, extra=extra, _mode=mode)

@dataclass
class DetectorSystemSettings:
    """
    Registry for all available detectors and their configurations.

    Maps detector IDs to their specific settings and hardware capabilities.

    Attributes:
        enabled: Master switch to enable/disable detector control.
        defaults_by_id: Mapping of detector IDs to their default startup settings.
        default_detector_id: The ID of the primary detector to use if none is specified.
        capabilities_by_id: Mapping of detector IDs to their read-only hardware capabilities.
        available_detector_ids: List of all valid detector IDs currently recognized.

    Notes:
        Validates that `default_detector_id` exists within the available detectors.
    """
    enabled: bool = True
    defaults_by_id: Dict[str, DetectorSettings] = field(default_factory=dict)
    default_detector_id: Optional[str] = None
    capabilities_by_id: Dict[str, DetectorCapabilities] = field(default_factory=dict)
    available_detector_ids: List[str] = field(default_factory=list)
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "DetectorSystemSettings")
        self.enabled = parse_bool_like(self.enabled, default=True)

        # Normalize mapping containers
        if self.defaults_by_id is None:
            self.defaults_by_id = {}
        elif not isinstance(self.defaults_by_id, dict):
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.defaults_by_id",
                TypeError(f"defaults_by_id must be dict, got {type(self.defaults_by_id)}"),
                mode=mode,
                raw=deepcopy(self.defaults_by_id),
            )
            self.defaults_by_id = {}

        if self.capabilities_by_id is None:
            self.capabilities_by_id = {}
        elif not isinstance(self.capabilities_by_id, dict):
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.capabilities_by_id",
                TypeError(f"capabilities_by_id must be dict, got {type(self.capabilities_by_id)}"),
                mode=mode,
                raw=deepcopy(self.capabilities_by_id),
            )
            self.capabilities_by_id = {}

        if self.available_detector_ids is None:
            self.available_detector_ids = []
        elif not isinstance(self.available_detector_ids, list):
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.available_detector_ids",
                TypeError(f"available_detector_ids must be list, got {type(self.available_detector_ids)}"),
                mode=mode,
                raw=deepcopy(self.available_detector_ids),
            )
            self.available_detector_ids = []

        # default_detector_id: allow missing; parse to optional id-like
        self.default_detector_id = parse_optional_id_like(
            self.default_detector_id, name="DetectorSystemSettings.default_detector_id", strict=False, extra=self.extra
        )

        # Normalize dict keys to str, and values to proper objects
        new_defaults: Dict[str, DetectorSettings] = {}
        for k, v in list(self.defaults_by_id.items()):
            det_id = parse_optional_id_like(
                k, name="DetectorSystemSettings.defaults_by_id.key", strict=strict, extra=self.extra
            )
            if det_id is None:
                self.extra.notes[f"DetectorSystemSettings.defaults_by_id.{k}_raw_key"] = k
                continue

            if isinstance(v, dict):
                try:
                    v = DetectorSettings.from_dict(v, mode=mode)
                except Exception as e:
                    note_or_raise(self.extra, f"DetectorSystemSettings.defaults_by_id.{det_id}", e, mode=mode, raw=deepcopy(v))
                    continue

            if isinstance(v, DetectorSettings):
                if is_dataclass(v) and getattr(v, "_mode", None) != mode:
                    v = replace(v, _mode=mode)
            else:
                note_or_raise(
                    self.extra,
                    f"DetectorSystemSettings.defaults_by_id.{det_id}",
                    TypeError(f"defaults_by_id['{det_id}'] must be DetectorSettings/dict, got {type(v)}"),
                    mode=mode,
                    raw=deepcopy(v),
                )
                continue

            # Align detector_id with key
            if v.detector_id is None:
                v.detector_id = det_id
            elif str(v.detector_id) != det_id:
                self.extra.notes[f"DetectorSystemSettings.defaults_by_id.{det_id}.detector_id_mismatch"] = {
                    "key": det_id,
                    "detector_id": v.detector_id,
                }
                v.detector_id = det_id

            new_defaults[det_id] = v
        self.defaults_by_id = new_defaults

        new_caps: Dict[str, DetectorCapabilities] = {}
        for k, v in list(self.capabilities_by_id.items()):
            det_id = parse_optional_id_like(
                k, name="DetectorSystemSettings.capabilities_by_id.key", strict=strict, extra=self.extra
            )
            if det_id is None:
                self.extra.notes[f"DetectorSystemSettings.capabilities_by_id.{k}_raw_key"] = k
                continue

            if isinstance(v, dict):
                v = DetectorCapabilities.from_dict(v)
            if isinstance(v, DetectorCapabilities):
                pass
            else:
                note_or_raise(
                    self.extra,
                    f"DetectorSystemSettings.capabilities_by_id.{det_id}",
                    TypeError(f"capabilities_by_id['{det_id}'] must be DetectorCapabilities/dict, got {type(v)}"),
                    mode=mode,
                    raw=deepcopy(v),
                )
                continue

            new_caps[det_id] = v
        self.capabilities_by_id = new_caps

        # Normalize available_detector_ids entries to str ids (drop invalid ones in lenient)
        norm_avail: List[str] = []
        for item in self.available_detector_ids:
            det_id = parse_optional_id_like(
                item, name="DetectorSystemSettings.available_detector_ids[]", strict=False, extra=self.extra
            )
            if det_id is None:
                self.extra.notes.setdefault("DetectorSystemSettings.available_detector_ids.invalid", []).append(repr(item))
                continue
            norm_avail.append(det_id)
        seen=set()
        self.available_detector_ids = [x for x in norm_avail if not (x in seen or seen.add(x))]

        # Semantic constraints live in validate().
    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate detector-system semantics."""
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        ok = True

        if self.default_detector_id:
            det_id = self.default_detector_id
            known = set(self.defaults_by_id.keys()) | set(self.capabilities_by_id.keys()) | set(self.available_detector_ids)
            if known and det_id not in known:
                note_or_raise(
                    self.extra,
                    "DetectorSystemSettings.default_detector_id",
                    ValueError(f"default_detector_id='{det_id}' not found in known detectors"),
                    mode=mode,
                    raw=det_id,
                )
                ok = False
                if not strict:
                    self.default_detector_id = None

        for det_id, ds in self.defaults_by_id.items():
            ok = ds.validate(mode=mode) and ok

        for det_id, cap in self.capabilities_by_id.items():
             ok = cap.validate(mode=mode) and ok

        return ok

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "default_detector_id": self.default_detector_id,
            "available_detector_ids": self.available_detector_ids,
            "defaults_by_id": {k: v.to_dict() for k, v in self.defaults_by_id.items()},
            "capabilities_by_id": {k: v.to_dict() for k, v in self.capabilities_by_id.items()},
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSystemSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, DetectorSystemSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return DetectorSystemSettings(_mode=mode)
        known = {"enabled", "available_detector_ids", "available_detectors", "default_detector_id", "defaults_by_id", "capabilities_by_id", "extra"}
        extra = collect_extra(d, known, owner="DetectorSystemSettings")
        return DetectorSystemSettings(
            enabled=d.get("enabled", True),
            available_detector_ids=d.get("available_detector_ids", d.get("available_detectors", [])),
            default_detector_id=d.get("default_detector_id"),
            defaults_by_id=d.get("defaults_by_id", {}),
            capabilities_by_id=d.get("capabilities_by_id", {}),
            extra=extra, _mode=mode
        )

@dataclass
class ImageOutputSettings:
    """
    Configuration for image file persistence.

    Controls the file format and destination path for saving acquired images.

    Attributes:
        file_format: The file extension/format (e.g., "tiff", "png", "jpg").
        path: The target directory or full file path for saving.
    """
    file_format: str = "tiff"
    path: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "ImageOutputSettings")
        self.file_format = parse_optional_str_like(self.file_format, name="ImageOutputSettings.file_format", strict=strict, extra=self.extra) or "tiff"
        self.file_format = self.file_format.lower()
        self.path = parse_optional_str_like(self.path, name="ImageOutputSettings.path", strict=strict, extra=self.extra)

    def validate(self, *, mode=None):
        if self.file_format not in {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}:
            if is_strict(mode or self._mode):
                raise ValueError(f"Unsupported format: {self.file_format}")
            self.file_format = "tiff"
            return False
        return True

    def to_dict(self):
        d = {"file_format": self.file_format, "path": self.path}
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d, *, mode=ParseMode.STRICT):
        if isinstance(d, ImageOutputSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return ImageOutputSettings(_mode=mode)
        return ImageOutputSettings(
            file_format=d.get("file_format", "tiff"),
            path=d.get("path"),
            extra=collect_extra(d, ("file_format", "path", "extra"), owner="ImageOutputSettings"),
            _mode=mode
        )

@dataclass
class AcquisitionRequest:
    """
    An executable command to acquire an image.

    A control-plane object that combines detector settings with output preferences.

    Attributes:
        detector_id: The ID of the detector to use (Required).
        detector: Specific settings for this acquisition.
        image: Output settings (format, path).

    Notes:
        In `STRICT` mode, this object requires a valid `detector_id` to be instantiated.
        It enforces consistency between the outer `detector_id` and the inner `detector.detector_id`.
    """
    detector_id: Optional[str] = None
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    image: ImageOutputSettings = field(default_factory=ImageOutputSettings)
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        # NORMALIZATION:
        # 1. Structure Coercion (Dict -> Dataclass)
        # 2. Patch missing holes (Copy IDs if one is missing)
        mode, strict, self.extra = _setup_init(self, self._mode, "AcquisitionRequest")

        self.detector = maybe_from_dict(DetectorSettings, self.detector, mode=mode) or DetectorSettings(_mode=mode)
        self.image = maybe_from_dict(ImageOutputSettings, self.image, mode=mode) or ImageOutputSettings(_mode=mode)
        self.detector_id = parse_optional_id_like(self.detector_id, name="id", strict=strict, extra=self.extra)

        # Logic: Patching holes.
        # DO NOT overwrite if both exist. That hides conflicts.
        if self.detector_id is None and self.detector.detector_id is not None:
            self.detector_id = self.detector.detector_id
        elif self.detector.detector_id is None and self.detector_id is not None:
            self.detector.detector_id = self.detector_id

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Consistency & Readiness
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        ok = True

        # Rule 1: Must have an ID to execute
        if not self.detector_id:
            note_or_raise(self.extra, "AcquisitionRequest.detector_id", ValueError("detector_id is required"), mode=mode)
            ok = False

        # Rule 2: Sub-objects must be valid
        ok = self.detector.validate(mode=mode) and ok
        ok = self.image.validate(mode=mode) and ok

        # Rule 3: Cross-field consistency (The Conflict Case)
        if self.detector.detector_id and self.detector_id and self.detector.detector_id != self.detector_id:
            note_or_raise(self.extra, "AcquisitionRequest.id_mismatch", ValueError(
                f"Ambiguous detector IDs: outer={self.detector_id}, inner={self.detector.detector_id}"), mode=mode)
            ok = False
            # Healing (Lenient only)
            if not strict:
                self.detector.detector_id = self.detector_id

        return ok

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "detector": self.detector.to_dict(),
            "image": self.image.to_dict(),
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "AcquisitionRequest":
        mode = as_parse_mode(mode)
        if isinstance(d, AcquisitionRequest): return replace(d, _mode=mode)
        if not isinstance(d, dict): return AcquisitionRequest(_mode=mode)
        extra = collect_extra(d, ("detector_id", "detector", "image", "extra"), owner="AcquisitionRequest")
        return AcquisitionRequest(
            detector_id=d.get("detector_id"),
            detector=d.get("detector"),
            image=d.get("image"),
            extra=extra, _mode=mode
        )

@dataclass
class Aperture:
    """
    State of a specific beam-limiting aperture mechanism.

    Tracks insertion status, selected size, and mechanical alignment.

    Attributes:
        aperture_id: Unique ID of the mechanism (e.g., "condenser", "objective").
        inserted: True if the aperture is currently inserted in the beam path.
        size_index: The selected aperture strip index.
        position: The physical alignment of the aperture mechanism.
    """
    aperture_id: Optional[str] = None
    inserted: bool = False
    size_index: Optional[int] = None
    position: Optional[Point] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "Aperture")
        self.aperture_id = parse_optional_id_like(self.aperture_id, name="Aperture.aperture_id", strict=strict, extra=self.extra)
        self.inserted = parse_bool_like(self.inserted, default=False)
        self.size_index = parse_optional_int_like(self.size_index, name="Aperture.size_index", strict=strict, extra=self.extra)
        self.position = maybe_from_dict(Point, self.position, extra=self.extra, key="Aperture.position", mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        ok = True

        if self.size_index is not None and self.size_index < 0:
            note_or_raise(
                self.extra, "Aperture.size_index",
                ValueError(f"size_index must be >= 0, got {self.size_index}"),
                mode=mode, raw=self.size_index
            )
            ok = False

        if self.position is not None:
            if not self.position.validate(mode=mode):
                note_or_raise(self.extra, "Aperture.position", ValueError("Invalid aperture position coordinates"),
                              mode=mode)
                ok = False
        return ok

    def to_dict(self) -> dict:
        d = {
            "aperture_id": self.aperture_id,
            "inserted": self.inserted,
            "size_index": self.size_index,
            "position": self.position.to_dict() if self.position else None
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Aperture":
        mode = as_parse_mode(mode)
        if isinstance(d, Aperture): return replace(d, _mode=mode)
        if not isinstance(d, dict): return Aperture(_mode=mode)
        extra = collect_extra(d, ("aperture_id", "name", "id", "inserted", "size_index", "position", "extra"), owner="Aperture")
        return Aperture(
            aperture_id=d.get("aperture_id", d.get("name", d.get("id"))),
            inserted=d.get("inserted", False),
            size_index=d.get("size_index"),
            position=d.get("position"),
            extra=extra, _mode=mode
        )

@dataclass
class MicroscopeState:
    """
    A comprehensive snapshot of the microscope hardware state.

    Aggregates the status of the stage, beam, apertures, and detectors at a specific timestamp.

    Attributes:
        timestamp: UTC timestamp of the snapshot.
        mode: The optical mode (e.g., "TEM", "STEM").
        stage_position: Current coordinates of the stage.
        beam: Current state of the electron beam.
        apertures: Dictionary of aperture states.
        detectors: Dictionary of detector settings.
        active_detector_ids: List of detectors currently marked as active.
        primary_detector_id: The ID of the currently selected main detector.

    Notes:
        Typically instantiated in `LENIENT` mode for logging/telemetry to preserve data despite partial failures.
    """
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).timestamp())
    mode: Optional[str] = None
    stage_position: StagePosition = field(default_factory=StagePosition)
    beam: BeamSettings = field(default_factory=BeamSettings)
    apertures: Dict[str, Aperture] = field(default_factory=dict)
    detectors: Dict[str, DetectorSettings] = field(default_factory=dict)
    active_detector_ids: List[str] = field(default_factory=list)
    primary_detector_id: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "MicroscopeState")
        if isinstance(self.timestamp, (int, float)):
            try:
                self.timestamp = datetime.datetime.fromtimestamp(self.timestamp, datetime.timezone.utc).isoformat()
            except Exception:
                pass
        self.timestamp = str(self.timestamp)
        self.mode = parse_optional_str_like(self.mode, name="MicroscopeState.mode", strict=strict, extra=self.extra)
        self.stage_position = maybe_from_dict(StagePosition, self.stage_position, mode=mode) or StagePosition()
        self.beam = maybe_from_dict(BeamSettings, self.beam, mode=mode) or BeamSettings()

        raw_aps = self.apertures or {}
        self.apertures = {}
        for k, v in raw_aps.items():
            ap_key = parse_optional_id_like(k, name="MicroscopeState.apertures.key", strict=strict, extra=self.extra)
            if ap_key is None:
                if self.extra:
                    self.extra.notes[f"MicroscopeState.apertures.{k}_raw_key"] = "Invalid ID"
                continue

            ap_obj = maybe_from_dict(Aperture, v, mode=mode)
            if ap_obj is None:
                ap_obj = Aperture(_mode=mode)

            if ap_obj.aperture_id is None:
                ap_obj.aperture_id = ap_key
            elif ap_obj.aperture_id != ap_key:
                note_or_raise(self.extra, f"MicroscopeState.apertures.{ap_key}.id_mismatch",
                              ValueError(f"Aperture key '{ap_key}' != internal id '{ap_obj.aperture_id}'"), mode=mode)
                ap_obj.aperture_id = ap_key

            self.apertures[ap_key] = ap_obj

        raw_dets = self.detectors or {}
        self.detectors = {}
        for k, v in raw_dets.items():
            det_key = parse_optional_id_like(k, name="MicroscopeState.detectors.key", strict=strict, extra=self.extra)
            if det_key is None:
                if self.extra:
                    self.extra.notes[f"MicroscopeState.detectors.{k}_raw_key"] = "Invalid ID"
                continue

            det_obj = maybe_from_dict(DetectorSettings, v, mode=mode)
            if det_obj is None:
                det_obj = DetectorSettings(_mode=mode)

            if det_obj.detector_id is None:
                det_obj.detector_id = det_key
            elif det_obj.detector_id != det_key:
                note_or_raise(self.extra, f"MicroscopeState.detectors.{det_key}.id_mismatch",
                              ValueError(f"Detector key '{det_key}' != internal id '{det_obj.detector_id}'"), mode=mode)
                det_obj.detector_id = det_key

            self.detectors[det_key] = det_obj

        raw_ids = self.active_detector_ids or []
        self.active_detector_ids = []
        for raw_id in raw_ids:
            norm_id = parse_optional_id_like(raw_id, name="MicroscopeState.active_detector_ids", strict=strict,
                                             extra=self.extra)
            if norm_id:
                self.active_detector_ids.append(norm_id)

        self.primary_detector_id = parse_optional_id_like(self.primary_detector_id,
                                                          name="MicroscopeState.primary_detector_id", strict=strict,
                                                          extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)
        ok = True

        ok = self.stage_position.validate(mode=mode) and ok
        ok = self.beam.validate(mode=mode) and ok

        for ds in self.detectors.values():
            ok = ds.validate(mode=mode) and ok
        for ap in self.apertures.values():
            ok = ap.validate(mode=mode) and ok

        valid_ids = []
        for det_id in self.active_detector_ids:
            if det_id not in self.detectors:
                note_or_raise(self.extra, "MicroscopeState.active_detector_ids",
                              ValueError(f"Active detector '{det_id}' not found in detectors list"), mode=mode)
                ok = False
            else:
                valid_ids.append(det_id)

        if not strict:
            self.active_detector_ids = valid_ids

        return ok

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "stage_position": self.stage_position.to_dict(),
            "beam": self.beam.to_dict(),
            "apertures": {k: v.to_dict() for k, v in self.apertures.items()},
            "detectors": {k: v.to_dict() for k, v in self.detectors.items()},
            "active_detector_ids": self.active_detector_ids,
            "primary_detector_id": self.primary_detector_id
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeState":
        mode = as_parse_mode(mode)
        if isinstance(d, MicroscopeState): return replace(d, _mode=mode)
        if not isinstance(d, dict): return MicroscopeState(_mode=mode)
        known = {f.name for f in fields(MicroscopeState)} | {"extra"}

        return MicroscopeState(
            timestamp=d.get("timestamp"),
            mode=d.get("mode"),
            stage_position=d.get("stage_position"),
            beam=d.get("beam"),
            apertures=d.get("apertures"),
            detectors=d.get("detectors") or {},
            active_detector_ids=d.get("active_detector_ids") or [],
            primary_detector_id=d.get("primary_detector_id"),
            extra=collect_extra(d, known, owner="MicroscopeState"),
            _mode=mode
        )

@dataclass
class MicroscopeImageMetadata:
    """
    Archival metadata associated with an acquired image.

    Contains flat, JSON-serializable acquisition parameters and the microscope state snapshot.

    Attributes:
        version: Metadata schema version.
        created_at: ISO8601 creation timestamp.
        magnification: The indicated magnification.
        camera_length_mm: The indicated camera length (Diffraction mode).
        pixel_size_nm: Tuple of (x, y) pixel size.
        image_size_px: Tuple of (width, height).
        accelerating_voltage_kv: High tension.
        beam_current_na: Beam current.
        exposure_ms: Exposure time.
        microscope_state: Full snapshot of the microscope state.
    """
    version: str = str(METADATA_VERSION)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    magnification: Optional[float] = None
    camera_length_mm: Optional[float] = None
    pixel_size_nm: Optional[Tuple[float, float]] = None
    image_size_px: Optional[Tuple[int, int]] = None
    accelerating_voltage_kv: Optional[float] = None
    beam_current_na: Optional[float] = None
    exposure_ms: Optional[float] = None
    microscope_state: Optional[MicroscopeState] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "MicroscopeImageMetadata")
        self.microscope_state = maybe_from_dict(MicroscopeState, self.microscope_state, mode=mode)

        # Scalar normalizations using parsers
        self.magnification = parse_optional_float_like(self.magnification, name="mag", strict=strict, extra=self.extra)
        self.camera_length_mm = parse_optional_float_like(self.camera_length_mm, name="cl", strict=strict, extra=self.extra)
        self.accelerating_voltage_kv = parse_optional_float_like(self.accelerating_voltage_kv, name="ht", strict=strict, extra=self.extra)
        self.beam_current_na = parse_optional_float_like(self.beam_current_na, name="beam", strict=strict, extra=self.extra)
        self.exposure_ms = parse_optional_float_like(self.exposure_ms, name="exp", strict=strict, extra=self.extra)
        self.pixel_size_nm = parse_optional_pair_float_like(self.pixel_size_nm, name="px", strict=strict, extra=self.extra)
        self.image_size_px = parse_optional_pair_int_like(self.image_size_px, name="res", strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        ok = True
        if self.microscope_state:
            ok = self.microscope_state.validate(mode=mode) and ok

        def _check_pos(val, name):
            if val is not None and val < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode, raw=val)
                return False
            return True

        ok = _check_pos(self.magnification, "magnification") and ok
        ok = _check_pos(self.exposure_ms, "exposure_ms") and ok
        ok = _check_pos(self.accelerating_voltage_kv, "accelerating_voltage_kv") and ok
        return ok

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "created_at": self.created_at,
            "magnification": self.magnification,
            "camera_length_mm": self.camera_length_mm,
            "pixel_size_nm": self.pixel_size_nm,
            "image_size_px": self.image_size_px,
            "accelerating_voltage_kv": self.accelerating_voltage_kv,
            "beam_current_na": self.beam_current_na,
            "exposure_ms": self.exposure_ms,
            "microscope_state": self.microscope_state.to_dict() if self.microscope_state else None
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeImageMetadata":
        mode = as_parse_mode(mode)
        if isinstance(d, MicroscopeImageMetadata): return replace(d, _mode=mode)
        if not isinstance(d, dict): return MicroscopeImageMetadata(_mode=mode)
        extra = collect_extra(d, ("version", "created_at", "magnification", "camera_length_mm", "pixel_size_nm",
                                  "image_size_px", "accelerating_voltage_kv", "beam_current_na", "exposure_ms",
                                  "microscope_state", "extra"), owner="MicroscopeImageMetadata")
        return MicroscopeImageMetadata(
            version=d.get("version", str(METADATA_VERSION)),
            created_at=d.get("created_at"),
            magnification=d.get("magnification"),
            camera_length_mm=d.get("camera_length_mm"),
            pixel_size_nm=d.get("pixel_size_nm"),
            image_size_px=d.get("image_size_px"),
            accelerating_voltage_kv=d.get("accelerating_voltage_kv"),
            beam_current_na=d.get("beam_current_na"),
            exposure_ms=d.get("exposure_ms"),
            microscope_state=d.get("microscope_state"),
            extra=extra, _mode=mode
        )

class MicroscopeImage:
    """
    Container for image data and associated metadata.

    Handles I/O operations (Load/Save) and manages the "sidecar" metadata relationship.

    Attributes:
        data: The raw 2D image data (numpy array).
        metadata: The associated acquisition parameters and state.

    Notes:
        Supports TIFF (with embedded metadata), PNG, and JPEG formats.
    """
    def __init__(self, data: np.ndarray, metadata: Optional[MicroscopeImageMetadata] = None):
        if not _check_data_format(data):
            if data.ndim == 3 and data.shape[0] == 1: data = data[0]
            elif data.ndim == 3 and data.shape[-1] == 1: data = data[..., 0]
            if not _check_data_format(data):
                raise ValueError("Invalid data format for MicroscopeImage. Must be 2D uint8/uint16.")
        self.data = data
        self.metadata = MicroscopeImageMetadata.from_dict(metadata) if isinstance(metadata, dict) else metadata

    @staticmethod
    def _decode_description(desc: Any) -> Optional[Dict[str, Any]]:
        if desc is None: return None
        if isinstance(desc, bytes):
            try: desc = desc.decode("utf-8", errors="replace")
            except Exception: return None
        if not isinstance(desc, str): return None
        desc = desc.strip()
        if not desc: return None
        try:
            obj = json.loads(desc)
            return obj if isinstance(obj, dict) else None
        except Exception: return None

    @staticmethod
    def _encode_description(md: Optional[MicroscopeImageMetadata]) -> str:
        if md is None: return ""
        try:
            safe = _jsonable(md.to_dict())
            return json.dumps(safe, ensure_ascii=False)
        except Exception:
            minimal = {"created_at": getattr(md, "created_at", None), "version": getattr(md, "version", None)}
            return json.dumps(_jsonable(minimal), ensure_ascii=False)

    @staticmethod
    def _to_uint8_preview(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        a = np.asarray(arr)
        if a.size == 0: return a.astype(np.uint8, copy=False)
        af = a.astype(np.float32, copy=False)
        finite = af[np.isfinite(af)]
        if finite.size == 0: return np.zeros_like(a, dtype=np.uint8)
        lo, hi = np.percentile(finite, [p_low, p_high])
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 1e-9:
            return np.zeros_like(a, dtype=np.uint8)
        scaled = (af - lo) * (255.0 / rng)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MicroscopeImage":
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")
        if ext in ("tif", "tiff"):
            with tff.TiffFile(str(path)) as tif:
                data = tif.asarray()
                if data.ndim == 3 and data.shape[0] == 1: data = data[0]
                if data.ndim == 3 and data.shape[-1] == 1: data = data[..., 0]
                if data.ndim != 2: raise ValueError(f"Expected single-frame 2D grayscale TIFF, got shape={data.shape}")
                if data.dtype == np.int16:
                    if data.min() >= 0: data = data.astype(np.uint16)
                    else: raise ValueError("Loaded TIFF is int16 with negative values.")
                if data.dtype in (np.int32, np.uint32):
                    if data.min() >= 0 and data.max() <= 65535: data = data.astype(np.uint16)
                    else: raise ValueError("TIFF is 32-bit with values outside uint16 range.")
                metadata = None
                try:
                    desc = tif.pages[0].tags["ImageDescription"].value
                    d = cls._decode_description(desc)
                    if d is not None: metadata = MicroscopeImageMetadata.from_dict(d)
                except Exception: metadata = None
            return cls(data=data, metadata=metadata)

        with Image.open(path) as img:
            if img.mode not in ("L", "I;16", "I;16B", "I;16L"): img = img.convert("L")
            data = np.array(img)
            if data.ndim == 3 and data.shape[-1] == 1: data = data[..., 0]
            if data.dtype == np.int32 and img.mode in ("I;16", "I;16B", "I;16L"): data = data.astype(np.uint16)
            if data.dtype not in (np.uint8, np.uint16):
                if np.issubdtype(data.dtype, np.number): data = np.clip(data, 0, 255).astype(np.uint8)
                else: data = data.astype(np.uint8)
        metadata = None
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(d, dict): metadata = MicroscopeImageMetadata.from_dict(d)
            except Exception: metadata = None
        return cls(data=data, metadata=metadata)

    def save(self, path: Union[str, Path], file_format: Optional[str] = None) -> Path:
        supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}
        path = Path(path)
        requested_ext = (file_format or path.suffix.lstrip(".") or "tiff").lower().strip()
        if requested_ext not in supported: raise ValueError(f"Unsupported file_format: {requested_ext!r}")
        fmt = "tiff" if requested_ext in ("tif", "tiff") else requested_ext
        suffix = ".tif" if requested_ext == "tif" else ".tiff" if requested_ext == "tiff" else f".{requested_ext}"
        path = path.with_suffix(suffix)
        os.makedirs(path.parent, exist_ok=True)
        desc = self._encode_description(self.metadata)
        if fmt == "tiff":
            tff.imwrite(str(path), self.data, description=desc)
            return path
        data_to_save = self.data
        if fmt in ("jpg", "jpeg", "bmp"):
            if data_to_save.dtype != np.uint8: data_to_save = self._to_uint8_preview(data_to_save)
        else:
            if data_to_save.dtype not in (np.uint8, np.uint16): data_to_save = np.clip(data_to_save, 0, 255).astype(np.uint8)
        img = Image.fromarray(data_to_save)
        pil_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "bmp": "BMP"}[fmt]
        img.save(path, format=pil_format)
        sidecar = path.with_suffix(path.suffix + ".json")
        try:
            if self.metadata is not None:
                sidecar.write_text(json.dumps(self.metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            elif sidecar.exists(): sidecar.unlink()
        except Exception: pass
        return path

@dataclass
class SystemInfo:
    """
    Static identity and version information for the system.

    Identifies hardware (Model, Serial) and software versions.

    Attributes:
        name: Human-readable name for this microscope instance.
        ip_address: Network address of the control PC.
        manufacturer: Vendor name.
        model: Model name.
        serial_number: Unique hardware serial number.
        hardware_version: Vendor hardware revision.
        software_version: Vendor control software version.
        application: Connected application name.

    Notes:
        Defaults to "Unknown" to prevent logging crashes on missing data.
    """
    name: str = "Unknown"
    ip_address: str = "Unknown"
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str = "Unknown"
    hardware_version: str = "Unknown"
    software_version: str = "Unknown"
    supertem_version: str = __version__
    application: Optional[str] = None
    application_version: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "SystemInfo")
        for f in fields(SystemInfo):
             if f.name == "extra" or f.name.startswith("_"): continue
             v = getattr(self, f.name)
             setattr(self, f.name, parse_optional_str_like(v, name=f"SystemInfo.{f.name}", strict=strict, extra=self.extra) or "Unknown")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        ip = self.ip_address.strip()
        if ip and ip != "Unknown":
            try:
                ipaddress.ip_address(ip)
            except Exception:
                note_or_raise(self.extra, "SystemInfo.ip_address", ValueError(f"Invalid IP: {self.ip_address!r}"), mode=mode, raw=self.ip_address)
                self.ip_address = "Unknown"
                return False
        return True

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(SystemInfo) if f.name not in ("extra", "_mode")}
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemInfo":
        mode = as_parse_mode(mode)
        if isinstance(d, SystemInfo): return replace(d, _mode=mode)
        if not isinstance(d, dict): return SystemInfo(_mode=mode)
        known = {f.name for f in fields(SystemInfo)} | {"extra"}
        extra = collect_extra(d, known, owner="SystemInfo")
        return SystemInfo(**{k: d.get(k) for k in known if k != "extra"}, extra=extra, _mode=mode)

@dataclass
class SystemSettings:
    """
    Root configuration object for the microscope hardware.

    Hierarchically aggregates settings for Stage, Beam, and Detectors.

    Attributes:
        stage_system: Settings and limits for the stage.
        beam_system: Settings and limits for the electron column.
        detector_system: Settings and capabilities for all detectors.
        info: Static system identity metadata.
    """
    stage_system: StageSystemSettings = field(default_factory=StageSystemSettings)
    beam_system: BeamSystemSettings = field(default_factory=BeamSystemSettings)
    detector_system: DetectorSystemSettings = field(default_factory=DetectorSystemSettings)
    info: SystemInfo = field(default_factory=SystemInfo)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        # Optim: No local mode/strict needed here as it delegates
        mode = as_parse_mode(self._mode)
        self.stage_system = maybe_from_dict(StageSystemSettings, self.stage_system, mode=mode) or StageSystemSettings(_mode=mode)
        self.beam_system = maybe_from_dict(BeamSystemSettings, self.beam_system, mode=mode) or BeamSystemSettings(_mode=mode)
        self.detector_system = maybe_from_dict(DetectorSystemSettings, self.detector_system, mode=mode) or DetectorSystemSettings(_mode=mode)
        self.info = maybe_from_dict(SystemInfo, self.info, mode=mode) or SystemInfo(_mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        return (self.stage_system.validate(mode=mode) and
                self.beam_system.validate(mode=mode) and
                self.detector_system.validate(mode=mode) and
                self.info.validate(mode=mode))

    def to_dict(self) -> dict:
        return _jsonable({
            "stage_system": self.stage_system.to_dict(),
            "beam_system": self.beam_system.to_dict(),
            "detector_system": self.detector_system.to_dict(),
            "info": self.info.to_dict(),
        })

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, SystemSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return SystemSettings(_mode=mode)
        return SystemSettings(
            stage_system=d.get("stage_system", d.get("stage")),
            beam_system=d.get("beam_system", d.get("beam")),
            detector_system=d.get("detector_system", d.get("detector")),
            info=d.get("info"),
            _mode=mode,
        )

@dataclass
class MicroscopeSettings:
    """
    Top-level application configuration.

    Combines hardware system settings with global application preferences and protocols.

    Attributes:
        system: Hardware configuration (Stage, Beam, Detectors).
        image: Global defaults for image output (Format, Path).
        protocol: Dictionary for experimental protocol parameters.
    """
    system: SystemSettings = field(default_factory=SystemSettings)
    image: ImageOutputSettings = field(default_factory=ImageOutputSettings)
    protocol: dict = field(default_factory=lambda: {"name": "demo"})
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        self.system = maybe_from_dict(SystemSettings, self.system, mode=mode) or SystemSettings(_mode=mode)
        self.image = maybe_from_dict(ImageOutputSettings, self.image, mode=mode) or ImageOutputSettings(_mode=mode)
        if not isinstance(self.protocol, dict): self.protocol = {"name": "demo"}

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        return self.system.validate(mode=mode) and self.image.validate(mode=mode)

    def to_dict(self) -> dict:
        return _jsonable({
            "system": self.system.to_dict(),
            "image": self.image.to_dict(),
            "protocol": self.protocol,
        })

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeSettings":
        mode = as_parse_mode(mode)
        if isinstance(d, MicroscopeSettings): return replace(d, _mode=mode)
        if not isinstance(d, dict): return MicroscopeSettings(_mode=mode)
        return MicroscopeSettings(
            system=d.get("system"),
            image=d.get("image"),
            protocol=d.get("protocol", {"name": "demo"}),
            _mode=mode,
        )