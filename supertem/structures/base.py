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
     - Goal:   Capture data without crashing, even if imperfect.

  2) Normalize (Integrity of Structure)
     - Happens during __post_init__ (via helper parsers)
     - Goal:   Produce a well-typed internal representation (Type Safety).
     - Action: Coerce types (str->int), populate structural defaults (None->[]),
               and park unparseable garbage in Extras.raw.
     - Note:   Does NOT check logic. Invalid values (e.g., width=-100) are
               preserved here to ensure data fidelity during ingestion.

  3) Validate (Integrity of Meaning)
     - Happens in validate(mode=...)
     - Goal:   Enforce domain constraints and logical invariants (Logic Safety).
     - Action (STRICT): Raise note_or_raise(...) on any violation.
     - Action (LENIENT): Record violation in Extras.notes and "Heal" the object
               (e.g., reset width=-100 -> 512, or disable the specific feature).
     - Result: The object is now guaranteed to be logically consistent.

  4) Serialize (JSON-capable representation)
     - Happens in to_dict()
     - Goal:   Produce JSON-serializable output for logging/storage/transport.

  5) Execute (Gatekeeping)
     - Only applicable to control-plane objects (requests/settings).
     - Policy: Validate STRICTLY immediately before hardware interaction.
     - Goal:   Ensure commands applied to the microscope are within hardware limits.

A key rule:
  Objects created via from_dict() are "Type-Safe" but "Logically Unverified."
  They MUST NOT be executed until validate() has been called and passed.

===============================================================================
II. ParseMode and Context
===============================================================================

ParseMode.LENIENT (data-plane default)
  Intended for metadata/state/log ingestion where completeness is not guaranteed.

  Behavior at Construction (__post_init__):
    - Prioritizes survival: construction will not fail due to malformed types.
    - Captures unparseable data in Extras.raw / Extras.notes.
    - Result: Object is type-safe but may contain logically unsafe values (e.g. exposure=-5).

  Behavior at Validation (.validate()):
    - "Heals" invalid logic: unsafe values are reset to defaults or None.
    - Records the intervention in Extras.notes.
    - Result: Object becomes safe for use.

ParseMode.STRICT (control-plane default)
  Intended for objects that will be applied to hardware (requests/settings).

  Behavior at Construction (__post_init__):
    - Prioritizes correctness: raises immediately on malformed types or missing structure.

  Behavior at Validation (.validate()):
    - Enforces constraints: raises note_or_raise(...) on any logical violation.
    - Result: Guaranteed safe to execute, or raises Exception.

===============================================================================
III. The Normalization, Validation, and Gatekeeping Rulebook
===============================================================================

To maintain safety without sacrificing robustness, this module enforces a strict
separation of concerns across three distinct lifecycles:

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
      If a field is `None` but required for the object to exist (e.g., to prevent
      AttributeError later), set a safe default here.
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
   GOAL:    Integrity of Meaning (Internal Logic & Self-Consistency)
   INPUT:   "Clean" data (guaranteed types from step 1)
   OUTPUT:  Boolean success flag (and populated Extras.notes)

   Rules:
   A. Trust the Types.
      Do not check `isinstance` or try/except AttributeErrors. If `__post_init__`
      did its job, variables have the correct type. Focus on *values*.

   B. Domain Constraints are Validation.
      Check physical and logical bounds of the object itself.
      (e.g., `width > 0`, `min_limit <= max_limit`).

   C. Internal Cross-Field Consistency.
      Check if two fields within the *same* object or hierarchy contradict each other.
      - "If axis is enabled (`can_tilt=True`), limits MUST be defined (`tilt_limits!=None`)."
      - "If `default_id` is set, it MUST exist in `available_ids`."

   D. Healing is Validation (Lenient Mode Only).
      If a value is structurally sound but logically invalid (e.g., `width=-50`):
        - STRICT Mode: Raise an Exception.
        - LENIENT Mode: "Heal" it to a safe value or disable the feature.

3. Gatekeeping (is_safe_... / is_supported)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Action (Runtime Safety & Hardware Compatibility)
   INPUT:   An external "Request" object (e.g., StagePosition, BeamSettings)
   OUTPUT:  Boolean allowed/rejected flag.

   Rules:
   A. Configs are Guardrails, Requests are Intent.
      The SystemSettings object acts as the Gatekeeper. It validates *external*
      requests against its *internal* limits.

   B. Specificity over Genericity.
      Use specific method names that describe the risk:
      - `is_safe_move(target)`: Checks collision/travel limits (Stage).
      - `is_safe_beam(target)`: Checks voltage/optical limits (Beam).
      - `is_supported(settings)`: Checks driver capabilities (Detector).

   Summary Table:
   +------------------+-----------------------+-----------------------------+
   | Phase            | Question Asked        | Example                     |
   +==================+=======================+=============================+
   | Normalization    | "Is it the right type?"| "10" -> 10 (int)           |
   | Validation       | "Is it logical?"      | min_limit < max_limit       |
   |                  | "Is it complete?"     | enabled=True -> limits!=None|
   +------------------+-----------------------+-----------------------------+
   | Gatekeeping      | "Is it safe/allowed?" | target_x < x_limit          |
   +------------------+-----------------------+-----------------------------+

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
from dataclasses import dataclass, field, replace, is_dataclass
from pathlib import Path
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, TypeVar, Type
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
    if isinstance(mode, ParseMode): return mode
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m == "lenient": return ParseMode.LENIENT
        if m == "strict": return ParseMode.STRICT
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
    if is_strict(mode): raise exc
    if extra is None: return
    try:
        if raw is not None: extra.raw[key] = _jsonable(raw)
        extra.notes[key] = {"error": repr(exc)}
    except Exception: pass


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
    if value is None: return None
    if isinstance(value, (bool, np.bool_)): return None
    if isinstance(value, (int, float, np.number)): return Q_(float(value), unit)

    try:
        if isinstance(value, Quantity):
            return Q_(value.magnitude, str(value.units)).to(unit)
        if isinstance(value, dict):
            mag = value.get("magnitude", value.get("value"))
            u = value.get("unit", value.get("units"))
            if mag is None: return None
            return Q_(mag, u or unit).to(unit)
        if isinstance(value, str):
            s = value.strip()
            if not s: return None
            try: return Q_(float(s), unit).to(unit)
            except Exception: pass
            return Q_(s).to(unit)
        return Q_(float(value), unit).to(unit)
    except Exception:
        return None

def serialize_quantity(q: Optional["Quantity"], target_unit: str) -> Optional[float]:
    """Convert a Quantity to a plain float magnitude in the target unit.

    This strips the unit information for safe JSON serialization.
    Example: serialize_quantity(Q_(300, 'kV'), 'V') -> 300000.0
    """
    if q is None: return None
    try:
        if not isinstance(q, Quantity): return float(q)
        return float(q.to(target_unit).magnitude)
    except Exception: return None

def _check_data_format(data: np.ndarray) -> bool:
    """Validate if numpy array is a valid 2D image (uint8/uint16)."""
    if data.ndim == 3:
        if data.shape[0] == 1: data = data[0]
        elif data.shape[2] == 1: data = data[:, :, 0]
    return (data.ndim == 2) and (data.dtype.kind == "u") and (data.dtype.itemsize in (1, 2))


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
        if self.vendor: out["vendor"] = deepcopy(self.vendor)
        if self.unknown: out["unknown"] = deepcopy(self.unknown)
        if self.raw: out["raw"] = deepcopy(self.raw)
        if self.notes: out["notes"] = deepcopy(self.notes)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe payload representation."""
        return _jsonable(self.to_native_dict())

    @staticmethod
    def from_any(value: Any, *, owner: str = "unknown") -> "Extras":
        """Intelligently parse 'extra' fields from various inputs."""
        if value is None: return Extras()
        if isinstance(value, Extras): return value
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
            # Flat dict fallback
            ex = Extras()
            try: ex.unknown = deepcopy(value)
            except Exception: ex.raw[f"{owner}.extra"] = repr(value)
            return ex
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(value)
        return ex

def _extra_put_raw(extra: Any, key: str, value: Any) -> None:
    if extra is None: return
    if isinstance(extra, Extras):
        extra.raw[key] = value
    elif isinstance(extra, dict):
        extra[f"{key}_raw"] = value

def collect_extra(d: Optional[Dict[str, Any]], known: Iterable[str], *, owner: str = "unknown") -> Extras:
    """Harvest unknown keys from a source dict into an Extras object.

    This ensures forward compatibility: if the hardware sends new fields we don't
    recognize yet, we preserve them in 'unknown' rather than discarding them.
    """
    if not isinstance(d, dict): return Extras()
    known_set = set(known)
    ex = normalize_extra_lenient(d.get("extra"), owner)
    for k, v in d.items():
        if k != "extra" and k not in known_set:
            try: ex.unknown[str(k)] = deepcopy(v)
            except Exception: ex.unknown[str(k)] = repr(v)
    return ex

def add_extra_if_any(out: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    if extra is None: return out
    ex_obj = extra if isinstance(extra, Extras) else Extras.from_any(extra)
    if ex_obj.is_empty(): return out
    payload = ex_obj.to_dict()
    clean_payload = {k: v for k, v in payload.items() if v}
    if clean_payload: out["extra"] = clean_payload
    return out

def drop_none_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in out.items() if v is not None}

def normalize_extra(extra: Any) -> Extras:
    if extra is None: return Extras()
    if isinstance(extra, Extras): return extra
    if isinstance(extra, dict): return Extras.from_any(extra)
    raise TypeError(f"extra must be Extras, dict, or None, got {type(extra)}")

def normalize_extra_lenient(extra: Any, owner: str) -> Extras:
    try: return normalize_extra(extra)
    except Exception:
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(extra)
        return ex

# =============================================================================
# Type Parsers
# =============================================================================

def parse_bool(value: Any, default: bool = False, *, name: str, strict: bool = False, extra: Any = None) -> bool:
    """Strict boolean parser (returns bool). Handles defaults internally to avoid falsy traps."""
    if value is None: return default
    if isinstance(value, bool): return value
    if isinstance(value, (int, float, np.number)): return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}: return True
        if s in {"0", "false", "f", "no", "n", "off", ""}: return False
    if extra is not None: _extra_put_raw(extra, name, value)
    if strict: raise ValueError(f"Invalid boolean for {name}: {value!r}")
    return default

def parse_opt_bool(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[bool]:
    """Optional boolean parser (returns Optional[bool]) for tristate logic."""
    if value is None: return None
    if isinstance(value, str) and not value.strip(): return None
    try: return parse_bool(value, default=False, name=name, strict=True)
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise
        return None

def parse_opt_int(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[int]:
    if value is None: return None
    if isinstance(value, bool):
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise TypeError(f"{name} cannot be bool")
        return None
    try:
        if isinstance(value, (int, np.integer)): return int(value)
        if isinstance(value, (float, np.floating)):
            f = float(value)
            if f.is_integer(): return int(f)
            raise ValueError(f"{name} must be an integer value, got {value!r}")
        if isinstance(value, str):
            s = value.strip()
            if s == "": return None
            f = float(s)
            if f.is_integer(): return int(f)
            raise ValueError(f"{name} must be an integer value, got {value!r}")
        raise TypeError(f"{name} must be int/float/str, got {type(value)}")
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise ValueError(f"{name} must be integer, got {value!r}")
        return None

def parse_opt_float(value: Any, *, name: str, unit: Optional[str] = None, strict: bool = False, extra: Any = None) -> Optional[float]:
    if value is None: return None
    if isinstance(value, (bool, np.bool_)):
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise TypeError(f"{name} cannot be bool")
        return None
    if unit:
        q = ensure_quantity(value, unit)
        if q is not None: return float(q.magnitude)
    try:
        # Prevent "Off-by-1000" errors: Reject Quantities if no unit was specified
        if isinstance(value, Quantity) and not unit:
            if strict: raise ValueError(f"{name} is a Quantity but no target unit defined.")
            return float(value.magnitude)
        if isinstance(value, (int, float, np.number)): return float(value)
        if isinstance(value, str):
            s = value.strip()
            if s == "": return None
            return float(s)
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise ValueError(f"{name} must be float-like")
        return None

def parse_opt_quantity(value: Any, unit: str, *, name: str, strict: bool = False, extra: Any = None) -> Optional["Quantity"]:
    q = ensure_quantity(value, unit)
    if q is not None: return q
    if value is not None:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise ValueError(f"'{name}' must be {unit}, got {value!r}")
    return None

def parse_opt_str(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[str]:
    if value is None: return None
    if isinstance(value, str): return value.strip() or None
    if strict:
        if extra is not None: _extra_put_raw(extra, name, value)
        raise TypeError(f"{name} must be str-like, got {type(value)}")
    if isinstance(value, bool):
        if extra is not None: _extra_put_raw(extra, name, value)
        return None
    try:
        s = str(value).strip()
        return s or None
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        return None

def parse_opt_id(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[str]:
    if value is None: return None
    if isinstance(value, str) and value.strip() == "":
        if extra is not None:
            _extra_put_raw(extra, name, value)
            if hasattr(extra, "notes"): extra.notes.setdefault("empty_id_fields", []).append(name)
        return None
    return parse_opt_str(value, name=name, strict=strict, extra=extra)

def parse_opt_pair_int(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[Tuple[int, int]]:
    if value is None: return None
    try:
        if not isinstance(value, (tuple, list)) or len(value) != 2: raise TypeError
        a = parse_opt_int(value[0], name=f"{name}[0]", strict=True)
        b = parse_opt_int(value[1], name=f"{name}[1]", strict=True)
        if a is None or b is None: raise ValueError
        return (a, b)
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise ValueError(f"{name} must be (int, int)")
        return None

def parse_opt_pair_float(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[Tuple[float, float]]:
    if value is None: return None
    try:
        if not isinstance(value, (tuple, list)) or len(value) != 2: raise TypeError
        a = parse_opt_float(value[0], name=f"{name}[0]", strict=True)
        b = parse_opt_float(value[1], name=f"{name}[1]", strict=True)
        if a is None or b is None: raise ValueError
        return (a, b)
    except Exception:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise ValueError(f"{name} must be (float, float)")
        return None

def parse_opt_pair_quantity(value: Any, unit: str, *, name: str, strict: bool = False, extra: Any = None) -> Optional[Tuple["Quantity", "Quantity"]]:
    if value is None: return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise TypeError(f"{name} must be (val, val)")
        return None
    q1 = parse_opt_quantity(value[0], unit, name=f"{name}[0]", strict=strict, extra=extra)
    q2 = parse_opt_quantity(value[1], unit, name=f"{name}[1]", strict=strict, extra=extra)
    if q1 is not None and q2 is not None: return (q1, q2)
    return None

def parse_str_list(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> List[str]:
    if value is None: return []
    if not isinstance(value, (list, tuple)):
        if extra is not None: _extra_put_raw(extra, name, value)
        if strict: raise TypeError(f"{name} must be list")
        return []
    out = []
    for i, item in enumerate(value):
        s = parse_opt_str(item, name=f"{name}[{i}]", strict=strict, extra=extra)
        if s is not None: out.append(s)
    return out

T = TypeVar("T")

def parse_model(cls: Type[T], raw: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT, extra: Optional["Extras"] = None, key: str = "", allow_empty: bool = False) -> Optional[T]:
    """Instantiate a Dataclass from a dict/list/tuple safely."""
    mode = as_parse_mode(mode)
    if raw is None: return None
    if isinstance(raw, dict) and (not raw) and (not allow_empty): return None

    # 1. Identity
    try:
        if isinstance(raw, cls):
            if is_dataclass(raw): return replace(raw, _mode=mode) # type: ignore
            return raw
    except TypeError: pass

    # 2. Type Check
    if not isinstance(raw, (dict, list, tuple)):
         note_or_raise(extra, key or f"{getattr(cls, '__name__', 'object')}", TypeError(f"expected structured data, got {type(raw)}"), mode=mode, raw=raw)
         return None

    # 3. Instantiate
    from_dict = getattr(cls, "from_dict", None)
    try:
        if callable(from_dict):
             try: return from_dict(raw, mode=mode) # type: ignore
             except TypeError: return from_dict(raw) # type: ignore
        elif is_dataclass(cls) and isinstance(raw, dict):
             return cls(**raw) # type: ignore
    except Exception as e:
        note_or_raise(extra, key or f"{getattr(cls, '__name__', 'object')}", e, mode=mode, raw=raw)
        return None
    return None

def parse_keyed_map(target_cls: Type[T], raw_map: Optional[Dict[str, Any]], id_field: Optional[str], owner_name: str, mode: ParseMode, extra: Extras) -> Dict[str, T]:
    out: Dict[str, T] = {}
    if not raw_map: return out
    strict = is_strict(mode)

    for k, v in raw_map.items():
        key_norm = parse_opt_id(k, name=f"{owner_name}.key", strict=strict, extra=extra)
        if key_norm is None: continue

        obj_key = f"{owner_name}.{key_norm}"
        obj = parse_model(target_cls, v, mode=mode, extra=extra, key=obj_key)

        if obj is None:
            note_or_raise(extra, obj_key, TypeError(f"Invalid object for key '{key_norm}'"), mode=mode, raw=v)
            if not strict:
                try: obj = target_cls(_mode=mode) # type: ignore
                except Exception: pass

        if obj is not None:
            if id_field and hasattr(obj, id_field):
                internal_id = getattr(obj, id_field, None)
                if internal_id is None: setattr(obj, id_field, key_norm)
                elif internal_id != key_norm:
                    note_or_raise(extra, f"{owner_name}.{key_norm}.id_mismatch", ValueError(f"Key '{key_norm}' != id '{internal_id}'"), mode=mode)
                    if not strict: setattr(obj, id_field, key_norm)
            out[key_norm] = obj
    return out


# =============================================================================
# Helpers
# =============================================================================

def _jsonable(obj: Any) -> Any:
    """Recursively convert object to JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    try:
        import numpy as _np
        if isinstance(obj, _np.generic): return obj.item()
        if isinstance(obj, _np.ndarray): return obj.tolist()
    except Exception: pass
    try:
        if isinstance(obj, Path): return str(obj)
    except Exception: pass
    try:
        if isinstance(obj, Quantity):
            return {"magnitude": float(obj.magnitude), "unit": str(obj.units)}
    except Exception: pass
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(x) for x in obj]
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

def _setup_validate(obj_mode: Any, override_mode: Any) -> Tuple[ParseMode, bool]:
    """
    Standardizes the start of validation methods.
    Returns: (effective_mode, is_strict_flag)
    """
    mode = as_parse_mode(obj_mode if override_mode is None else override_mode)
    return mode, is_strict(mode)

def _finish_to_dict(payload: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    """Standardizes the final steps of serialization: extras injection and cleanup."""
    add_extra_if_any(payload, extra)
    return _jsonable(drop_none_keys(payload))

def _setup_from_dict(cls: Type[T], data: Any, mode: Union[ParseMode, str, None], known_keys: Iterable[str] = (), aliases: Iterable[str] = ()) -> Tuple[Optional[Dict[str, Any]], ParseMode, Optional[Extras]]:
    mode = as_parse_mode(mode)
    if isinstance(data, cls): return None, mode, None
    if not isinstance(data, dict):
        if is_strict(mode): raise TypeError(f"{cls.__name__} expects dict")
        ex = Extras()
        if data is not None:
            _extra_put_raw(ex, "source_type_error", data)
            ex.notes["source_type_error"] = {"error": f"Expected dict, got {type(data).__name__}"}
        return {}, mode, ex
    extra = collect_extra(data, tuple(known_keys) + tuple(aliases), owner=cls.__name__)
    return data, mode, extra

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
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)
        self.x = parse_opt_float(self.x, name="Point.x", strict=strict) or 0.0
        self.y = parse_opt_float(self.y, name="Point.y", strict=strict) or 0.0
        self.z = parse_opt_float(self.z, name="Point.z", strict=strict) or 0.0
        self.name = parse_opt_str(self.name, name="Point.name", strict=False)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        if not (math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)):
            note_or_raise(None, "Point.coordinates", ValueError(f"Coordinates must be finite"), mode=mode, raw=(self.x, self.y, self.z))
            if strict: return False
            self.x = 0.0 if not math.isfinite(self.x) else self.x
            self.y = 0.0 if not math.isfinite(self.y) else self.y
            self.z = 0.0 if not math.isfinite(self.z) else self.z
        return True

    def to_dict(self) -> dict:
        return _jsonable(drop_none_keys({"x": self.x, "y": self.y, "z": self.z, "name": self.name}))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Point":
        mode = as_parse_mode(mode)
        if isinstance(d, Point): return replace(d, _mode=mode)
        if isinstance(d, (list, tuple)) and len(d) in (2, 3):
            return Point(x=d[0], y=d[1], z=d[2] if len(d) == 3 else 0.0, _mode=mode)
        if not isinstance(d, dict):
            if is_strict(mode) and d is not None: raise TypeError("Point expects dict/list")
            return Point(_mode=mode)
        return Point(x=d.get("x"), y=d.get("y"), z=d.get("z"), name=d.get("name"), _mode=mode)

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
        mode, strict, self.extra = _setup_init(self, self._mode, "ROI")
        self.x = parse_opt_int(self.x, name="ROI.x", strict=strict, extra=self.extra) or 0
        self.y = parse_opt_int(self.y, name="ROI.y", strict=strict, extra=self.extra) or 0
        self.width = parse_opt_int(self.width, name="ROI.width", strict=strict, extra=self.extra) or 512
        self.height = parse_opt_int(self.height, name="ROI.height", strict=strict, extra=self.extra) or 512

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True
        if self.x < 0 or self.y < 0:
            note_or_raise(self.extra, "ROI.xy", ValueError("ROI.x/ROI.y must be >= 0"), mode=mode, raw=(self.x, self.y))
            if strict: ok = False
            else: self.x, self.y = max(self.x, 0), max(self.y, 0)
        if (self.width <= 0) or (self.height <= 0):
            note_or_raise(self.extra, "ROI.size", ValueError("ROI.width/height must be > 0"), mode=mode, raw=(self.width, self.height))
            if strict: ok = False
            else: self.width, self.height = max(self.width, 512), max(self.height, 512)
        return ok

    def to_dict(self) -> dict:
        d = {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ROI":
        if isinstance(d, (list, tuple)) and len(d) == 4:
            m = as_parse_mode(mode)
            ex = Extras() if is_strict(m) else normalize_extra_lenient(None, "ROI")
            return ROI(x=d[0], y=d[1], width=d[2], height=d[3], extra=ex, _mode=m)
        d_dict, mode, extra = _setup_from_dict(
            ROI, d, mode,
            known_keys=("x", "y", "width", "height", "extra"),
            aliases=("w", "h")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return ROI(
            x=d_dict.get("x"), y=d_dict.get("y"),
            width=d_dict.get("width", d_dict.get("w")),
            height=d_dict.get("height", d_dict.get("h")),
            extra=extra, _mode=mode,
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
        mode, strict, self.extra = _setup_init(self, self._mode, "StagePosition")
        # 1. Scalars
        self.name = parse_opt_str(self.name, name="StagePosition.name", strict=strict, extra=self.extra)
        self.coordinate_system = parse_opt_str(self.coordinate_system, name="StagePosition.coordinate_system", strict=strict, extra=self.extra)
        # 2. Quantities
        self.x = parse_opt_quantity(self.x, "nm", name="StagePosition.x", strict=strict, extra=self.extra)
        self.y = parse_opt_quantity(self.y, "nm", name="StagePosition.y", strict=strict, extra=self.extra)
        self.z = parse_opt_quantity(self.z, "nm", name="StagePosition.z", strict=strict, extra=self.extra)
        self.r = parse_opt_quantity(self.r, "degree", name="StagePosition.r", strict=strict, extra=self.extra)
        self.tilt_x = parse_opt_quantity(self.tilt_x, "degree", name="StagePosition.tilt_x", strict=strict, extra=self.extra)
        self.tilt_y = parse_opt_quantity(self.tilt_y, "degree", name="StagePosition.tilt_y", strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True
        for name, q in [("x", self.x), ("y", self.y), ("z", self.z), ("r", self.r), ("tilt_x", self.tilt_x), ("tilt_y", self.tilt_y)]:
            if q is not None and not math.isfinite(q.magnitude):
                note_or_raise(self.extra, f"StagePosition.{name}", ValueError(f"{name} must be finite"), mode=mode, raw=q)
                ok = False
        return ok

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
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> 'StagePosition':
        d_dict, mode, extra = _setup_from_dict(
            StagePosition, d, mode,
            known_keys=("name", "x", "y", "z", "r", "tilt_x", "tilt_y", "coordinate_system", "extra"),
            aliases=("x_nm", "y_nm", "z_nm", "r_deg", "tilt_x_deg", "tilt_y_deg")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return StagePosition(
            name=d_dict.get("name"),
            x=d_dict.get("x", d_dict.get("x_nm")),
            y=d_dict.get("y", d_dict.get("y_nm")),
            z=d_dict.get("z", d_dict.get("z_nm")),
            r=d_dict.get("r", d_dict.get("r_deg")),
            tilt_x=d_dict.get("tilt_x", d_dict.get("tilt_x_deg")),
            tilt_y=d_dict.get("tilt_y", d_dict.get("tilt_y_deg")),
            coordinate_system=d_dict.get("coordinate_system"),
            extra=extra, _mode=mode,
        )

    def __add__(self, other: 'StagePosition') -> 'StagePosition':
        """Enable vector addition for relative movements."""
        if not isinstance(other, StagePosition): return NotImplemented
        def add(a, b, u):
            if a is None and b is None: return None
            return (a or Q_(0, u)) + (b or Q_(0, u))
        return StagePosition(
            name=self.name,
            x=add(self.x, other.x, "nm"), y=add(self.y, other.y, "nm"), z=add(self.z, other.z, "nm"),
            r=add(self.r, other.r, "degree"), tilt_x=add(self.tilt_x, other.tilt_x, "degree"), tilt_y=add(self.tilt_y, other.tilt_y, "degree"),
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: 'StagePosition') -> 'StagePosition':
        if not isinstance(other, StagePosition): return NotImplemented
        def sub(a, b): return (a - b) if (a is not None and b is not None) else None
        return StagePosition(
            name=self.name,
            x=sub(self.x, other.x), y=sub(self.y, other.y), z=sub(self.z, other.z),
            r=sub(self.r, other.r), tilt_x=sub(self.tilt_x, other.tilt_x), tilt_y=sub(self.tilt_y, other.tilt_y),
            coordinate_system=self.coordinate_system,
        )

    def is_close(self, other: 'StagePosition', tol_nm: float = 1.0, tol_deg: float = 1e-3) -> bool:
        def chk(a, b, u, t):
            if a is None or b is None: return False
            return abs(a.to(u).magnitude - b.to(u).magnitude) <= t
        return (chk(self.x, other.x, "nm", tol_nm) and chk(self.y, other.y, "nm", tol_nm) and
                chk(self.z, other.z, "nm", tol_nm) and chk(self.r, other.r, "degree", tol_deg) and
                chk(self.tilt_x, other.tilt_x, "degree", tol_deg) and chk(self.tilt_y, other.tilt_y, "degree", tol_deg))

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
    settle_time: "Quantity" = field(default_factory=lambda: Q_(0.2, "seconds"))
    timeout: "Quantity" = field(default_factory=lambda: Q_(10.0, "seconds"))
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "StageSystemSettings")
        # 1. Scalars
        self.enabled = parse_bool(self.enabled, default=True, name="StageSystemSettings.enabled", strict=strict, extra=self.extra)
        self.can_x = parse_bool(self.can_x, default=True, name="StageSystemSettings.can_x", strict=strict, extra=self.extra)
        self.can_y = parse_bool(self.can_y, default=True, name="StageSystemSettings.can_y", strict=strict, extra=self.extra)
        self.can_z = parse_bool(self.can_z, default=True, name="StageSystemSettings.can_z", strict=strict, extra=self.extra)
        self.can_r = parse_bool(self.can_r, default=False, name="StageSystemSettings.can_r", strict=strict, extra=self.extra)
        self.can_tilt_x = parse_bool(self.can_tilt_x, default=False, name="StageSystemSettings.can_tilt_x", strict=strict, extra=self.extra)
        self.can_tilt_y = parse_bool(self.can_tilt_y, default=False, name="StageSystemSettings.can_tilt_y", strict=strict, extra=self.extra)

        # 2. Quantities
        self.x_limits = parse_opt_pair_quantity(self.x_limits, "nm", name="StageSystemSettings.x_limits", strict=strict, extra=self.extra)
        self.y_limits = parse_opt_pair_quantity(self.y_limits, "nm", name="StageSystemSettings.y_limits", strict=strict, extra=self.extra)
        self.z_limits = parse_opt_pair_quantity(self.z_limits, "nm", name="StageSystemSettings.z_limits", strict=strict, extra=self.extra)
        self.r_limits = parse_opt_pair_quantity(self.r_limits, "degree", name="StageSystemSettings.r_limits", strict=strict, extra=self.extra)
        self.tilt_x_limits = parse_opt_pair_quantity(self.tilt_x_limits, "degree", name="StageSystemSettings.tilt_x_limits", strict=strict, extra=self.extra)
        self.tilt_y_limits = parse_opt_pair_quantity(self.tilt_y_limits, "degree", name="StageSystemSettings.tilt_y_limits", strict=strict, extra=self.extra)
        self.max_step_distance = parse_opt_quantity(self.max_step_distance, "nm", name="StageSystemSettings.max_step_distance", strict=strict, extra=self.extra) or Q_(50000.0, "nm")
        self.max_step_angle = parse_opt_quantity(self.max_step_angle, "degree", name="StageSystemSettings.max_step_angle", strict=strict, extra=self.extra) or Q_(1.0, "degree")
        self.eucentric_z = parse_opt_quantity(self.eucentric_z, "nm", name="StageSystemSettings.eucentric_z", strict=strict, extra=self.extra)
        self.settle_time = parse_opt_quantity(self.settle_time, "seconds", name="StageSystemSettings.settle_time", strict=strict, extra=self.extra) or Q_(0.2, "seconds")
        self.timeout = parse_opt_quantity(self.timeout, "seconds", name="StageSystemSettings.timeout", strict=strict, extra=self.extra) or Q_(10.0, "seconds")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True
        def _check_limits(lims, name):
            if lims and lims[0] > lims[1]:
                note_or_raise(self.extra, f"StageSystemSettings.{name}", ValueError("min > max"), mode=mode)
                if strict: return False
                try:
                    setattr(self, name, (lims[1], lims[0]))
                except:
                    pass
            return True

        ok = _check_limits(self.x_limits, "x") and ok
        ok = _check_limits(self.y_limits, "y") and ok
        ok = _check_limits(self.z_limits, "z") and ok
        ok = _check_limits(self.r_limits, "r") and ok
        ok = _check_limits(self.tilt_x_limits, "tilt_x") and ok
        ok = _check_limits(self.tilt_y_limits, "tilt_y") and ok

        # 2. Cross-Field Consistency
        def _check_completeness(enabled: bool, limits: Any, name: str):
            if enabled and limits is None:
                note_or_raise(self.extra, f"StageSystemSettings.{name}_safety",
                              ValueError(f"Axis {name} is enabled but has no safety limits defined."), mode=mode)
                if not strict:
                     # Heal: Disable the unsafe axis
                     try: setattr(self, f"can_{name}", False)
                     except: pass
                else:
                     return False
            return True

        ok = _check_completeness(self.can_x, self.x_limits, "x") and ok
        ok = _check_completeness(self.can_y, self.y_limits, "y") and ok
        ok = _check_completeness(self.can_z, self.z_limits, "z") and ok
        ok = _check_completeness(self.can_r, self.r_limits, "r") and ok
        ok = _check_completeness(self.can_tilt_x, self.tilt_x_limits, "tilt_x") and ok
        ok = _check_completeness(self.can_tilt_y, self.tilt_y_limits, "tilt_y") and ok

        # 3. Parameter Safety
        if self.max_step_distance.magnitude <= 0:
            note_or_raise(self.extra, "StageSystemSettings.max_step_distance",
                          ValueError("max_step_distance must be > 0"), mode=mode)
            if strict:
                ok = False
            else:
                # Heal: Reset to default safe step (50um)
                self.max_step_distance = Q_(50000.0, "nm")

        if self.eucentric_z is not None and self.z_limits:
            z_min, z_max = self.z_limits
            if not (z_min <= self.eucentric_z <= z_max):
                note_or_raise(self.extra, "StageSystemSettings.eucentric_z",
                              ValueError(f"eucentric_z ({self.eucentric_z}) outside z_limits"), mode=mode)
                if strict:
                    ok = False
                else:
                    # Heal: Mark as unknown/unsafe to use
                    self.eucentric_z = None

        if self.settle_time.magnitude < 0:
            note_or_raise(self.extra, "StageSystemSettings.settle_time",
                          ValueError("Settle time must be >= 0"), mode=mode)
            if strict:
                ok = False
            else:
                self.settle_time = Q_(0.2, "seconds")

        return ok

    def is_safe_move(self, target: StagePosition, current: Optional[StagePosition] = None) -> bool:
        """
        RUNTIME CHECK: External Safety.
        Checks if a specific request complies with the validated limits.
        """
        mode = as_parse_mode(self._mode)
        def check(val, lims, name):
            if val is not None and lims and not (lims[0] <= val <= lims[1]):
                note_or_raise(self.extra, f"Safety.{name}", ValueError(f"Target {val} outside {lims}"), mode=mode)
                return False
            return True

        ok = True
        ok = check(target.x, self.x_limits, "x") and ok
        ok = check(target.y, self.y_limits, "y") and ok
        ok = check(target.z, self.z_limits, "z") and ok
        ok = check(target.r, self.r_limits, "r") and ok
        ok = check(target.tilt_x, self.tilt_x_limits, "tilt_x") and ok
        ok = check(target.tilt_y, self.tilt_y_limits, "tilt_y") and ok

        # 2. Relative Step Size Checks (Dynamics)
        if current is not None:
            # Euclidean distance for XY stage movement
            dx = (target.x - current.x) if (target.x is not None and current.x is not None) else Q_(0, 'nm')
            dy = (target.y - current.y) if (target.y is not None and current.y is not None) else Q_(0, 'nm')

            # Simple magnitude check without sqrt optimization for clarity/units
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance > self.max_step_distance:
                note_or_raise(self.extra, "Safety.max_step_distance",
                              ValueError(f"XY move distance {distance} exceeds limit {self.max_step_distance}"),
                              mode=mode)
                ok = False

            # Check tilt step
            if target.tilt_x is not None and current.tilt_x is not None:
                d_tilt = abs(target.tilt_x - current.tilt_x)
                if d_tilt > self.max_step_angle:
                    note_or_raise(self.extra, "Safety.max_step_angle",
                                  ValueError(f"Tilt X step {d_tilt} exceeds limit {self.max_step_angle}"), mode=mode)
                    ok = False

        return ok

    def to_dict(self) -> dict:
        def _sl(v, u): return [serialize_quantity(x, u) for x in v] if v else None
        d = {
            "enabled": self.enabled,
            "can_x": self.can_x, "can_y": self.can_y, "can_z": self.can_z,
            "can_r": self.can_r, "can_tilt_x": self.can_tilt_x, "can_tilt_y": self.can_tilt_y,
            "x_limits_nm": _sl(self.x_limits, "nm"),
            "y_limits_nm": _sl(self.y_limits, "nm"),
            "z_limits_nm": _sl(self.z_limits, "nm"),
            "r_limits_deg": _sl(self.r_limits, "degree"),
            "tilt_x_limits_deg": _sl(self.tilt_x_limits, "degree"),
            "tilt_y_limits_deg": _sl(self.tilt_y_limits, "degree"),
            "max_step_nm": serialize_quantity(self.max_step_distance, "nm"),
            "max_step_deg": serialize_quantity(self.max_step_angle, "degree"),
            "eucentric_z_nm": serialize_quantity(self.eucentric_z, "nm"),
            "settle_time_s": serialize_quantity(self.settle_time, "seconds"),
            "timeout_s": serialize_quantity(self.timeout, "seconds")
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageSystemSettings":
        d_dict, mode, extra = _setup_from_dict(
            StageSystemSettings, d, mode,
            known_keys=("enabled", "can_x", "can_y", "can_z", "can_r", "can_tilt_x", "can_tilt_y",
                        "x_limits", "y_limits", "z_limits", "r_limits", "tilt_x_limits", "tilt_y_limits",
                        "max_step_distance", "max_step_angle", "eucentric_z", "settle_time", "timeout", "extra"),
            aliases=("x_limits_nm", "y_limits_nm", "z_limits_nm", "r_limits_deg", "tilt_x_limits_deg",
                     "tilt_y_limits_deg",
                     "max_step_nm", "max_step_deg", "eucentric_z_nm", "settle_time_s", "timeout_s")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return StageSystemSettings(
            enabled=d_dict.get("enabled", True),
            can_x=d_dict.get("can_x", True), can_y=d_dict.get("can_y", True), can_z=d_dict.get("can_z", True),
            can_r=d_dict.get("can_r", False), can_tilt_x=d_dict.get("can_tilt_x", False),
            can_tilt_y=d_dict.get("can_tilt_y", False),
            x_limits=d_dict.get("x_limits", d_dict.get("x_limits_nm")),
            y_limits=d_dict.get("y_limits", d_dict.get("y_limits_nm")),
            z_limits=d_dict.get("z_limits", d_dict.get("z_limits_nm")),
            r_limits=d_dict.get("r_limits", d_dict.get("r_limits_deg")),
            tilt_x_limits=d_dict.get("tilt_x_limits", d_dict.get("tilt_x_limits_deg")),
            tilt_y_limits=d_dict.get("tilt_y_limits", d_dict.get("tilt_y_limits_deg")),
            max_step_distance=d_dict.get("max_step_distance", d_dict.get("max_step_nm")),
            max_step_angle=d_dict.get("max_step_angle", d_dict.get("max_step_deg")),
            eucentric_z=d_dict.get("eucentric_z", d_dict.get("eucentric_z_nm")),
            settle_time=d_dict.get("settle_time", d_dict.get("settle_time_s")),
            timeout=d_dict.get("timeout", d_dict.get("timeout_s")),
            extra=extra, _mode=mode
        )

@dataclass
class BeamSettings:
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
        # 1. Scalars
        self.spot_size = parse_opt_int(self.spot_size, name="BeamSettings.spot_size", strict=strict, extra=self.extra)
        # 2. Quantities
        self.voltage = parse_opt_quantity(self.voltage, "kV", name="BeamSettings.voltage", strict=strict, extra=self.extra)
        self.beam_current = parse_opt_quantity(self.beam_current, "nA", name="BeamSettings.beam_current", strict=strict, extra=self.extra)
        self.convergence_angle = parse_opt_quantity(self.convergence_angle, "mrad", name="BeamSettings.convergence_angle", strict=strict, extra=self.extra)
        self.defocus = parse_opt_quantity(self.defocus, "nm", name="BeamSettings.defocus", strict=strict, extra=self.extra)
        self.scan_rotation = parse_opt_quantity(self.scan_rotation, "degree", name="BeamSettings.scan_rotation", strict=strict, extra=self.extra)
        # 3. Complex
        self.stigmation = parse_model(Point, self.stigmation, extra=self.extra, key="BeamSettings.stigmation", mode=mode)
        self.beam_shift = parse_model(Point, self.beam_shift, extra=self.extra, key="BeamSettings.beam_shift", mode=mode)
        self.image_shift = parse_model(Point, self.image_shift, extra=self.extra, key="BeamSettings.image_shift", mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True
        if self.stigmation: ok = self.stigmation.validate(mode=mode) and ok
        if self.beam_shift: ok = self.beam_shift.validate(mode=mode) and ok
        if self.image_shift: ok = self.image_shift.validate(mode=mode) and ok

        # Helper to check positivity and heal
        def _check_pos_qty(val, name, attr_name):
            if val is not None and val.magnitude < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode)
                if strict:
                    return False
                setattr(self, attr_name, None)  # Heal: Set to Unknown
            return True

        def _check_pos_int(val, name, attr_name):
            if val is not None and val < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode)
                if strict:
                    return False
                setattr(self, attr_name, None)  # Heal: Set to Unknown
            return True

        ok = _check_pos_qty(self.convergence_angle, "BeamSettings.convergence_angle", "convergence_angle") and ok
        ok = _check_pos_qty(self.voltage, "BeamSettings.voltage", "voltage") and ok
        ok = _check_pos_qty(self.beam_current, "BeamSettings.beam_current", "beam_current") and ok
        ok = _check_pos_int(self.spot_size, "BeamSettings.spot_size", "spot_size") and ok

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
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSettings":
        d_dict, mode, extra = _setup_from_dict(
            BeamSettings, d, mode,
            known_keys=("voltage", "beam_current", "spot_size", "convergence_angle", "defocus", "scan_rotation",
                        "stigmation", "beam_shift", "image_shift", "extra"),
            aliases=("voltage_kv", "beam_current_na", "convergence_angle_mrad", "defocus_nm", "scan_rotation_deg")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return BeamSettings(
            voltage=d_dict.get("voltage", d_dict.get("voltage_kv")),
            beam_current=d_dict.get("beam_current", d_dict.get("beam_current_na")),
            spot_size=d_dict.get("spot_size"),
            convergence_angle=d_dict.get("convergence_angle", d_dict.get("convergence_angle_mrad")),
            defocus=d_dict.get("defocus", d_dict.get("defocus_nm")),
            scan_rotation=d_dict.get("scan_rotation", d_dict.get("scan_rotation_deg")),
            stigmation=d_dict.get("stigmation"), beam_shift=d_dict.get("beam_shift"),
            image_shift=d_dict.get("image_shift"),
            extra=extra, _mode=mode
        )

# Alias for semantic clarity in Read-Only contexts
BeamState = BeamSettings

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
        # 1. Scalars
        self.enabled = parse_bool(self.enabled, default=True, name="BeamSystemSettings.enabled", strict=strict, extra=self.extra)
        self.spot_size_limits = parse_opt_pair_int(self.spot_size_limits, name="BeamSystemSettings.spot_size_limits", strict=strict, extra=self.extra)
        # 2. Quantities
        self.voltage_limits = parse_opt_pair_quantity(self.voltage_limits, "kV", name="BeamSystemSettings.voltage_limits", strict=strict, extra=self.extra)
        self.beam_current_limits = parse_opt_pair_quantity(self.beam_current_limits, "nA", name="BeamSystemSettings.beam_current_limits", strict=strict, extra=self.extra)
        self.convergence_angle_limits = parse_opt_pair_quantity(self.convergence_angle_limits, "mrad", name="BeamSystemSettings.convergence_angle_limits", strict=strict, extra=self.extra)
        # 3. Complex
        self.default_beam = parse_model(BeamSettings, self.default_beam, mode=mode, extra=self.extra) or BeamSettings(_mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = self.default_beam.validate(mode=mode)

        def _check(rng, name):
            if rng:
                mn, mx = rng
                if mn > mx:
                    note_or_raise(self.extra, name, ValueError(f"{name} invalid: min > max"), mode=mode)
                    if strict:
                        return False
                    # Heal: Swap
                    try: setattr(self, name, (mx, mn))
                    except Exception: pass

                # Check for negative lower bounds
                if mn.magnitude < 0:
                    note_or_raise(self.extra, f"BeamSystemSettings.{name}",
                                  ValueError(f"{name} invalid: min < 0"), mode=mode)
                    if strict:
                         return False
                    # Heal: Clamp min to 0
                    try: setattr(self, name, (Q_(0, mn.units), mx))
                    except: pass
            return True

        ok = _check(self.voltage_limits, "voltage_limits") and ok
        ok = _check(self.beam_current_limits, "beam_current_limits") and ok
        ok = _check(self.convergence_angle_limits, "convergence_angle_limits") and ok
        ok = _check(self.spot_size_limits, "spot_size_limits") and ok

        return ok

    def is_safe_beam(self, target: BeamSettings) -> bool:
        """
        Runtime Gatekeeper: Checks if a target beam configuration respects system limits.
        """
        mode = as_parse_mode(self._mode)

        # Helper for limit checking
        def _check(val, limit_tuple, name):
            if val is None or limit_tuple is None: return True
            min_lim, max_lim = limit_tuple
            if not (min_lim <= val <= max_lim):
                note_or_raise(self.extra, f"Safety.beam_{name}",
                              ValueError(f"{name} {val} outside limits {limit_tuple}"), mode=mode)
                return False
            return True
        ok = True
        ok = _check(target.voltage, self.voltage_limits, "voltage") and ok
        ok = _check(target.beam_current, self.beam_current_limits, "current") and ok
        ok = _check(target.convergence_angle, self.convergence_angle_limits, "convergence") and ok

        # Discrete checks
        if target.spot_size is not None and self.spot_size_limits:
            min_s, max_s = self.spot_size_limits
            if not (min_s <= target.spot_size <= max_s):
                note_or_raise(self.extra, "Safety.beam_spot",
                              ValueError(f"Spot size {target.spot_size} outside {self.spot_size_limits}"), mode=mode)
                ok = False

        return ok

    def to_dict(self) -> dict:
        def _sl(v, u): return [serialize_quantity(x, u) for x in v] if v else None
        d = {
            "enabled": self.enabled,
            "default_beam": self.default_beam.to_dict(),
            "voltage_limits_kv": _sl(self.voltage_limits, "kV"),
            "beam_current_limits_na": _sl(self.beam_current_limits, "nA"),
            "spot_size_limits": self.spot_size_limits,
            "convergence_angle_limits_mrad": _sl(self.convergence_angle_limits, "mrad"),
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSystemSettings":
        d_dict, mode, extra = _setup_from_dict(
            BeamSystemSettings, d, mode,
            known_keys=("enabled", "default_beam", "voltage_limits", "beam_current_limits",
                        "spot_size_limits", "convergence_angle_limits", "extra"),
            aliases=("voltage_limits_kv", "beam_current_limits_na", "convergence_angle_limits_mrad")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return BeamSystemSettings(
            enabled=d_dict.get("enabled", True),
            default_beam=d_dict.get("default_beam"),
            voltage_limits=d_dict.get("voltage_limits", d_dict.get("voltage_limits_kv")),
            beam_current_limits=d_dict.get("beam_current_limits", d_dict.get("beam_current_limits_na")),
            spot_size_limits=d_dict.get("spot_size_limits"),
            convergence_angle_limits=d_dict.get("convergence_angle_limits",
                                                d_dict.get("convergence_angle_limits_mrad")),
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
    exposure: Optional["Quantity"] = None
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
        mode, strict, self.extra = _setup_init(self, self._mode, "DetectorSettings")
        # 1. Scalars
        self.detector_id = parse_opt_id(self.detector_id, name="DetectorSettings.detector_id", strict=False, extra=self.extra)
        self.binning_index = parse_opt_int(self.binning_index, name="DetectorSettings.binning_index", strict=strict, extra=self.extra)
        self.binning_xy = parse_opt_pair_int(self.binning_xy, name="DetectorSettings.binning_xy", strict=strict, extra=self.extra)
        self.frame_integration = parse_opt_int(self.frame_integration, name="DetectorSettings.frame_integration", strict=strict, extra=self.extra)
        self.gain_index = parse_opt_int(self.gain_index, name="DetectorSettings.gain_index", strict=strict, extra=self.extra)
        self.offset_index = parse_opt_int(self.offset_index, name="DetectorSettings.offset_index", strict=strict, extra=self.extra)
        self.digital_rotation_deg = parse_opt_float(self.digital_rotation_deg, name="DetectorSettings.digital_rotation_deg", strict=strict, extra=self.extra)
        # 2. Quantities
        self.exposure = parse_opt_quantity(self.exposure, "ms", name="DetectorSettings.exposure", strict=strict, extra=self.extra)
        # 3. Complex
        self.roi = parse_model(ROI, self.roi, mode=mode, extra=self.extra, key="DetectorSettings.roi")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True

        if self.exposure is not None:
             if self.exposure.magnitude <= 0:
                 note_or_raise(self.extra, "DetectorSettings.exposure", ValueError("Exposure must be > 0"), mode=mode)
                 if strict:
                     ok = False
                 else:
                     self.exposure = None

        if self.binning_xy:
             # Semantic check: Binning must be positive integers
             if self.binning_xy[0] <= 0 or self.binning_xy[1] <= 0:
                 note_or_raise(self.extra, "DetectorSettings.binning_xy", ValueError("binning_xy must be > 0"), mode=mode, raw=self.binning_xy)
                 if strict:
                     ok = False
                 else:
                     self.binning_xy = None

        if self.frame_integration is not None and self.frame_integration < 1:
            note_or_raise(self.extra, "DetectorSettings.frame_integration",
                          ValueError(f"Frame integration must be >= 1, got {self.frame_integration}"), mode=mode)
            if strict:
                ok = False
            else:
                self.frame_integration = 1

        if self.gain_index is not None and self.gain_index < 0:
            note_or_raise(self.extra, "DetectorSettings.gain_index",
                          ValueError("Gain index must be >= 0"), mode=mode)
            if strict:
                ok = False
            else:
                self.gain_index = 0

        if self.roi:
            if not self.roi.validate(mode=mode):
                ok = False
        return ok

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "exposure_ms": serialize_quantity(self.exposure, "ms"),
            "binning_index": self.binning_index,
            "binning_xy": self.binning_xy,
            "frame_integration": self.frame_integration,
            "gain_index": self.gain_index,
            "offset_index": self.offset_index,
            "digital_rotation_deg": self.digital_rotation_deg,
            "roi": self.roi.to_dict() if self.roi else None
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSettings":
        d_dict, mode, extra = _setup_from_dict(
            DetectorSettings, d, mode,
            known_keys=("detector_id", "exposure", "binning_index", "binning_xy", "roi",
                        "frame_integration", "gain_index", "offset_index", "digital_rotation_deg", "extra"),
            aliases=("exposure_ms",)
        )
        if d_dict is None: return replace(d, _mode=mode)
        return DetectorSettings(
            detector_id=d_dict.get("detector_id"),
            exposure=d_dict.get("exposure", d_dict.get("exposure_ms")),
            binning_index=d_dict.get("binning_index"),
            binning_xy=d_dict.get("binning_xy"),
            roi=d_dict.get("roi"),
            frame_integration=d_dict.get("frame_integration"),
            gain_index=d_dict.get("gain_index"),
            offset_index=d_dict.get("offset_index"),
            digital_rotation_deg=d_dict.get("digital_rotation_deg"),
            extra=extra, _mode=mode
        )

# Alias for semantic clarity in Read-Only contexts
DetectorState = DetectorSettings

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
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "DetectorCapabilities")
        # 1. Booleans (Fail Fast)
        self.can_binning = parse_opt_bool(self.can_binning, name="DetectorCapabilities.can_binning", strict=strict, extra=self.extra)
        self.can_gain = parse_opt_bool(self.can_gain, name="DetectorCapabilities.can_gain", strict=strict, extra=self.extra)
        self.can_offset = parse_opt_bool(self.can_offset, name="DetectorCapabilities.can_offset", strict=strict, extra=self.extra)
        self.can_digital_rotation = parse_opt_bool(self.can_digital_rotation, name="DetectorCapabilities.can_digital_rotation", strict=strict, extra=self.extra)
        # 2. Scalars/Ranges
        self.binning_index_min = parse_opt_int(self.binning_index_min, name="DetectorCapabilities.binning_index_min", strict=strict, extra=self.extra)
        self.binning_index_max = parse_opt_int(self.binning_index_max, name="DetectorCapabilities.binning_index_max", strict=strict, extra=self.extra)
        self.binning_xy_min = parse_opt_pair_int(self.binning_xy_min, name="DetectorCapabilities.binning_xy_min", strict=strict, extra=self.extra)
        self.binning_xy_max = parse_opt_pair_int(self.binning_xy_max, name="DetectorCapabilities.binning_xy_max", strict=strict, extra=self.extra)
        self.exposure_ms_min = parse_opt_float(self.exposure_ms_min, name="DetectorCapabilities.exposure_ms_min", strict=strict, extra=self.extra)
        self.exposure_ms_max = parse_opt_float(self.exposure_ms_max, name="DetectorCapabilities.exposure_ms_max", strict=strict, extra=self.extra)
        self.frame_integration_min = parse_opt_int(self.frame_integration_min, name="DetectorCapabilities.frame_integration_min", strict=strict, extra=self.extra)
        self.frame_integration_max = parse_opt_int(self.frame_integration_max, name="DetectorCapabilities.frame_integration_max", strict=strict, extra=self.extra)
        self.roi_size_min = parse_opt_pair_int(self.roi_size_min, name="DetectorCapabilities.roi_size_min", strict=strict, extra=self.extra)
        self.roi_size_max = parse_opt_pair_int(self.roi_size_max, name="DetectorCapabilities.roi_size_max", strict=strict, extra=self.extra)
        self.gain_index_min = parse_opt_int(self.gain_index_min, name="DetectorCapabilities.gain_index_min", strict=strict, extra=self.extra)
        self.gain_index_max = parse_opt_int(self.gain_index_max, name="DetectorCapabilities.gain_index_max", strict=strict, extra=self.extra)
        self.offset_index_min = parse_opt_int(self.offset_index_min, name="DetectorCapabilities.offset_index_min", strict=strict, extra=self.extra)
        self.offset_index_max = parse_opt_int(self.offset_index_max, name="DetectorCapabilities.offset_index_max", strict=strict, extra=self.extra)
        self.digital_rotation_deg_min = parse_opt_float(self.digital_rotation_deg_min, name="DetectorCapabilities.digital_rotation_deg_min", strict=strict, extra=self.extra)
        self.digital_rotation_deg_max = parse_opt_float(self.digital_rotation_deg_max, name="DetectorCapabilities.digital_rotation_deg_max", strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True

        def _check_range(min_val, max_val, name, min_attr, max_attr):
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    note_or_raise(self.extra, name, ValueError(f"{name} invalid: min > max ({min_val} > {max_val})"), mode=mode)
                    if strict:
                        return False
                    # Heal: Swap limits
                    try:
                        setattr(self, min_attr, max_val)
                        setattr(self, max_attr, min_val)
                    except Exception:
                        pass
            return True

        def _check_pos(val, name, attr_name):
            if val is not None and val < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode, raw=val)
                if strict:
                    return False
                # Heal: Clamp to 0
                setattr(self, attr_name, 0.0)
            return True

        # 1. Range Consistency
        # Note: We pass attribute names to allow swapping in _check_range
        ok = _check_range(self.binning_index_min, self.binning_index_max, "DetectorCapabilities.binning_index", "binning_index_min", "binning_index_max") and ok
        ok = _check_range(self.frame_integration_min, self.frame_integration_max, "DetectorCapabilities.frame_integration", "frame_integration_min", "frame_integration_max") and ok
        ok = _check_range(self.gain_index_min, self.gain_index_max, "DetectorCapabilities.gain_index", "gain_index_min", "gain_index_max") and ok
        ok = _check_range(self.offset_index_min, self.offset_index_max, "DetectorCapabilities.offset_index", "offset_index_min", "offset_index_max") and ok
        ok = _check_range(self.exposure_ms_min, self.exposure_ms_max, "DetectorCapabilities.exposure_ms", "exposure_ms_min", "exposure_ms_max") and ok

        # 2. Physical Non-negativity
        ok = _check_pos(self.exposure_ms_min, "DetectorCapabilities.exposure_ms_min", "exposure_ms_min") and ok

        # 3. Tuple consistency (ROI/Binning)
        if self.roi_size_min and self.roi_size_max:
             if self.roi_size_min[0] > self.roi_size_max[0] or self.roi_size_min[1] > self.roi_size_max[1]:
                 note_or_raise(self.extra, "DetectorCapabilities.roi_size", ValueError("ROI size min > max"), mode=mode)
                 if strict:
                     ok = False
                 else:
                     # Heal: Swap X and Y components individually
                     new_min_x = min(self.roi_size_min[0], self.roi_size_max[0])
                     new_max_x = max(self.roi_size_min[0], self.roi_size_max[0])
                     new_min_y = min(self.roi_size_min[1], self.roi_size_max[1])
                     new_max_y = max(self.roi_size_min[1], self.roi_size_max[1])
                     self.roi_size_min = (new_min_x, new_min_y)
                     self.roi_size_max = (new_max_x, new_max_y)

        return ok

    def supports(self, settings: DetectorSettings) -> bool:
        if settings.detector_id:
             pass

        if settings.binning_xy:
            bx, by = settings.binning_xy
            if self.binning_xy_max:
                max_x, max_y = self.binning_xy_max
                if bx > max_x or by > max_y: return False
            if self.can_binning is False and (bx > 1 or by > 1): return False

        if settings.exposure:
            if self.exposure_ms_min and settings.exposure < Q_(self.exposure_ms_min, 'ms'): return False
            if self.exposure_ms_max and settings.exposure > Q_(self.exposure_ms_max, 'ms'): return False

        return True

    def to_dict(self) -> dict:
        # FIX: Explicit listing instead of dynamic iteration
        d = {
            "can_binning": self.can_binning,
            "binning_index_min": self.binning_index_min,
            "binning_index_max": self.binning_index_max,
            "binning_xy_min": list(self.binning_xy_min) if self.binning_xy_min else None,
            "binning_xy_max": list(self.binning_xy_max) if self.binning_xy_max else None,
            "exposure_ms_min": self.exposure_ms_min,
            "exposure_ms_max": self.exposure_ms_max,
            "frame_integration_min": self.frame_integration_min,
            "frame_integration_max": self.frame_integration_max,
            "roi_size_min": list(self.roi_size_min) if self.roi_size_min else None,
            "roi_size_max": list(self.roi_size_max) if self.roi_size_max else None,
            "can_gain": self.can_gain,
            "gain_index_min": self.gain_index_min,
            "gain_index_max": self.gain_index_max,
            "can_offset": self.can_offset,
            "offset_index_min": self.offset_index_min,
            "offset_index_max": self.offset_index_max,
            "can_digital_rotation": self.can_digital_rotation,
            "digital_rotation_deg_min": self.digital_rotation_deg_min,
            "digital_rotation_deg_max": self.digital_rotation_deg_max,
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "DetectorCapabilities":
        d_dict, mode, extra = _setup_from_dict(
            DetectorCapabilities, d, mode,
            known_keys=("can_binning", "binning_index_min", "binning_index_max", "binning_xy_min", "binning_xy_max",
                        "exposure_ms_min", "exposure_ms_max", "frame_integration_min", "frame_integration_max",
                        "roi_size_min", "roi_size_max", "can_gain", "gain_index_min", "gain_index_max",
                        "can_offset", "offset_index_min", "offset_index_max",
                        "can_digital_rotation", "digital_rotation_deg_min", "digital_rotation_deg_max", "extra"),
            aliases=("roi_min", "roi_max")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return DetectorCapabilities(
            can_binning=d_dict.get("can_binning"),
            binning_index_min=d_dict.get("binning_index_min"),
            binning_index_max=d_dict.get("binning_index_max"),
            binning_xy_min=d_dict.get("binning_xy_min"),
            binning_xy_max=d_dict.get("binning_xy_max"),
            exposure_ms_min=d_dict.get("exposure_ms_min"),
            exposure_ms_max=d_dict.get("exposure_ms_max"),
            frame_integration_min=d_dict.get("frame_integration_min"),
            frame_integration_max=d_dict.get("frame_integration_max"),
            roi_size_min=d_dict.get("roi_size_min", d_dict.get("roi_min")),
            roi_size_max=d_dict.get("roi_size_max", d_dict.get("roi_max")),
            can_gain=d_dict.get("can_gain"),
            gain_index_min=d_dict.get("gain_index_min"),
            gain_index_max=d_dict.get("gain_index_max"),
            can_offset=d_dict.get("can_offset"),
            offset_index_min=d_dict.get("offset_index_min"),
            offset_index_max=d_dict.get("offset_index_max"),
            can_digital_rotation=d_dict.get("can_digital_rotation"),
            digital_rotation_deg_min=d_dict.get("digital_rotation_deg_min"),
            digital_rotation_deg_max=d_dict.get("digital_rotation_deg_max"),
            extra=extra, _mode=mode
        )

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
        # 1. Scalars
        self.enabled = parse_bool(self.enabled, default=True, name="DetectorSystemSettings.enabled", strict=strict, extra=self.extra)
        self.default_detector_id = parse_opt_id(self.default_detector_id, name="DetectorSystemSettings.default_detector_id", strict=False, extra=self.extra)
        # 2. Lists
        self.available_detector_ids = parse_str_list(self.available_detector_ids, name="DetectorSystemSettings.available_detector_ids", strict=False, extra=self.extra)
        self.available_detector_ids = list(dict.fromkeys(self.available_detector_ids))
        # 3. Maps
        self.defaults_by_id = parse_keyed_map(DetectorSettings, self.defaults_by_id, "detector_id", "DetectorSystemSettings.defaults_by_id", mode, self.extra)
        self.capabilities_by_id = parse_keyed_map(DetectorCapabilities, self.capabilities_by_id, None, "DetectorSystemSettings.capabilities_by_id", mode, self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate detector-system semantics and cross-field consistency."""
        mode, strict = _setup_validate(self._mode, mode)
        ok = True

        # Rule 2B: Cross-Field Consistency
        # We must ensure that the sets of IDs make sense together.

        ids_available = set(self.available_detector_ids)
        ids_defaults = set(self.defaults_by_id.keys())
        ids_caps = set(self.capabilities_by_id.keys())

        # Check 1: Default Selection Validity
        if self.default_detector_id:
            # Note: We trust default_detector_id is str or None (from __post_init__)
            known_anywhere = ids_available | ids_defaults | ids_caps
            if known_anywhere and self.default_detector_id not in known_anywhere:
                note_or_raise(
                    self.extra,
                    "DetectorSystemSettings.default_detector_id",
                    ValueError(f"Selected default '{self.default_detector_id}' is unknown."),
                    mode=mode
                )
                if strict:
                    ok = False
                else:
                    self.default_detector_id = None

        # Check 2: Configuration Completeness
        # If a detector is 'available', it MUST have settings and capabilities to be usable.
        missing_defaults = ids_available - ids_defaults
        if missing_defaults:
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.completeness",
                ValueError(f"Available detectors missing default settings: {missing_defaults}"),
                mode=mode
            )
            # Generally broken config, but in lenient we might just proceed.
            if strict:
                ok = False

        missing_caps = ids_available - ids_caps
        if missing_caps:
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.completeness",
                ValueError(f"Available detectors missing capabilities: {missing_caps}"),
                mode=mode
            )
            if strict:
                ok = False

        for ds in self.defaults_by_id.values():
            ok = ds.validate(mode=mode) and ok

        for cap in self.capabilities_by_id.values():
            ok = cap.validate(mode=mode) and ok

        return ok

    def is_supported(self, settings: DetectorSettings) -> bool:
        if not settings.detector_id: return False

        caps = self.capabilities_by_id.get(settings.detector_id)
        return caps.supports(settings) if caps else True

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "default_detector_id": self.default_detector_id,
            "available_detector_ids": self.available_detector_ids,
            "defaults_by_id": {k: v.to_dict() for k, v in self.defaults_by_id.items()},
            "capabilities_by_id": {k: v.to_dict() for k, v in self.capabilities_by_id.items()},
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSystemSettings":
        d_dict, mode, extra = _setup_from_dict(
            DetectorSystemSettings, d, mode,
            known_keys=("enabled", "default_detector_id", "available_detector_ids", "defaults_by_id",
                        "capabilities_by_id", "extra"),
            aliases=("available_detectors",)
        )
        if d_dict is None: return replace(d, _mode=mode)
        return DetectorSystemSettings(
            enabled=d_dict.get("enabled", True),
            available_detector_ids=d_dict.get("available_detector_ids", d_dict.get("available_detectors", [])),
            default_detector_id=d_dict.get("default_detector_id"),
            defaults_by_id=d_dict.get("defaults_by_id"),
            capabilities_by_id=d_dict.get("capabilities_by_id"),
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
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "ImageOutputSettings")
        self.file_format = parse_opt_str(self.file_format, name="ImageOutputSettings.file_format", strict=strict, extra=self.extra) or "tiff"
        self.file_format = self.file_format.lower()
        self.path = parse_opt_str(self.path, name="ImageOutputSettings.path", strict=strict, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        if self.file_format not in {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}:
            note_or_raise(self.extra, "ImageOutputSettings.file_format", ValueError(f"Unsupported format: {self.file_format}"), mode=mode)
            if strict: return False
            self.file_format = "tiff"
        return True

    def to_dict(self) -> dict:
        d = {"file_format": self.file_format, "path": self.path}
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ImageOutputSettings":
        d_dict, mode, extra = _setup_from_dict(
            ImageOutputSettings, d, mode,
            known_keys=("file_format", "path", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return ImageOutputSettings(
            file_format=d_dict.get("file_format", "tiff"),
            path=d_dict.get("path"),
            extra=extra, _mode=mode
        )

@dataclass
class Aperture:
    aperture_id: Optional[str] = None
    inserted: bool = False
    size_index: Optional[int] = None
    position: Optional[Point] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "Aperture")
        # 1. Scalars
        self.aperture_id = parse_opt_id(self.aperture_id, name="Aperture.aperture_id", strict=strict, extra=self.extra)
        self.inserted = parse_bool(self.inserted, default=False, name="Aperture.inserted", strict=strict, extra=self.extra)
        self.size_index = parse_opt_int(self.size_index, name="Aperture.size_index", strict=strict, extra=self.extra)
        # 2. Complex
        self.position = parse_model(Point, self.position, extra=self.extra, key="Aperture.position", mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True
        if self.size_index is not None and self.size_index < 0:
            note_or_raise(
                self.extra, "Aperture.size_index",
                ValueError(f"size_index must be >= 0, got {self.size_index}"),
                mode=mode, raw=self.size_index
            )
            if strict: ok = False
            else:
                self.size_index = None  # Heal: Unknown size

        if self.position is not None:
            # Check if the nested object is valid
            if not self.position.validate(mode=mode):
                note_or_raise(self.extra, "Aperture.position", ValueError("Invalid aperture position coordinates"),
                              mode=mode)
                if strict:
                    ok = False
                else:
                    # Heal: Discard the invalid position data, keep the aperture info
                    self.position = None

        return ok

    def to_dict(self) -> dict:
        d = {
            "aperture_id": self.aperture_id,
            "inserted": self.inserted,
            "size_index": self.size_index,
            "position": self.position.to_dict() if self.position else None
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Aperture":
        d_dict, mode, extra = _setup_from_dict(
            Aperture, d, mode,
            known_keys=("aperture_id", "inserted", "size_index", "position", "extra"),
            aliases=("id",)
        )
        if d_dict is None: return replace(d, _mode=mode)
        return Aperture(
            aperture_id=d_dict.get("aperture_id", d_dict.get("id")),
            inserted=d_dict.get("inserted", False),
            size_index=d_dict.get("size_index"),
            position=d_dict.get("position"),
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
        apertures: Dictionary of current aperture states.
        detectors: Dictionary of current detector States.
        active_detector_ids: List of detectors currently marked as active.
        primary_detector_id: The ID of the currently selected main detector.

    Notes:
        Typically instantiated in `LENIENT` mode for logging/telemetry to preserve data despite partial failures.
    """
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    mode: Optional[str] = None
    stage_position: StagePosition = field(default_factory=StagePosition)
    beam: BeamState = field(default_factory=BeamState)
    apertures: Dict[str, Aperture] = field(default_factory=dict)
    detectors: Dict[str, DetectorState] = field(default_factory=dict)
    active_detector_ids: List[str] = field(default_factory=list)
    primary_detector_id: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "MicroscopeState")
        # 1. Scalars
        self.timestamp = parse_opt_str(self.timestamp, name="timestamp", strict=False) or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.mode = parse_opt_str(self.mode, name="MicroscopeState.mode", strict=strict, extra=self.extra)
        self.primary_detector_id = parse_opt_id(self.primary_detector_id, name="MicroscopeState.primary_detector_id", strict=strict, extra=self.extra)
        # 2. Lists
        self.active_detector_ids = parse_str_list(self.active_detector_ids, name="MicroscopeState.active_detector_ids", strict=strict, extra=self.extra)
        # 3. Complex
        self.stage_position = parse_model(StagePosition, self.stage_position, mode=mode, extra=self.extra) or StagePosition()
        self.beam = parse_model(BeamState, self.beam, mode=mode, extra=self.extra) or BeamState()
        self.apertures = parse_keyed_map(Aperture, self.apertures, "aperture_id", "MicroscopeState.apertures", mode, self.extra)
        self.detectors = parse_keyed_map(DetectorState, self.detectors, "detector_id", "MicroscopeState.detectors", mode, self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
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
                if strict:
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
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeState":
        d_dict, mode, extra = _setup_from_dict(
            MicroscopeState, d, mode,
            known_keys=("timestamp", "mode", "stage_position", "beam", "apertures", "detectors",
                        "active_detector_ids", "primary_detector_id", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return MicroscopeState(
            timestamp=d_dict.get("timestamp"), mode=d_dict.get("mode"),
            stage_position=d_dict.get("stage_position"), beam=d_dict.get("beam"),
            apertures=d_dict.get("apertures"), detectors=d_dict.get("detectors"),
            active_detector_ids=d_dict.get("active_detector_ids"),
            primary_detector_id=d_dict.get("primary_detector_id"),
            extra=extra, _mode=mode
        )

@dataclass
class MicroscopeImageMetadata:
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
        # 1. Scalars
        self.version = parse_opt_str(self.version, name="version", strict=False) or str(METADATA_VERSION)
        self.created_at = parse_opt_str(self.created_at, name="created_at", strict=False) or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.magnification = parse_opt_float(self.magnification, name="MicroscopeImageMetadata.magnification", strict=strict, extra=self.extra)
        self.camera_length_mm = parse_opt_float(self.camera_length_mm, unit="mm", name="MicroscopeImageMetadata.camera_length_mm", strict=strict, extra=self.extra)
        self.accelerating_voltage_kv = parse_opt_float(self.accelerating_voltage_kv, unit="kV", name="MicroscopeImageMetadata.accelerating_voltage_kv", strict=strict, extra=self.extra)
        self.beam_current_na = parse_opt_float(self.beam_current_na, unit="nA", name="MicroscopeImageMetadata.beam_current_na", strict=strict, extra=self.extra)
        self.exposure_ms = parse_opt_float(self.exposure_ms, unit="ms", name="MicroscopeImageMetadata.exposure_ms", strict=strict, extra=self.extra)
        self.pixel_size_nm = parse_opt_pair_float(self.pixel_size_nm, name="MicroscopeImageMetadata.pixel_size_nm", strict=strict, extra=self.extra)
        self.image_size_px = parse_opt_pair_int(self.image_size_px, name="MicroscopeImageMetadata.image_size_px", strict=strict, extra=self.extra)
        # 2. Complex
        self.microscope_state = parse_model(MicroscopeState, self.microscope_state, mode=mode, extra=self.extra)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True

        if self.microscope_state:
            ok = self.microscope_state.validate(mode=mode) and ok

        def _check_pos(val, name, attr_name):
            if val is not None and val < 0:
                note_or_raise(self.extra, name, ValueError(f"{name} must be >= 0"), mode=mode, raw=val)
                if strict:
                    return False
                setattr(self, attr_name, None)  # Heal: Set to Unknown
            return True

        ok = _check_pos(self.magnification, "magnification", "magnification") and ok
        ok = _check_pos(self.exposure_ms, "exposure_ms", "exposure_ms") and ok
        ok = _check_pos(self.accelerating_voltage_kv, "accelerating_voltage_kv", "accelerating_voltage_kv") and ok

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
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeImageMetadata":
        d_dict, mode, extra = _setup_from_dict(
            MicroscopeImageMetadata, d, mode,
            known_keys=("version", "created_at", "magnification", "camera_length_mm", "pixel_size_nm", "image_size_px",
                        "accelerating_voltage_kv", "beam_current_na", "exposure_ms", "microscope_state", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return MicroscopeImageMetadata(
            version=d_dict.get("version", str(METADATA_VERSION)), created_at=d_dict.get("created_at"),
            magnification=d_dict.get("magnification"), camera_length_mm=d_dict.get("camera_length_mm"),
            pixel_size_nm=d_dict.get("pixel_size_nm"), image_size_px=d_dict.get("image_size_px"),
            accelerating_voltage_kv=d_dict.get("accelerating_voltage_kv"),
            beam_current_na=d_dict.get("beam_current_na"),
            exposure_ms=d_dict.get("exposure_ms"), microscope_state=d_dict.get("microscope_state"),
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
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "SystemInfo")
        self.name = parse_opt_str(self.name, name="SystemInfo.name", strict=strict, extra=self.extra) or "Unknown"
        self.ip_address = parse_opt_str(self.ip_address, name="SystemInfo.ip_address", strict=strict, extra=self.extra) or "Unknown"
        self.manufacturer = parse_opt_str(self.manufacturer, name="SystemInfo.manufacturer", strict=strict, extra=self.extra) or "Unknown"
        self.model = parse_opt_str(self.model, name="SystemInfo.model", strict=strict, extra=self.extra) or "Unknown"
        self.serial_number = parse_opt_str(self.serial_number, name="SystemInfo.serial_number", strict=strict, extra=self.extra) or "Unknown"
        self.hardware_version = parse_opt_str(self.hardware_version, name="SystemInfo.hardware_version", strict=strict, extra=self.extra) or "Unknown"
        self.software_version = parse_opt_str(self.software_version, name="SystemInfo.software_version", strict=strict, extra=self.extra) or "Unknown"
        self.supertem_version = parse_opt_str(self.supertem_version, name="SystemInfo.supertem_version", strict=strict, extra=self.extra) or __version__
        self.application = parse_opt_str(self.application, name="SystemInfo.application", strict=strict, extra=self.extra) or "Unknown"
        self.application_version = parse_opt_str(self.application_version, name="SystemInfo.application_version", strict=strict, extra=self.extra) or "Unknown"

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)

        ip = self.ip_address.strip()
        if ip and ip != "Unknown":
            try:
                ipaddress.ip_address(ip)
            except Exception:
                note_or_raise(self.extra, "SystemInfo.ip_address", ValueError(f"Invalid IP: {self.ip_address!r}"), mode=mode, raw=self.ip_address)
                if strict:
                    return False
                self.ip_address = "Unknown"
        return True

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "ip_address": self.ip_address,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "hardware_version": self.hardware_version,
            "software_version": self.software_version,
            "supertem_version": self.supertem_version,
            "application": self.application,
            "application_version": self.application_version,
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemInfo":
        d_dict, mode, extra = _setup_from_dict(
            SystemInfo, d, mode,
            known_keys=("name", "ip_address", "manufacturer", "model", "serial_number",
                        "hardware_version", "software_version", "supertem_version",
                        "application", "application_version", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return SystemInfo(
            name=d_dict.get("name"),
            ip_address=d_dict.get("ip_address"),
            manufacturer=d_dict.get("manufacturer"),
            model=d_dict.get("model"),
            serial_number=d_dict.get("serial_number"),
            hardware_version=d_dict.get("hardware_version"),
            software_version=d_dict.get("software_version"),
            supertem_version=d_dict.get("supertem_version", __version__),
            application=d_dict.get("application"),
            application_version=d_dict.get("application_version"),
            extra=extra,
            _mode=mode
        )

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
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "SystemSettings")
        self.stage_system = parse_model(StageSystemSettings, self.stage_system, mode=mode, extra=self.extra) or StageSystemSettings(_mode=mode)
        self.beam_system = parse_model(BeamSystemSettings, self.beam_system, mode=mode, extra=self.extra) or BeamSystemSettings(_mode=mode)
        self.detector_system = parse_model(DetectorSystemSettings, self.detector_system, mode=mode, extra=self.extra) or DetectorSystemSettings(_mode=mode)
        self.info = parse_model(SystemInfo, self.info, mode=mode, extra=self.extra) or SystemInfo(_mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        return (self.stage_system.validate(mode=mode) and
                self.beam_system.validate(mode=mode) and
                self.detector_system.validate(mode=mode) and
                self.info.validate(mode=mode))

    def to_dict(self) -> dict:
        d = {
            "stage_system": self.stage_system.to_dict(),
            "beam_system": self.beam_system.to_dict(),
            "detector_system": self.detector_system.to_dict(),
            "info": self.info.to_dict(),
        }
        # FIX: Include extras
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemSettings":
        d_dict, mode, extra = _setup_from_dict(
            SystemSettings, d, mode,
            known_keys=("stage_system", "beam_system", "detector_system", "info", "extra"),
            aliases=("stage", "beam", "detector")
        )
        if d_dict is None: return replace(d, _mode=mode)
        return SystemSettings(
            stage_system=d_dict.get("stage_system", d_dict.get("stage")),
            beam_system=d_dict.get("beam_system", d_dict.get("beam")),
            detector_system=d_dict.get("detector_system", d_dict.get("detector")),
            info=d_dict.get("info"), extra=extra, _mode=mode
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
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "MicroscopeSettings")
        self.system = parse_model(SystemSettings, self.system, mode=mode, extra=self.extra) or SystemSettings(_mode=mode)
        self.image = parse_model(ImageOutputSettings, self.image, mode=mode, extra=self.extra) or ImageOutputSettings(_mode=mode)
        if not isinstance(self.protocol, dict): self.protocol = {"name": "demo"}

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        return self.system.validate(mode=mode) and self.image.validate(mode=mode)

    def to_dict(self) -> dict:
        d = {
            "system": self.system.to_dict(),
            "image": self.image.to_dict(),
            "protocol": self.protocol,
        }
        # FIX: Include extras
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeSettings":
        d_dict, mode, extra = _setup_from_dict(
            MicroscopeSettings, d, mode,
            known_keys=("system", "image", "protocol", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return MicroscopeSettings(system=d_dict.get("system"), image=d_dict.get("image"),
                                  protocol=d_dict.get("protocol"), extra=extra,
                                  _mode=mode)

# =============================================================================
# Requests
# =============================================================================

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
        mode, strict, self.extra = _setup_init(self, self._mode, "AcquisitionRequest")
        # 1. Scalars
        self.detector_id = parse_opt_id(self.detector_id, name="AcquisitionRequest.detector_id", strict=strict, extra=self.extra)
        # 2. Complex
        self.detector = parse_model(DetectorSettings, self.detector, mode=mode, extra=self.extra) or DetectorSettings(_mode=mode)
        self.image = parse_model(ImageOutputSettings, self.image, mode=mode, extra=self.extra) or ImageOutputSettings(_mode=mode)
        # 3. Patching
        if self.detector_id is None and self.detector.detector_id is not None:
            self.detector_id = self.detector.detector_id
        elif self.detector.detector_id is None and self.detector_id is not None:
            self.detector.detector_id = self.detector_id

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        # VALIDATION: Consistency & Readiness
        mode, strict = _setup_validate(self._mode, mode)
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
            if strict:
                ok = False
            else:
                self.detector.detector_id = self.detector_id

        return ok

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "detector": self.detector.to_dict(),
            "image": self.image.to_dict(),
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "AcquisitionRequest":
        d_dict, mode, extra = _setup_from_dict(
            AcquisitionRequest, d, mode,
            known_keys=("detector_id", "detector", "image", "extra"),
            aliases=()
        )
        if d_dict is None: return replace(d, _mode=mode)
        return AcquisitionRequest(detector_id=d_dict.get("detector_id"), detector=d_dict.get("detector"),
                                  image=d_dict.get("image"), extra=extra, _mode=mode)


@dataclass
class StageMoveRequest:
    """
    Explicit intent to move the microscope stage.

    Attributes:
        target: The coordinate goals (absolute or relative vectors).
        relative: If True, target values are added to current position (deltas).
        backlash_correction: Whether to perform hardware backlash compensation.
        wait_for_settle: If True, blocks until movement and settling are complete.
        settle_time: Optional duration to wait after movement stops.
                     If None, uses the default from StageSystemSettings.
    """
    target: StagePosition = field(default_factory=StagePosition)
    relative: bool = False
    backlash_correction: bool = True
    wait_for_settle: bool = True
    settle_time: Optional["Quantity"] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        mode, strict, self.extra = _setup_init(self, self._mode, "StageMoveRequest")
        # 1. Scalars
        self.relative = parse_bool(self.relative, default=False, name="StageMoveRequest.relative", strict=strict, extra=self.extra)
        self.backlash_correction = parse_bool(self.backlash_correction, default=True, name="StageMoveRequest.backlash_correction", strict=strict, extra=self.extra)
        self.wait_for_settle = parse_bool(self.wait_for_settle, default=True, name="StageMoveRequest.wait_for_settle", strict=strict, extra=self.extra)
        # 2. Quantities
        self.settle_time = parse_opt_quantity(self.settle_time, "seconds", name="StageMoveRequest.settle_time", strict=strict, extra=self.extra)
        # 3. Complex
        self.target = parse_model(StagePosition, self.target, mode=mode, extra=self.extra) or StagePosition(_mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode, strict = _setup_validate(self._mode, mode)
        ok = True

        if not self.target.validate(mode=mode):
            ok = False

        # Check for empty request
        axes = [self.target.x, self.target.y, self.target.z,
                self.target.r, self.target.tilt_x, self.target.tilt_y]
        if all(a is None for a in axes):
            note_or_raise(self.extra, "StageMoveRequest.empty",
                          ValueError("StageMoveRequest has no target coordinates"),
                          mode=mode)
            if strict: ok = False

        # Check for negative time
        if self.settle_time is not None and self.settle_time.magnitude < 0:
            note_or_raise(self.extra, "StageMoveRequest.settle_time",
                          ValueError("Settle time cannot be negative"), mode=mode)
            if strict:
                ok = False
            else:
                self.settle_time = None  # Heal

        return ok

    def to_dict(self) -> dict:
        d = {
            "target": self.target.to_dict(),
            "relative": self.relative,
            "backlash_correction": self.backlash_correction,
            "wait_for_settle": self.wait_for_settle,
            "settle_time_s": serialize_quantity(self.settle_time, "seconds"),
        }
        return _finish_to_dict(d, self.extra)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageMoveRequest":
        d_dict, mode, extra = _setup_from_dict(
            StageMoveRequest, d, mode,
            known_keys=("target", "relative", "backlash_correction", "wait_for_settle", "settle_time", "extra"),
            aliases=("settle_time_s",)
        )
        if d_dict is None: return replace(d, _mode=mode)
        return StageMoveRequest(
            target=d_dict.get("target"), relative=d_dict.get("relative"),
            backlash_correction=d_dict.get("backlash_correction"),
            wait_for_settle=d_dict.get("wait_for_settle"),
            settle_time=d_dict.get("settle_time", d_dict.get("settle_time_s")),
            extra=extra, _mode=mode
        )