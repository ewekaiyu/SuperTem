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
III. Normalization vs Validation (Separation of Concerns)
===============================================================================

Normalization (parse step)
  Purpose:
    Convert untrusted input into a typed internal structure without discarding
    information.

  Typical operations:
    - dict -> dataclass conversion for nested fields
    - scalar coercion where appropriate (e.g., "128" -> 128)
    - Extras normalization and unknown-key capture
    - move rejected values into Extras.raw and record diagnostics in Extras.notes

  Policy:
    - __post_init__ performs normalization only
    - normalization should not enforce domain/physics constraints as business rules

Validation (semantic step)
  Purpose:
    Enforce domain constraints and invariants.

  Typical checks:
    - range / non-negativity constraints
    - required fields for executable requests
    - cross-field consistency
    - capability-limited bounds when available

  Policy:
    - validate(mode=...) is the single authoritative place for semantic constraints
    - STRICT: raise via note_or_raise(...)
    - LENIENT: record issue and repair-to-safe / disable unsafe fields

Rule of thumb:
  - Wrong type/shape => normalization concern
  - Right type but invalid/unsafe value => validation concern

===============================================================================
IV. Extras: Preservation and Diagnostics
===============================================================================

`Extras` is the structured container for non-canonical information:
  - vendor_properties: vendor-specific fields (prefer namespaced keys)
  - unknown_keys: unrecognized keys observed during parsing
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
    3) call validate(mode=_mode) in a mode-aware manner (or defer to boundary)
- from_dict(...) should be thin:
    - construct with _mode set
    - rely on __post_init__ + validate for consistent behavior

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
from dataclasses import dataclass, field, fields
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


# -----------------------
# Parsing modes
# -----------------------

class ParseMode(str, Enum):
    STRICT = "strict"
    LENIENT = "lenient"


def as_parse_mode(mode: Union["ParseMode", str, None]) -> "ParseMode":
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
    return as_parse_mode(mode) == ParseMode.STRICT


def note_or_raise(
    extra: Optional["Extras"],
    key: str,
    exc: Exception,
    *,
    mode: Union["ParseMode", str, None] = ParseMode.STRICT,
    raw: Any = None,
) -> None:
    """In STRICT mode, raise. In LENIENT mode, record and keep going."""
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

# -----------------------
# Quantity
# -----------------------

try:
    from pint import UnitRegistry
except ImportError as e:
    raise ImportError(
        "Dependency missing: 'pint' is required. Install it with `pip install pint`."
    ) from e

ureg = UnitRegistry()
Q_ = ureg.Quantity

# Quantity type import is version-dependent across Pint releases.
try:  # Pint >= 0.20 often exposes Quantity at top-level
    from pint import Quantity  # type: ignore
except Exception:  # pragma: no cover
    from pint.facets.plain.quantity import Quantity  # type: ignore

def ensure_quantity(value: Any, unit: str) -> Optional["Quantity"]:
    """Coerce `value` into a Pint Quantity using the project registry, in `unit`.

    Robust version: returns None instead of raising on malformed inputs.
    Accepts:
      - Pint Quantity (any registry)
      - numbers
      - numeric strings ("12.3")
      - quantity strings ("12.3 nm", "5 degree")
      - dicts like {"magnitude": 5, "unit": "nm"} or {"value": 5, "unit": "nm"}
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None

    try:
        # Already a Pint Quantity (possibly from another registry)
        if isinstance(value, Quantity):
            q = Q_(value.magnitude, str(value.units))
            return q.to(unit)

        # Dict forms
        if isinstance(value, dict):
            mag = value.get("magnitude", value.get("value", None))
            u = value.get("unit", value.get("units", None))
            if mag is None or isinstance(mag, (bool, np.bool_)):
                return None
            q = Q_(mag, u) if u else Q_(mag, unit)
            return q.to(unit)

        # Plain numbers
        if isinstance(value, (int, float, np.number)):
            q = Q_(float(value), unit)
            return q.to(unit)

        # Strings: "5", "5 nm", "5degree"
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            # try numeric first (supports 1e-3)
            try:
                q = Q_(float(s), unit)
                return q.to(unit)
            except Exception:
                pass
            # then try quantity string like "12 nm"
            q = Q_(s)
            return q.to(unit)

        # Last resort: try numeric cast
        q = Q_(float(value), unit)
        return q.to(unit)

    except Exception:
        return None

def magnitude(value: Any, unit: str) -> Optional[float]:
    """Return magnitude as a plain float in `unit`."""
    q = ensure_quantity(value, unit)
    if q is None:
        return None
    return float(q.magnitude)

def _check_data_format(data: np.ndarray) -> bool:
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[2] == 1:
            data = data[:, :, 0]
    if data.ndim != 2:
        return False
    return (data.dtype.kind == "u") and (data.dtype.itemsize in (1, 2))

# -----------------------
# Extras
# -----------------------

@dataclass
class Extras:
    """Structured extras bucket.

    - vendor: vendor-specific extension payloads (namespaced by vendor key)
    - unknown: unknown top-level keys swept during from_dict (forward compatibility)
    - raw: raw/unparsed values for known fields
    - notes: non-fatal validation / normalization notes
    """

    vendor: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unknown: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.vendor or self.unknown or self.raw or self.notes)

    def to_native_dict(self) -> Dict[str, Any]:
        """Return a deep-copied native-python representation.

        Note: this may include non-JSON-serializable objects. Use `to_dict()`
        when you need a JSON-safe payload.
        """
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
        if value is None:
            return Extras()
        if isinstance(value, Extras):
            return value
        if isinstance(value, dict):
            # New (bucketed) format
            ex = Extras()
            if any(k in value for k in ("vendor", "unknown", "raw", "notes")):

                v = value.get("vendor", {})
                if isinstance(v, dict):
                    # enforce vendor namespace -> dict
                    for vend, payload in v.items():
                        if isinstance(payload, dict):
                            ex.vendor[str(vend)] = deepcopy(payload)
                        else:
                            ex.vendor[str(vend)] = {"_value": deepcopy(payload)}
                elif v is not None:
                    ex.raw[f"{owner}.extra.vendor"] = deepcopy(v)

                u = value.get("unknown", {})
                if isinstance(u, dict):
                    ex.unknown = deepcopy(u)
                elif u is not None:
                    ex.raw[f"{owner}.extra.unknown"] = deepcopy(u)

                r = value.get("raw", {})
                if isinstance(r, dict):
                    ex.raw.update(deepcopy(r))
                elif r is not None:
                    ex.raw[f"{owner}.extra.raw"] = deepcopy(r)

                n = value.get("notes", {})
                if isinstance(n, dict):
                    ex.notes = deepcopy(n)
                elif n is not None:
                    ex.raw[f"{owner}.extra.notes"] = deepcopy(n)

                return ex
            else:
                # treat as legacy/flat extras
                try:
                    ex.unknown = deepcopy(value)
                except Exception:
                    ex.raw[f"{owner}.extra"] = repr(value)
                return ex

        # Totally unexpected: keep something rather than crash
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

def _extra_put_note(extra: Any, key: str, value: Any) -> None:
    if extra is None:
        return
    if isinstance(extra, Extras):
        extra.notes[key] = value
        return
    if isinstance(extra, dict):
        extra[key] = value

def _extra_put_unknown(extra: Any, key: str, value: Any) -> None:
    if extra is None:
        return
    if isinstance(extra, Extras):
        extra.unknown[key] = value
        return
    if isinstance(extra, dict):
        extra[key] = value

def collect_extra(d: Optional[Dict[str, Any]], known: Iterable[str], *, owner: str = "unknown") -> Extras:
    """
    Start from d.get("extra", {}), then sweep *all* unknown top-level keys into extra.unknown.

    - Preserves raw parse failures under extra.raw
    - Keeps forward compatibility by not dropping unknown keys.
    """
    if not isinstance(d, dict):
        return Extras()

    known_set: Set[str] = set(known)

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
    if extra is None:
        return out

    if isinstance(extra, Extras):
        payload = extra.to_dict()
    else:
        payload = Extras.from_any(extra).to_dict()

    # Remove empty buckets
    payload = {k: v for k, v in payload.items() if v}

    if payload:
        out["extra"] = payload
    return out

def drop_none_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of out with any keys whose value is None removed."""
    return {k: v for k, v in out.items() if v is not None}

def normalize_extra(extra: Any) -> Extras:
    if extra is None:
        return Extras()
    if isinstance(extra, Extras):
        return extra
    if isinstance(extra, dict):
        return Extras.from_any(extra)
    raise TypeError(f"extra must be Extras, dict, or None, got {type(extra)}")

def normalize_extra_lenient(extra: Any, owner: str) -> Extras:
    try:
        return normalize_extra(extra)
    except Exception:
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(extra)
        return ex

def _deep_merge_dict_inplace(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (dict-to-dict only).

    - If both dst[k] and src[k] are dicts, merge recursively.
    - Otherwise, src overwrites dst (deepcopied).
    """
    for k, v in src.items():
        if k in dst and isinstance(dst.get(k), dict) and isinstance(v, dict):
            _deep_merge_dict_inplace(dst[k], v)  # type: ignore[arg-type]
        else:
            dst[k] = deepcopy(v)
    return dst

def merge_extras(dst: Extras, src: Any, *, owner: str) -> Extras:
    s = Extras.from_any(src, owner=owner)

    # vendor: deep-merge per vendor namespace to avoid clobbering nested payloads
    for vend, payload in s.vendor.items():
        if vend in dst.vendor and isinstance(dst.vendor.get(vend), dict) and isinstance(payload, dict):
            _deep_merge_dict_inplace(dst.vendor[vend], payload)
        else:
            # Extras.from_any enforces payload is a dict (or wraps it as {"_value": ...})
            dst.vendor[vend] = deepcopy(payload)

    # other buckets are flat
    dst.unknown.update(deepcopy(s.unknown))
    dst.raw.update(deepcopy(s.raw))
    dst.notes.update(deepcopy(s.notes))
    return dst

# -----------------------
# Parsers
# -----------------------

def parse_bool(value: Any, default: bool = False, *, strict: bool = False) -> bool:
    """Parse booleans robustly (handles YAML/CLI strings like 'false', '0', 'no')."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    # Avoid bool(True) being treated as int
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
    if value is None:
        return None

    # bool is a subclass of int -> reject it
    if isinstance(value, bool):
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise TypeError(f"{name} must be int-like, got bool")
        return None

    try:
        # real ints (including numpy ints)
        if isinstance(value, (int, np.integer)):
            return int(value)

        # floats that are actually integers (including numpy floats)
        if isinstance(value, (float, np.floating)):
            f = float(value)
            if f.is_integer():
                return int(f)
            raise ValueError(f"{name} must be an integer value, got {value!r}")

        # strings like "2", "2.0", "  2 "
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            f = float(s)  # raises if not numeric
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
    """Parse an optional float from int/float/np scalar or numeric string.

    If strict=False, returns None on invalid input and (optionally) stores the raw value in extra.
    If strict=True, raises on invalid input.
    """
    if value is None:
        return None

    # bool is a subclass of int -> reject it
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
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return parse_bool(value, default=False, strict=True)
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        if strict:
            raise
        return None

def parse_optional_str_like(value: Any, *, name: str, strict: bool = False, extra: Any = None) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip()
        return s or None

    # If strict, non-str is an error. Capture raw only on error.
    if strict:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        raise TypeError(f"{name} must be str-like, got {type(value)}")

    # Be consistent with your int/float parsers: bool is suspicious.
    if isinstance(value, bool):
        if extra is not None:
            _extra_put_raw(extra, name, value)
        return None

    # Non-str input: coerce without polluting extra unless it fails.
    try:
        s = str(value).strip()
    except Exception:
        if extra is not None:
            _extra_put_raw(extra, name, value)
        return None

    return s or None

def parse_optional_pair_int_like(value: Any, *, name: str, sort: bool = False, strict: bool = False, extra: Any = None) -> Optional[Tuple[int, int]]:
    """Parse an optional 2-tuple of int-like values (keeps order unless sort=True)."""
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
    """Parse an optional 2-tuple of float-like values (keeps order unless sort=True)."""
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
    """
    Like parse_optional_str_like, but records empty-string inputs into extra.raw/notes.
    Intended for identifiers (detector_id, primary_detector_id, default_detector_id, etc.).
    """
    if value is None:
        return None

    if isinstance(value, str) and value.strip() == "":
        if extra is not None:
            _extra_put_raw(extra, name, value)
            # optional breadcrumb (keeps raw for exact value, note for intent)
            try:
                extra.notes.setdefault("empty_id_fields", []).append(name)
            except Exception:
                pass
        return None

    return parse_optional_str_like(value, name=name, strict=strict, extra=extra)

def _maybe_point(v: Any, *, extra: Any = None, name: str = "point") -> Optional["Point"]:
    if v is None:
        return None
    if isinstance(v, Point):
        return v
    if isinstance(v, (dict, list, tuple)):
        return Point.from_dict(v)
    if extra is not None:
        _extra_put_raw(extra, name, v)
    return None

# -----------------------
# Generic parsing helpers
# -----------------------

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
    """Best-effort parse for optional nested dataclasses.

    - STRICT: raises on invalid input.
    - LENIENT: records the issue into `extra` (if provided) and returns None.

    Works with classes whose `from_dict` signature is either:
        - from_dict(d, *, mode=...)
        - from_dict(d)
    """
    mode = as_parse_mode(mode)

    if raw is None:
        return None

    # Treat {} as "missing" by default for optional sub-objects.
    if isinstance(raw, dict) and (not raw) and (not allow_empty_dict):
        return None

    # Already parsed
    try:
        if isinstance(raw, cls):
            # Avoid mutating _mode (it doesn't re-run parsing). Enforce mode via validate if available.
            validate = getattr(raw, "validate", None)
            if callable(validate):
                try:
                    validate(mode=mode)
                except TypeError:
                    validate()
            return raw
    except TypeError:
        pass

    if not isinstance(raw, dict):
        note_or_raise(
            extra,
            key or f"{getattr(cls, '__name__', 'object')}",
            TypeError(f"expected dict for {getattr(cls, '__name__', 'object')}, got {type(raw)}"),
            mode=mode,
            raw=raw,
        )
        return None

    from_dict = getattr(cls, "from_dict", None)
    if not callable(from_dict):
        note_or_raise(
            extra,
            key or f"{getattr(cls, '__name__', 'object')}",
            AttributeError(f"{getattr(cls, '__name__', 'object')} has no from_dict(...)"),
            mode=mode,
            raw=raw,
        )
        return None

    try:
        try:
            return from_dict(raw, mode=mode)  # type: ignore[misc]
        except TypeError:
            return from_dict(raw)  # type: ignore[misc]
    except Exception as e:
        note_or_raise(extra, key or f"{getattr(cls, '__name__', 'object')}", e, mode=mode, raw=raw)
        return None

# -----------------------
# Helpers
# -----------------------

def _jsonable(obj: Any) -> Any:
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars / arrays
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Path
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:
        pass

    # Pint Quantity (store as {magnitude, unit})
    try:
        if isinstance(obj, Quantity):
            return {"magnitude": float(obj.magnitude), "unit": str(obj.units)}
    except Exception:
        pass

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]

    # fallback: stringify (better than crashing or losing everything)
    return str(obj)



@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        vx = parse_optional_float_like(self.x, name="Point.x", strict=False)
        vy = parse_optional_float_like(self.y, name="Point.y", strict=False)
        vz = parse_optional_float_like(self.z, name="Point.z", strict=False)
        self.x = 0.0 if vx is None else float(vx)
        self.y = 0.0 if vy is None else float(vy)
        self.z = 0.0 if vz is None else float(vz)
        self.name = parse_optional_str_like(self.name, name="Point.name", strict=False)

    def to_dict(self) -> dict:
        return _jsonable(drop_none_keys({"x": self.x, "y": self.y, "z": self.z, "name": self.name}))

    @staticmethod
    def from_dict(d: Any) -> "Point":
        if isinstance(d, Point):
            return d

        def _f(v: Any, default: float = 0.0) -> float:
            out = parse_optional_float_like(v, name="Point", strict=False)
            return default if out is None else float(out)

        if isinstance(d, dict):
            return Point(
                x=_f(d.get("x", 0.0)),
                y=_f(d.get("y", 0.0)),
                z=_f(d.get("z", 0.0)),
                name=parse_optional_str_like(d.get("name", None), name="Point.name", strict=False)
            )

        if isinstance(d, (list, tuple)) and len(d) in (2, 3):
            x = _f(d[0], 0.0)
            y = _f(d[1], 0.0)
            z = _f(d[2], 0.0) if len(d) == 3 else 0.0
            return Point(x=x, y=y, z=z)

        return Point()

    def to_list(self) -> list:
        return [self.x, self.y, self.z]

@dataclass
class ROI:
    x: int = 0
    y: int = 0
    width: int = 512
    height: int = 512

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        ix = parse_optional_int_like(self.x, name="ROI.x", strict=strict)
        iy = parse_optional_int_like(self.y, name="ROI.y", strict=strict)
        iw = parse_optional_int_like(self.width, name="ROI.width", strict=strict)
        ih = parse_optional_int_like(self.height, name="ROI.height", strict=strict)

        self.x = 0 if ix is None else int(ix)
        self.y = 0 if iy is None else int(iy)
        self.width = 512 if iw is None else int(iw)
        self.height = 512 if ih is None else int(ih)

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate ROI semantics.

        - parse/normalization belongs in __post_init__.
        - semantic constraints belong here.
        """
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        if self.x < 0 or self.y < 0:
            err = ValueError(f"ROI.x/ROI.y must be >= 0, got x={self.x}, y={self.y}")
            if strict:
                raise err
            self.x = max(self.x, 0)
            self.y = max(self.y, 0)

        invalid_size = (self.width <= 0) or (self.height <= 0)
        if invalid_size:
            err = ValueError(f"ROI.width/ROI.height must be > 0, got width={self.width}, height={self.height}")
            if strict:
                raise err
            # lenient: keep a safe placeholder size, but report invalid so callers can decide to drop ROI.
            self.width = 512 if self.width <= 0 else self.width
            self.height = 512 if self.height <= 0 else self.height

        return not invalid_size

    def to_dict(self) -> dict:
        return _jsonable(drop_none_keys({"x": self.x, "y": self.y, "width": self.width, "height": self.height}))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ROI":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(d, ROI):
            d.validate(mode=mode)
            return d
        if isinstance(d, (list, tuple)):
            if len(d) != 4:
                if strict:
                    raise ValueError(f"ROI list/tuple must have len 4, got {len(d)}")
                return ROI(_mode=mode)
            return ROI(x=d[0], y=d[1], width=d[2], height=d[3], _mode=mode)
        if not isinstance(d, dict):
            if strict:
                raise TypeError(f"ROI must be dict/list/tuple/ROI, got {type(d)}")
            return ROI(_mode=mode)
        return ROI(
            x=d.get("x", 0),
            y=d.get("y", 0),
            width=d.get("width", d.get("w", 512)),
            height=d.get("height", d.get("h", 512)),
            _mode=mode,
        )

@dataclass
class TemImageMetadata:
    """
    Universal, vendor-agnostic image metadata.
    """

    # ---- Schema / provenance ----
    version: str = str(METADATA_VERSION)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    user: Optional[str] = None

    # ---- Instrument identity (minimal + optional) ----
    manufacturer: Optional[str] = None
    device: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    software_version: Optional[str] = None

    # ---- Imaging summary (canonical, cross-vendor-ish) ----
    mode: Optional[str] = None                 # "TEM" / "STEM" (or vendor string)
    detector_id: Optional[str] = None
    detector_name: Optional[str] = None

    magnification: Optional[float] = None
    camera_length_mm: Optional[float] = None

    pixel_size_nm: Optional[Tuple[float, float]] = None   # (px_x_nm, px_y_nm)
    image_size_px: Optional[Tuple[int, int]] = None       # (width, height), often "resolution"

    accelerating_voltage_kv: Optional[float] = None
    beam_current_na: Optional[float] = None
    exposure_ms: Optional[float] = None
    dwell_time_us: Optional[float] = None
    working_distance_mm: Optional[float] = None

    # ---- Structured snapshots (reuse your base structures) ----
    microscope_state: Optional["MicroscopeState"] = None
    acquisition: Optional["AcquisitionRequest"] = None

    # ---- Vendor-specific or unknown stuff ----
    extra: Extras = field(default_factory=Extras)

    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)
        # ---- extra: be lenient here (metadata should not brick loading) ----
        self.extra = normalize_extra_lenient(self.extra, "TemImageMetadata")

        # ---- schema/provenance ----
        self.version = (
                parse_optional_str_like(
                    self.version,
                    name="TemImageMetadata.version",
                    strict=strict,
                    extra=self.extra,
                )
                or str(METADATA_VERSION)
        )

        # created_at: accept ISO string, unix seconds (int/float), or numeric string
        raw_created = self.created_at
        created_iso: Optional[str] = None

        if isinstance(raw_created, (int, float)) and not isinstance(raw_created, bool):
            try:
                ts = float(raw_created)
                if np.isfinite(ts):
                    created_iso = datetime.datetime.fromtimestamp(
                        ts, tz=datetime.timezone.utc
                    ).isoformat()
                else:
                    self.extra.raw["TemImageMetadata.created_at"] = raw_created
                    created_iso = None
            except Exception:
                self.extra.raw["TemImageMetadata.created_at"] = raw_created
                created_iso = None
        else:
            s = parse_optional_str_like(raw_created, name="TemImageMetadata.created_at", strict=strict, extra=self.extra)
            if s is not None:
                ss = s.strip()
                # ISO first
                try:
                    dt = datetime.datetime.fromisoformat(ss.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    created_iso = dt.astimezone(datetime.timezone.utc).isoformat()
                except Exception:
                    # numeric string
                    try:
                        ts = float(ss)
                        if np.isfinite(ts):
                            created_iso = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
                        else:
                            self.extra.raw["TemImageMetadata.created_at"] = ss
                    except Exception:
                        self.extra.raw["TemImageMetadata.created_at"] = ss

        if not created_iso:
            # TODO: didn't use noteandraise
            if strict and raw_created not in (None, ""):
                raise ValueError(f"TemImageMetadata.created_at is invalid: {raw_created!r}")
            created_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.created_at = created_iso

        # ---- identity strings ----
        self.user = parse_optional_str_like(self.user, name="TemImageMetadata.user", strict=strict, extra=self.extra)
        self.manufacturer = parse_optional_str_like(self.manufacturer, name="TemImageMetadata.manufacturer",
                                                    strict=strict, extra=self.extra)
        self.device = parse_optional_str_like(self.device, name="TemImageMetadata.device", strict=strict,
                                              extra=self.extra)
        self.model = parse_optional_str_like(self.model, name="TemImageMetadata.model", strict=strict, extra=self.extra)
        self.serial_number = parse_optional_str_like(self.serial_number, name="TemImageMetadata.serial_number",
                                                     strict=strict, extra=self.extra)
        self.software_version = parse_optional_str_like(self.software_version, name="TemImageMetadata.software_version",
                                                        strict=strict, extra=self.extra)

        # ---- imaging summary ----
        self.mode = parse_optional_str_like(self.mode, name="TemImageMetadata.mode", strict=strict, extra=self.extra)
        self.detector_id = parse_optional_id_like(self.detector_id, name="TemImageMetadata.detector_id", strict=strict,
                                                   extra=self.extra)
        self.detector_name = parse_optional_str_like(self.detector_name, name="TemImageMetadata.detector_name",
                                                     strict=strict, extra=self.extra)

        self.magnification = parse_optional_float_like(self.magnification, name="TemImageMetadata.magnification",
                                                       strict=strict, extra=self.extra)
        self.camera_length_mm = parse_optional_float_like(self.camera_length_mm,
                                                          name="TemImageMetadata.camera_length_mm", strict=strict,
                                                          extra=self.extra)

        # ---- geometry ----
        self.pixel_size_nm = parse_optional_pair_float_like(
            self.pixel_size_nm,
            name="TemImageMetadata.pixel_size_nm",
            sort=False,
            strict=strict,
            extra=self.extra,
        )
        self.image_size_px = parse_optional_pair_int_like(
            self.image_size_px,
            name="TemImageMetadata.image_size_px",
            sort=False,
            strict=strict,
            extra=self.extra,
        )

        # ---- acquisition params ----
        self.accelerating_voltage_kv = parse_optional_float_like(
            self.accelerating_voltage_kv, name="TemImageMetadata.accelerating_voltage_kv", strict=strict,
            extra=self.extra
        )
        self.beam_current_na = parse_optional_float_like(
            self.beam_current_na, name="TemImageMetadata.beam_current_na", strict=strict, extra=self.extra
        )
        self.exposure_ms = parse_optional_float_like(
            self.exposure_ms, name="TemImageMetadata.exposure_ms", strict=strict, extra=self.extra
        )
        self.dwell_time_us = parse_optional_float_like(
            self.dwell_time_us, name="TemImageMetadata.dwell_time_us", strict=strict, extra=self.extra
        )
        self.working_distance_mm = parse_optional_float_like(
            self.working_distance_mm, name="TemImageMetadata.working_distance_mm", strict=strict, extra=self.extra
        )
        # ---- nested objects (best-effort; don't raise in metadata) ----
        self.microscope_state = maybe_from_dict(
            MicroscopeState,
            self.microscope_state,
            mode=mode,
            extra=self.extra,
            key="TemImageMetadata.microscope_state",
            allow_empty_dict=False,
        )

        self.acquisition = maybe_from_dict(
            AcquisitionRequest,
            self.acquisition,
            mode=mode,
            extra=self.extra,
            key="TemImageMetadata.acquisition",
            allow_empty_dict=False,
        )

        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        # created_at should be ISO; in lenient mode we already backfilled it, but keep it sane.
        if not isinstance(self.created_at, str) or not self.created_at:
            note_or_raise(self.extra, "TemImageMetadata.created_at", TypeError("created_at must be a non-empty str"), mode=mode, raw=self.created_at)
            self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Nested snapshots: validate (propagate mode).
        if self.microscope_state is not None:
            try:
                self.microscope_state.validate(mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "TemImageMetadata.microscope_state", e, mode=mode, raw=self.microscope_state)
                if not strict:
                    self.microscope_state = None

        if self.acquisition is not None:
            try:
                self.acquisition.validate(mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "TemImageMetadata.acquisition", e, mode=mode, raw=self.acquisition)
                if not strict:
                    self.acquisition = None

        # Basic sanity for a few numeric summaries (do not over-police; vendors vary).
        def _pos(name: str, v: Any):
            if v is None:
                return
            try:
                fv = float(v)
            except Exception as e:
                note_or_raise(self.extra, f"TemImageMetadata.{name}", e, mode=mode, raw=v)
                if not strict:
                    setattr(self, name, None)
                return
            if not math.isfinite(fv) or fv <= 0:
                note_or_raise(self.extra, f"TemImageMetadata.{name}", ValueError(f"{name} must be finite and > 0"), mode=mode, raw=v)
                if not strict:
                    setattr(self, name, None)

        _pos("magnification", self.magnification)
        _pos("camera_length_mm", self.camera_length_mm)
        _pos("pixel_size_nm", self.pixel_size_nm)
        _pos("accelerating_voltage_kv", self.accelerating_voltage_kv)
        _pos("exposure_ms", self.exposure_ms)
        _pos("dwell_time_us", self.dwell_time_us)

        return True

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "version": self.version,
            "created_at": self.created_at,
            "user": self.user,
            "manufacturer": self.manufacturer,
            "device": self.device,
            "model": self.model,
            "serial_number": self.serial_number,
            "software_version": self.software_version,
            "mode": self.mode,
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "magnification": self.magnification,
            "camera_length_mm": self.camera_length_mm,
            "pixel_size_nm": self.pixel_size_nm,
            "image_size_px": self.image_size_px,
            "accelerating_voltage_kv": self.accelerating_voltage_kv,
            "beam_current_na": self.beam_current_na,
            "exposure_ms": self.exposure_ms,
            "dwell_time_us": self.dwell_time_us,
            "working_distance_mm": self.working_distance_mm,
        }

        if self.microscope_state is not None:
            d["microscope_state"] = self.microscope_state.to_dict()

        if self.acquisition is not None:
            d["acquisition"] = self.acquisition.to_dict()

        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "TemImageMetadata":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(d, TemImageMetadata):
            d.validate(mode=mode)
            return d
        if not isinstance(d, dict):
            if strict:
                raise TypeError(f"TemImageMetadata.from_dict expects a dict, got {type(d)}")
            return TemImageMetadata(_mode=mode)

        known = {
            # schema/provenance
            "version", "metadata_version",
            "created_at", "timestamp",
            "user",

            # identity
            "manufacturer",
            "device",
            "model",
            "serial_number",
            "software_version",

            # imaging summary
            "mode",
            "detector_id",
            "detector_name",
            "magnification",
            "camera_length_mm", "camera_length",

            # geometry
            "pixel_size_nm",
            "image_size_px",

            # acquisition params
            "accelerating_voltage_kv", "accelerating_voltage",
            "beam_current_na", "beam_current",
            "exposure_ms",
            "dwell_time_us", "dwell_time",
            "working_distance_mm", "working_distance",

            # nested
            "microscope_state",
            "acquisition",

            # extras
            "extra",
        }
        extra = collect_extra(d, known, owner="TemImageMetadata")

        def _default_created_at() -> str:
            return datetime.datetime.now(datetime.timezone.utc).isoformat()

        def _opt_str(val: Any, name: str) -> Optional[str]:
            if name == "detector_id":
                return parse_optional_id_like(val, name=f"TemImageMetadata.{name}", strict=strict, extra=extra)
            return parse_optional_str_like(val, name=f"TemImageMetadata.{name}", strict=strict, extra=extra)

        # version: keep as str (METADATA_VERSION is your canonical default)
        version_raw = d.get("version", d.get("metadata_version", METADATA_VERSION))
        version = _opt_str(version_raw, "version") or str(METADATA_VERSION)

        # created_at: accept ISO string or unix seconds
        created_raw = d.get("created_at", d.get("timestamp", None))

        if isinstance(created_raw, (int, float)) and not isinstance(created_raw, bool):
            try:
                ts = float(created_raw)
                if np.isfinite(ts):
                    created_at = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
                else:
                    extra.raw["TemImageMetadata.created_at"] = created_raw
                    created_at = _default_created_at()
            except Exception:
                extra.raw["TemImageMetadata.created_at"] = created_raw
                created_at = _default_created_at()
        else:
            # let __post_init__ deal with numeric strings like "1e9"
            created_at = _opt_str(created_raw, "created_at") or _default_created_at()

        # identity (best-effort string conversion, but don't accept bool)
        user = _opt_str(d.get("user", None), "user")
        manufacturer = _opt_str(d.get("manufacturer", None), "manufacturer")
        device = _opt_str(d.get("device", None), "device")
        model = _opt_str(d.get("model", None), "model")
        serial_number = _opt_str(d.get("serial_number", None), "serial_number")
        software_version = _opt_str(d.get("software_version", None), "software_version")

        # imaging summary
        mode = _opt_str(d.get("mode", None), "mode")
        detector_id = _opt_str(d.get("detector_id", None), "detector_id")
        detector_name = _opt_str(d.get("detector_name", None), "detector_name")

        magnification = parse_optional_float_like(d.get("magnification", None), name="TemImageMetadata.magnification", strict=strict, extra=extra)
        camera_length_mm = parse_optional_float_like(
            d.get("camera_length_mm", d.get("camera_length", None)),
            name="TemImageMetadata.camera_length_mm",
            strict=strict,
            extra=extra,
        )

        accelerating_voltage_kv = parse_optional_float_like(
            d.get("accelerating_voltage_kv", d.get("accelerating_voltage", None)),
            name="TemImageMetadata.accelerating_voltage_kv",
            strict=strict,
            extra=extra,
        )
        beam_current_na = parse_optional_float_like(
            d.get("beam_current_na", d.get("beam_current", None)),
            name="TemImageMetadata.beam_current_na",
            strict=strict,
            extra=extra,
        )
        exposure_ms = parse_optional_float_like(d.get("exposure_ms", None), name="TemImageMetadata.exposure_ms", strict=strict, extra=extra)
        dwell_time_us = parse_optional_float_like(
            d.get("dwell_time_us", d.get("dwell_time", None)),
            name="TemImageMetadata.dwell_time_us",
            strict=strict,
            extra=extra,
        )
        working_distance_mm = parse_optional_float_like(
            d.get("working_distance_mm", d.get("working_distance", None)),
            name="TemImageMetadata.working_distance_mm",
            strict=strict,
            extra=extra,
        )

        pixel_size_nm = parse_optional_pair_float_like(
            d.get("pixel_size_nm", None), name="TemImageMetadata.pixel_size_nm", sort=False, strict=strict, extra=extra
        )
        image_size_px = parse_optional_pair_int_like(
            d.get("image_size_px", None), name="TemImageMetadata.image_size_px", sort=False, strict=strict, extra=extra
        )

        # nested objects
        ms_raw = d.get("microscope_state", None)
        microscope_state = maybe_from_dict(
            MicroscopeState,
            ms_raw,
            mode=mode,
            extra=extra,
            key="TemImageMetadata.microscope_state",
            allow_empty_dict=False,
        )

        acq_raw = d.get("acquisition", None)
        acquisition = maybe_from_dict(
            AcquisitionRequest,
            acq_raw,
            mode=mode,
            extra=extra,
            key="TemImageMetadata.acquisition",
            allow_empty_dict=False,
        )
        return TemImageMetadata(
            version=version,
            created_at=created_at,
            user=user,
            manufacturer=manufacturer,
            device=device,
            model=model,
            serial_number=serial_number,
            software_version=software_version,
            mode=mode,
            detector_id=detector_id,
            detector_name=detector_name,
            magnification=magnification,
            camera_length_mm=camera_length_mm,
            pixel_size_nm=pixel_size_nm,
            image_size_px=image_size_px,
            accelerating_voltage_kv=accelerating_voltage_kv,
            beam_current_na=beam_current_na,
            exposure_ms=exposure_ms,
            dwell_time_us=dwell_time_us,
            working_distance_mm=working_distance_mm,
            microscope_state=microscope_state,
            acquisition=acquisition,
            extra=extra,
        )

@dataclass
class TemStagePosition:
    """Stage position (Quantity-based).

    Conventions used across this project:
      - x, y, z: length (stored as **nanometer**)
      - r, tilt_x, tilt_y: angles (stored as **degree**)
    """

    name: Optional[str] = None
    x: Optional["Quantity"] = None
    y: Optional["Quantity"] = None
    z: Optional["Quantity"] = None
    r: Optional["Quantity"] = None
    tilt_x: Optional["Quantity"] = None
    tilt_y: Optional["Quantity"] = None
    coordinate_system: Optional[str] = None

    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)


    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.name = parse_optional_str_like(self.name, name="TemStagePosition.name", strict=strict)
        self.coordinate_system = parse_optional_str_like(
            self.coordinate_system, name="TemStagePosition.coordinate_system", strict=strict
        )

        def coerce_axis(raw: Any, unit: str, field_name: str) -> Optional["Quantity"]:
            q = ensure_quantity(raw, unit)
            if strict and (raw is not None) and (q is None):
                raise ValueError(f"{field_name} must be convertible to {unit}, got {raw!r}")
            return q

        # Normalize individual axes into canonical units.
        self.x = coerce_axis(self.x, "nanometer", "TemStagePosition.x")
        self.y = coerce_axis(self.y, "nanometer", "TemStagePosition.y")
        self.z = coerce_axis(self.z, "nanometer", "TemStagePosition.z")
        self.r = coerce_axis(self.r, "degree", "TemStagePosition.r")
        self.tilt_x = coerce_axis(self.tilt_x, "degree", "TemStagePosition.tilt_x")
        self.tilt_y = coerce_axis(self.tilt_y, "degree", "TemStagePosition.tilt_y")

        self.validate(mode=mode)

    @property
    def stage(self) -> List[Optional["Quantity"]]:
        return [self.x, self.y, self.z, self.r, self.tilt_x, self.tilt_y]

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "x": magnitude(self.x, "nanometer"),
            "y": magnitude(self.y, "nanometer"),
            "z": magnitude(self.z, "nanometer"),
            "r": magnitude(self.r, "degree"),
            "tilt_x": magnitude(self.tilt_x, "degree"),
            "tilt_y": magnitude(self.tilt_y, "degree"),
            "coordinate_system": self.coordinate_system,
        }
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "TemStagePosition":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(d, TemStagePosition):
            d.validate(mode=mode)
            return d
        if d is None:
            return TemStagePosition(_mode=mode)
        if not isinstance(d, dict):
            if strict:
                raise TypeError(f"TemStagePosition.from_dict expects a dict, got {type(d)}")
            return TemStagePosition(_mode=mode)

        return TemStagePosition(
            name=d.get("name", None),
            x=d.get("x", None),
            y=d.get("y", None),
            z=d.get("z", None),
            r=d.get("r", None),
            tilt_x=d.get("tilt_x", None),
            tilt_y=d.get("tilt_y", None),
            coordinate_system=d.get(
                "coordinate_system",
                d.get("coord_system", d.get("cs", None)),
            ),
            _mode=mode,
        )

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        def cleaned(q: Any, unit: str, field_name: str) -> Optional["Quantity"]:
            qq = ensure_quantity(q, unit)
            if qq is None:
                if q is None:
                    return None
                note_or_raise(None, field_name, ValueError(f"{field_name} must be convertible to {unit}"), mode=mode, raw=q)
                return None
            try:
                mag = float(qq.to(unit).magnitude)
            except Exception as e:
                note_or_raise(None, field_name, e, mode=mode, raw=qq)
                return None
            if not math.isfinite(mag):
                note_or_raise(None, field_name, ValueError(f"{field_name} magnitude must be finite"), mode=mode, raw=mag)
                return None
            return qq

        self.x = cleaned(self.x, "nanometer", "TemStagePosition.x")
        self.y = cleaned(self.y, "nanometer", "TemStagePosition.y")
        self.z = cleaned(self.z, "nanometer", "TemStagePosition.z")
        self.r = cleaned(self.r, "degree", "TemStagePosition.r")
        self.tilt_x = cleaned(self.tilt_x, "degree", "TemStagePosition.tilt_x")
        self.tilt_y = cleaned(self.tilt_y, "degree", "TemStagePosition.tilt_y")

        return True


    def __add__(self, other: "TemStagePosition") -> "TemStagePosition":
        if not isinstance(other, TemStagePosition):
            return NotImplemented

        def add_axis(a, b, unit: str):
            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)

            if qa is None and qb is None:
                return None
            if qa is None:
                return qb
            if qb is None:
                return qa
            return qa + qb

        return TemStagePosition(
            name=self.name,
            x=add_axis(self.x, other.x, "nanometer"),
            y=add_axis(self.y, other.y, "nanometer"),
            z=add_axis(self.z, other.z, "nanometer"),
            r=add_axis(self.r, other.r, "degree"),
            tilt_x=add_axis(self.tilt_x, other.tilt_x, "degree"),
            tilt_y=add_axis(self.tilt_y, other.tilt_y, "degree"),
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: "TemStagePosition") -> "TemStagePosition":
        if not isinstance(other, TemStagePosition):
            return NotImplemented

        def sub_axis(a, b, unit: str):
            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)

            if qa is None and qb is None:
                return None
            if qa is None:
                return -qb if qb is not None else None
            if qb is None:
                return qa
            return qa - qb

        return TemStagePosition(
            name=self.name,
            x=sub_axis(self.x, other.x, "nanometer"),
            y=sub_axis(self.y, other.y, "nanometer"),
            z=sub_axis(self.z, other.z, "nanometer"),
            r=sub_axis(self.r, other.r, "degree"),
            tilt_x=sub_axis(self.tilt_x, other.tilt_x, "degree"),
            tilt_y=sub_axis(self.tilt_y, other.tilt_y, "degree"),
            coordinate_system=self.coordinate_system,
        )

    def is_close(self, other: "TemStagePosition", tol_nm: float = 1.0, tol_deg: float = 1e-3, *, compare_only_specified: bool = True) -> bool:
        def close_axis(a, b, unit: str, tol: float) -> bool:
            # If either side doesn't specify this axis, ignore it (don't care).
            if compare_only_specified and (a is None or b is None):
                return True

            if a is None or b is None:
                return False

            qa = ensure_quantity(a, unit)
            qb = ensure_quantity(b, unit)
            if qa is None or qb is None:
                return False

            da = abs(qa - qb)
            return float(da.m_as(unit)) <= float(tol)

        return (
                close_axis(self.x, other.x, "nanometer", tol_nm)
                and close_axis(self.y, other.y, "nanometer", tol_nm)
                and close_axis(self.z, other.z, "nanometer", tol_nm)
                and close_axis(self.r, other.r, "degree", tol_deg)
                and close_axis(self.tilt_x, other.tilt_x, "degree", tol_deg)
                and close_axis(self.tilt_y, other.tilt_y, "degree", tol_deg)
        )

@dataclass
class StageSystemSettings:
    """Stage system configuration for safe TEM automation.

    Units:
      - *_limits_nm are in nanometer
      - *_limits_deg are in degree
      - max_step_nm / max_step_deg define the largest single move you allow automation to command
    """

    enabled: bool = True

    # Capabilities / axes availability
    can_x: bool = True
    can_y: bool = True
    can_z: bool = True
    can_r: bool = False
    can_tilt_x: bool = False
    can_tilt_y: bool = False

    # Soft limits (optional; None means "unknown / not enforced here")
    x_limits_nm: Optional[Tuple[float, float]] = None
    y_limits_nm: Optional[Tuple[float, float]] = None
    z_limits_nm: Optional[Tuple[float, float]] = None
    r_limits_deg: Optional[Tuple[float, float]] = None
    tilt_x_limits_deg: Optional[Tuple[float, float]] = None
    tilt_y_limits_deg: Optional[Tuple[float, float]] = None

    # Motion safety defaults
    max_step_nm: float = 50000.0        # 50 µm
    max_step_deg: float = 1.0
    settle_time_s: float = 0.2
    timeout_s: float = 10.0

    # Common TEM calibration hint (optional)
    eucentric_z_nm: Optional[float] = None

    # Everything vendor-specific goes here instead of polluting the core schema
    extra: Extras = field(default_factory=Extras)

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "StageSystemSettings")

        self.enabled = parse_bool(self.enabled, default=True)
        self.can_x = parse_bool(self.can_x, default=True)
        self.can_y = parse_bool(self.can_y, default=True)
        self.can_z = parse_bool(self.can_z, default=True)
        self.can_r = parse_bool(self.can_r, default=False)
        self.can_tilt_x = parse_bool(self.can_tilt_x, default=False)
        self.can_tilt_y = parse_bool(self.can_tilt_y, default=False)

        # Robust pair parser (rejects bool, enforces length==2)
        self.x_limits_nm = parse_optional_pair_float_like(
            self.x_limits_nm, name="StageSystemSettings.x_limits_nm", sort=True, strict=strict, extra=self.extra
        )
        self.y_limits_nm = parse_optional_pair_float_like(
            self.y_limits_nm, name="StageSystemSettings.y_limits_nm", sort=True, strict=strict, extra=self.extra
        )
        self.z_limits_nm = parse_optional_pair_float_like(
            self.z_limits_nm, name="StageSystemSettings.z_limits_nm", sort=True, strict=strict, extra=self.extra
        )
        self.r_limits_deg = parse_optional_pair_float_like(
            self.r_limits_deg, name="StageSystemSettings.r_limits_deg", sort=True, strict=strict, extra=self.extra
        )
        self.tilt_x_limits_deg = parse_optional_pair_float_like(
            self.tilt_x_limits_deg, name="StageSystemSettings.tilt_x_limits_deg", sort=True, strict=strict, extra=self.extra
        )
        self.tilt_y_limits_deg = parse_optional_pair_float_like(
            self.tilt_y_limits_deg, name="StageSystemSettings.tilt_y_limits_deg", sort=True, strict=strict, extra=self.extra
        )

        # Floats: reject bool, allow numeric strings, keep defaults if None
        v = parse_optional_float_like(self.max_step_nm, name="StageSystemSettings.max_step_nm", strict=strict, extra=self.extra)
        self.max_step_nm = 50000.0 if v is None else v

        v = parse_optional_float_like(self.max_step_deg, name="StageSystemSettings.max_step_deg", strict=strict, extra=self.extra)
        self.max_step_deg = 1.0 if v is None else v

        v = parse_optional_float_like(self.settle_time_s, name="StageSystemSettings.settle_time_s", strict=strict, extra=self.extra)
        self.settle_time_s = 0.2 if v is None else v

        v = parse_optional_float_like(self.timeout_s, name="StageSystemSettings.timeout_s", strict=strict, extra=self.extra)
        self.timeout_s = 10.0 if v is None else v

        self.eucentric_z_nm = parse_optional_float_like(
            self.eucentric_z_nm, name="StageSystemSettings.eucentric_z_nm", strict=strict, extra=self.extra
        )

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate StageSystemSettings semantics."""
        mode = as_parse_mode(self._mode if mode is None else mode)

        if self.max_step_nm <= 0:
            note_or_raise(
                self.extra,
                "StageSystemSettings.max_step_nm",
                ValueError(f"max_step_nm must be > 0, got {self.max_step_nm}"),
                mode=mode,
                raw=self.max_step_nm,
            )
            self.max_step_nm = 50000.0

        if self.max_step_deg <= 0:
            note_or_raise(
                self.extra,
                "StageSystemSettings.max_step_deg",
                ValueError(f"max_step_deg must be > 0, got {self.max_step_deg}"),
                mode=mode,
                raw=self.max_step_deg,
            )
            self.max_step_deg = 1.0

        if self.settle_time_s < 0:
            note_or_raise(
                self.extra,
                "StageSystemSettings.settle_time_s",
                ValueError(f"settle_time_s must be >= 0, got {self.settle_time_s}"),
                mode=mode,
                raw=self.settle_time_s,
            )
            self.settle_time_s = 0.2

        if self.timeout_s <= 0:
            note_or_raise(
                self.extra,
                "StageSystemSettings.timeout_s",
                ValueError(f"timeout_s must be > 0, got {self.timeout_s}"),
                mode=mode,
                raw=self.timeout_s,
            )
            self.timeout_s = 10.0

        return True

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "can_x": self.can_x,
            "can_y": self.can_y,
            "can_z": self.can_z,
            "can_r": self.can_r,
            "can_tilt_x": self.can_tilt_x,
            "can_tilt_y": self.can_tilt_y,
            "x_limits_nm": self.x_limits_nm,
            "y_limits_nm": self.y_limits_nm,
            "z_limits_nm": self.z_limits_nm,
            "r_limits_deg": self.r_limits_deg,
            "tilt_x_limits_deg": self.tilt_x_limits_deg,
            "tilt_y_limits_deg": self.tilt_y_limits_deg,
            "max_step_nm": self.max_step_nm,
            "max_step_deg": self.max_step_deg,
            "settle_time_s": self.settle_time_s,
            "timeout_s": self.timeout_s,
            "eucentric_z_nm": self.eucentric_z_nm,
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageSystemSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, StageSystemSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"StageSystemSettings.from_dict expects a dict, got {type(settings)}")
            return StageSystemSettings(_mode=mode)

        known = {
            "enabled",
            "can_x", "can_y", "can_z", "can_r", "can_tilt_x", "can_tilt_y",
            "x_limits_nm", "y_limits_nm", "z_limits_nm",
            "r_limits_deg", "tilt_x_limits_deg", "tilt_y_limits_deg",
            "max_step_nm", "max_step_deg", "settle_time_s", "timeout_s",
            "eucentric_z_nm",
            # alias you support:
            "eucentric_height",
            "extra",
        }
        extra = collect_extra(settings, known, owner="StageSystemSettings")

        # Keep from_dict thin: parsing/validation happens in __post_init__.
        euc_raw = settings.get("eucentric_z_nm", settings.get("eucentric_height", None))

        return StageSystemSettings(
            enabled=settings.get("enabled", True),
            can_x=settings.get("can_x", True),
            can_y=settings.get("can_y", True),
            can_z=settings.get("can_z", True),
            can_r=settings.get("can_r", False),
            can_tilt_x=settings.get("can_tilt_x", False),
            can_tilt_y=settings.get("can_tilt_y", False),
            x_limits_nm=settings.get("x_limits_nm", None),
            y_limits_nm=settings.get("y_limits_nm", None),
            z_limits_nm=settings.get("z_limits_nm", None),
            r_limits_deg=settings.get("r_limits_deg", None),
            tilt_x_limits_deg=settings.get("tilt_x_limits_deg", None),
            tilt_y_limits_deg=settings.get("tilt_y_limits_deg", None),
            max_step_nm=settings.get("max_step_nm", 50000.0),
            max_step_deg=settings.get("max_step_deg", 1.0),
            settle_time_s=settings.get("settle_time_s", 0.2),
            timeout_s=settings.get("timeout_s", 10.0),
            eucentric_z_nm=euc_raw,
            extra=extra,
            _mode=mode,
        )

@dataclass
class BeamSettings:
    """Beam settings for TEM/STEM automation.

    This is kept intentionally generic across TEM + STEM:
      - voltage: accelerating voltage (kV)
      - beam_current: probe/beam current (nA) if available
      - spot_size: instrument-specific index (optional)
      - convergence_angle_mrad: mainly for STEM probe formation (optional)

    Shifts/stigs are kept as `Point` because different vendors expose different units/axes.
    If you need strict units, store the vendor values plus a unit hint in `extra`.
    """

    voltage: Optional[float] = None  # kV
    beam_current: Optional[float] = None  # nA (if known)
    spot_size: Optional[int] = None
    convergence_angle_mrad: Optional[float] = None

    stigmation: Optional[Point] = None
    beam_shift: Optional[Point] = None
    image_shift: Optional[Point] = None

    # For STEM scan coordinate systems (optional)
    scan_rotation_deg: Optional[float] = None

    extra: Extras = field(default_factory=Extras)

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "BeamSettings")

        # Numeric-ish inputs are accepted (e.g. "200", "200.0", numpy scalars).
        self.voltage = parse_optional_float_like(self.voltage, name="BeamSettings.voltage", strict=strict, extra=self.extra)
        self.beam_current = parse_optional_float_like(
            self.beam_current, name="BeamSettings.beam_current", strict=strict, extra=self.extra
        )
        self.convergence_angle_mrad = parse_optional_float_like(
            self.convergence_angle_mrad, name="BeamSettings.convergence_angle_mrad", strict=strict, extra=self.extra
        )
        self.scan_rotation_deg = parse_optional_float_like(
            self.scan_rotation_deg, name="BeamSettings.scan_rotation_deg", strict=strict, extra=self.extra
        )

        self.spot_size = parse_optional_int_like(
            self.spot_size, name="BeamSettings.spot_size", strict=strict, extra=self.extra
        )

        # Normalize Point-ish inputs (Point / dict / [x,y] / (x,y,z) / None)
        self.stigmation = _maybe_point(self.stigmation, extra=self.extra, name="BeamSettings.stigmation")
        self.beam_shift = _maybe_point(self.beam_shift, extra=self.extra, name="BeamSettings.beam_shift")
        self.image_shift = _maybe_point(self.image_shift, extra=self.extra, name="BeamSettings.image_shift")

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate beam-setting semantics.

        Parsing/coercion belongs in __post_init__ / from_dict.
        """
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        def _finite_or_none(v: Optional[float], key: str) -> Optional[float]:
            if v is None:
                return None
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                note_or_raise(self.extra, key, ValueError(f"{key} must be a finite number"), mode=mode, raw=v)
                return None
            return float(v)

        self.voltage = _finite_or_none(self.voltage, "BeamSettings.voltage")
        self.beam_current = _finite_or_none(self.beam_current, "BeamSettings.beam_current")
        self.convergence_angle_mrad = _finite_or_none(self.convergence_angle_mrad, "BeamSettings.convergence_angle_mrad")
        self.scan_rotation_deg = _finite_or_none(self.scan_rotation_deg, "BeamSettings.scan_rotation_deg")

        if self.voltage is not None and self.voltage <= 0:
            note_or_raise(self.extra, "BeamSettings.voltage", ValueError(f"voltage must be > 0 kV, got {self.voltage}"), mode=mode, raw=self.voltage)
            self.voltage = None

        if self.beam_current is not None and self.beam_current < 0:
            note_or_raise(self.extra, "BeamSettings.beam_current", ValueError(f"beam_current must be >= 0 nA, got {self.beam_current}"), mode=mode, raw=self.beam_current)
            self.beam_current = None

        if self.convergence_angle_mrad is not None and self.convergence_angle_mrad < 0:
            note_or_raise(self.extra, "BeamSettings.convergence_angle_mrad", ValueError(f"convergence_angle_mrad must be >= 0, got {self.convergence_angle_mrad}"), mode=mode, raw=self.convergence_angle_mrad)
            self.convergence_angle_mrad = None

        if self.spot_size is not None and self.spot_size < 0:
            note_or_raise(self.extra, "BeamSettings.spot_size", ValueError(f"spot_size must be >= 0, got {self.spot_size}"), mode=mode, raw=self.spot_size)
            self.spot_size = None

        return True

    def to_dict(self) -> dict:
        d = {
            "voltage": self.voltage,
            "beam_current": self.beam_current,
            "spot_size": self.spot_size,
            "convergence_angle_mrad": self.convergence_angle_mrad,
            "scan_rotation_deg": self.scan_rotation_deg,
            "stigmation": None if self.stigmation is None else self.stigmation.to_dict(),
            "beam_shift": None if self.beam_shift is None else self.beam_shift.to_dict(),
            "image_shift": None if self.image_shift is None else self.image_shift.to_dict(),
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, BeamSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"BeamSettings.from_dict expects a dict, got {type(settings)}")
            return BeamSettings(_mode=mode)

        known = {
            "voltage", "accelerating_voltage_kv",
            "beam_current", "current",
            "spot_size", "spot",
            "convergence_angle_mrad", "convergence_mrad",
            "stigmation", "beam_shift", "image_shift",
            "scan_rotation_deg",
            "extra",
        }
        extra = collect_extra(settings, known, owner="BeamSettings")

        # Keep from_dict thin: collect extras + propagate aliases; parsing happens in __post_init__.
        return BeamSettings(
            voltage=settings.get("voltage", settings.get("accelerating_voltage_kv", None)),
            beam_current=settings.get("beam_current", settings.get("current", None)),
            spot_size=settings.get("spot_size", settings.get("spot", None)),
            convergence_angle_mrad=settings.get("convergence_angle_mrad", settings.get("convergence_mrad", None)),
            stigmation=settings.get("stigmation", None),
            beam_shift=settings.get("beam_shift", None),
            image_shift=settings.get("image_shift", None),
            scan_rotation_deg=settings.get("scan_rotation_deg", None),
            extra=extra,
            _mode=mode,
        )

@dataclass
class BeamSystemSettings:
    """Beam subsystem configuration (defaults + soft constraints).

    If you want detector defaults, put them in `ImageSettings` / `TemDetectorSettings`.
    If you want vendor-specific quirks, put them in `extra`.
    """

    enabled: bool = True

    # A "known good" default for automation sessions (optional).
    default_beam: BeamSettings = field(default_factory=BeamSettings)

    # Soft constraints (optional)
    voltage_range_kv: Optional[Tuple[float, float]] = None
    beam_current_range_na: Optional[Tuple[float, float]] = None
    spot_size_range: Optional[Tuple[int, int]] = None
    convergence_angle_range_mrad: Optional[Tuple[float, float]] = None

    extra: Extras = field(default_factory=Extras)

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "BeamSystemSettings")
        self.enabled = parse_bool(self.enabled, default=True)

        # default_beam normalization
        if self.default_beam is None:
            self.default_beam = BeamSettings(_mode=mode)
        elif isinstance(self.default_beam, dict):
            try:
                self.default_beam = BeamSettings.from_dict(self.default_beam, mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "BeamSystemSettings.default_beam", e, mode=mode, raw=deepcopy(self.default_beam))
                self.default_beam = BeamSettings(_mode=mode)
        elif isinstance(self.default_beam, BeamSettings):
            self.default_beam.validate(mode=mode)
        else:
            note_or_raise(
                self.extra,
                "BeamSystemSettings.default_beam",
                TypeError(f"default_beam must be BeamSettings/dict, got {type(self.default_beam)}"),
                mode=mode,
                raw=deepcopy(self.default_beam),
            )
            self.default_beam = BeamSettings(_mode=mode)

        self.voltage_range_kv = parse_optional_pair_float_like(
            self.voltage_range_kv, name="BeamSystemSettings.voltage_range_kv", sort=True, strict=strict, extra=self.extra
        )
        self.beam_current_range_na = parse_optional_pair_float_like(
            self.beam_current_range_na, name="BeamSystemSettings.beam_current_range_na", sort=True, strict=strict, extra=self.extra
        )
        self.convergence_angle_range_mrad = parse_optional_pair_float_like(
            self.convergence_angle_range_mrad,
            name="BeamSystemSettings.convergence_angle_range_mrad",
            sort=True,
            strict=strict,
            extra=self.extra,
        )
        self.spot_size_range = parse_optional_pair_int_like(
            self.spot_size_range, name="BeamSystemSettings.spot_size_range", sort=True, strict=strict, extra=self.extra
        )

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate beam-system semantics."""
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        ok = True

        # default_beam must be a BeamSettings
        if not isinstance(self.default_beam, BeamSettings):
            note_or_raise(
                self.extra,
                "BeamSystemSettings.default_beam",
                TypeError(f"default_beam must be BeamSettings, got {type(self.default_beam)}"),
                mode=mode,
                raw=deepcopy(self.default_beam),
            )
            self.default_beam = BeamSettings(_mode=mode)
            ok = False
        else:
            ok = self.default_beam.validate(mode=mode) and ok

        def _check_nonnegative_range(rng: Optional[Tuple[float, float]], key: str) -> Optional[Tuple[float, float]]:
            nonlocal ok
            if rng is None:
                return None
            lo, hi = rng
            if lo < 0 or hi < 0:
                note_or_raise(self.extra, key, ValueError(f"{key} must be >= 0"), mode=mode, raw=rng)
                ok = False
                return None
            return (lo, hi)

        def _check_nonnegative_int_range(rng: Optional[Tuple[int, int]], key: str) -> Optional[Tuple[int, int]]:
            nonlocal ok
            if rng is None:
                return None
            lo, hi = rng
            if lo < 0 or hi < 0:
                note_or_raise(self.extra, key, ValueError(f"{key} must be >= 0"), mode=mode, raw=rng)
                ok = False
                return None
            return (lo, hi)

        self.voltage_range_kv = _check_nonnegative_range(self.voltage_range_kv, "BeamSystemSettings.voltage_range_kv")
        self.beam_current_range_na = _check_nonnegative_range(
            self.beam_current_range_na, "BeamSystemSettings.beam_current_range_na"
        )
        self.convergence_angle_range_mrad = _check_nonnegative_range(
            self.convergence_angle_range_mrad, "BeamSystemSettings.convergence_angle_range_mrad"
        )
        self.spot_size_range = _check_nonnegative_int_range(self.spot_size_range, "BeamSystemSettings.spot_size_range")

        return ok

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "default_beam": self.default_beam.to_dict() if self.default_beam is not None else None,
            "voltage_range_kv": self.voltage_range_kv,
            "beam_current_range_na": self.beam_current_range_na,
            "spot_size_range": self.spot_size_range,
            "convergence_angle_range_mrad": self.convergence_angle_range_mrad,
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSystemSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, BeamSystemSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"BeamSystemSettings.from_dict expects a dict, got {type(settings)}")
            return BeamSystemSettings(_mode=mode)

        known = {
            "enabled",
            "default_beam",
            "voltage_range_kv",
            "voltage_limits_kv",
            "beam_current_range_na",
            "spot_size_range",
            "convergence_angle_range_mrad",
            "extra",
        }
        extra = collect_extra(settings, known, owner="BeamSystemSettings")

        # accept aliases
        voltage_raw = settings.get("voltage_range_kv", settings.get("voltage_limits_kv", None))

        return BeamSystemSettings(
            enabled=settings.get("enabled", True),
            default_beam=settings.get("default_beam", None),
            voltage_range_kv=voltage_raw,
            beam_current_range_na=settings.get("beam_current_range_na", None),
            spot_size_range=settings.get("spot_size_range", None),
            convergence_angle_range_mrad=settings.get("convergence_angle_range_mrad", None),
            extra=extra,
            _mode=mode,
        )

@dataclass
class DetectorSettings:
    """Per-acquisition detector settings (the *requested* values).

    Keep this free of hardware capability metadata; that belongs in `DetectorSystemSettings`.
    """

    detector_id: Optional[str] = None

    # Acquisition / camera controls
    exposure_ms: Optional[float] = None
    binning_index: Optional[int] = None
    binning_xy: Optional[Tuple[int, int]] = None
    roi: Optional[ROI] = None

    frame_integration: Optional[int] = None
    gain_index: Optional[int] = None  # (often "contrast")
    offset_index: Optional[int] = None  # (often "brightness")
    digital_rotation_deg: Optional[float] = None

    extra: Extras = field(default_factory=Extras)

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "DetectorSettings")

        # detector_id is intentionally lenient: allow missing; enforce at request boundary.
        self.detector_id = parse_optional_id_like(
            self.detector_id, name="DetectorSettings.detector_id", strict=False, extra=self.extra
        )

        self.exposure_ms = parse_optional_float_like(
            self.exposure_ms, name="DetectorSettings.exposure_ms", strict=strict, extra=self.extra
        )

        self.binning_index = parse_optional_int_like(
            self.binning_index, name="DetectorSettings.binning_index", strict=strict, extra=self.extra
        )

        self.binning_xy = parse_optional_pair_int_like(
            self.binning_xy, name="DetectorSettings.binning_xy", sort=False, strict=strict, extra=self.extra
        )

        self.frame_integration = parse_optional_int_like(
            self.frame_integration, name="DetectorSettings.frame_integration", strict=strict, extra=self.extra
        )

        self.gain_index = parse_optional_int_like(
            self.gain_index, name="DetectorSettings.gain_index", strict=strict, extra=self.extra
        )

        self.offset_index = parse_optional_int_like(
            self.offset_index, name="DetectorSettings.offset_index", strict=strict, extra=self.extra
        )

        self.digital_rotation_deg = parse_optional_float_like(
            self.digital_rotation_deg, name="DetectorSettings.digital_rotation_deg", strict=strict, extra=self.extra
        )

        # ROI normalization (accept ROI/dict/list/tuple)
        if self.roi is not None:
            raw_roi = self.roi

            if isinstance(raw_roi, ROI):
                raw_roi.validate(mode=mode)
                self.roi = raw_roi

            elif isinstance(raw_roi, (list, tuple)):
                if len(raw_roi) != 4:
                    note_or_raise(
                        self.extra,
                        "DetectorSettings.roi",
                        ValueError(f"ROI list/tuple must have len 4, got {len(raw_roi)}"),
                        mode=mode,
                        raw=deepcopy(raw_roi),
                    )
                    self.roi = None
                else:
                    try:
                        self.roi = ROI.from_dict(raw_roi, mode=mode)
                    except Exception as e:
                        note_or_raise(self.extra, "DetectorSettings.roi", e, mode=mode, raw=deepcopy(raw_roi))
                        self.roi = None

            elif isinstance(raw_roi, dict):
                try:
                    self.roi = ROI.from_dict(raw_roi, mode=mode)
                except Exception as e:
                    note_or_raise(self.extra, "DetectorSettings.roi", e, mode=mode, raw=deepcopy(raw_roi))
                    self.roi = None

            else:
                note_or_raise(
                    self.extra,
                    "DetectorSettings.roi",
                    TypeError(f"roi must be ROI/dict/list/tuple, got {type(raw_roi)}"),
                    mode=mode,
                    raw=deepcopy(raw_roi),
                )
                self.roi = None



        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate detector-setting semantics.

        Parsing/coercion belongs in __post_init__ / from_dict.
        """
        mode = as_parse_mode(self._mode if mode is None else mode)

        if self.exposure_ms is not None and self.exposure_ms < 0:
            note_or_raise(self.extra, "DetectorSettings.exposure_ms", ValueError(f"exposure_ms must be >= 0, got {self.exposure_ms}"), mode=mode, raw=self.exposure_ms)
            self.exposure_ms = None

        if self.binning_index is not None and self.binning_index < 0:
            note_or_raise(self.extra, "DetectorSettings.binning_index", ValueError(f"binning_index must be >= 0, got {self.binning_index}"), mode=mode, raw=self.binning_index)
            self.binning_index = None

        if self.binning_xy is not None and (self.binning_xy[0] < 0 or self.binning_xy[1] < 0):
            note_or_raise(self.extra, "DetectorSettings.binning_xy", ValueError(f"binning_xy must be >= 0, got {self.binning_xy}"), mode=mode, raw=self.binning_xy)
            self.binning_xy = None

        if self.frame_integration is not None and self.frame_integration < 0:
            note_or_raise(self.extra, "DetectorSettings.frame_integration", ValueError(f"frame_integration must be >= 0, got {self.frame_integration}"), mode=mode, raw=self.frame_integration)
            self.frame_integration = None

        if self.gain_index is not None and self.gain_index < 0:
            note_or_raise(self.extra, "DetectorSettings.gain_index", ValueError(f"gain_index must be >= 0, got {self.gain_index}"), mode=mode, raw=self.gain_index)
            self.gain_index = None

        if self.offset_index is not None and self.offset_index < 0:
            note_or_raise(self.extra, "DetectorSettings.offset_index", ValueError(f"offset_index must be >= 0, got {self.offset_index}"), mode=mode, raw=self.offset_index)
            self.offset_index = None

        if self.digital_rotation_deg is not None and not math.isfinite(float(self.digital_rotation_deg)):
            note_or_raise(self.extra, "DetectorSettings.digital_rotation_deg", ValueError("digital_rotation_deg must be finite"), mode=mode, raw=self.digital_rotation_deg)
            self.digital_rotation_deg = None

        if self.roi is not None:
            # Let ROI validate itself (STRICT will raise).
            ok = self.roi.validate(mode=mode)
            if (not ok) or (self.roi.width <= 0 or self.roi.height <= 0):
                note_or_raise(self.extra, "DetectorSettings.roi_invalid_or_empty", ValueError("ROI width/height must be > 0"), mode=mode, raw=self.roi.to_dict())
                self.roi = None

        return True

    @classmethod
    def from_kwargs(cls, **kwargs):
        field_names = {f.name for f in fields(cls)}

        init_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}

        user_extra = init_kwargs.pop("extra", None)
        obj = cls(**init_kwargs)

        # merge provided extra (bucketed)
        merge_extras(obj.extra, user_extra, owner=f"{cls.__name__}")

        # unknown kwargs go to unknown bucket (not vendor)
        for k, v in extra_kwargs.items():
            try:
                obj.extra.unknown[str(k)] = deepcopy(v)
            except Exception:
                obj.extra.unknown[str(k)] = repr(v)

        obj.extra = normalize_extra(obj.extra)
        return obj

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "exposure_ms": self.exposure_ms,
            "binning_index": self.binning_index,
            "binning_xy": None if self.binning_xy is None else list(self.binning_xy),
            "frame_integration": self.frame_integration,
            "gain_index": self.gain_index,
            "offset_index": self.offset_index,
            "digital_rotation_deg": self.digital_rotation_deg,
            "roi": None if self.roi is None else self.roi.to_dict(),
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, DetectorSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"DetectorSettings.from_dict expects a dict, got {type(settings)}")
            return DetectorSettings(_mode=mode)

        known = {
            "detector_id",
            "exposure_ms",
            "binning_index",
            "binning_xy",
            "frame_integration",
            "roi",
            "gain_index",
            "offset_index",
            "digital_rotation_deg",
            # aliases
            "detector_roi",
            "imaging_area",
            "extra",
        }
        extra = collect_extra(settings, known, owner="DetectorSettings")

        # Keep from_dict thin: collect extras + propagate aliases; parsing/validation happens in __post_init__.
        roi_raw = settings.get("roi", None)
        if roi_raw is None:
            for alias in ("detector_roi", "imaging_area"):
                if alias in settings:
                    roi_raw = settings.get(alias)
                    break

        return DetectorSettings(
            detector_id=settings.get("detector_id", None),
            exposure_ms=settings.get("exposure_ms", None),
            binning_index=settings.get("binning_index", None),
            binning_xy=settings.get("binning_xy", None),
            frame_integration=settings.get("frame_integration", None),
            roi=roi_raw,
            gain_index=settings.get("gain_index", None),
            offset_index=settings.get("offset_index", None),
            digital_rotation_deg=settings.get("digital_rotation_deg", None),
            extra=extra,
            _mode=mode,
        )

@dataclass
class DetectorCapabilities:
    """Static capability description for a detector.

    Put *what the detector can do* here (ranges, supported features), not in `DetectorSettings`.
    Per-acquisition requests belong in `DetectorSettings`.

    Notes:
      - Many vendors expose different knobs. Anything you don't want to standardize goes in `extra`.
    """

    # Binning
    can_binning: Optional[bool] = None
    binning_index_min: Optional[int] = None
    binning_index_max: Optional[int] = None
    binning_xy_min: Optional[Tuple[int, int]] = None
    binning_xy_max: Optional[Tuple[int, int]] = None

    # Exposure / timing
    exposure_ms_min: Optional[float] = None
    exposure_ms_max: Optional[float] = None
    frame_integration_min: Optional[int] = None
    frame_integration_max: Optional[int] = None

    # ROI bounds (width, height)
    roi_min: Optional[Tuple[int, int]] = None
    roi_max: Optional[Tuple[int, int]] = None

    # Gain / offset
    can_gain: Optional[bool] = None
    gain_index_min: Optional[int] = None
    gain_index_max: Optional[int] = None

    can_offset: Optional[bool] = None
    offset_index_min: Optional[int] = None
    offset_index_max: Optional[int] = None

    # Digital rotation
    can_digital_rotation: Optional[bool] = None
    digital_rotation_deg_min: Optional[float] = None
    digital_rotation_deg_max: Optional[float] = None

    extra: Extras = field(default_factory=Extras)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        # Tuple-ish
        self.binning_xy_min = parse_optional_pair_int_like(self.binning_xy_min, name="DetectorCapabilities.binning_xy_min", strict=False, extra=self.extra)
        self.binning_xy_max = parse_optional_pair_int_like(self.binning_xy_max, name="DetectorCapabilities.binning_xy_max", strict=False, extra=self.extra)
        self.roi_min = parse_optional_pair_int_like(self.roi_min, name="DetectorCapabilities.roi_min", strict=False, extra=self.extra)
        self.roi_max = parse_optional_pair_int_like(self.roi_max, name="DetectorCapabilities.roi_max", strict=False, extra=self.extra)

        # Int-like ranges
        for name in (
            "binning_index_min", "binning_index_max",
            "frame_integration_min", "frame_integration_max",
            "gain_index_min", "gain_index_max",
            "offset_index_min", "offset_index_max",
        ):
            v = getattr(self, name)
            setattr(self, name, parse_optional_int_like(v, name=f"DetectorCapabilities.{name}", strict=False, extra=self.extra))

        # Float-like ranges
        for name in ("exposure_ms_min", "exposure_ms_max", "digital_rotation_deg_min", "digital_rotation_deg_max"):
            v = getattr(self, name)
            setattr(self, name, parse_optional_float_like(v, name=f"DetectorCapabilities.{name}", strict=False, extra=self.extra))

        # Optional bools
        for name in ("can_binning", "can_gain", "can_offset", "can_digital_rotation"):
            v = getattr(self, name)
            setattr(self, name, parse_optional_bool_like(v, name=f"DetectorCapabilities.{name}", strict=False, extra=self.extra))

        def _repair_minmax(min_name: str, max_name: str) -> None:
            v_min = getattr(self, min_name)
            v_max = getattr(self, max_name)
            if v_min is None or v_max is None:
                return
            if v_min > v_max:
                self.extra.notes[f"DetectorCapabilities.{min_name}_gt_{max_name}"] = {"min": v_min, "max": v_max}
                setattr(self, min_name, v_max)
                setattr(self, max_name, v_min)

        def _repair_pair_minmax(min_name: str, max_name: str) -> None:
            v_min = getattr(self, min_name)
            v_max = getattr(self, max_name)
            if v_min is None or v_max is None:
                return
            # component-wise repair
            min_w, min_h = v_min
            max_w, max_h = v_max
            if min_w > max_w or min_h > max_h:
                self.extra.notes[f"DetectorCapabilities.{min_name}_gt_{max_name}"] = {"min": v_min, "max": v_max}
                min_w, max_w = sorted((min_w, max_w))
                min_h, max_h = sorted((min_h, max_h))
                setattr(self, min_name, (min_w, min_h))
                setattr(self, max_name, (max_w, max_h))

        _repair_minmax("binning_index_min", "binning_index_max")
        _repair_pair_minmax("binning_xy_min", "binning_xy_max")
        _repair_minmax("exposure_ms_min", "exposure_ms_max")
        _repair_minmax("frame_integration_min", "frame_integration_max")
        _repair_pair_minmax("roi_min", "roi_max")
        _repair_minmax("gain_index_min", "gain_index_max")
        _repair_minmax("offset_index_min", "offset_index_max")
        _repair_minmax("digital_rotation_deg_min", "digital_rotation_deg_max")


    def to_dict(self) -> dict:
        # Serialize only non-None, and only include extra if non-empty
        d: Dict[str, Any] = {}
        for f in fields(DetectorCapabilities):
            if f.name == "extra":
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            d[f.name] = v
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any) -> "DetectorCapabilities":
        if isinstance(d, DetectorCapabilities):
            return d
        if not isinstance(d, dict):
            return DetectorCapabilities()

        known = {f.name for f in fields(DetectorCapabilities)} | {"extra"}
        extra = collect_extra(d, known, owner="DetectorCapabilities")

        # Parse into canonical types, but don't die if vendor dumps weird stuff into the JSON.
        kwargs: Dict[str, Any] = {}

        pair_fields = {"binning_xy_min", "binning_xy_max", "roi_min", "roi_max"}
        int_fields = {
            "binning_index_min", "binning_index_max",
            "frame_integration_min", "frame_integration_max",
            "gain_index_min", "gain_index_max",
            "offset_index_min", "offset_index_max",
        }
        float_fields = {"exposure_ms_min", "exposure_ms_max", "digital_rotation_deg_min", "digital_rotation_deg_max"}
        bool_fields = {"can_binning", "can_gain", "can_offset", "can_digital_rotation"}

        for f in fields(DetectorCapabilities):
            if f.name == "extra" or f.name not in d:
                continue

            v = d.get(f.name)

            if f.name in pair_fields:
                kwargs[f.name] = parse_optional_pair_int_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in int_fields:
                kwargs[f.name] = parse_optional_int_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in float_fields:
                kwargs[f.name] = parse_optional_float_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            elif f.name in bool_fields:
                kwargs[f.name] = parse_optional_bool_like(v, name=f"DetectorCapabilities.{f.name}", strict=False, extra=extra)
            else:
                kwargs[f.name] = v

        return DetectorCapabilities(**kwargs, extra=extra)

@dataclass
class DetectorSystemSettings:
    """Detector subsystem configuration (defaults + capabilities).

    TEM automation almost always deals with *multiple* detectors (camera, HAADF, BF, etc.).
    A single `default_detector` becomes ambiguous fast, so we store **per-detector** defaults.

    Conventions:
      - `defaults_by_id[detector_id]` holds a known-good baseline settings object for that detector.
      - `default_detector_id` is an *optional* session-level choice used when the caller doesn't specify
        a detector explicitly.
      - `capabilities_by_id[detector_id]` holds static capability/range info for that detector.
    """

    enabled: bool = True

    # Per-detector baseline settings (keyed by detector_id).
    defaults_by_id: Dict[str, DetectorSettings] = field(default_factory=dict)

    # Optional: which detector should be used by default if none is specified.
    default_detector_id: Optional[str] = None

    # Capability map keyed by detector_id.
    capabilities_by_id: Dict[str, DetectorCapabilities] = field(default_factory=dict)

    # Optional list of detectors you want to advertise/allow in automation UI.
    # If empty, it can be inferred from `capabilities_by_id` or `defaults_by_id`.
    available_detectors: List[str] = field(default_factory=list)

    extra: Extras = field(default_factory=Extras)

    # Parsing / validation mode (strict by default).
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "DetectorSystemSettings")
        self.enabled = parse_bool(self.enabled, default=True)

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

        if self.available_detectors is None:
            self.available_detectors = []
        elif not isinstance(self.available_detectors, list):
            note_or_raise(
                self.extra,
                "DetectorSystemSettings.available_detectors",
                TypeError(f"available_detectors must be list, got {type(self.available_detectors)}"),
                mode=mode,
                raw=deepcopy(self.available_detectors),
            )
            self.available_detectors = []

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
                v.validate(mode=mode)
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

        # Normalize available_detectors entries to str ids (drop invalid ones in lenient)
        norm_avail: List[str] = []
        for item in self.available_detectors:
            det_id = parse_optional_id_like(
                item, name="DetectorSystemSettings.available_detectors[]", strict=False, extra=self.extra
            )
            if det_id is None:
                self.extra.notes.setdefault("DetectorSystemSettings.available_detectors.invalid", []).append(repr(item))
                continue
            norm_avail.append(det_id)
        seen=set()
        self.available_detectors = [x for x in norm_avail if not (x in seen or seen.add(x))]

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate detector-system semantics."""
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        ok = True

        if self.default_detector_id:
            det_id = str(self.default_detector_id)
            known = set(self.defaults_by_id.keys()) | set(self.capabilities_by_id.keys()) | set(self.available_detectors)
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
            if not isinstance(ds, DetectorSettings):
                note_or_raise(
                    self.extra,
                    f"DetectorSystemSettings.defaults_by_id.{det_id}",
                    TypeError(f"defaults_by_id['{det_id}'] must be DetectorSettings, got {type(ds)}"),
                    mode=mode,
                    raw=deepcopy(ds),
                )
                ok = False
                continue
            ok = ds.validate(mode=mode) and ok

        return ok

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "default_detector_id": self.default_detector_id,
            "available_detectors": list(self.available_detectors) if self.available_detectors else [],
            "defaults_by_id": {k: v.to_dict() for k, v in self.defaults_by_id.items()},
            "capabilities_by_id": {k: v.to_dict() for k, v in self.capabilities_by_id.items()},
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSystemSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, DetectorSystemSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"DetectorSystemSettings.from_dict expects a dict, got {type(settings)}")
            return DetectorSystemSettings(_mode=mode)

        known = {
            "enabled",
            "available_detectors",
            "default_detector_id",
            "defaults_by_id",
            "default_settings_by_id",
            "defaults",
            "capabilities_by_id",
            "capabilities",
            "extra",
        }
        extra = collect_extra(settings, known, owner="DetectorSystemSettings")

        defaults_raw = settings.get("defaults_by_id", settings.get("default_settings_by_id", settings.get("defaults", {})))
        caps_raw = settings.get("capabilities_by_id", settings.get("capabilities", {}))

        return DetectorSystemSettings(
            enabled=settings.get("enabled", True),
            available_detectors=settings.get("available_detectors", []),
            default_detector_id=settings.get("default_detector_id", None),
            defaults_by_id=defaults_raw,
            capabilities_by_id=caps_raw,
            extra=extra,
            _mode=mode,
        )

@dataclass
class ImageOutputSettings:
    file_format: Optional[str] = "tiff"  # supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}
    path: Optional[Union[str, Path]] = None  # default output directory (session dir)
    extra: Extras = field(default_factory=Extras)

    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "ImageOutputSettings")

        ff = parse_optional_str_like(
            self.file_format, name="ImageOutputSettings.file_format", strict=strict, extra=self.extra
        )
        self.file_format = (ff or "tiff").lower()

        if isinstance(self.path, str) and self.path.strip() == "":
            self.path = None

        if self.path is not None and not isinstance(self.path, Path):
            if isinstance(self.path, (str, bytes)):
                try:
                    self.path = Path(self.path)
                except Exception as e:
                    note_or_raise(self.extra, "ImageOutputSettings.path", e, mode=mode, raw=self.path)
                    self.path = None
            else:
                note_or_raise(
                    self.extra,
                    "ImageOutputSettings.path",
                    TypeError(f"path must be str/Path, got {type(self.path)}"),
                    mode=mode,
                    raw=self.path,
                )
                self.path = None

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate ImageOutputSettings semantics."""
        mode = as_parse_mode(self._mode if mode is None else mode)

        supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}
        if self.file_format is not None and self.file_format not in supported:
            note_or_raise(
                self.extra,
                "ImageOutputSettings.file_format",
                ValueError(f"Unsupported file_format: {self.file_format!r}. Supported: {sorted(supported)}"),
                mode=mode,
                raw=self.file_format,
            )
            self.file_format = "tiff"

        return True

    def to_dict(self) -> dict:
        d = {"file_format": self.file_format, "path": None if self.path is None else str(self.path)}
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ImageOutputSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, ImageOutputSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"ImageOutputSettings.from_dict expects a dict, got {type(settings)}")
            return ImageOutputSettings(_mode=mode)

        known = ("file_format", "path", "extra")
        extra = collect_extra(settings, known=known, owner="ImageOutputSettings")

        # Keep from_dict thin: parsing/validation happens in __post_init__.
        return ImageOutputSettings(
            file_format=settings.get("file_format", "tiff"),
            path=settings.get("path", None),
            extra=extra,
            _mode=mode,
        )

@dataclass
class AcquisitionRequest:
    detector_id: Optional[str] = None
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    image: ImageOutputSettings = field(default_factory=ImageOutputSettings)
    extra: Extras = field(default_factory=Extras)

    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra,
                                                                                        "AcquisitionRequest")

        # ---- normalize nested objects (never leave them as raw dicts/garbage) ----
        det_raw = self.detector
        if det_raw is None:
            self.detector = DetectorSettings(_mode=mode)
        elif isinstance(det_raw, DetectorSettings):
            det_raw.validate(mode=mode)
            self.detector = det_raw
        elif isinstance(det_raw, dict):
            try:
                self.detector = DetectorSettings.from_dict(det_raw, mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "AcquisitionRequest.detector", e, mode=mode, raw=deepcopy(det_raw))
                self.detector = DetectorSettings(_mode=mode)
        else:
            note_or_raise(
                self.extra,
                "AcquisitionRequest.detector",
                TypeError(f"detector must be DetectorSettings/dict/None, got {type(det_raw)}"),
                mode=mode,
                raw=deepcopy(det_raw),
            )
            self.detector = DetectorSettings(_mode=mode)

        img_raw = self.image
        if img_raw is None:
            self.image = ImageOutputSettings(_mode=mode)
        elif isinstance(img_raw, ImageOutputSettings):
            img_raw.validate(mode=mode)
            self.image = img_raw
        elif isinstance(img_raw, dict):
            try:
                self.image = ImageOutputSettings.from_dict(img_raw, mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "AcquisitionRequest.image", e, mode=mode, raw=deepcopy(img_raw))
                self.image = ImageOutputSettings(_mode=mode)
        else:
            note_or_raise(
                self.extra,
                "AcquisitionRequest.image",
                TypeError(f"image must be ImageOutputSettings/dict/None, got {type(img_raw)}"),
                mode=mode,
                raw=deepcopy(img_raw),
            )
            self.image = ImageOutputSettings(_mode=mode)


        # ---- normalize detector_id (do not enforce required-ness here) ----
        self.detector_id = parse_optional_id_like(
            self.detector_id, name="AcquisitionRequest.detector_id", strict=strict, extra=self.extra
        )

        # If missing, adopt from nested detector (best-effort).
        if self.detector_id is None and self.detector.detector_id:
            self.detector_id = str(self.detector.detector_id)

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Enforce request invariants (strictness depends on mode).

        Keep this free of parsing/normalization; that belongs in __post_init__ / from_dict.
        """
        mode = as_parse_mode(self._mode if mode is None else mode)
        strict = is_strict(mode)

        if not isinstance(self.detector, DetectorSettings):
            if strict:
                note_or_raise(
                    self.extra,
                    "AcquisitionRequest.detector",
                    TypeError(f"AcquisitionRequest.detector must be DetectorSettings, got {type(self.detector)}"),
                    mode=mode,
                    raw=self.detector,
                )
                return False
            self.detector = DetectorSettings()
        if not isinstance(self.image, ImageOutputSettings):
            if strict:
                note_or_raise(
                    self.extra,
                    "AcquisitionRequest.image",
                    TypeError(f"AcquisitionRequest.image must be ImageOutputSettings, got {type(self.image)}"),
                    mode=mode,
                    raw=self.image
                )
                return False
            self.image = ImageOutputSettings()

        # Resolve the effective detector_id (request field wins).
        det_id = self.detector_id
        if det_id is None and self.detector.detector_id:
            det_id = self.detector.detector_id

        det_id = None if det_id is None else str(det_id).strip()
        if not det_id:
            if strict:
                note_or_raise(
                    self.extra,
                    "AcquisitionRequest.detector_id",
                    ValueError("detector_id is required in STRICT mode"),
                    mode=mode,
                    raw=self.detector_id,
                )
                return False
            return True  # lenient: allow missing

        self.detector_id = det_id

        # Keep nested detector settings consistent.
        if self.detector.detector_id is None:
            self.detector.detector_id = det_id
        elif str(self.detector.detector_id) != det_id:
            note_or_raise(
                self.extra,
                "AcquisitionRequest.detector_id_mismatch",
                ValueError(
                    f"detector_id mismatch: request={det_id!r} detector.detector_id={self.detector.detector_id!r}"
                ),
                mode=mode,
                raw={"request": det_id, "detector.detector_id": self.detector.detector_id},
            )
            if strict:
                return False
            self.detector.detector_id = det_id

        return True

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
        strict = is_strict(mode)

        if isinstance(d, AcquisitionRequest):
            d.validate(mode=mode)
            return d
        if not isinstance(d, dict):
            if strict:
                raise ValueError("AcquisitionRequest.from_dict expects a dict")
            return AcquisitionRequest(_mode=mode)

        used_keys = {"detector_id", "detector", "image", "extra"}
        extra = collect_extra(d, used_keys, owner="AcquisitionRequest")

        det_raw = d.get("detector")
        if isinstance(det_raw, DetectorSettings):
            det = det_raw
        elif isinstance(det_raw, dict):
            det = DetectorSettings.from_dict(det_raw, mode=mode)
        else:
            det = DetectorSettings(_mode=mode)
            if det_raw is not None:
                extra.raw["AcquisitionRequest.detector"] = _jsonable(det_raw)

        img_raw = d.get("image")
        if isinstance(img_raw, ImageOutputSettings):
            img = img_raw
        elif isinstance(img_raw, dict):
            img = ImageOutputSettings.from_dict(img_raw, mode=mode)
        else:
            img = ImageOutputSettings(_mode=mode)
            if img_raw is not None:
                extra.raw["AcquisitionRequest.image"] = _jsonable(img_raw)

        raw_id = d.get("detector_id", None)
        if raw_id is None and isinstance(det, DetectorSettings):
            raw_id = det.detector_id

        det_id = parse_optional_id_like(raw_id, name="AcquisitionRequest.detector_id", strict=strict, extra=extra)

        return AcquisitionRequest(detector_id=det_id, detector=det, image=img, extra=extra, _mode=mode)

@dataclass
class MicroscopeState:
    """
    Snapshot of microscope state at a moment in time.

    Notes:
      - timestamp is stored as unix seconds (float) for easy logging/ordering.
      - stage_position stores Quantity internally, but serializes to plain numbers in nm/deg.
    """

    timestamp: float = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).timestamp()
    )
    stage_position: TemStagePosition = field(default_factory=TemStagePosition)
    beam: BeamSettings = field(default_factory=BeamSettings)

    # All known/current detector configurations (by ID)
    detectors: Dict[str, DetectorSettings] = field(default_factory=dict)

    # Which detectors are currently active / producing signal
    active_detector_ids: List[str] = field(default_factory=list)

    # Optional: the “main” detector the UI/operator considers selected
    primary_detector_id: Optional[str] = None

    # Anything session/protocol-ish you want to carry along
    protocol: Dict[str, Any] = field(default_factory=dict)

    # Vendor-specific / unknown fields live here
    extra: Extras = field(default_factory=Extras)

    # Parsing mode: states are lenient by default.
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "MicroscopeState")

        # timestamp
        self.timestamp = float(self._parse_timestamp(self.timestamp))

        # stage_position
        sp = self.stage_position
        if sp is None:
            self.stage_position = TemStagePosition()
        elif isinstance(sp, TemStagePosition):
            self.stage_position = sp
        elif isinstance(sp, dict):
            try:
                self.stage_position = TemStagePosition.from_dict(sp, mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "MicroscopeState.stage_position", e, mode=mode, raw=sp)
                self.stage_position = TemStagePosition()
        else:
            note_or_raise(
                self.extra,
                "MicroscopeState.stage_position",
                TypeError(f"stage_position must be TemStagePosition/dict, got {type(sp)}"),
                mode=mode,
                raw=sp,
            )
            self.stage_position = TemStagePosition()

        # beam
        b = self.beam
        if b is None:
            self.beam = BeamSettings(_mode=mode)
        elif isinstance(b, BeamSettings):
            b.validate(mode=mode)
            self.beam = b
        elif isinstance(b, dict):
            try:
                self.beam = BeamSettings.from_dict(b, mode=mode)
            except Exception as e:
                note_or_raise(self.extra, "MicroscopeState.beam", e, mode=mode, raw=b)
                self.beam = BeamSettings(_mode=mode)
        else:
            note_or_raise(
                self.extra,
                "MicroscopeState.beam",
                TypeError(f"beam must be BeamSettings/dict, got {type(b)}"),
                mode=mode,
                raw=b,
            )
            self.beam = BeamSettings(_mode=mode)

        # detectors
        dets_raw = self.detectors
        if dets_raw is None:
            dets_raw = {}
        if not isinstance(dets_raw, dict):
            note_or_raise(
                self.extra,
                "MicroscopeState.detectors",
                TypeError(f"detectors must be dict, got {type(dets_raw)}"),
                mode=mode,
                raw=dets_raw,
            )
            dets_raw = {}

        fixed: Dict[str, DetectorSettings] = {}
        for det_id, det in dets_raw.items():
            key = parse_optional_id_like(det_id, name="MicroscopeState.detectors.key", strict=strict, extra=self.extra)
            if key is None:
                self.extra.notes[f"MicroscopeState.detectors.{det_id}_raw_key"] = det_id
                continue
            try:
                if isinstance(det, DetectorSettings):
                    ds = det
                    ds.validate(mode=mode)
                elif isinstance(det, dict):
                    ds = DetectorSettings.from_dict(det, mode=mode)
                else:
                    self.extra.raw[f"MicroscopeState.detectors.{key}"] = _jsonable(det)
                    continue

                if ds.detector_id is None:
                    ds.detector_id = key
                elif str(ds.detector_id) != key:
                    self.extra.notes[f"MicroscopeState.detectors.{key}.detector_id_mismatch"] = ds.detector_id
                    ds.detector_id = key

                fixed[key] = ds
            except Exception as e:
                note_or_raise(self.extra, f"MicroscopeState.detectors.{key}", e, mode=mode, raw=det)

        self.detectors = fixed

        # active_detector_ids
        ids = self.active_detector_ids
        if ids is None:
            ids = []
        if not isinstance(ids, (list, tuple)):
            note_or_raise(
                self.extra,
                "MicroscopeState.active_detector_ids",
                TypeError("active_detector_ids must be a list"),
                mode=mode,
                raw=ids,
            )
            ids = []

        cleaned: List[str] = []
        for item in ids:
            s = parse_optional_id_like(item, name="MicroscopeState.active_detector_ids[]", strict=strict, extra=self.extra)
            if s:
                cleaned.append(s)
        self.active_detector_ids = cleaned

        # primary_detector_id
        self.primary_detector_id = parse_optional_id_like(
            self.primary_detector_id, name="MicroscopeState.primary_detector_id", strict=strict, extra=self.extra
        )

        # protocol: keep dict-ish
        if self.protocol is None:
            self.protocol = {}
        elif not isinstance(self.protocol, dict):
            self.extra.raw["MicroscopeState.protocol"] = _jsonable(self.protocol)
            self.protocol = {}

        # Semantic constraints live in validate().
        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        """Validate MicroscopeState semantics.

        This is intentionally *lightweight*: state snapshots are often partial.
        In LENIENT mode we repair/trim obvious inconsistencies; in STRICT we raise.
        """
        mode = as_parse_mode(self._mode if mode is None else mode)

        # timestamp should be finite
        if not math.isfinite(float(self.timestamp)):
            note_or_raise(self.extra, "MicroscopeState.timestamp", ValueError("timestamp must be finite"), mode=mode, raw=self.timestamp)
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # active_detector_ids must be subset of detectors
        if self.active_detector_ids:
            missing = [d for d in self.active_detector_ids if d not in (self.detectors or {})]
            if missing:
                note_or_raise(
                    self.extra,
                    "MicroscopeState.active_detector_ids_missing",
                    ValueError(f"active_detector_ids not present in detectors: {missing}"),
                    mode=mode,
                    raw=missing,
                )
                self.active_detector_ids = [d for d in self.active_detector_ids if d in (self.detectors or {})]

        # primary_detector_id must exist if provided
        if self.primary_detector_id and self.primary_detector_id not in (self.detectors or {}):
            note_or_raise(
                self.extra,
                "MicroscopeState.primary_detector_id_missing",
                ValueError(f"primary_detector_id {self.primary_detector_id!r} not present in detectors"),
                mode=mode,
                raw=self.primary_detector_id,
            )
            self.primary_detector_id = None

        return True

    @staticmethod
    def _parse_timestamp(v: Any) -> float:
        """Accept float/int, numeric str, or ISO datetime string."""
        if v is None:
            return datetime.datetime.now(datetime.timezone.utc).timestamp()
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return datetime.datetime.now(datetime.timezone.utc).timestamp()
            # numeric string
            try:
                return float(s)
            except Exception:
                pass
            # ISO string
            try:
                dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.timestamp()
            except Exception:
                return datetime.datetime.now(datetime.timezone.utc).timestamp()
        try:
            return float(v)
        except Exception:
            return datetime.datetime.now(datetime.timezone.utc).timestamp()

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "stage_position": None if self.stage_position is None else self.stage_position.to_dict(),
            "beam": None if self.beam is None else self.beam.to_dict(),
            "detectors": {k: v.to_dict() for k, v in (self.detectors or {}).items()},
            "active_detector_ids": list(self.active_detector_ids or []),
            "primary_detector_id": self.primary_detector_id,
            "protocol": deepcopy(self.protocol) if self.protocol else {},
        }
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeState":
        mode = as_parse_mode(mode)

        if isinstance(d, MicroscopeState):
            d.validate(mode=mode)
            return d
        if not isinstance(d, dict):
            return MicroscopeState(_mode=mode)

        known = {
            "timestamp",
            "stage_position",
            "beam",
            "detectors",
            "active_detector_ids",
            "primary_detector_id",
            "protocol",
            "extra",
        }
        extra = collect_extra(d, known, owner="MicroscopeState")

        return MicroscopeState(
            timestamp=d.get("timestamp"),
            stage_position=d.get("stage_position"),
            beam=d.get("beam"),
            detectors=d.get("detectors") or {},
            active_detector_ids=d.get("active_detector_ids") or [],
            primary_detector_id=d.get("primary_detector_id"),
            protocol=d.get("protocol") or {},
            extra=extra,
            _mode=mode,
        )

class TemImage:
    """
    Generic TEM Image object with universal metadata handling.

    Attributes:
        data (np.ndarray): image data.
        metadata (TemImageMetadata): associated metadata.
    Supports:
        - ThermoFisher API (AdornedImage)
        - Tescan API (Header)
        - Other vendors via vendor-specific factory functions.
    """

    def __init__(self, data: np.ndarray, metadata: Optional[TemImageMetadata] = None):
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]

        if not _check_data_format(data):
            raise ValueError("Invalid data format for TemImage.")
        self.data = data
        if isinstance(metadata, dict):
            metadata = TemImageMetadata.from_dict(metadata)
        elif metadata is not None and not isinstance(metadata, TemImageMetadata):
            raise TypeError(f"metadata must be TemImageMetadata/dict/None, got {type(metadata)}")
        self.metadata = metadata

    # -------------------------- helpers --------------------------

    @staticmethod
    def _decode_description(desc: Any) -> Optional[Dict[str, Any]]:
        """Try to parse TIFF ImageDescription as JSON dict."""
        if desc is None:
            return None
        if isinstance(desc, bytes):
            try:
                desc = desc.decode("utf-8", errors="replace")
            except Exception:
                return None
        if not isinstance(desc, str):
            return None
        desc = desc.strip()
        if not desc:
            return None
        try:
            obj = json.loads(desc)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _encode_description(md: Optional[TemImageMetadata]) -> str:
        if md is None:
            return ""
        try:
            safe = _jsonable(md.to_dict())
            return json.dumps(safe, ensure_ascii=False)
        except Exception:
            # last-ditch: at least keep something
            minimal = {"created_at": getattr(md, "created_at", None), "version": getattr(md, "version", None)}
            return json.dumps(_jsonable(minimal), ensure_ascii=False)

    @staticmethod
    def _to_uint8_preview(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        """
        Convert arbitrary numeric image to uint8 for preview exports (JPEG/BMP).
        Uses percentile scaling to avoid single hot pixels ruining contrast.
        """
        a = np.asarray(arr)

        if a.size == 0:
            return a.astype(np.uint8, copy=False)

        # Work in float for scaling
        af = a.astype(np.float32, copy=False)

        finite = af[np.isfinite(af)]
        if finite.size == 0:
            return np.zeros_like(a, dtype=np.uint8)

        lo, hi = np.percentile(finite, [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # constant image or broken stats
            return np.zeros_like(a, dtype=np.uint8)

        scaled = (af - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    # -------------------------- I/O --------------------------

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemImage":
        """Load an image from disk.

        - For TIFF (.tif/.tiff): metadata is read from ImageDescription (JSON), if present.
        - For other supported formats (png/jpg/jpeg/bmp): image pixels are loaded via Pillow.
          Metadata is loaded from a sidecar JSON file: <filename>.<ext>.json, if present.
        """
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")

        if ext in ("tif", "tiff"):
            with tff.TiffFile(str(path)) as tif:
                data = tif.asarray()
                # If it's (1, H, W), drop the frame axis
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]

                # If it's (H, W, 1), drop the singleton channel axis
                if data.ndim == 3 and data.shape[-1] == 1:
                    data = data[..., 0]

                if data.ndim != 2:
                    raise ValueError(f"Expected single-frame 2D grayscale TIFF, got shape={data.shape}")

                if data.dtype == np.int16:
                    # If it never goes below 0, it's basically unsigned data stored signed.
                    if data.min() >= 0:
                        data = data.astype(np.uint16)
                    else:
                        raise ValueError(
                            "Loaded TIFF is int16 with negative values. "
                            "TemImage expects unsigned intensity images (uint8/uint16). "
                            "Convert to uint16 (with an appropriate offset/clamp) or store this as a signed map type."
                        )
                if data.dtype in (np.int32, np.uint32):
                    if data.min() >= 0 and data.max() <= 65535:
                        data = data.astype(np.uint16)
                    else:
                        raise ValueError(
                            "TIFF is 32-bit with values outside uint16 range; not a standard intensity image.")

                metadata: Optional[TemImageMetadata] = None
                try:
                    desc = tif.pages[0].tags["ImageDescription"].value
                    d = cls._decode_description(desc)
                    if d is not None:
                        metadata = TemImageMetadata.from_dict(d)
                except Exception:
                    metadata = None

            return cls(data=data, metadata=metadata)

        # Non-TIFF: load pixels with Pillow
        with Image.open(path) as img:
            if img.mode not in ("L", "I;16", "I;16B", "I;16L"):
                img = img.convert("L")

            data = np.array(img)
            if data.ndim == 3 and data.shape[-1] == 1:
                data = data[..., 0]

            # Pillow may produce int32 for some modes; normalize to uint16/uint8.
            if data.dtype == np.int32 and img.mode in ("I;16", "I;16B", "I;16L"):
                data = data.astype(np.uint16)

            if data.dtype not in (np.uint8, np.uint16):
                # Be conservative: coerce to uint8 preview-style
                if np.issubdtype(data.dtype, np.number):
                    data = np.clip(data, 0, 255).astype(np.uint8)
                else:
                    data = data.astype(np.uint8)

        metadata: Optional[TemImageMetadata] = None
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    metadata = TemImageMetadata.from_dict(d)
            except Exception:
                metadata = None

        return cls(data=data, metadata=metadata)

    def save(self, path: Union[str, Path], file_format: Optional[str] = None) -> Path:
        supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}

        path = Path(path)

        # Keep the user's requested extension (jpg vs jpeg, tif vs tiff) for naming,
        # but use a canonical fmt for logic.
        requested_ext = (file_format or path.suffix.lstrip(".") or "tiff").lower().strip()
        if requested_ext not in supported:
            raise ValueError(f"Unsupported file_format: {requested_ext!r}. Supported: {sorted(supported)}")

        fmt = "tiff" if requested_ext in ("tif", "tiff") else requested_ext

        # Enforce suffix WITHOUT rewriting jpg->jpeg (or tif<->tiff)
        if requested_ext == "tif":
            suffix = ".tif"
        elif requested_ext == "tiff":
            suffix = ".tiff"
        else:
            suffix = f".{requested_ext}"
        path = path.with_suffix(suffix)

        os.makedirs(path.parent, exist_ok=True)

        desc = self._encode_description(self.metadata)

        if fmt == "tiff":
            tff.imwrite(str(path), self.data, description=desc)
            return path

        # Other formats: save pixels via Pillow (metadata via sidecar)
        data_to_save = self.data

        # JPEG/BMP can't store uint16 grayscale reliably; create a uint8 preview.
        if fmt in ("jpg", "jpeg", "bmp"):
            if data_to_save.dtype != np.uint8:
                data_to_save = self._to_uint8_preview(data_to_save)
        else:
            # PNG supports uint16, but keep it uint8/uint16 only.
            if data_to_save.dtype not in (np.uint8, np.uint16):
                data_to_save = np.clip(data_to_save, 0, 255).astype(np.uint8)

        img = Image.fromarray(data_to_save)

        pil_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "bmp": "BMP"}[fmt]
        img.save(path, format=pil_format)

        # Sidecar metadata (best-effort)
        sidecar = path.with_suffix(path.suffix + ".json")
        try:
            if self.metadata is not None:
                sidecar.write_text(
                    json.dumps(_jsonable(self.metadata.to_dict()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                if sidecar.exists():
                    sidecar.unlink()
        except Exception:
            pass

        return path

@dataclass
class SystemInfo:
    """Basic microscope/system identity information (mostly informational).

    Default behavior is LENIENT for ingestion; pass mode=STRICT when you want fail-fast.
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
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        self.extra = normalize_extra(self.extra) if strict else normalize_extra_lenient(self.extra, "SystemInfo")

        self.name = parse_optional_str_like(self.name, name="SystemInfo.name", strict=strict, extra=self.extra) or "Unknown"
        self.ip_address = (
            parse_optional_str_like(self.ip_address, name="SystemInfo.ip_address", strict=strict, extra=self.extra) or "Unknown"
        )
        self.manufacturer = (
            parse_optional_str_like(self.manufacturer, name="SystemInfo.manufacturer", strict=strict, extra=self.extra) or "Unknown"
        )
        self.model = parse_optional_str_like(self.model, name="SystemInfo.model", strict=strict, extra=self.extra) or "Unknown"
        self.serial_number = (
            parse_optional_str_like(self.serial_number, name="SystemInfo.serial_number", strict=strict, extra=self.extra) or "Unknown"
        )
        self.hardware_version = (
            parse_optional_str_like(self.hardware_version, name="SystemInfo.hardware_version", strict=strict, extra=self.extra)
            or "Unknown"
        )
        self.software_version = (
            parse_optional_str_like(self.software_version, name="SystemInfo.software_version", strict=strict, extra=self.extra)
            or "Unknown"
        )
        self.supertem_version = (
            parse_optional_str_like(self.supertem_version, name="SystemInfo.supertem_version", strict=strict, extra=self.extra)
            or __version__
        )
        self.application = parse_optional_str_like(
            self.application, name="SystemInfo.application", strict=strict, extra=self.extra
        )
        self.application_version = parse_optional_str_like(
            self.application_version, name="SystemInfo.application_version", strict=strict, extra=self.extra
        )

        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)

        # ip_address: if present and not "Unknown", it should look like an IP.
        ip = self.ip_address
        if isinstance(ip, str):
            ip = ip.strip()

        if ip and ip != "Unknown":
            try:
                import ipaddress

                ipaddress.ip_address(ip)
            except Exception:
                note_or_raise(
                    self.extra,
                    "SystemInfo.ip_address",
                    ValueError(f"Invalid IP address: {self.ip_address!r}"),
                    mode=mode,
                    raw=self.ip_address,
                )
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
        add_extra_if_any(d, self.extra)
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemInfo":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, SystemInfo):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"SystemInfo.from_dict expects a dict, got {type(settings)}")
            return SystemInfo(_mode=mode)

        known = {
            "name",
            "ip_address",
            "manufacturer",
            "model",
            "serial_number",
            "hardware_version",
            "software_version",
            "supertem_version",
            "application",
            "application_version",
        }
        extra = collect_extra(settings, known=known, owner="SystemInfo")

        return SystemInfo(
            name=settings.get("name", "Unknown"),
            ip_address=settings.get("ip_address", "Unknown"),
            manufacturer=settings.get("manufacturer", "Unknown"),
            model=settings.get("model", "Unknown"),
            serial_number=settings.get("serial_number", "Unknown"),
            hardware_version=settings.get("hardware_version", "Unknown"),
            software_version=settings.get("software_version", "Unknown"),
            supertem_version=settings.get("supertem_version", __version__),
            application=settings.get("application", None),
            application_version=settings.get("application_version", None),
            extra=extra,
            _mode=mode,
        )

@dataclass
class SystemSettings:
    """Microscope system settings (stage/beam/detector + identity info).

    This is a config boundary:
      - Use STRICT when you are about to *apply* settings to hardware.
      - Use LENIENT when loading older/partial configs.
    """

    stage: StageSystemSettings = field(default_factory=StageSystemSettings)
    beam: BeamSystemSettings = field(default_factory=BeamSystemSettings)
    detector: DetectorSystemSettings = field(default_factory=DetectorSystemSettings)
    info: SystemInfo = field(default_factory=SystemInfo)

    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        # stage
        if isinstance(self.stage, dict):
            self.stage = StageSystemSettings.from_dict(self.stage, mode=mode)
        elif self.stage is None:
            self.stage = StageSystemSettings(_mode=mode)
        elif not isinstance(self.stage, StageSystemSettings):
            if strict:
                raise TypeError(f"SystemSettings.stage must be StageSystemSettings/dict, got {type(self.stage)}")
            self.stage = StageSystemSettings(_mode=mode)

        # beam
        if isinstance(self.beam, dict):
            self.beam = BeamSystemSettings.from_dict(self.beam, mode=mode)
        elif self.beam is None:
            self.beam = BeamSystemSettings(_mode=mode)
        elif not isinstance(self.beam, BeamSystemSettings):
            if strict:
                raise TypeError(f"SystemSettings.beam must be BeamSystemSettings/dict, got {type(self.beam)}")
            self.beam = BeamSystemSettings(_mode=mode)

        # detector
        if isinstance(self.detector, dict):
            self.detector = DetectorSystemSettings.from_dict(self.detector, mode=mode)
        elif self.detector is None:
            self.detector = DetectorSystemSettings(_mode=mode)
        elif not isinstance(self.detector, DetectorSystemSettings):
            if strict:
                raise TypeError(f"SystemSettings.detector must be DetectorSystemSettings/dict, got {type(self.detector)}")
            self.detector = DetectorSystemSettings(_mode=mode)

        # info
        if isinstance(self.info, dict):
            self.info = SystemInfo.from_dict(self.info, mode=mode)
        elif self.info is None:
            self.info = SystemInfo(_mode=mode)
        elif not isinstance(self.info, SystemInfo):
            if strict:
                raise TypeError(f"SystemSettings.info must be SystemInfo/dict, got {type(self.info)}")
            self.info = SystemInfo(_mode=mode)

        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)

        self.stage.validate(mode=mode)
        self.beam.validate(mode=mode)
        self.detector.validate(mode=mode)
        self.info.validate(mode=mode)

        return True

    def to_dict(self) -> dict:
        d = {
            "stage": self.stage.to_dict(),
            "beam": self.beam.to_dict(),
            "detector": self.detector.to_dict(),
            "info": self.info.to_dict(),
        }
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(settings: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, SystemSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"SystemSettings.from_dict expects a dict, got {type(settings)}")
            return SystemSettings(_mode=mode)

        return SystemSettings(
            stage=settings.get("stage"),
            beam=settings.get("beam"),
            detector=settings.get("detector"),
            info=settings.get("info"),
            _mode=mode,
        )

@dataclass
class MicroscopeSettings:
    """Top-level settings bundle (system + image + protocol blob)."""

    system: SystemSettings = field(default_factory=SystemSettings)
    image: ImageOutputSettings = field(default_factory=ImageOutputSettings)

    # Protocol is intentionally a free-form dict (owned by higher-level logic).
    protocol: dict = field(default_factory=lambda: {"name": "demo"})

    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        mode = as_parse_mode(self._mode)
        strict = is_strict(mode)

        if isinstance(self.system, dict):
            self.system = SystemSettings.from_dict(self.system, mode=mode)
        elif self.system is None:
            self.system = SystemSettings(_mode=mode)
        elif not isinstance(self.system, SystemSettings):
            if strict:
                raise TypeError(f"MicroscopeSettings.system must be SystemSettings/dict, got {type(self.system)}")
            self.system = SystemSettings(_mode=mode)

        if isinstance(self.image, dict):
            self.image = ImageOutputSettings.from_dict(self.image, mode=mode)
        elif self.image is None:
            self.image = ImageOutputSettings(_mode=mode)
        elif not isinstance(self.image, ImageOutputSettings):
            if strict:
                raise TypeError(f"MicroscopeSettings.image must be ImageOutputSettings/dict, got {type(self.image)}")
            self.image = ImageOutputSettings(_mode=mode)

        if self.protocol is None:
            self.protocol = {"name": "demo"}
        elif not isinstance(self.protocol, dict):
            if strict:
                raise TypeError(f"MicroscopeSettings.protocol must be dict, got {type(self.protocol)}")
            self.protocol = {"name": "demo"}

        self.validate(mode=mode)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        mode = as_parse_mode(self._mode if mode is None else mode)

        self.system.validate(mode=mode)
        self.image.validate(mode=mode)

        # Protocol is free-form, but must stay JSON-able; `_jsonable` handles conversion.
        if not isinstance(self.protocol, dict):
            note_or_raise(None, "MicroscopeSettings.protocol", TypeError("protocol must be a dict"), mode=mode, raw=self.protocol)
            self.protocol = {"name": "demo"}

        return True

    def to_dict(self) -> dict:
        d = {
            "system": self.system.to_dict(),
            "image": self.image.to_dict(),
            "protocol": _jsonable(self.protocol),
        }
        return _jsonable(drop_none_keys(d))

    @staticmethod
    def from_dict(
        settings: Any,
        protocol: Optional[dict] = None,
        *,
        mode: Union[ParseMode, str, None] = ParseMode.LENIENT,
    ) -> "MicroscopeSettings":
        mode = as_parse_mode(mode)
        strict = is_strict(mode)

        if isinstance(settings, MicroscopeSettings):
            settings.validate(mode=mode)
            return settings
        if not isinstance(settings, dict):
            if strict:
                raise TypeError(f"MicroscopeSettings.from_dict expects a dict, got {type(settings)}")
            return MicroscopeSettings(_mode=mode)

        settings_proto = settings.get("protocol", None)
        if protocol is None:
            protocol = settings_proto if isinstance(settings_proto, dict) else {"name": "demo"}
        elif not isinstance(protocol, dict):
            protocol = settings_proto if isinstance(settings_proto, dict) else {"name": "demo"}

        return MicroscopeSettings(
            system=settings.get("system"),
            image=settings.get("image"),
            protocol=protocol,
            _mode=mode,
        )
