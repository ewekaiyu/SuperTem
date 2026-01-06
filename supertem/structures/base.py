import datetime
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Set
from pint import UnitRegistry
import numpy as np
from PIL import Image

import tifffile as tff

from supertem.config import METADATA_VERSION

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("supertem")
except PackageNotFoundError:
    try:
        __version__ = version("SuperTem")
    except PackageNotFoundError:
        __version__ = "unknown"


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

def collect_extra(d: Optional[Dict[str, Any]], known: Iterable[str]) -> Dict[str, Any]:
    """
    Start from d.get("extra", {}), then sweep *all* unknown top-level keys into extra.
    This makes from_dict forward-compatible and prevents silent drops.
    """
    if not isinstance(d, dict):
        return {}

    known_set: Set[str] = set(known)

    base = d.get("extra", {})
    extra: Dict[str, Any] = deepcopy(base) if isinstance(base, dict) else {}

    for k, v in d.items():
        if k == "extra":
            continue
        if k not in known_set:
            extra[k] = v
    extra = {k: v for k, v in extra.items() if v is not None}

    return extra

def add_extra_if_any(out: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(extra, dict):
        cleaned = {k: v for k, v in extra.items() if v is not None}
        if cleaned:
            out["extra"] = deepcopy(cleaned)
    return out

def drop_none_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of out with any keys whose value is None removed."""
    return {k: v for k, v in out.items() if v is not None}

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

def parse_optional_int_like(value: Any, *, name: str, strict: bool = False, extra: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if value is None:
        return None

    # bool is a subclass of int -> reject it
    if isinstance(value, bool):
        if extra is not None:
            extra[f"{name}_raw"] = value
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
            extra[f"{name}_raw"] = value
        if strict:
            raise
        return None

def parse_optional_float_like(value: Any, *, name: str, strict: bool = False, extra: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Parse an optional float from int/float/np scalar or numeric string.

    If strict=False, returns None on invalid input and (optionally) stores the raw value in extra.
    If strict=True, raises on invalid input.
    """
    if value is None:
        return None

    # bool is a subclass of int -> reject it
    if isinstance(value, bool):
        if extra is not None:
            extra[f"{name}_raw"] = value
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
            extra[f"{name}_raw"] = value
        if strict:
            raise
        return None

def parse_optional_bool_like(value: Any, *, name: str, strict: bool = False, extra: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return parse_bool(value, default=False, strict=True)
    except Exception:
        if extra is not None:
            extra[f"{name}_raw"] = value
        if strict:
            raise
        return None

def parse_optional_str_like(value: Any, *, name: str, strict: bool = False, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip()
        return s or None

    # If strict, non-str is an error. Capture raw only on error.
    if strict:
        if extra is not None:
            extra[f"{name}_raw"] = value
        raise TypeError(f"{name} must be str-like, got {type(value)}")

    # Be consistent with your int/float parsers: bool is suspicious.
    if isinstance(value, bool):
        if extra is not None:
            extra[f"{name}_raw"] = value
        return None

    # Non-str input: coerce without polluting extra unless it fails.
    try:
        s = str(value).strip()
    except Exception:
        if extra is not None:
            extra[f"{name}_raw"] = value
        return None

    return s or None

def parse_optional_pair_int_like(value: Any, *, name: str, sort: bool = False, strict: bool = False, extra: Optional[Dict[str, Any]] = None) -> Optional[Tuple[int, int]]:
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
            extra[f"{name}_raw"] = value
        if strict:
            raise
        return None

def parse_optional_pair_float_like(value: Any, *, name: str, sort: bool = False, strict: bool = False, extra: Optional[Dict[str, Any]] = None,) -> Optional[Tuple[float, float]]:
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
            extra[f"{name}_raw"] = value
        if strict:
            raise
        return None

def normalize_extra(extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise TypeError(f"extra must be dict or None, got {type(extra)}")
    return {k: v for k, v in extra.items() if v is not None}

def normalize_extra_lenient(extra: Any, owner: str) -> Dict[str, Any]:
    try:
        return normalize_extra(extra)  # strict function
    except Exception:
        out: Dict[str, Any] = {}
        if extra is not None:
            out[f"{owner}.extra_raw"] = repr(extra)
        return out

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

def _maybe_point(v: Any, *, extra: Optional[Dict[str, Any]] = None, name: str = "point") -> Optional["Point"]:
    if v is None:
        return None
    if isinstance(v, Point):
        return v
    if isinstance(v, (dict, list, tuple)):
        return Point.from_dict(v)
    if extra is not None:
        extra[f"{name}_raw"] = v
    return None


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
        return drop_none_keys({"x": self.x, "y": self.y, "z": self.z, "name": self.name})

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
    width: int = 0
    height: int = 0

    def __post_init__(self):
        ix = parse_optional_int_like(self.x, name="ROI.x", strict=True)
        iy = parse_optional_int_like(self.y, name="ROI.y", strict=True)
        iw = parse_optional_int_like(self.width, name="ROI.width", strict=True)
        ih = parse_optional_int_like(self.height, name="ROI.height", strict=True)
        self.x = 0 if ix is None else int(ix)
        self.y = 0 if iy is None else int(iy)
        self.width = 0 if iw is None else int(iw)
        self.height = 0 if ih is None else int(ih)

        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI x/y must be >= 0, got x={self.x}, y={self.y}")
        if self.width < 0 or self.height < 0:
            raise ValueError(f"ROI width/height must be >= 0, got w={self.width}, h={self.height}")

    def to_dict(self) -> dict:
        return drop_none_keys({"x": self.x, "y": self.y, "width": self.width, "height": self.height})

    @staticmethod
    def from_dict(d: Any) -> "ROI":
        if isinstance(d, ROI):
            return d
        try:
            if isinstance(d, (list, tuple)) and len(d) == 4:
                return ROI(x=d[0], y=d[1], width=d[2], height=d[3])
            if not isinstance(d, dict):
                return ROI()

            def _i(v: Any, default: int = 0) -> int:
                out = parse_optional_int_like(v, name="ROI", strict=False)
                return default if out is None else int(out)

            return ROI(
                x=_i(d.get("x", 0)),
                y=_i(d.get("y", 0)),
                width=_i(d.get("width", d.get("w", 0))),
                height=_i(d.get("height", d.get("h", 0))),
            )
        except Exception:
            return ROI()

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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # ---- extra: be lenient here (metadata should not brick loading) ----
        self.extra = normalize_extra_lenient(self.extra, "TemImageMetadata")

        # ---- schema/provenance ----
        self.version = (
                parse_optional_str_like(
                    self.version,
                    name="TemImageMetadata.version",
                    strict=False,
                    extra=self.extra,
                )
                or str(METADATA_VERSION)
        )

        # created_at: accept ISO string, unix seconds (int/float), or numeric string
        raw_created = self.created_at
        created_iso: Optional[str] = None

        if isinstance(raw_created, (int, float)) and not isinstance(raw_created, bool):
            try:
                created_iso = datetime.datetime.fromtimestamp(
                    float(raw_created), tz=datetime.timezone.utc
                ).isoformat()
            except Exception:
                self.extra["TemImageMetadata.created_at_raw"] = raw_created
        else:
            s = parse_optional_str_like(
                raw_created, name="TemImageMetadata.created_at", strict=False, extra=self.extra
            )
            if s is not None:
                ss = s.strip()
                # numeric string (supports e.g. "1700000000", "1700000000.5", "1e9")
                try:
                    ts = float(ss)
                    if np.isfinite(ts):
                        created_iso = datetime.datetime.fromtimestamp(
                            ts, tz=datetime.timezone.utc
                        ).isoformat()
                    else:
                        self.extra["TemImageMetadata.created_at_raw"] = ss
                        created_iso = None
                except Exception:
                    created_iso = ss

        if not created_iso:
            created_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.created_at = created_iso

        # ---- identity strings ----
        self.user = parse_optional_str_like(self.user, name="TemImageMetadata.user", strict=False, extra=self.extra)
        self.manufacturer = parse_optional_str_like(self.manufacturer, name="TemImageMetadata.manufacturer",
                                                    strict=False, extra=self.extra)
        self.device = parse_optional_str_like(self.device, name="TemImageMetadata.device", strict=False,
                                              extra=self.extra)
        self.model = parse_optional_str_like(self.model, name="TemImageMetadata.model", strict=False, extra=self.extra)
        self.serial_number = parse_optional_str_like(self.serial_number, name="TemImageMetadata.serial_number",
                                                     strict=False, extra=self.extra)
        self.software_version = parse_optional_str_like(self.software_version, name="TemImageMetadata.software_version",
                                                        strict=False, extra=self.extra)

        # ---- imaging summary ----
        self.mode = parse_optional_str_like(self.mode, name="TemImageMetadata.mode", strict=False, extra=self.extra)
        self.detector_id = parse_optional_str_like(self.detector_id, name="TemImageMetadata.detector_id", strict=False,
                                                   extra=self.extra)
        self.detector_name = parse_optional_str_like(self.detector_name, name="TemImageMetadata.detector_name",
                                                     strict=False, extra=self.extra)

        self.magnification = parse_optional_float_like(self.magnification, name="TemImageMetadata.magnification",
                                                       strict=False, extra=self.extra)
        self.camera_length_mm = parse_optional_float_like(self.camera_length_mm,
                                                          name="TemImageMetadata.camera_length_mm", strict=False,
                                                          extra=self.extra)

        # ---- geometry ----
        self.pixel_size_nm = parse_optional_pair_float_like(
            self.pixel_size_nm,
            name="TemImageMetadata.pixel_size_nm",
            sort=False,
            strict=False,
            extra=self.extra,
        )
        self.image_size_px = parse_optional_pair_int_like(
            self.image_size_px,
            name="TemImageMetadata.image_size_px",
            sort=False,
            strict=False,
            extra=self.extra,
        )

        # ---- acquisition params ----
        self.accelerating_voltage_kv = parse_optional_float_like(
            self.accelerating_voltage_kv, name="TemImageMetadata.accelerating_voltage_kv", strict=False,
            extra=self.extra
        )
        self.beam_current_na = parse_optional_float_like(
            self.beam_current_na, name="TemImageMetadata.beam_current_na", strict=False, extra=self.extra
        )
        self.exposure_ms = parse_optional_float_like(
            self.exposure_ms, name="TemImageMetadata.exposure_ms", strict=False, extra=self.extra
        )
        self.dwell_time_us = parse_optional_float_like(
            self.dwell_time_us, name="TemImageMetadata.dwell_time_us", strict=False, extra=self.extra
        )
        self.working_distance_mm = parse_optional_float_like(
            self.working_distance_mm, name="TemImageMetadata.working_distance_mm", strict=False, extra=self.extra
        )

        # ---- nested objects (best-effort; don't raise in metadata) ----
        ms = self.microscope_state
        ms_parsed = MicroscopeState.from_dict_lenient(ms)
        if ms_parsed is None and ms is not None:
            self.extra["TemImageMetadata.microscope_state_raw"] = repr(ms)
        self.microscope_state = ms_parsed

        acq = self.acquisition
        acq_parsed = MicroscopeState.from_dict_lenient(ms)
        if acq_parsed is None and ms is not None:
            self.extra["TemImageMetadata.acquisition_raw"] = repr(ms)
        self.acquisition = acq_parsed

        self.extra = normalize_extra(self.extra)

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(d: Any) -> "TemImageMetadata":
        if isinstance(d, TemImageMetadata):
            return d
        if not isinstance(d, dict):
            return TemImageMetadata()

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
        extra = collect_extra(d, known)

        def _default_created_at() -> str:
            return datetime.datetime.now(datetime.timezone.utc).isoformat()

        def _opt_str(val: Any, name: str) -> Optional[str]:
            return parse_optional_str_like(val, name=name, strict=False, extra=extra)

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
                    extra["TemImageMetadata.created_at_raw"] = created_raw
                    created_at = _default_created_at()
            except Exception:
                extra["TemImageMetadata.created_at_raw"] = created_raw
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

        magnification = parse_optional_float_like(d.get("magnification", None), name="magnification", strict=False, extra=extra)
        camera_length_mm = parse_optional_float_like(
            d.get("camera_length_mm", d.get("camera_length", None)),
            name="camera_length_mm",
            strict=False,
            extra=extra,
        )

        accelerating_voltage_kv = parse_optional_float_like(
            d.get("accelerating_voltage_kv", d.get("accelerating_voltage", None)),
            name="accelerating_voltage_kv",
            strict=False,
            extra=extra,
        )
        beam_current_na = parse_optional_float_like(
            d.get("beam_current_na", d.get("beam_current", None)),
            name="beam_current_na",
            strict=False,
            extra=extra,
        )
        exposure_ms = parse_optional_float_like(d.get("exposure_ms", None), name="exposure_ms", strict=False, extra=extra)
        dwell_time_us = parse_optional_float_like(
            d.get("dwell_time_us", d.get("dwell_time", None)),
            name="dwell_time_us",
            strict=False,
            extra=extra,
        )
        working_distance_mm = parse_optional_float_like(
            d.get("working_distance_mm", d.get("working_distance", None)),
            name="working_distance_mm",
            strict=False,
            extra=extra,
        )

        pixel_size_nm = parse_optional_pair_float_like(
            d.get("pixel_size_nm", None), name="pixel_size_nm", sort=False, strict=False, extra=extra
        )
        image_size_px = parse_optional_pair_int_like(
            d.get("image_size_px", None), name="image_size_px", sort=False, strict=False, extra=extra
        )

        # nested objects
        ms_raw = d.get("microscope_state", None)
        microscope_state = MicroscopeState.from_dict_lenient(ms_raw)
        if microscope_state  is None and ms_raw is not None:
            extra["TemImageMetadata.microscope_state_raw"] = repr(ms_raw)

        acq_raw = d.get("acquisition", None)
        acquisition = AcquisitionRequest.from_dict_lenient(acq_raw)
        if acquisition is None and acq_raw is not None:
            extra["TemImageMetadata.acquisition_raw"] = repr(acq_raw)

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

    def __post_init__(self):
        self.name = parse_optional_str_like(self.name, name="TemStagePosition.name", strict=False)
        self.coordinate_system = parse_optional_str_like(
            self.coordinate_system, name="TemStagePosition.coordinate_system", strict=False
        )
        # Normalize individual axes into canonical units.
        self.x = ensure_quantity(self.x, "nanometer")
        self.y = ensure_quantity(self.y, "nanometer")
        self.z = ensure_quantity(self.z, "nanometer")
        self.r = ensure_quantity(self.r, "degree")
        self.tilt_x = ensure_quantity(self.tilt_x, "degree")
        self.tilt_y = ensure_quantity(self.tilt_y, "degree")

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(d: Any) -> "TemStagePosition":
        if isinstance(d, TemStagePosition):
            return d
        if not isinstance(d, dict):
            return TemStagePosition()
        return TemStagePosition(
            name=parse_optional_str_like(d.get("name", None), name="TemStagePosition.name", strict=False),
            x=ensure_quantity(d.get("x", None), "nanometer"),
            y=ensure_quantity(d.get("y", None), "nanometer"),
            z=ensure_quantity(d.get("z", None), "nanometer"),
            r=ensure_quantity(d.get("r", None), "degree"),
            tilt_x=ensure_quantity(d.get("tilt_x", None), "degree"),
            tilt_y=ensure_quantity(d.get("tilt_y", None), "degree"),
            coordinate_system=parse_optional_str_like(d.get("coordinate_system", None), name="TemStagePosition.coordinate_system", strict=False),
        )

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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.enabled = parse_bool(self.enabled, default=True)
        self.can_x = parse_bool(self.can_x, default=True)
        self.can_y = parse_bool(self.can_y, default=True)
        self.can_z = parse_bool(self.can_z, default=True)
        self.can_r = parse_bool(self.can_r, default=False)
        self.can_tilt_x = parse_bool(self.can_tilt_x, default=False)
        self.can_tilt_y = parse_bool(self.can_tilt_y, default=False)

        # Use your robust pair parser (rejects bool, enforces length==2)
        self.x_limits_nm = parse_optional_pair_float_like(self.x_limits_nm, name="x_limits_nm", sort=True, strict=True)
        self.y_limits_nm = parse_optional_pair_float_like(self.y_limits_nm, name="y_limits_nm", sort=True, strict=True)
        self.z_limits_nm = parse_optional_pair_float_like(self.z_limits_nm, name="z_limits_nm", sort=True, strict=True)
        self.r_limits_deg = parse_optional_pair_float_like(self.r_limits_deg, name="r_limits_deg", sort=True,
                                                           strict=True)
        self.tilt_x_limits_deg = parse_optional_pair_float_like(self.tilt_x_limits_deg, name="tilt_x_limits_deg",
                                                                sort=True, strict=True)
        self.tilt_y_limits_deg = parse_optional_pair_float_like(self.tilt_y_limits_deg, name="tilt_y_limits_deg",
                                                                sort=True, strict=True)

        # Floats: reject bool, allow numeric strings, keep defaults if None
        v = parse_optional_float_like(self.max_step_nm, name="max_step_nm", strict=True)
        self.max_step_nm = 50000.0 if v is None else v

        v = parse_optional_float_like(self.max_step_deg, name="max_step_deg", strict=True)
        self.max_step_deg = 1.0 if v is None else v

        v = parse_optional_float_like(self.settle_time_s, name="settle_time_s", strict=True)
        self.settle_time_s = 0.2 if v is None else v

        v = parse_optional_float_like(self.timeout_s, name="timeout_s", strict=True)
        self.timeout_s = 10.0 if v is None else v

        if self.max_step_nm <= 0:
            raise ValueError(f"max_step_nm must be > 0, got {self.max_step_nm}")
        if self.max_step_deg <= 0:
            raise ValueError(f"max_step_deg must be > 0, got {self.max_step_deg}")
        if self.settle_time_s < 0:
            raise ValueError(f"settle_time_s must be >= 0, got {self.settle_time_s}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")

        self.eucentric_z_nm = parse_optional_float_like(self.eucentric_z_nm, name="eucentric_z_nm", strict=True)

        self.extra = normalize_extra(self.extra)

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "StageSystemSettings":
        if isinstance(settings, StageSystemSettings):
            return settings
        if not isinstance(settings, dict):
            return StageSystemSettings()

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
        extra = collect_extra(settings, known)

        def _f_pos(key: str, default: float) -> float:
            raw = settings.get(key, None)
            v = parse_optional_float_like(raw, name=key, strict=False, extra=extra)
            if v is None:
                return default
            fv = float(v)
            if fv <= 0:
                # avoid __post_init__ raising; treat as "not provided" and stash raw
                extra[f"{key}_raw"] = raw
                return default
            return fv

        def _f_nonneg(key: str, default: float) -> float:
            raw = settings.get(key, None)
            v = parse_optional_float_like(raw, name=key, strict=False, extra=extra)
            if v is None:
                return default
            fv = float(v)
            if fv < 0:
                extra[f"{key}_raw"] = raw
                return default
            return fv

        def _pair(key: str) -> Optional[Tuple[float, float]]:
            raw = settings.get(key, None)
            return parse_optional_pair_float_like(raw, name=key, sort=True, strict=False, extra=extra)

        x_limits_nm = _pair("x_limits_nm")
        y_limits_nm = _pair("y_limits_nm")
        z_limits_nm = _pair("z_limits_nm")
        r_limits_deg = _pair("r_limits_deg")
        tilt_x_limits_deg = _pair("tilt_x_limits_deg")
        tilt_y_limits_deg = _pair("tilt_y_limits_deg")

        max_step_nm = _f_pos("max_step_nm", 50000.0)
        max_step_deg = _f_pos("max_step_deg", 1.0)
        settle_time_s = _f_nonneg("settle_time_s", 0.2)
        timeout_s = _f_pos("timeout_s", 10.0)

        euc_raw = settings.get("eucentric_z_nm", settings.get("eucentric_height", None))
        eucentric_z_nm = parse_optional_float_like(euc_raw, name="eucentric_z_nm", strict=False, extra=extra)

        return StageSystemSettings(
            enabled=parse_bool(settings.get("enabled"), default=True),
            can_x=parse_bool(settings.get("can_x"), default=True),
            can_y=parse_bool(settings.get("can_y"), default=True),
            can_z=parse_bool(settings.get("can_z"), default=True),
            can_r=parse_bool(settings.get("can_r"), default=False),
            can_tilt_x=parse_bool(settings.get("can_tilt_x"), default=False),
            can_tilt_y=parse_bool(settings.get("can_tilt_y"), default=False),
            x_limits_nm=x_limits_nm,
            y_limits_nm=y_limits_nm,
            z_limits_nm=z_limits_nm,
            r_limits_deg=r_limits_deg,
            tilt_x_limits_deg=tilt_x_limits_deg,
            tilt_y_limits_deg=tilt_y_limits_deg,
            max_step_nm=max_step_nm,
            max_step_deg=max_step_deg,
            settle_time_s=settle_time_s,
            timeout_s=timeout_s,
            eucentric_z_nm=eucentric_z_nm,
            extra=extra,
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

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Numeric-ish inputs are accepted (e.g. "200", "200.0", numpy scalars).
        # Truly invalid values raise, because you don't want silent nonsense in automation.
        self.voltage = parse_optional_float_like(self.voltage, name="voltage", strict=True)
        self.beam_current = parse_optional_float_like(self.beam_current, name="beam_current", strict=True)
        self.convergence_angle_mrad = parse_optional_float_like(
            self.convergence_angle_mrad, name="convergence_angle_mrad", strict=True
        )
        self.scan_rotation_deg = parse_optional_float_like(self.scan_rotation_deg, name="scan_rotation_deg", strict=True)

        self.spot_size = parse_optional_int_like(self.spot_size, name="spot_size", strict=True)
        if self.spot_size is not None and self.spot_size < 0:
            raise ValueError(f"spot_size must be >= 0, got {self.spot_size}")

        # Normalize Point-ish inputs (Point / dict / [x,y] / (x,y,z) / None)
        self.stigmation = _maybe_point(self.stigmation, extra=self.extra, name="BeamSettings.stigmation")
        self.beam_shift = _maybe_point(self.beam_shift, extra=self.extra, name="BeamSettings.beam_shift")
        self.image_shift = _maybe_point(self.image_shift, extra=self.extra, name="BeamSettings.image_shift")

        self.extra = normalize_extra(self.extra)

    def to_dict(self) -> dict:
        d = {
            "voltage": float(self.voltage) if self.voltage is not None else None,
            "beam_current": float(self.beam_current) if self.beam_current is not None else None,
            "spot_size": self.spot_size,
            "convergence_angle_mrad": float(self.convergence_angle_mrad) if self.convergence_angle_mrad is not None else None,
            "stigmation": self.stigmation.to_dict() if self.stigmation is not None else None,
            "beam_shift": self.beam_shift.to_dict() if self.beam_shift is not None else None,
            "image_shift": self.image_shift.to_dict() if self.image_shift is not None else None,
            "scan_rotation_deg": float(self.scan_rotation_deg) if self.scan_rotation_deg is not None else None,
        }
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "BeamSettings":
        if isinstance(settings, BeamSettings):
            return settings
        if not isinstance(settings, dict):
            return BeamSettings()

        known = {
            "voltage", "accelerating_voltage_kv",
            "beam_current", "current",
            "spot_size", "spot",
            "convergence_angle_mrad", "convergence_mrad",
            "stigmation", "beam_shift", "image_shift",
            "scan_rotation_deg",
            "extra",
        }
        extra = collect_extra(settings, known)

        # Points
        stigmation = _maybe_point(settings.get("stigmation"), extra=extra, name="BeamSettings.stigmation")
        beam_shift = _maybe_point(settings.get("beam_shift"), extra=extra, name="BeamSettings.beam_shift")
        image_shift = _maybe_point(settings.get("image_shift"), extra=extra, name="BeamSettings.image_shift")

        # Numbers (be permissive here; stash raw garbage into extra instead of crashing loads)
        voltage_raw = settings.get("voltage", settings.get("accelerating_voltage_kv", None))
        current_raw = settings.get("beam_current", settings.get("current", None))
        conv_raw = settings.get("convergence_angle_mrad", settings.get("convergence_mrad", None))
        scan_rot_raw = settings.get("scan_rotation_deg", None)

        voltage = parse_optional_float_like(voltage_raw, name="voltage", strict=False, extra=extra)
        beam_current = parse_optional_float_like(current_raw, name="beam_current", strict=False, extra=extra)
        convergence = parse_optional_float_like(conv_raw, name="convergence_angle_mrad", strict=False, extra=extra)
        scan_rot = parse_optional_float_like(scan_rot_raw, name="scan_rotation_deg", strict=False, extra=extra)

        spot_raw = settings.get("spot_size", settings.get("spot", None))
        spot = parse_optional_int_like(spot_raw, name="spot_size", strict=False, extra=extra)
        if spot is not None and spot < 0:
            extra["spot_size_raw"] = spot_raw
            spot = None

        return BeamSettings(
            voltage=voltage,
            beam_current=beam_current,
            spot_size=spot,
            convergence_angle_mrad=convergence,
            stigmation=stigmation,
            beam_shift=beam_shift,
            image_shift=image_shift,
            scan_rotation_deg=scan_rot,
            extra=extra,
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

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.enabled = parse_bool(self.enabled, default=True)
        if self.default_beam is None:
            self.default_beam = BeamSettings()
        elif isinstance(self.default_beam, dict):
            self.default_beam = BeamSettings.from_dict(self.default_beam)
        elif not isinstance(self.default_beam, BeamSettings):
            raise TypeError(f"default_beam must be BeamSettings/dict, got {type(self.default_beam)}")

        self.voltage_range_kv = parse_optional_pair_float_like(
            self.voltage_range_kv, name="voltage_range_kv", sort=True, strict=True
        )
        self.beam_current_range_na = parse_optional_pair_float_like(
            self.beam_current_range_na, name="beam_current_range_na", sort=True, strict=True
        )
        self.convergence_angle_range_mrad = parse_optional_pair_float_like(
            self.convergence_angle_range_mrad, name="convergence_angle_range_mrad", sort=True, strict=True
        )
        self.spot_size_range = parse_optional_pair_int_like(
            self.spot_size_range, name="spot_size_range", sort=True, strict=True
        )

        self.extra = normalize_extra(self.extra)

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "BeamSystemSettings":
        if isinstance(settings, BeamSystemSettings):
            return settings
        if not isinstance(settings, dict):
            return BeamSystemSettings()

        known = {
            "enabled",
            "default_beam",
            "voltage_range_kv", "voltage_limits_kv",
            "beam_current_range_na",
            "spot_size_range",
            "convergence_angle_range_mrad",
            "extra",
        }
        extra = collect_extra(settings, known)

        default_beam = BeamSettings.from_dict(settings.get("default_beam", None))

        voltage_raw = settings.get("voltage_range_kv", settings.get("voltage_limits_kv", None))
        voltage_range_kv = parse_optional_pair_float_like(
            voltage_raw, name="voltage_range_kv", sort=True, strict=False, extra=extra
        )
        beam_current_range_na = parse_optional_pair_float_like(
            settings.get("beam_current_range_na", None), name="beam_current_range_na", sort=True, strict=False, extra=extra
        )
        spot_size_range = parse_optional_pair_int_like(
            settings.get("spot_size_range", None), name="spot_size_range", sort=True, strict=False, extra=extra
        )
        convergence_angle_range_mrad = parse_optional_pair_float_like(
            settings.get("convergence_angle_range_mrad", None), name="convergence_angle_range_mrad", sort=True, strict=False, extra=extra
        )

        return BeamSystemSettings(
            enabled=parse_bool(settings.get("enabled"), default=True),
            default_beam=default_beam,
            voltage_range_kv=voltage_range_kv,
            beam_current_range_na=beam_current_range_na,
            spot_size_range=spot_size_range,
            convergence_angle_range_mrad=convergence_angle_range_mrad,
            extra=extra,
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

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        self.detector_id = parse_optional_str_like(self.detector_id, name="detector_id", strict=False, extra=self.extra)

        self.exposure_ms = parse_optional_float_like(self.exposure_ms, name="exposure_ms", strict=True)
        if self.exposure_ms is not None and self.exposure_ms < 0:
            raise ValueError(f"exposure_ms must be >= 0, got {self.exposure_ms}")

        self.binning_index = parse_optional_int_like(self.binning_index, name="binning_index", strict=True)
        if self.binning_index is not None and self.binning_index < 0:
            raise ValueError(f"binning_index must be >= 0, got {self.binning_index}")

        self.binning_xy = parse_optional_pair_int_like(self.binning_xy, name="binning_xy", sort=False, strict=True)
        if self.binning_xy is not None and (self.binning_xy[0] < 0 or self.binning_xy[1] < 0):
            raise ValueError(f"binning_xy values must be >= 0, got {self.binning_xy}")

        if self.roi is not None:
            if isinstance(self.roi, (ROI, dict, list, tuple)):
                self.roi = ROI.from_dict(self.roi)
            else:
                raise TypeError(f"roi must be ROI/dict/list/tuple, got {type(self.roi)}")
        if self.roi is not None and (self.roi.width == 0 or self.roi.height == 0):
            self.extra["roi_invalid_or_empty"] = {"parsed": self.roi.to_dict()}
            self.roi = None

        self.frame_integration = parse_optional_int_like(self.frame_integration, name="frame_integration", strict=True)
        if self.frame_integration is not None and self.frame_integration < 0:
            raise ValueError(f"frame_integration must be >= 0, got {self.frame_integration}")

        self.gain_index = parse_optional_int_like(self.gain_index, name="gain_index", strict=True)
        if self.gain_index is not None and self.gain_index < 0:
            raise ValueError(f"gain_index must be >= 0, got {self.gain_index}")

        self.offset_index = parse_optional_int_like(self.offset_index, name="offset_index", strict=True)
        if self.offset_index is not None and self.offset_index < 0:
            raise ValueError(f"offset_index must be >= 0, got {self.offset_index}")

        self.digital_rotation_deg = parse_optional_float_like(
            self.digital_rotation_deg, name="digital_rotation_deg", strict=True
        )

        self.extra = normalize_extra(self.extra)

    @classmethod
    def from_kwargs(cls, **kwargs):
        field_names = {f.name for f in fields(cls)}

        init_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}

        # If caller already provided extra=..., merge it
        user_extra = init_kwargs.pop("extra", None)
        obj = cls(**init_kwargs)

        if isinstance(user_extra, dict):
            obj.extra.update(user_extra)
        obj.extra.update(extra_kwargs)
        obj.extra = normalize_extra(obj.extra)

        return obj

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "detector_id": self.detector_id,
            "exposure_ms": self.exposure_ms,
            "binning_index": self.binning_index,
            "binning_xy": self.binning_xy,
            "frame_integration": self.frame_integration,
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "gain_index": self.gain_index,
            "offset_index": self.offset_index,
            "digital_rotation_deg": self.digital_rotation_deg,
        }
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "DetectorSettings":
        if isinstance(settings, DetectorSettings):
            return settings
        if not isinstance(settings, dict):
            return DetectorSettings()

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
        extra = collect_extra(settings, known)

        detector_id = parse_optional_str_like(settings.get("detector_id", None), name="detector_id", strict=False, extra=extra)

        exposure_raw = settings.get("exposure_ms", None)
        exposure_ms = parse_optional_float_like(exposure_raw, name="exposure_ms", strict=False, extra=extra)
        if exposure_ms is not None and exposure_ms < 0:
            extra["exposure_ms_raw"] = exposure_raw
            exposure_ms = None

        binning_index_raw = settings.get("binning_index", None)
        binning_index = parse_optional_int_like(binning_index_raw, name="binning_index", strict=False, extra=extra)
        if binning_index is not None and binning_index < 0:
            extra["binning_index_raw"] = binning_index_raw
            binning_index = None

        binning_xy_raw = settings.get("binning_xy", None)
        binning_xy = parse_optional_pair_int_like(binning_xy_raw, name="binning_xy", sort=False, strict=False, extra=extra)
        if binning_xy is not None and (binning_xy[0] < 0 or binning_xy[1] < 0):
            extra["binning_xy_raw"] = binning_xy_raw
            binning_xy = None

        frame_integration_raw = settings.get("frame_integration", None)
        frame_integration = parse_optional_int_like(frame_integration_raw, name="frame_integration", strict=False, extra=extra)
        if frame_integration is not None and frame_integration < 0:
            extra["frame_integration_raw"] = frame_integration_raw
            frame_integration = None

        gain_index_raw = settings.get("gain_index", None)
        gain_index = parse_optional_int_like(gain_index_raw, name="gain_index", strict=False, extra=extra)
        if gain_index is not None and gain_index < 0:
            extra["gain_index_raw"] = gain_index_raw
            gain_index = None

        offset_index_raw = settings.get("offset_index", None)
        offset_index = parse_optional_int_like(offset_index_raw, name="offset_index", strict=False, extra=extra)
        if offset_index is not None and offset_index < 0:
            extra["offset_index_raw"] = offset_index_raw
            offset_index = None

        digital_rotation_raw = settings.get("digital_rotation_deg", None)
        digital_rotation_deg = parse_optional_float_like(
            digital_rotation_raw, name="digital_rotation_deg", strict=False, extra=extra
        )

        roi_raw = settings.get("roi", None)
        if roi_raw is None:
            for alias in ("detector_roi", "imaging_area"):
                if alias in settings:
                    roi_raw = settings.get(alias)
                    break
        roi = ROI.from_dict(roi_raw) if roi_raw is not None else None
        # If ROI was provided but parses to an empty/invalid ROI, drop it and record why.
        if roi_raw is not None and roi is not None and (roi.width == 0 or roi.height == 0):
            extra["roi_invalid_or_empty"] = {"roi_raw": roi_raw, "parsed": roi.to_dict()}
            roi = None

        return DetectorSettings(
            detector_id=detector_id,
            exposure_ms=exposure_ms,
            binning_index=binning_index,
            binning_xy=binning_xy,
            roi=roi,
            frame_integration=frame_integration,
            gain_index=gain_index,
            offset_index=offset_index,
            digital_rotation_deg=digital_rotation_deg,
            extra=extra,
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

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        # Tuple-ish
        self.binning_xy_min = parse_optional_pair_int_like(self.binning_xy_min, name="binning_xy_min", strict=False, extra=self.extra)
        self.binning_xy_max = parse_optional_pair_int_like(self.binning_xy_max, name="binning_xy_max", strict=False, extra=self.extra)
        self.roi_min = parse_optional_pair_int_like(self.roi_min, name="roi_min", strict=False, extra=self.extra)
        self.roi_max = parse_optional_pair_int_like(self.roi_max, name="roi_max", strict=False, extra=self.extra)

        # Int-like ranges
        for name in (
            "binning_index_min", "binning_index_max",
            "frame_integration_min", "frame_integration_max",
            "gain_index_min", "gain_index_max",
            "offset_index_min", "offset_index_max",
        ):
            v = getattr(self, name)
            setattr(self, name, parse_optional_int_like(v, name=name, strict=False, extra=self.extra))

        # Float-like ranges
        for name in ("exposure_ms_min", "exposure_ms_max", "digital_rotation_deg_min", "digital_rotation_deg_max"):
            v = getattr(self, name)
            setattr(self, name, parse_optional_float_like(v, name=name, strict=False, extra=self.extra))

        # Optional bools
        for name in ("can_binning", "can_gain", "can_offset", "can_digital_rotation"):
            v = getattr(self, name)
            setattr(self, name, parse_optional_bool_like(v, name=name, strict=False, extra=self.extra))

        def _repair_minmax(min_name: str, max_name: str) -> None:
            v_min = getattr(self, min_name)
            v_max = getattr(self, max_name)
            if v_min is None or v_max is None:
                return
            if v_min > v_max:
                self.extra[f"{min_name}_gt_{max_name}"] = {"min": v_min, "max": v_max}
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
                self.extra[f"{min_name}_gt_{max_name}"] = {"min": v_min, "max": v_max}
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

        self.extra = normalize_extra(self.extra)

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(d: Any) -> "DetectorCapabilities":
        if isinstance(d, DetectorCapabilities):
            return d
        if not isinstance(d, dict):
            return DetectorCapabilities()

        known = {f.name for f in fields(DetectorCapabilities)} | {"extra"}
        extra = collect_extra(d, known)

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
                kwargs[f.name] = parse_optional_pair_int_like(v, name=f.name, strict=False, extra=extra)
            elif f.name in int_fields:
                kwargs[f.name] = parse_optional_int_like(v, name=f.name, strict=False, extra=extra)
            elif f.name in float_fields:
                kwargs[f.name] = parse_optional_float_like(v, name=f.name, strict=False, extra=extra)
            elif f.name in bool_fields:
                kwargs[f.name] = parse_optional_bool_like(v, name=f.name, strict=False, extra=extra)
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
    # If empty, it can be inferred from `capabilities_by_id` / `defaults_by_id`.
    available_detectors: List[str] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        self.enabled = parse_bool(self.enabled, default=True)
        if self.defaults_by_id is None:
            self.defaults_by_id = {}
        if self.capabilities_by_id is None:
            self.capabilities_by_id = {}
        if self.available_detectors is None:
            self.available_detectors = []

        # Normalize dict keys to str, and values to proper objects
        new_defaults: Dict[str, DetectorSettings] = {}
        for k, v in self.defaults_by_id.items():
            det_id = parse_optional_str_like(k, name="defaults_by_id.key", strict=False, extra=self.extra)
            if det_id is None:
                self.extra[f"defaults_by_id.{k}_raw_key"] = k
                continue
            if isinstance(v, dict):
                v = DetectorSettings.from_dict(v)
            if not isinstance(v, DetectorSettings):
                raise TypeError(f"defaults_by_id['{det_id}'] must be DetectorSettings/dict, got {type(v)}")
            if v.detector_id is None:
                v.detector_id = det_id
            elif str(v.detector_id) != det_id:
                self.extra[f"defaults_by_id.{det_id}.detector_id_mismatch"] = {
                    "key": det_id,
                    "detector_id": v.detector_id,
                }
                v.detector_id = det_id
            new_defaults[det_id] = v
        self.defaults_by_id = new_defaults

        new_caps: Dict[str, DetectorCapabilities] = {}
        for k, v in self.capabilities_by_id.items():
            det_id = parse_optional_str_like(k, name="capabilities_by_id.key", strict=False, extra=self.extra)
            if det_id is None:
                self.extra[f"capabilities_by_id.{k}_raw_key"] = k
                continue
            if isinstance(v, dict):
                v = DetectorCapabilities.from_dict(v)
            if not isinstance(v, DetectorCapabilities):
                raise TypeError(f"capabilities_by_id['{det_id}'] must be DetectorCapabilities/dict, got {type(v)}")
            new_caps[det_id] = v
        self.capabilities_by_id = new_caps

        # Normalize available list
        cleaned: List[str] = []
        for x in self.available_detectors:
            sx = parse_optional_str_like(x, name="available_detectors", strict=False, extra=self.extra)
            if sx is not None:
                cleaned.append(sx)
        self.available_detectors = cleaned

        # Infer available detectors if empty
        if not self.available_detectors:
            key_union = set(self.defaults_by_id.keys()) | set(self.capabilities_by_id.keys())
            self.available_detectors = list(sorted(key_union))

        # Normalize default detector id
        ddi = parse_optional_str_like(self.default_detector_id, name="default_detector_id", strict=False,
                                       extra=self.extra)
        if ddi is None:
            self.default_detector_id = self.available_detectors[0] if self.available_detectors else None
        else:
            self.default_detector_id = ddi
            if self.default_detector_id not in self.available_detectors:
                # be consistent with from_dict(): keep it, and add it
                self.extra["default_detector_id_not_in_available"] = {
                       "default_detector_id": self.default_detector_id,
                        "available_detectors": list(self.available_detectors),
                }
                self.available_detectors.append(self.default_detector_id)
                self.available_detectors = list(dict.fromkeys(self.available_detectors))

        self.extra = normalize_extra(self.extra)

    def to_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "default_detector_id": self.default_detector_id,
            "defaults_by_id": {k: v.to_dict() for k, v in self.defaults_by_id.items()},
            "capabilities_by_id": {k: v.to_dict() for k, v in self.capabilities_by_id.items()},
            "available_detectors": deepcopy(self.available_detectors),
        }
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "DetectorSystemSettings":
        if isinstance(settings, DetectorSystemSettings):
            return settings
        if not isinstance(settings, dict):
            return DetectorSystemSettings()

        known = {
            "enabled",
            "available_detectors",
            "default_detector_id",
            "defaults_by_id", "default_settings_by_id", "default_detector_settings_by_id",
            "capabilities_by_id",
            "extra",
        }
        extra = collect_extra(settings, known)

        # defaults
        if "defaults_by_id" in settings:
            defaults_raw = settings.get("defaults_by_id")
        elif "default_settings_by_id" in settings:
            defaults_raw = settings.get("default_settings_by_id")
        elif "default_detector_settings_by_id" in settings:
            defaults_raw = settings.get("default_detector_settings_by_id")
        else:
            defaults_raw = {}
        if defaults_raw is None:
            defaults_raw = {}
        elif not isinstance(defaults_raw, dict):
            extra["defaults_by_id_raw"] = defaults_raw
            defaults_raw = {}

        defaults_by_id: Dict[str, DetectorSettings] = {}
        for det_id, det_cfg in defaults_raw.items():
            key = parse_optional_str_like(det_id, name="defaults_by_id.key", strict=False, extra=extra)
            if key is None:
                extra[f"defaults_by_id.{det_id}_raw_key"] = det_id
                continue
            if isinstance(det_cfg, DetectorSettings):
                ds = det_cfg
            elif isinstance(det_cfg, dict):
                ds = DetectorSettings.from_dict(det_cfg)
            else:
                extra[f"defaults_by_id.{key}_raw"] = det_cfg
                continue
            if ds.detector_id is None:
                ds.detector_id = key
            defaults_by_id[key] = ds

        # capabilities
        cap_raw = settings.get("capabilities_by_id", None)
        if cap_raw is None:
            cap_raw = {}
        elif not isinstance(cap_raw, dict):
            extra["capabilities_by_id_raw"] = cap_raw
            cap_raw = {}

        capabilities_by_id: Dict[str, DetectorCapabilities] = {}
        for det_id, cap_cfg in cap_raw.items():
            key = parse_optional_str_like(det_id, name="capabilities_by_id.key", strict=False, extra=extra)
            if key is None:
                extra[f"capabilities_by_id.{det_id}_raw_key"] = det_id
                continue

            if isinstance(cap_cfg, DetectorCapabilities):
                cap = cap_cfg
            elif isinstance(cap_cfg, dict):
                cap = DetectorCapabilities.from_dict(cap_cfg)
            else:
                extra[f"capabilities_by_id.{key}_raw"] = cap_cfg
                continue

            capabilities_by_id[key] = cap

        # available detectors
        available_raw = settings.get("available_detectors", None)
        if available_raw is None:
            available: Optional[List[str]] = None
        elif isinstance(available_raw, (list, tuple)):
            cleaned: List[str] = []
            for i, x in enumerate(available_raw):
                sx = parse_optional_str_like(x, name=f"available_detectors[{i}]", strict=False, extra=extra)
                if sx is not None:
                    cleaned.append(sx)
            # treat empty list as "not provided"
            available = cleaned or None
        else:
            extra["available_detectors_raw"] = available_raw
            available = None

        if available is None:
            key_union = set(defaults_by_id.keys()) | set(capabilities_by_id.keys())
            available = sorted(key_union)
        else:
            # dedupe while preserving order
            available = list(dict.fromkeys(available))

        # default detector id
        ddi_raw = settings.get("default_detector_id", None)
        ddi = parse_optional_str_like(ddi_raw, name="default_detector_id", strict=False, extra=extra)

        if ddi is None:
            default_detector_id = available[0] if available else None
        else:
            default_detector_id = ddi
            if default_detector_id not in available:
                # keep it consistent (and avoid __post_init__ raising)
                extra["default_detector_id_not_in_available"] = {
                    "default_detector_id": default_detector_id,
                    "available_detectors": list(available),
                }
                available.append(default_detector_id)
                available = list(dict.fromkeys(available))

        return DetectorSystemSettings(
            enabled=parse_bool(settings.get("enabled"), default=True),
            available_detectors=available,
            default_detector_id=default_detector_id,
            defaults_by_id=defaults_by_id,
            capabilities_by_id=capabilities_by_id,
            extra=extra,
        )

@dataclass
class ImageOutputSettings:
    file_format: Optional[str] = "tiff"  #  supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}
    path: Optional[Union[str, Path]] = None  # default output directory (session dir)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}

        ff = parse_optional_str_like(self.file_format, name="file_format", strict=False, extra=self.extra)
        self.file_format = (ff or "tiff").lower()

        if self.file_format is not None and self.file_format not in supported:
            raise ValueError(f"Unsupported file_format: {self.file_format!r}. Supported: {sorted(supported)}")

        if isinstance(self.path, str) and self.path.strip() == "":
            self.path = None

        if self.path is not None and not isinstance(self.path, Path):
            if isinstance(self.path, (str, bytes)):
                self.path = Path(self.path)
            else:
                self.extra.setdefault("path_raw", self.path)
                self.path = None

        self.extra = normalize_extra(self.extra)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "file_format": self.file_format,
            "path": str(self.path) if self.path is not None else None,
        }
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "ImageOutputSettings":
        if isinstance(settings, ImageOutputSettings):
            return settings
        if not isinstance(settings, dict):
            return ImageOutputSettings()

        known = ("file_format", "path", "extra")
        extra = collect_extra(settings, known=known)

        ff = parse_optional_str_like(settings.get("file_format", "tiff"), name="file_format", strict=False,
                                     extra=extra) or "tiff"
        p = parse_optional_str_like(settings.get("path", None), name="path", strict=False, extra=extra)

        return ImageOutputSettings(file_format=ff, path=p, extra=extra)

@dataclass
class AcquisitionRequest:
    detector_id: str
    detector: DetectorSettings
    image: ImageOutputSettings
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # normalize first
        if isinstance(self.detector, dict):
            self.detector = DetectorSettings.from_dict(self.detector)
        if isinstance(self.image, dict):
            self.image = ImageOutputSettings.from_dict(self.image)

        self.extra = normalize_extra(self.extra)

        # normalize detector_id
        if isinstance(self.detector_id, str):
            self.detector_id = self.detector_id.strip()

        # then enforce rules
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.detector_id, str) or not self.detector_id.strip():
            raise ValueError(f"detector_id must be a non-empty str, got {self.detector_id!r}")

        if not isinstance(self.detector, DetectorSettings):
            raise TypeError(f"detector must be DetectorSettings/dict, got {type(self.detector)}")

        if not isinstance(self.image, ImageOutputSettings):
            raise TypeError(f"image must be ImageOutputSettings/dict, got {type(self.image)}")

        # keep them consistent
        if self.detector.detector_id is None:
            self.detector.detector_id = self.detector_id
        elif str(self.detector.detector_id) != self.detector_id:
            raise ValueError(
                f"detector.detector_id ({self.detector.detector_id!r}) != detector_id ({self.detector_id!r})"
            )

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "detector_id": self.detector_id,
            "detector": self.detector.to_dict() if self.detector is not None else None,
            "image": self.image.to_dict() if self.image is not None else None,
        }
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(d: Any) -> "AcquisitionRequest":
        if isinstance(d, AcquisitionRequest):
            return d
        if not isinstance(d, dict):
            raise ValueError("AcquisitionRequest.from_dict expects a non-empty dict")

        used_keys = {"detector_id", "detector", "image", "extra"}
        extra = collect_extra(d, used_keys)

        det_raw = d.get("detector")
        if isinstance(det_raw, DetectorSettings):
            det = det_raw
        elif isinstance(det_raw, dict):
            det = DetectorSettings.from_dict(det_raw)
        else:
            det = DetectorSettings()

        img_raw = d.get("image")
        if isinstance(img_raw, ImageOutputSettings):
            img = img_raw
        elif isinstance(img_raw, dict):
            img = ImageOutputSettings.from_dict(img_raw)
        else:
            img = ImageOutputSettings()

        raw = d.get("detector_id", None)
        if raw is None:
            raw = det.detector_id
        det_id = parse_optional_str_like(raw, name="detector_id", strict=False, extra=extra)
        if det_id is None:
            raise ValueError("AcquisitionRequest missing detector_id (and detector.detector_id)")

        obj = AcquisitionRequest(detector_id=det_id, detector=det, image=img, extra=extra)
        return obj

    @staticmethod
    def from_dict_lenient(d: Any) -> Optional["AcquisitionRequest"]:
        if d is None:
            return None
        if isinstance(d, AcquisitionRequest):
            return d
        if not isinstance(d, dict) or not d:
            return None

        try:
            return AcquisitionRequest.from_dict(d)
        except Exception:
            return None

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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        # timestamp
        self.timestamp = float(self._parse_timestamp(self.timestamp))

        # stage_position
        sp = self.stage_position
        if sp is None:
            self.stage_position = TemStagePosition()
        elif isinstance(sp, TemStagePosition):
            self.stage_position = sp
        elif isinstance(sp, dict):
            self.stage_position = TemStagePosition.from_dict(sp)
        else:
            self.extra["stage_position_raw"] = repr(sp)
            self.stage_position = TemStagePosition()

        # beam
        b = self.beam
        if b is None:
            self.beam = BeamSettings()
        elif isinstance(b, BeamSettings):
            self.beam = b
        elif isinstance(b, dict):
            self.beam = BeamSettings.from_dict(b)
        else:
            self.extra["beam_raw"] = repr(b)
            self.beam = BeamSettings()

        # detectors
        dets_raw = self.detectors
        if dets_raw is None:
            dets_raw = {}
        if not isinstance(dets_raw, dict):
            self.extra["detectors_raw"] = dets_raw
            dets_raw = {}

        fixed: Dict[str, DetectorSettings] = {}
        for det_id, det in dets_raw.items():
            key = parse_optional_str_like(det_id, name="detectors.key", strict=False, extra=self.extra)
            if key is None:
                self.extra[f"detectors.{det_id}_raw_key"] = det_id
                continue
            try:
                if isinstance(det, DetectorSettings):
                    ds = det
                elif isinstance(det, dict):
                    ds = DetectorSettings.from_dict(det)
                else:
                    self.extra[f"detectors.{key}_raw"] = det
                    continue

                if ds.detector_id is None:
                    ds.detector_id = key
                elif str(ds.detector_id) != key:
                    self.extra[f"detectors.{key}.detector_id_mismatch"] = ds.detector_id
                    ds.detector_id = key

                fixed[key] = ds
            except Exception:
                self.extra[f"detectors.{key}_raw"] = det

        self.detectors = fixed

        # active detector ids
        active_raw = self.active_detector_ids
        if active_raw is None:
            active_raw = []
        if isinstance(active_raw, (list, tuple)):
            cleaned: List[str] = []
            for i, x in enumerate(active_raw):
                sx = parse_optional_str_like(x, name=f"active_detector_ids[{i}]", strict=False, extra=self.extra)
                if sx is not None:
                    cleaned.append(sx)
            self.active_detector_ids = cleaned
        else:
            self.extra["active_detector_ids_raw"] = active_raw
            self.active_detector_ids = []

        # protocol
        proto_raw = self.protocol
        if proto_raw is None:
            proto_raw = {}
        if isinstance(proto_raw, dict):
            self.protocol = proto_raw
        else:
            self.extra["protocol_raw"] = proto_raw
            self.protocol = {}

        # primary detector id
        self.primary_detector_id = parse_optional_str_like(self.primary_detector_id, name="primary_detector_id", strict=False, extra=self.extra)

        # choose a stable primary if needed
        if self.primary_detector_id is None and self.detectors:
            self.primary_detector_id = sorted(self.detectors.keys())[0]

        if self.detectors:
            known = set(self.detectors.keys())

            dropped = [x for x in self.active_detector_ids if x not in known]
            if dropped:
                self.extra["active_detector_ids_unknown"] = dropped
                self.active_detector_ids = [x for x in self.active_detector_ids if x in known]

            if self.primary_detector_id is not None and self.primary_detector_id not in known:
                self.extra["primary_detector_id_unknown"] = self.primary_detector_id
                self.primary_detector_id = sorted(known)[0]

        self.extra = normalize_extra(self.extra)

    def to_dict(self) -> dict:
        ts_iso = datetime.datetime.fromtimestamp(float(self.timestamp), tz=datetime.timezone.utc).isoformat()
        d = {
            "timestamp": float(self.timestamp),
            "timestamp_iso": ts_iso,
            "stage_position": self.stage_position.to_dict() if self.stage_position else None,
            "beam": self.beam.to_dict() if self.beam else None,
            "detectors": {k: v.to_dict() for k, v in self.detectors.items()},
            "active_detector_ids": list(self.active_detector_ids),
            "primary_detector_id": self.primary_detector_id,
        }

        if self.protocol:
            d["protocol"] = deepcopy(self.protocol)
        add_extra_if_any(d, self.extra)
        return drop_none_keys(d)

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        """Accept float seconds, int, or ISO string."""
        if value is None:
            return datetime.datetime.now(datetime.timezone.utc).timestamp()

        if isinstance(value, bool):
            return datetime.datetime.now(datetime.timezone.utc).timestamp()

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # ISO first
            try:
                dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
            # stringified float
            try:
                return float(value)
            except Exception:
                return datetime.datetime.now(datetime.timezone.utc).timestamp()

        return datetime.datetime.now(datetime.timezone.utc).timestamp()

    @staticmethod
    def from_dict(state_dict: Any) -> "MicroscopeState":
        if isinstance(state_dict, MicroscopeState):
            return state_dict
        if not isinstance(state_dict, dict):
            return MicroscopeState()

        known = {
            "timestamp", "timestamp_iso",
            "stage_position", "stage", "absolute_position",
            "beam",
            "detectors",
            "active_detector_ids",
            "primary_detector_id", "detector_id",
            "protocol",
            "extra",
        }
        extra = collect_extra(state_dict, known)

        # timestamp
        ts_raw = state_dict.get("timestamp", state_dict.get("timestamp_iso", None))
        if isinstance(ts_raw, bool):
            extra["timestamp_raw"] = ts_raw
            ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        else:
            ts = MicroscopeState._parse_timestamp(ts_raw)

        # stage
        if "stage_position" in state_dict:
            sp_raw = state_dict.get("stage_position")
        elif "stage" in state_dict:
            sp_raw = state_dict.get("stage")
        elif "absolute_position" in state_dict:
            sp_raw = state_dict.get("absolute_position")
        else:
            sp_raw = None
        if isinstance(sp_raw, TemStagePosition):
            stage_position = sp_raw
        elif isinstance(sp_raw, dict):
            stage_position = TemStagePosition.from_dict(sp_raw)
        else:
            stage_position = TemStagePosition()

        # beam
        b_raw = state_dict.get("beam", None)
        if isinstance(b_raw, BeamSettings):
            beam = b_raw
        elif isinstance(b_raw, dict):
            beam = BeamSettings.from_dict(b_raw)
        else:
            beam = BeamSettings()

        # detectors
        detectors: Dict[str, DetectorSettings] = {}
        dets_raw = state_dict.get("detectors", None)
        if isinstance(dets_raw, dict):
            for det_id, det_val in dets_raw.items():
                key = parse_optional_str_like(det_id, name="detectors.key", strict=False, extra=extra)
                if key is None:
                    extra[f"detectors.{det_id}_raw_key"] = det_id
                    continue

                if isinstance(det_val, DetectorSettings):
                    ds = det_val
                elif isinstance(det_val, dict):
                    ds = DetectorSettings.from_dict(det_val)
                else:
                    extra[f"detectors.{key}_raw"] = det_val
                    continue

                if ds.detector_id is None:
                    ds.detector_id = key
                elif str(ds.detector_id) != key:
                    extra[f"detectors.{key}.detector_id_mismatch"] = ds.detector_id
                    ds.detector_id = key

                detectors[key] = ds
        elif dets_raw is not None:
            extra["detectors_raw"] = dets_raw

        # active ids
        active_ids: List[str] = []
        active_raw = state_dict.get("active_detector_ids", None)
        if isinstance(active_raw, (list, tuple)):
            for i, x in enumerate(active_raw):
                sx = parse_optional_str_like(x, name=f"active_detector_ids[{i}]", strict=False, extra=extra)
                if sx is not None:
                    active_ids.append(sx)
        elif active_raw is not None:
            extra["active_detector_ids_raw"] = active_raw

        # primary id
        primary_raw = state_dict.get("primary_detector_id", state_dict.get("detector_id"))
        if primary_raw is None:
            primary_id = None
        elif isinstance(primary_raw, bool):
            extra["primary_detector_id_raw"] = primary_raw
            primary_id = None
        else:
            primary_id = parse_optional_str_like(primary_raw, name="primary_detector_id", strict=False, extra=extra)

        # protocol
        protocol_raw = state_dict.get("protocol", None)
        if protocol_raw is None:
            protocol: Dict[str, Any] = {}
        elif isinstance(protocol_raw, dict):
            protocol = protocol_raw
        else:
            extra["protocol_raw"] = protocol_raw
            protocol = {}

        return MicroscopeState(
            timestamp=ts,
            stage_position=stage_position,
            beam=beam,
            detectors=detectors,
            active_detector_ids=active_ids,
            primary_detector_id=primary_id,
            protocol=protocol,
            extra=extra,
        )

    @staticmethod
    def from_dict_lenient(d: Any) -> Optional["MicroscopeState"]:
        if d is None:
            return None
        if isinstance(d, MicroscopeState):
            return d
        if not isinstance(d, dict) or not d:
            return None

        try:
            return MicroscopeState.from_dict(d)
        except Exception:
            return None

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
        if not _check_data_format(data):
            raise ValueError("Invalid data format for TemImage.")
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
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
            tff.imwrite(path, self.data, description=desc)
            return path

        # Other formats: save pixels via Pillow (metadata via sidecar)
        data_to_save = self.data

        # JPEG/BMP can't store uint16 grayscale reliably; create a uint8 preview.
        if fmt in ("jpg", "jpeg", "bmp"):
            if data_to_save.dtype == np.uint16:
                data_to_save = (data_to_save >> 8).astype(np.uint8)
            elif data_to_save.dtype != np.uint8:
                data_to_save = np.clip(data_to_save, 0, 255).astype(np.uint8)
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

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extra = normalize_extra(self.extra)

        self.name = parse_optional_str_like(self.name, name="name", strict=False, extra=self.extra) or "Unknown"
        self.ip_address = parse_optional_str_like(self.ip_address, name="ip_address", strict=False,
                                                  extra=self.extra) or "Unknown"
        self.manufacturer = parse_optional_str_like(self.manufacturer, name="manufacturer", strict=False,
                                                    extra=self.extra) or "Unknown"
        self.model = parse_optional_str_like(self.model, name="model", strict=False, extra=self.extra) or "Unknown"
        self.serial_number = parse_optional_str_like(self.serial_number, name="serial_number", strict=False,
                                                     extra=self.extra) or "Unknown"
        self.hardware_version = parse_optional_str_like(self.hardware_version, name="hardware_version", strict=False,
                                                        extra=self.extra) or "Unknown"
        self.software_version = parse_optional_str_like(self.software_version, name="software_version", strict=False,
                                                        extra=self.extra) or "Unknown"
        self.supertem_version = parse_optional_str_like(self.supertem_version, name="supertem_version", strict=False,
                                                        extra=self.extra) or __version__

        self.application = parse_optional_str_like(self.application, name="application", strict=False, extra=self.extra)
        self.application_version = parse_optional_str_like(self.application_version, name="application_version",
                                                           strict=False, extra=self.extra)

        self.extra = normalize_extra(self.extra)

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
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "SystemInfo":
        if isinstance(settings, SystemInfo):
            return settings
        if not isinstance(settings, dict):
            return SystemInfo()

        known = (
            "name","ip_address","manufacturer","model","serial_number",
            "hardware_version","software_version","supertem_version",
            "application","application_version",
            "extra",
        )
        extra = collect_extra(settings, known=known)

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
        )

@dataclass
class SystemSettings:
    stage: StageSystemSettings = field(default_factory=StageSystemSettings)
    beam: BeamSystemSettings = field(default_factory=BeamSystemSettings)
    detector: DetectorSystemSettings = field(default_factory=DetectorSystemSettings)
    info: SystemInfo = field(default_factory=SystemInfo)

    def to_dict(self) -> dict:
        d = {
            "stage": self.stage.to_dict(),
            "beam": self.beam.to_dict(),
            "detector": self.detector.to_dict(),
            "info": self.info.to_dict(),
        }
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any) -> "SystemSettings":
        if isinstance(settings, SystemSettings):
            return settings
        if not isinstance(settings, dict):
            return SystemSettings()

        return SystemSettings(
            stage=StageSystemSettings.from_dict(settings.get("stage")),
            beam=BeamSystemSettings.from_dict(settings.get("beam")),
            detector=DetectorSystemSettings.from_dict(settings.get("detector")),
            info=SystemInfo.from_dict(settings.get("info")),
        )

@dataclass
class MicroscopeSettings:

    """
    A data class representing the settings for a microscope system.

    Attributes:
        system (SystemSettings): An instance of the `SystemSettings` class that holds the system settings.
        image (ImageOutputSettings): An instance of the `ImageOutputSettings` class that holds the image settings.
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to {"name": "demo"}.

    Methods:
        to_dict(): Returns a dictionary representation of the `MicroscopeSettings` object.
        from_dict(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """
    system: SystemSettings = field(default_factory=SystemSettings)
    image: ImageOutputSettings = field(default_factory=ImageOutputSettings)
    protocol: Dict[str, Any] = field(default_factory=lambda: {"name":"demo"})

    def to_dict(self) -> dict:
        d = {
            "system": self.system.to_dict(),
            "image": self.image.to_dict(),
        }
        if self.protocol:
            d["protocol"] = deepcopy(self.protocol)
        return drop_none_keys(d)

    @staticmethod
    def from_dict(settings: Any, protocol: Optional[Dict[str, Any]] = None) -> "MicroscopeSettings":
        if isinstance(settings, MicroscopeSettings):
            return settings
        if not isinstance(settings, dict):
            return MicroscopeSettings()

        settings_proto = settings.get("protocol", None)
        if protocol is None:
            protocol = settings_proto if isinstance(settings_proto, dict) else {"name": "demo"}
        elif not isinstance(protocol, dict):
            protocol = settings_proto if isinstance(settings_proto, dict) else {"name": "demo"}

        return MicroscopeSettings(
            system=SystemSettings.from_dict(settings.get("system")),
            image=ImageOutputSettings.from_dict(settings.get("image")),
            protocol=protocol,
        )

