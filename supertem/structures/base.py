import datetime
import importlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from pint import UnitRegistry

# ---- Units (Pint) ----
# Single project-wide UnitRegistry to avoid "mixed registry" issues.
ureg = UnitRegistry()
Q_ = ureg.Quantity

# Quantity type import is version-dependent across Pint releases.
try:  # Pint >= 0.20 often exposes Quantity at top-level
    from pint import Quantity  # type: ignore
except Exception:  # pragma: no cover
    from pint.facets.plain.quantity import Quantity  # type: ignore


def ensure_quantity(value: Any, unit: str) -> Optional["Quantity"]:
    """Coerce `value` into a Pint Quantity using the project registry, in `unit`.

    - If `value` is already a Quantity (even from another registry), it's re-created in `ureg`.
    - If `value` is a number, it's interpreted as being in `unit`.
    """
    if value is None:
        return None
    if isinstance(value, Quantity):
        q = Q_(value.magnitude, str(value.units))
    else:
        q = Q_(value, unit)
    return q.to(unit)


def magnitude(value: Any, unit: str) -> Optional[float]:
    """Return magnitude as a plain float in `unit`."""
    q = ensure_quantity(value, unit)
    if q is None:
        return None
    return float(q.magnitude)

from supertem.config import METADATA_VERSION

import numpy as np
import tifffile as tff

try:
    __version__ = importlib.metadata.version('SuperTem')
except ModuleNotFoundError:
    __version__ = "unknown"

try:
    from PyJEM import TEM3

    JEOL = True
except ImportError:
    JEOL = False

def _check_data_format(data: np.ndarray) -> bool:
    """Checks that data is in the correct format."""
    # assert data.ndim == 2  # or data.ndim == 3
    # assert data.dtype in [np.uint8, np.uint16]
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    return data.ndim == 2 and data.dtype in [np.uint8, np.uint16]

@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(d: dict) -> "Point":
        x = float(d["x"])
        y = float(d["y"])
        z = float(d["z"])
        return Point(x, y, z)

    def to_list(self) -> list:
        return [self.x, self.y, self.z]



# assumes METADATA_VERSION, MicroscopeState, AcquisitionRequest,
# DetectorSettings, ImageOutputSettings already exist in this module

@dataclass
class TemImageMetadata:
    """
    Universal, vendor-agnostic image metadata.
    """

    # ---- Schema / provenance ----
    version: str = METADATA_VERSION
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

    # ---------------- Serialization ----------------

    def to_dict(self) -> Dict[str, Any]:
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
            "pixel_size_nm": list(self.pixel_size_nm) if self.pixel_size_nm else None,
            "image_size_px": list(self.image_size_px) if self.image_size_px else None,
            "accelerating_voltage_kv": self.accelerating_voltage_kv,
            "beam_current_na": self.beam_current_na,
            "exposure_ms": self.exposure_ms,
            "dwell_time_us": self.dwell_time_us,
            "working_distance_mm": self.working_distance_mm,
        }

        if self.microscope_state is not None:
            d["microscope_state"] = self.microscope_state.to_dict()

        if self.acquisition is not None:
            # AcquisitionRequest doesn't currently have to_dict(), so serialize explicitly
            d["acquisition"] = {
                "detector_id": self.acquisition.detector_id,
                "detector": self.acquisition.detector.to_dict() if self.acquisition.detector else None,
                "image": self.acquisition.image.to_dict() if self.acquisition.image else None,
            }

        if self.extra:
            d["extra"] = deepcopy(self.extra)

        # Drop keys with None to keep TIFF metadata smaller (optional)
        return {k: v for k, v in d.items() if v is not None}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TemImageMetadata":
        if not d:
            return TemImageMetadata()

        used_keys = set()

        # --- New-style keys ---
        version = d.get("version", d.get("metadata_version", METADATA_VERSION)); used_keys |= {"version", "metadata_version"}
        created_at = d.get("created_at", d.get("timestamp", None)); used_keys |= {"created_at", "timestamp"}
        user = d.get("user", None); used_keys.add("user")

        obj = TemImageMetadata(
            version=version,
            created_at=created_at if isinstance(created_at, str) else TemImageMetadata().created_at,
            user=user,
            manufacturer=d.get("manufacturer", None),
            device=d.get("device", None),
            model=d.get("model", None),
            serial_number=d.get("serial_number", None),
            software_version=d.get("software_version", None),
            mode=d.get("mode", None),
            detector_id=d.get("detector_id", None),
            detector_name=d.get("detector_name", None),
            magnification=d.get("magnification", None),
            camera_length_mm=d.get("camera_length_mm", d.get("camera_length", None)),
            accelerating_voltage_kv=d.get("accelerating_voltage_kv", d.get("accelerating_voltage", None)),
            beam_current_na=d.get("beam_current_na", d.get("beam_current", None)),
            exposure_ms=d.get("exposure_ms", None),
            dwell_time_us=d.get("dwell_time_us", d.get("dwell_time", None)),
            working_distance_mm=d.get("working_distance_mm", d.get("working_distance", None)),
        )

        used_keys |= {
            "manufacturer","device","model","serial_number","software_version",
            "mode","detector_id","detector_name","magnification","camera_length_mm","camera_length",
            "accelerating_voltage_kv","accelerating_voltage","beam_current_na","beam_current",
            "exposure_ms","dwell_time_us","dwell_time","working_distance_mm","working_distance",
        }

        # pixel_size_nm
        px = d.get("pixel_size_nm", None)
        if isinstance(px, (list, tuple)) and len(px) == 2:
            obj.pixel_size_nm = (float(px[0]), float(px[1]))
            used_keys.add("pixel_size_nm")

        # image_size_px
        imsz = d.get("image_size_px", None)
        if isinstance(imsz, (list, tuple)) and len(imsz) == 2:
            obj.image_size_px = (int(imsz[0]), int(imsz[1]))
            used_keys.add("image_size_px")

        # microscope_state
        ms = d.get("microscope_state", None)
        if isinstance(ms, dict):
            obj.microscope_state = MicroscopeState.from_dict(ms)
            used_keys.add("microscope_state")

        # acquisition
        acq = d.get("acquisition", None)
        if isinstance(acq, dict):
            det = DetectorSettings.from_dict(acq.get("detector")) if isinstance(acq.get("detector"), dict) else DetectorSettings()
            img = ImageOutputSettings.from_dict(acq.get("image")) if isinstance(acq.get("image"), dict) else ImageOutputSettings()
            det_id = acq.get("detector_id", det.detector_id or "")
            if det_id:
                obj.acquisition = AcquisitionRequest(detector_id=str(det_id), detector=det, image=img)
            used_keys.add("acquisition")

        # extra: everything else
        extra = deepcopy(d.get("extra", {})) if isinstance(d.get("extra", None), dict) else {}
        used_keys.add("extra")

        for k, v in d.items():
            if k not in used_keys:
                extra[k] = v
        obj.extra = extra

        return obj

    # ---------- Vendor-specific constructors ----------
    @staticmethod
    def from_jeol(header: Dict[str, Any]) -> "TemImageMetadata":
        """
        If you want: parse JEOL header -> fill the fields above + stash the rest in extra.
        Keep it conservative: only map what you're confident about.
        """
        md = TemImageMetadata()
        md.extra["jeol_header"] = header
        return md

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
        return {
            "name": self.name if self.name is not None else None,
            "x": magnitude(self.x, "nanometer"),
            "y": magnitude(self.y, "nanometer"),
            "z": magnitude(self.z, "nanometer"),
            "r": magnitude(self.r, "degree"),
            "tilt_x": magnitude(self.tilt_x, "degree"),
            "tilt_y": magnitude(self.tilt_y, "degree"),
            "coordinate_system": self.coordinate_system,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemStagePosition":
        return cls(
            name=data.get("name", None),
            x=ensure_quantity(data.get("x", None), "nanometer"),
            y=ensure_quantity(data.get("y", None), "nanometer"),
            z=ensure_quantity(data.get("z", None), "nanometer"),
            r=ensure_quantity(data.get("r", None), "degree"),
            tilt_x=ensure_quantity(data.get("tilt_x", None), "degree"),
            tilt_y=ensure_quantity(data.get("tilt_y", None), "degree"),
            coordinate_system=data.get("coordinate_system", None),
        )

    def __add__(self, other: "TemStagePosition") -> "TemStagePosition":
        if not isinstance(other, TemStagePosition):
            return NotImplemented

        def add_axis(a, b, unit: str):
            if a is None and b is None:
                return None
            if a is None:
                return ensure_quantity(b, unit)
            if b is None:
                return ensure_quantity(a, unit)
            return ensure_quantity(a, unit) + ensure_quantity(b, unit)

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
            if a is None and b is None:
                return None
            if a is None:
                return -ensure_quantity(b, unit)
            if b is None:
                return ensure_quantity(a, unit)
            return ensure_quantity(a, unit) - ensure_quantity(b, unit)

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

    def is_close(
        self,
        other: "TemStagePosition",
        tol_nm: float = 1.0,
        tol_deg: float = 1e-3,
    ) -> bool:
        """Return True if axes differ by <= tolerances.

        tol_nm: tolerance for x/y/z in nanometer
        tol_deg: tolerance for r/tilts in degree
        """

        def close_axis(a, b, unit: str, tol: float) -> bool:
            if a is None or b is None:
                return False
            da = abs(ensure_quantity(a, unit) - ensure_quantity(b, unit))
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
    can_r: bool = True
    can_tilt_x: bool = True
    can_tilt_y: bool = True

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

    def to_dict(self) -> dict:
        return {
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
            "extra": deepcopy(self.extra),
        }

    @staticmethod
    def from_dict(settings: dict) -> "StageSystemSettings":
        if settings is None:
            return StageSystemSettings()

        # ---- Backward-compat mapping for older, weird keys ----
        extra: Dict[str, Any] = deepcopy(settings.get("extra", {}))

        return StageSystemSettings(
            enabled=bool(settings.get("enabled", True)),
            can_x=bool(settings.get("can_x", True)),
            can_y=bool(settings.get("can_y", True)),
            can_z=bool(settings.get("can_z", True)),
            can_r=bool(settings.get("can_r", True)),
            can_tilt_x=bool(settings.get("can_tilt_x", True)),
            can_tilt_y=bool(settings.get("can_tilt_y", True)),
            x_limits_nm=settings.get("x_limits_nm", None),
            y_limits_nm=settings.get("y_limits_nm", None),
            z_limits_nm=settings.get("z_limits_nm", None),
            r_limits_deg=settings.get("r_limits_deg", None),
            tilt_x_limits_deg=settings.get("tilt_x_limits_deg", None),
            tilt_y_limits_deg=settings.get("tilt_y_limits_deg", None),
            max_step_nm=float(settings.get("max_step_nm", 50000.0)),
            max_step_deg=float(settings.get("max_step_deg", 1.0)),
            settle_time_s=float(settings.get("settle_time_s", 0.2)),
            timeout_s=float(settings.get("timeout_s", 10.0)),
            eucentric_z_nm=settings.get("eucentric_z_nm", settings.get("eucentric_height", None)),
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

    stigmation: Point = field(default_factory=Point)

    # Separate "beam shift" and "image shift" (older code used a single `shift`)
    beam_shift: Point = field(default_factory=Point)
    image_shift: Point = field(default_factory=Point)

    # For STEM scan coordinate systems (optional)
    scan_rotation_deg: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.voltage is not None:
            assert isinstance(self.voltage, (float, int)), f"voltage must be float/int, got {type(self.voltage)}"
        if self.beam_current is not None:
            assert isinstance(self.beam_current, (float, int)), f"beam_current must be float/int, got {type(self.beam_current)}"
        if self.spot_size is not None:
            assert isinstance(self.spot_size, int), f"spot_size must be int, got {type(self.spot_size)}"
        if self.convergence_angle_mrad is not None:
            assert isinstance(self.convergence_angle_mrad, (float, int)), f"convergence_angle_mrad must be float/int, got {type(self.convergence_angle_mrad)}"
        if self.scan_rotation_deg is not None:
            assert isinstance(self.scan_rotation_deg, (float, int)), f"scan_rotation_deg must be float/int, got {type(self.scan_rotation_deg)}"

        if self.stigmation is None:
            self.stigmation = Point()
        if self.beam_shift is None:
            self.beam_shift = Point()
        if self.image_shift is None:
            self.image_shift = Point()
        if self.extra is None:
            self.extra = {}

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
            "extra": deepcopy(self.extra),
        }
        return d

    @staticmethod
    def from_dict(state_dict: dict) -> "BeamSettings":
        if state_dict is None:
            return BeamSettings()

        extra: Dict[str, Any] = deepcopy(state_dict.get("extra", {}))

        # stigmation
        if "stigmation" in state_dict and state_dict["stigmation"] is not None:
            stigmation = Point.from_dict(state_dict["stigmation"])
        else:
            stigmation = Point()

        # new preferred keys
        beam_shift = None
        image_shift = None
        if "beam_shift" in state_dict and state_dict["beam_shift"] is not None:
            beam_shift = Point.from_dict(state_dict["beam_shift"])
        if "image_shift" in state_dict and state_dict["image_shift"] is not None:
            image_shift = Point.from_dict(state_dict["image_shift"])

        if beam_shift is None:
            beam_shift = Point()
        if image_shift is None:
            image_shift = Point()

        # voltage key is kept for compatibility; interpret as kV
        voltage = state_dict.get("voltage", state_dict.get("accelerating_voltage_kv", None))

        # common aliases for current
        current = state_dict.get("beam_current", state_dict.get("current", None))
        spot_size = state_dict.get("spot_size", state_dict.get("spot", None))
        conv = state_dict.get("convergence_angle_mrad", state_dict.get("convergence_mrad", None))
        scan_rot = state_dict.get("scan_rotation_deg", None)
        return BeamSettings(
            voltage=voltage,
            beam_current=current,
            spot_size=spot_size if spot_size is None else int(spot_size),
            convergence_angle_mrad=conv,
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

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "default_beam": self.default_beam.to_dict() if self.default_beam is not None else None,
            "voltage_range_kv": self.voltage_range_kv,
            "beam_current_range_na": self.beam_current_range_na,
            "spot_size_range": self.spot_size_range,
            "convergence_angle_range_mrad": self.convergence_angle_range_mrad,
            "extra": deepcopy(self.extra),
        }

    @staticmethod
    def from_dict(settings: dict) -> "BeamSystemSettings":
        if settings is None:
            return BeamSystemSettings()

        extra: Dict[str, Any] = deepcopy(settings.get("extra", {}))

        # Backward-compat: older schema stuffed beam+detector fields at this level.
        # We'll treat the whole dict as a beam default if `default_beam` isn't provided.
        default_beam_dict = settings.get("default_beam", None)
        default_beam = BeamSettings.from_dict(default_beam_dict)


        return BeamSystemSettings(
            enabled=bool(settings.get("enabled", True)),
            default_beam=default_beam,
            voltage_range_kv=settings.get("voltage_range_kv", settings.get("voltage_limits_kv", None)),
            beam_current_range_na=settings.get("beam_current_range_na", None),
            spot_size_range=settings.get("spot_size_range", None),
            convergence_angle_range_mrad=settings.get("convergence_angle_range_mrad", None),
            extra=extra,
        )

@dataclass
class ROI:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

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

        return obj

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "detector_id": self.detector_id,
            "exposure_ms": self.exposure_ms,
            "binning_index": self.binning_index,
            "binning_xy": list(self.binning_xy) if self.binning_xy is not None else None,
            "frame_integration": self.frame_integration,
            "gain_index": self.gain_index,
            "offset_index": self.offset_index,
            "digital_rotation_deg": self.digital_rotation_deg,
        }
        if self.roi is not None:
            d["roi"] = asdict(self.roi)
        if self.extra:
            d["extra"] = deepcopy(self.extra)
        return d

    @staticmethod
    def from_dict(settings: Dict[str, Any]) -> "DetectorSettings":
        if settings is None:
            return DetectorSettings()

        kwargs: Dict[str, Any] = {}
        direct_keys = [
            "detector_id",
            "exposure_ms",
            "binning_index",
            "binning_xy",
            "frame_integration",
            "gain_index",
            "offset_index",
            "digital_rotation_deg",
            "roi",
            "extra",
        ]
        for k in direct_keys:
            if k in settings and k not in kwargs:
                kwargs[k] = settings.get(k)

        # ROI parsing
        roi_val = kwargs.get("roi")
        if isinstance(roi_val, dict):
            kwargs["roi"] = ROI(
                x=int(roi_val.get("x", 0)),
                y=int(roi_val.get("y", 0)),
                width=int(roi_val.get("width", roi_val.get("w", 0))),
                height=int(roi_val.get("height", roi_val.get("h", 0))),
            )
        elif roi_val is None:
            # backward-compat aliases
            for alias in ("detector_roi", "imaging_area"):
                if isinstance(settings.get(alias), dict):
                    r = settings[alias]
                    kwargs["roi"] = ROI(
                        x=int(r.get("x", 0)),
                        y=int(r.get("y", 0)),
                        width=int(r.get("width", r.get("w", 0))),
                        height=int(r.get("height", r.get("h", 0))),
                    )
                    break

        # Normalize binning_xy
        if isinstance(kwargs.get("binning_xy"), (list, tuple)) and kwargs.get("binning_xy") is not None:
            bx = kwargs["binning_xy"]
            if len(bx) == 2:
                kwargs["binning_xy"] = (int(bx[0]), int(bx[1]))

        user_extra = kwargs.pop("extra", None)
        obj = DetectorSettings(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(DetectorSettings)}})

        if isinstance(user_extra, dict):
            obj.extra.update(user_extra)
        return obj

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

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "default_detector_id": self.default_detector_id,
            "defaults_by_id": {k: v.to_dict() for k, v in self.defaults_by_id.items()},
            "capabilities_by_id": {k: asdict(v) for k, v in self.capabilities_by_id.items()},
            "available_detectors": list(self.available_detectors),
            "extra": deepcopy(self.extra),
        }

    @staticmethod
    def from_dict(settings: Optional[dict]) -> "DetectorSystemSettings":
        if not settings:
            return DetectorSystemSettings()

        extra: Dict[str, Any] = deepcopy(settings.get("extra", {}))

        defaults_raw = (
            settings.get("defaults_by_id")
            or settings.get("default_detectors_by_id")
            or settings.get("default_detector_by_id")
            or {}
        )

        defaults_by_id: Dict[str, DetectorSettings] = {}

        if isinstance(defaults_raw, dict):
            for det_id, val in defaults_raw.items():
                if isinstance(val, DetectorSettings):
                    ds = val
                else:
                    ds = DetectorSettings.from_dict(val)
                # Ensure detector_id is set consistently
                if ds.detector_id is None:
                    ds.detector_id = str(det_id)
                defaults_by_id[str(det_id)] = ds

        # ---- Capabilities map ----
        cap_map_raw = settings.get("capabilities_by_id", {}) or {}
        cap_map: Dict[str, DetectorCapabilities] = {}
        if isinstance(cap_map_raw, dict):
            for det_id, cap in cap_map_raw.items():
                if isinstance(cap, DetectorCapabilities):
                    cap_map[str(det_id)] = cap
                elif isinstance(cap, dict):
                    cap_kwargs = {f.name: cap.get(f.name) for f in fields(DetectorCapabilities) if f.name in cap}
                    cap_extra = {k: v for k, v in cap.items() if k not in {f.name for f in fields(DetectorCapabilities)}}
                    obj = DetectorCapabilities(**cap_kwargs)
                    if cap_extra:
                        obj.extra.update(cap_extra)
                    cap_map[str(det_id)] = obj

        # ---- Available detectors ----
        available = settings.get("available_detectors", None)
        if available is None:
            # Prefer explicit list; otherwise infer from union of keys.
            key_union = set(defaults_by_id.keys()) | set(cap_map.keys())
            available = list(sorted(key_union))

        # ---- Default detector id ----
        default_detector_id = settings.get("default_detector_id", None)
        if default_detector_id is None:
            if len(available) == 1:
                default_detector_id = available[0]
            elif len(defaults_by_id) == 1:
                default_detector_id = next(iter(defaults_by_id.keys()))

        return DetectorSystemSettings(
            enabled=bool(settings.get("enabled", True)),
            defaults_by_id=defaults_by_id,
            default_detector_id=str(default_detector_id) if default_detector_id is not None else None,
            capabilities_by_id=cap_map,
            available_detectors=[str(x) for x in (available or [])],
            extra=extra,
        )

@dataclass
class ImageOutputSettings:
    file_format: Optional[str] = "tiff"  # "tiff", "jpg", "bmp", ...
    path: Optional[Union[str, Path]] = None  # default output directory (session dir)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "file_format": self.file_format,
            "path": str(self.path) if self.path is not None else None,
        }

        return d

    @staticmethod
    def from_dict(settings: dict) -> "ImageOutputSettings":
        setting = ImageOutputSettings(
            file_format=settings.get("file_format", "tiff"),
            path=settings.get("path", None),
        )
        return setting

@dataclass
class AcquisitionRequest:
    detector_id: str
    detector: DetectorSettings
    image: ImageOutputSettings
    
@dataclass
class MicroscopeState:

    """Data Class representing the state of a microscope with various parameters.

    Attributes:

        timestamp (float): A float representing the timestamp at which the state of the microscope was recorded. Defaults to the timestamp of the current datetime.
        stage_position (TemStagePosition): An instance of TemStagePosition representing the current absolute position of the stage. Defaults to an empty instance of TemStagePosition.
        beam (BeamSettings): An instance of BeamSettings representing the beam settings. Defaults to to an empty instance of BeamSettings.
        detector (TemDetectorSettings): An instance of TemDetectorSettings representing the detector settings. Defaults to an empty instance of TemDetectorSettings.

    Methods:

        to_dict(self) -> dict: Converts the current state of the Microscope to a dictionary and returns it.
        from_dict(state_dict: dict) -> "MicroscopeState": Returns a new instance of MicroscopeState with attributes created from the passed dictionary.
    """

    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    stage_position: TemStagePosition = field(default_factory=TemStagePosition)
    beam: BeamSettings = field(default_factory=BeamSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    protocol: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            isinstance(self.stage_position, TemStagePosition)
            or self.stage_position is None
        ), f"absolute position must be of type TemStagePosition, currently is {type(self.stage_position)}"
        assert (
            isinstance(self.beam, BeamSettings) or self.beam is None
        ), f"beam must be of type BeamSettings, currently is {type(self.beam)}"
        assert (
            isinstance(self.detector, DetectorSettings) or self.detector is None
        ), f"detector must be of type DetectorSettings, currently is {type(self.detector)}"

    def to_dict(self) -> dict:
        state_dict = {
            "timestamp": self.timestamp,
            "stage_position": self.stage_position.to_dict()
            if self.stage_position is not None
            else None,
            "beam": self.beam.to_dict()
            if self.beam is not None
            else None,
            "detector": self.detector.to_dict()
            if self.detector is not None
            else None,
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "MicroscopeState":

        if state_dict.get("beam", None) is not None:
            beam = BeamSettings.from_dict(state_dict["beam"])
        if state_dict.get("detector", None) is not None:
            detector = DetectorSettings.from_dict(state_dict["detector"])

        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            stage_position=TemStagePosition.from_dict(
                state_dict["stage_position"]
            ),
            beam=beam,
            detector=detector,
        )

        return microscope_state


class TemImage:
    """
    Generic TEM Image object with universal metadata handling.

    Attributes:
        data (np.ndarray): image data.
        metadata (TemImageMetadataRefined): associated metadata.
    Supports:
        - ThermoFisher API (AdornedImage)
        - Tescan API (Header)
        - Other vendors via vendor-specific factory functions.
    """

    def __init__(self, data: np.ndarray, metadata: Optional[TemImageMetadata] = None):
        if not _check_data_format(data):
            raise ValueError("Invalid data format for Tem Image.")
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        self.data = data
        self.metadata = metadata

    # -------------------------- I/O --------------------------

    @classmethod
    def load(cls, tiff_path: str) -> "TemImage":
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                desc = tiff_image.pages[0].tags["ImageDescription"].value
                metadata = TemImageMetadata.from_dict(json.loads(desc))
            except Exception:
                metadata = None
        return cls(data=data, metadata=metadata)

    def save(self, path: Path) -> None:
        path = Path(path).with_suffix(".tif")
        os.makedirs(path.parent, exist_ok=True)
        metadata_dict = self.metadata.to_dict() if self.metadata else {}
        tff.imwrite(path, self.data, metadata=metadata_dict)

    # ---------------------- Vendor-specific ----------------------

    @classmethod
    def from_jeol(cls, image, image_settings: ImageOutputSettings, state: MicroscopeState, detector: DetectorSettings):
        """Convert Jeol image object (with Header) to TemImage."""
        pixel_size = Point(
            float(image.Header["MAIN"]["PixelSizeX"]),
            float(image.Header["MAIN"]["PixelSizeY"]),
        )

        metadata = TemImageMetadata(
            image_settings=image_settings,
            pixel_size=pixel_size,
            microscope_state=state,
            detector_settings=detector,
            version=METADATA_VERSION,
        )
        return cls(data=np.array(image.Image), metadata=metadata)
    
    @classmethod
    def from_jeol_image(
        cls,
        image,
        image_settings: Optional["ImageSettings"] = None,
        state: Optional["MicroscopeState"] = None,
        detector: Optional["DetectorSettings"] = None,
    ) -> "TemImage":
        """
        Create a TemImage from a Jeol microscope image output.

        Args:
            image: Jeol image object (with .Header and .Image)
            image_settings: optional, capture parameters used for acquisition
            state: optional, current microscope state (stage, beam, etc.)
            detector: optional, detector configuration

        Returns:
            FibsemImage: standardized image object with unified metadata
        """

        # --- Convert raw pixel data ---
        data = np.array(image.Image)

        # --- Extract standardized metadata from JEOL header ---
        header = {section: dict(image.Header.items(section)) for section in image.Header.sections()}
        metadata_refined = TemImageMetadata.from_jeol(header)

        # --- Construct the TemImage ---
        tem_image = cls(
            data=data,
            metadata=metadata_refined,
        )

        return tem_image

    # ---------------------- Generic Adapter ----------------------

    @classmethod
    def from_vendor(cls, vendor: str, *args, **kwargs) -> "TemImage":
        """
        Universal adapter for different vendor inputs.
        vendor: 'tescan', 'thermofisher', 'jeol', 'hitachi', etc.
        """
        vendor = vendor.lower()
        if vendor == "jeol":
            return cls.from_jeol(*args, **kwargs)
        else:
            # For other vendors, fallback to minimal metadata
            data = kwargs.get("data")
            pixel_size = kwargs.get("pixel_size", Point(1, 1))
            image_settings = kwargs.get("image_settings", ImageOutputSettings(resolution=data.shape))
            metadata = TemImageMetadata(image_settings=image_settings, pixel_size=pixel_size)
            return cls(data=data, metadata=metadata)
    
@dataclass
class SystemInfo:
    name: str
    ip_address: str
    manufacturer: str
    model: str
    serial_number: str
    hardware_version: str
    software_version: str
    supertem_version: str = __version__
    application: str = None
    application_version: str = None

    def to_dict(self):
        return {
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
    
    @staticmethod
    def from_dict(settings: dict):
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
        )


@dataclass
class SystemSettings:
    stage: StageSystemSettings
    beam: BeamSystemSettings
    detector: DetectorSystemSettings
    info: SystemInfo

    def to_dict(self):
        return {
            "stage": self.stage.to_dict(),
            "beam": self.beam.to_dict(),
            "detector": self.detector.to_dict(),
            "info": self.info.to_dict(),
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return SystemSettings(
            stage=StageSystemSettings.from_dict(settings["stage"]),
            beam=BeamSystemSettings.from_dict(settings["beam"]),
            detector=DetectorSystemSettings.from_dict(settings["detector"]),
            info=SystemInfo.from_dict(settings["info"]),
        )

@dataclass
class MicroscopeSettings:

    """
    A data class representing the settings for a microscope system.

    Attributes:
        system (SystemSettings): An instance of the `SystemSettings` class that holds the system settings.
        image (ImageSettings): An instance of the `ImageSettings` class that holds the image settings.
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to None.

    Methods:
        to_dict(): Returns a dictionary representation of the `MicroscopeSettings` object.
        from_dict(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """

    system: SystemSettings
    image: ImageOutputSettings
    protocol: dict = None

    def to_dict(self) -> dict:
        settings_dict = {
            "image": self.image.to_dict(),
            "protocol": self.protocol,
        }
        settings_dict.update(self.system.to_dict())

        return settings_dict

    @staticmethod
    def from_dict(
        settings: dict, protocol: dict = None
    ) -> "MicroscopeSettings":
        
        if protocol is None:
            protocol = settings.get("protocol", {"name": "demo"})
     
        return MicroscopeSettings(
            system=SystemSettings.from_dict(settings),
            image=ImageOutputSettings.from_dict(settings["image"]),
            protocol=protocol,
        )
