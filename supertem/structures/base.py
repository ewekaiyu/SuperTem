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

@dataclass
class TemImageMetadataRefined:
    """Universal metadata for TemImage, compatible across SEM/FIB vendors."""

    # --- Basic Info (from MAIN section) ---
    device: Optional[str] = None              # e.g. "TESCAN SOLARIS X"
    model: Optional[str] = None               # e.g. "S9251X"
    serial_number: Optional[str] = None
    software_version: Optional[str] = None
    user: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

    # --- Imaging Info ---
    magnification: Optional[float] = None
    magnification_reference: Optional[float] = None   # from MagnificationReference
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    resolution: Optional[List[int]] = None            # (width, height)

    # --- Beam & Optics ---
    accelerating_voltage: Optional[float] = None      # HV or AcceleratorVoltage
    emission_current: Optional[float] = None
    beam_current: Optional[float] = None              # SpecimenCurrent or PredictedBeamCurrent
    dwell_time: Optional[float] = None
    spot_size: Optional[float] = None
    gun_type: Optional[str] = None                    # e.g. "Schottky", "Mistral"

    # --- Imaging Environment ---
    detector: Optional[str] = None                    # e.g. "In-Beam SE" or "SE"
    chamber_pressure: Optional[float] = None
    working_distance: Optional[float] = None
    scan_rotation: Optional[float] = None
    scan_speed: Optional[float] = None
    injected_gas: Optional[str] = None                # e.g. "N2" (SEM specific)
    column_tilt: Optional[float] = None               # (FIB specific)
    column_type: Optional[str] = None                 # (FIB specific)

    # --- Stage & Geometry ---
    stage_x: Optional[float] = None
    stage_y: Optional[float] = None
    stage_z: Optional[float] = None
    stage_tilt: Optional[float] = None
    stage_rotation: Optional[float] = None

    # --- Image Processing / LUT ---
    lut_minimum: Optional[float] = None
    lut_maximum: Optional[float] = None
    lut_gamma: Optional[float] = None

    # --- Misc ---
    session_id: Optional[str] = None
    stigmator_x: Optional[float] = None
    stigmator_y: Optional[float] = None
    tilt_correction: Optional[float] = None
    objective: Optional[float] = None

    # --- For vendor-specific or unknown values ---
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a flat dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TemImageMetadataRefined":
        """Construct from dictionary, storing unknown keys in extra."""
        field_names = {f.name for f in fields(TemImageMetadataRefined)}
        known = {k: v for k, v in d.items() if k in field_names}
        extra_data = {k: v for k, v in d.items() if k not in field_names}
        obj = TemImageMetadataRefined(**known)
        if extra_data:
            obj.extra.update(extra_data)
        return obj

    # ---------- Vendor-specific constructors ----------
    @staticmethod
    def from_jeol(header: Dict[str, Any]) -> "TemImageMetadataRefined":
        """Construct metadata from JEOL HEADER (supports SEM or FIB)."""
        pass


@dataclass
class ImageSettings:
    width: Optional[int] = None
    height: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    binning: Optional[int] = None
    exposure_ms: Optional["Quantity"] = None
    dwell_us: Optional["Quantity"] = None
    file_format: Optional[str] = "tiff"  # "tiff", "jpg", "bmp", ...

    def __post_init__(self):
        # Normalize timing units
        if self.exposure_ms is not None:
            self.exposure_ms = ensure_quantity(self.exposure_ms, "millisecond")
        if self.dwell_us is not None:
            self.dwell_us = ensure_quantity(self.dwell_us, "microsecond")

    def to_dict(self) -> dict:
        # Keep JSON-friendly magnitudes (ms / µs)
        return {
            "width": self.width,
            "height": self.height,
            "x": self.x,
            "y": self.y,
            "binning": self.binning,
            "exposure_ms": magnitude(self.exposure_ms, "millisecond"),
            "dwell_us": magnitude(self.dwell_us, "microsecond"),
            "file_format": self.file_format,
        }

    @staticmethod
    def from_dict(settings: dict) -> "ImageSettings":
        return ImageSettings(
            width=settings["width"],
            height=settings["height"],
            x=settings.get("x", None),
            y=settings.get("y", None),
            binning=settings.get("binning", None),
            exposure_ms=ensure_quantity(settings.get("exposure_ms", None), "millisecond"),
            dwell_us=ensure_quantity(settings.get("dwell_us", None), "microsecond"),
            file_format=settings.get("file_format", "tiff"),
        )

@dataclass
class StageSystemSettings:
    rotation_reference: float
    rotation_180: float
    shuttle_pre_tilt: float
    manipulator_height_limit: float
    enabled: bool = True
    rotation: bool = True
    tilt: bool  = True

    def to_dict(self):
        return {
            "rotation_reference": self.rotation_reference,
            "rotation_180": self.rotation_180,
            "shuttle_pre_tilt": self.shuttle_pre_tilt,
            "manipulator_height_limit": self.manipulator_height_limit,
            "enabled": self.enabled,
            "rotation": self.rotation,
            "tilt": self.tilt,
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return StageSystemSettings(
            rotation_reference=settings["rotation_reference"],
            rotation_180=settings["rotation_180"],
            shuttle_pre_tilt=settings["shuttle_pre_tilt"],
            manipulator_height_limit=settings["manipulator_height_limit"],
            enabled=settings.get("enabled", True),
            rotation=settings.get("rotation", True),
            tilt=settings.get("tilt", True),
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
    stage: List[Any] = field(default_factory=list)

    def __post_init__(self):
        # Normalize individual axes
        self.x = ensure_quantity(self.x, "nanometer")
        self.y = ensure_quantity(self.y, "nanometer")
        self.z = ensure_quantity(self.z, "nanometer")
        self.r = ensure_quantity(self.r, "degree")
        self.tilt_x = ensure_quantity(self.tilt_x, "degree")
        self.tilt_y = ensure_quantity(self.tilt_y, "degree")

        if self.stage is None:
            self.stage = [self.x, self.y, self.z, self.r, self.tilt_x, self.tilt_y]

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
            if a is None:
                return None
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

    def _scale_repr(self, scale: float, precision: int = 2):
        return f"x:{self.x*scale:.{precision}f}, y:{self.y*scale:.{precision}f}, z:{self.z*scale:.{precision}f}"

    def is_close(self, pos2: 'TemStagePosition', tol: float = 1e-6) -> bool:
        """Check if two positions are close to each other."""
        return ((abs(self.x - pos2.x) < tol) and 
                (abs(self.y - pos2.y) < tol) and 
                (abs(self.z - pos2.z) < tol) and 
                (abs(self.t - pos2.t) < tol) and 
                (abs(self.r - pos2.r) < tol) and 
                (abs(self.tilt_y - pos2.tilt_y) < tol))

    
@dataclass
class BeamSettings:
    """
    Dataclass representing the beam settings for an imaging session.

    Attributes:
        working_distance (float): The working distance for the microscope, in meters.
        beam_current (float): The beam current for the microscope, in amps.
        hfw (float): The horizontal field width for the microscope, in meters.
        resolution (list): The desired resolution for the image.
        dwell_time (float): The dwell time for the microscope.
        stigmation (Point): The point for stigmation correction.
        shift (Point): The point for shift correction.

    Methods:
        to_dict(): Returns a dictionary representation of the object.
        from_dict(state_dict: dict) -> BeamSettings: Returns a new BeamSettings object created from a dictionary.

    """
    working_distance: float = None
    beam_current: float = None
    voltage: float = None
    hfw: float = None
    resolution: List[int] = field(default_factory=list)
    dwell_time: float = None
    stigmation: Point = field(default_factory=Point)
    shift: Point = field(default_factory=Point)
    scan_rotation: float = None

    def __post_init__(self):
        assert (
            isinstance(self.working_distance, (float, int))
            or self.working_distance is None
        ), f"Working distance must be float or int, currently is {type(self.working_distance)}"
        assert (
            isinstance(self.beam_current, (float, int)) or self.beam_current is None
        ), f"beam current must be float or int, currently is {type(self.beam_current)}"
        assert (
            isinstance(self.voltage, (float, int)) or self.voltage is None
        ), f"voltage must be float or int, currently is {type(self.voltage)}"
        assert (
            isinstance(self.hfw, (float, int)) or self.hfw is None
        ), f"horizontal field width (HFW) must be float or int, currently is {type(self.hfw)}"
        assert (
            isinstance(self.resolution, list) or self.resolution is None
        ), f"resolution must be a list, currently is {type(self.resolution)}"
        assert (
            isinstance(self.dwell_time, (float, int)) or self.dwell_time is None
        ), f"dwell_time must be float or int, currently is {type(self.dwell_time)}"
        assert (
            isinstance(self.stigmation, Point) or self.stigmation is None
        ), f"stigmation must be a Point instance, currently is {type(self.stigmation)}"
        assert (
            isinstance(self.shift, Point) or self.shift is None
        ), f"shift must be a Point instance, currently is {type(self.shift)}"


    def to_dict(self) -> dict:
        state_dict = {
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "voltage": self.voltage,
            "hfw": self.hfw,
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            "stigmation": self.stigmation.to_dict()
            if self.stigmation is not None
            else None,
            "shift": self.shift.to_dict() if self.shift is not None else None,
            "scan_rotation": self.scan_rotation,
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "BeamSettings":
        if "stigmation" in state_dict and state_dict["stigmation"] is not None:
            stigmation = Point.from_dict(state_dict["stigmation"])
        else:
            stigmation = Point()
        if "shift" in state_dict and state_dict["shift"] is not None:
            shift = Point.from_dict(state_dict["shift"])
        else:
            shift = Point()
        
        wd = state_dict.get("working_distance", state_dict.get("eucentric_height", None))
        current = state_dict.get("beam_current", state_dict.get("current", None))

        beam_settings = BeamSettings(
            working_distance=wd,
            beam_current=current,
            voltage=state_dict["voltage"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            stigmation=stigmation,
            shift=shift,
            scan_rotation=state_dict.get("scan_rotation", 0.0),
        )

        return beam_settings
    
@dataclass
class TemDetectorSettings:
    type: str = None
    mode: str = None
    brightness: float = 0.5
    contrast: float = 0.5

    def __post_init__(self):
        assert (
            isinstance(self.type, str) or self.type is None
        ), f"type must be input as str, currently is {type(self.type)}"
        assert (
            isinstance(self.mode, str) or self.mode is None
        ), f"mode must be input as str, currently is {type(self.mode)}"
        assert (
            isinstance(self.brightness, (float, int)) or self.brightness is None
        ), f"brightness must be int or float value, currently is {type(self.brightness)}"
        assert (
            isinstance(self.contrast, (float, int)) or self.contrast is None
        ), f"contrast must be int or float value, currently is {type(self.contrast)}"

    if JEOL:

        def to_jeol(self):
            """Converts to jeol format."""
            jeol_brightness = self.brightness * 100
            jeol_contrast = self.contrast * 100
            return jeol_brightness, jeol_contrast

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "type": self.type,
            "mode": self.mode,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }

    @staticmethod
    def from_dict(settings: dict) -> "TemDetectorSettings":
        """Converts from a dictionary."""
        return TemDetectorSettings(
            type=settings.get("type", "Unknown"),
            mode=settings.get("mode", "Unknown"),
            brightness=settings.get("brightness", 0.0),
            contrast=settings.get("contrast", 0.0),
        )
    
@dataclass
class BeamSystemSettings:
    enabled: bool
    beam: BeamSettings
    detector: TemDetectorSettings
    eucentric_height: float
    column_tilt: float
    plasma: bool = False
    plasma_gas: str = None

    def to_dict(self):
        ddict = {
            "enabled": self.enabled,
            "eucentric_height": self.eucentric_height,
            "column_tilt": self.column_tilt,
            "plasma": self.plasma,
            "plasma_gas": self.plasma_gas,
        }
        ddict.update(self.beam.to_dict())
        ddict.update(self.detector.to_dict())
        
        # rename keys to match config
        ddict["detector_mode"] = ddict.pop("mode")
        ddict["detector_type"] = ddict.pop("type")
        ddict["detector_brightness"] = ddict.pop("brightness")
        ddict["detector_contrast"] = ddict.pop("contrast")
        ddict["current"] = ddict.pop("beam_current")

        return ddict
    
    @staticmethod
    def from_dict(settings: dict) -> 'BeamSystemSettings':
        return BeamSystemSettings(
            enabled=settings["enabled"],
            beam=BeamSettings.from_dict(settings),
            detector=TemDetectorSettings.from_dict(settings),
            eucentric_height=settings["eucentric_height"],
            column_tilt=settings["column_tilt"],
            plasma=settings.get("plasma", False),
            plasma_gas=settings.get("plasma_gas", None),
        )
    
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
    beam: BeamSettings = field(default_factory=lambda: BeamSettings)
    detector: TemDetectorSettings = field(default_factory=TemDetectorSettings)
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
            isinstance(self.detector, TemDetectorSettings) or self.detector is None
        ), f"detector must be of type TemDetectorSettings, currently is {type(self.detector)}"

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
            detector = TemDetectorSettings.from_dict(state_dict["detector"])

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

    def __init__(self, data: np.ndarray, metadata: Optional[TemImageMetadataRefined] = None):
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
                metadata = TemImageMetadataRefined.from_dict(json.loads(desc))
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
    def from_jeol(cls, image, image_settings: ImageSettings, state: MicroscopeState, detector: TemDetectorSettings):
        """Convert Jeol image object (with Header) to TemImage."""
        pixel_size = Point(
            float(image.Header["MAIN"]["PixelSizeX"]),
            float(image.Header["MAIN"]["PixelSizeY"]),
        )

        metadata = TemImageMetadataRefined(
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
        detector: Optional["TemDetectorSettings"] = None,
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
        metadata_refined = TemImageMetadataRefined.from_jeol(header)

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
            image_settings = kwargs.get("image_settings", ImageSettings(resolution=data.shape))
            metadata = TemImageMetadataRefined(image_settings=image_settings, pixel_size=pixel_size)
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
    info: SystemInfo    

    def to_dict(self):
        return {
            "stage": self.stage.to_dict(),
            "beam": self.beam.to_dict(),
            "info": self.info.to_dict(),
        }
    
    @staticmethod
    def from_dict(settings: dict):

            
        return SystemSettings(
            stage=StageSystemSettings.from_dict(settings["stage"]),
            beam=BeamSystemSettings.from_dict(settings["beam"]),
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
    image: ImageSettings
    protocol: dict = None

    def to_dict(self) -> dict:
        settings_dict = {
            "imaging": self.image.to_dict(),
            "protocol": self.protocol,
            "milling": self.milling.to_dict(),
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
            image=ImageSettings.from_dict(settings["imaging"]),
            protocol=protocol,
        )
