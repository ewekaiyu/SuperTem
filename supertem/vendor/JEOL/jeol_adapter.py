"""
supertem.vendor.JEOL.jeol_adapter

The Data Translation Layer for JEOL PyJEM.

This module is responsible for converting between:
1. Raw Python types returned by PyJEM (dicts, lists, int codes).
2. Strictly typed SuperTEM structures.

It isolates the 'Dirty' logic of parsing vendor keys/units from the 'Clean' logic of the microscope driver.
All unmapped data is preserved in the `extra.vendor['JEOL']` dictionary.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from supertem.structures.base import (
    DetectorSettings,
    DetectorCapabilities,
    ROI,
    StagePosition,
    Extras,
    Q_,
    Units,
    Point
)

# =============================================================================
# 0. Helpers
# =============================================================================

def _pack_vendor_extras(data: Dict[str, Any]) -> Optional[Extras]:
    """
    Wraps a dictionary of raw vendor data into the standard Extras.vendor structure.
    Strictly namespaces under 'JEOL'.
    """
    if not data:
        return None
    return Extras(vendor={"JEOL": data})

def _jeol_roi_to_struct(x: Any) -> Optional[ROI]:
    """Convert JEOL 'ImagingArea' dict to ROI object."""
    if not isinstance(x, dict):
        return None
    # Robust casing check: X vs x, Width vs width
    return ROI(
        x=int(x.get("X", x.get("x", 0))),
        y=int(x.get("Y", x.get("y", 0))),
        width=int(x.get("Width", x.get("width", 512))),
        height=int(x.get("Height", x.get("height", 512))),
    )

def _struct_to_jeol_roi(roi: Optional[ROI]) -> Optional[Dict[str, int]]:
    if not roi:
        return None
    return {
        "X": int(roi.x or 0),
        "Y": int(roi.y or 0),
        "Width": int(roi.width or 512),
        "Height": int(roi.height or 512),
    }

def _jeol_bin_to_tuple(x: Any) -> Optional[Tuple[int, int]]:
    """Convert JEOL 'BinningSize' dict to tuple (w, h)."""
    if not isinstance(x, dict):
        return None
    return (int(x.get("Width", 1)), int(x.get("Height", 1)))

def _parse_jeol_roi_limit(val: Any) -> Optional[Tuple[int, int]]:
    """
    Robustly parse JEOL size limits.
    Handles both dict {'Width': w, 'Height': h} and string 'w, h' formats.
    Returns: (width, height)
    """
    if isinstance(val, dict):
        return (int(val.get("Width", 0)), int(val.get("Height", 0)))

    # Handle string format "1072, 1072" found in some PyJEM versions
    if isinstance(val, str) and "," in val:
        try:
            parts = [p.strip() for p in val.split(",")]
            if len(parts) >= 2:
                # Assuming "Width, Height" or "X, Y" (usually symmetric)
                return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass

    # Fallback for single integer strings or numbers (assuming square)
    try:
        val_int = int(val)
        return (val_int, val_int)
    except (TypeError, ValueError):
        pass

    return None


# =============================================================================
# 1. Detector Adapters
# =============================================================================

def from_jeol_detector_response(payload: Dict[str, Any], detector_id: str) -> Tuple[
    DetectorSettings, DetectorCapabilities]:
    """
    Parses dictionary from Detector.get_detectorsetting().
    Handles dynamic ROI source (ImagingArea vs AreaModeImagingArea) based on Scan Mode.
    Strictly separates Camera Exposure (ms) from STEM Dwell (us).
    """
    p = copy.deepcopy(payload)

    def _pop_any(keys, default=None):
        for k in keys:
            if k in p: return p.pop(k)
        return default

    # --- 1. Detect Detector Type (STEM vs TEM) ---
    is_stem = False
    if "AreaModeImagingArea" in payload or "SpotPosition" in payload:
        is_stem = True
    elif "BinningSize" in payload:
        is_stem = False
    elif any(x in detector_id.upper() for x in ["DFI", "BFI", "BEI", "SEI", "EXT"]):
        is_stem = True

    # --- 2. Determine Active ROI Source ---
    # Default to ImagingArea (standard for TEM and STEM Full Scan)
    roi_source_key = "ImagingArea"

    if is_stem:
        # Check Scan Mode to see if we should use AreaModeImagingArea
        # JEOL Code 3 usually denotes 'Area' (Sub-scan)
        scan_mode = p.get("ScanMode") or p.get("scanMode") or p.get("ScanModeValue")

        is_area_mode = False
        if isinstance(scan_mode, int) and scan_mode == 3:
            is_area_mode = True
        elif isinstance(scan_mode, str) and "AREA" in scan_mode.upper():
            is_area_mode = True

        if is_area_mode and "AreaModeImagingArea" in p:
            roi_source_key = "AreaModeImagingArea"

    # --- 3. Extract Settings ---
    # Use the dynamically selected key for the primary ROI
    roi_data = _pop_any([roi_source_key, "ImagingArea", "imagingArea"])

    bin_size_data = _pop_any(["BinningSize", "binningSize"])
    exp_val = _pop_any(["ExposureTimeValue", "Exposure", "ExposureTime", "exposureTime"], 0.0)

    settings_kwargs = {
        "detector_id": detector_id,
        "binning_index": _pop_any(["BinningIndex", "binningIndex"]),
        "frame_integration": _pop_any(["frameIntegration", "AccumulationCount", "accumulationCount"]),
        "gain_index": _pop_any(["GainIndex", "gainIndex"]),
        "offset_index": _pop_any(["OffsetIndex", "offsetIndex"]),
        "_mode": "lenient"
    }

    # Strict Exposure Logic
    if is_stem:
        settings_kwargs["exposure"] = None
    else:
        settings_kwargs["exposure"] = Q_(float(exp_val), Units.MS)

    fr_val = _pop_any(["FrameRate", "frameRate"])
    if fr_val:
        try:
            if isinstance(fr_val, (int, float)):
                settings_kwargs["frame_rate"] = Q_(float(fr_val), Units.HZ)
            elif isinstance(fr_val, str) and fr_val.strip():
                clean_fr = fr_val.lower().replace("fps", "").replace("hz", "").strip()
                if clean_fr:
                    settings_kwargs["frame_rate"] = Q_(float(clean_fr), Units.HZ)
        except Exception:
            pass

    rot_val = _pop_any(["DigitalRotation", "RotationAngle"])
    if rot_val is not None:
        settings_kwargs["digital_rotation"] = Q_(float(rot_val), Units.DEG)

    if roi_data:
        settings_kwargs["roi"] = _jeol_roi_to_struct(roi_data)

    if bin_size_data:
        settings_kwargs["binning_xy"] = _jeol_bin_to_tuple(bin_size_data)

    _pop_any(["ExposureTimeIndex", "ExposureTimeString"])

    # --- 4. Extract Capabilities ---
    caps_kwargs = {
        "can_binning": bool(_pop_any(["CanBinning", "canBinning"], False)),
        "binning_index_min": _pop_any(["BinningIndexMinimum"]),
        "binning_index_max": _pop_any(["BinningIndexMaximum"]),

        "can_gain": bool(_pop_any(["CanGain", "canGain"], False)),
        "gain_index_min": _pop_any(["GainIndexMinimum"]),
        "gain_index_max": _pop_any(["GainIndexMaximum"]),

        "can_offset": bool(_pop_any(["CanOffset", "canOffset"], False)),
        "offset_index_min": _pop_any(["OffsetIndexMinimum"]),
        "offset_index_max": _pop_any(["OffsetIndexMaximum"]),

        "_mode": "lenient"
    }

    # Exposure Limits (Only for Camera)
    exp_min = _pop_any(["ExposureTimeMinimum", "ExposureTimeMin", "ExposureTimeIndexMinimum"])
    exp_max = _pop_any(["ExposureTimeMaximum", "ExposureTimeMax", "ExposureTimeIndexMaximum"])
    if not is_stem:
        if exp_min is not None:
            try:
                caps_kwargs["exposure_min"] = Q_(float(exp_min), Units.MS)
            except Exception:
                pass
        if exp_max is not None:
            try:
                caps_kwargs["exposure_max"] = Q_(float(exp_max), Units.MS)
            except Exception:
                pass

    # Rotation Limits
    rot_min = _pop_any(["DigitalRotationMinimum", "RotationAngleMinimum"])
    if rot_min is not None:
        caps_kwargs["digital_rotation_min"] = Q_(float(rot_min), Units.DEG)
    rot_max = _pop_any(["DigitalRotationMaximum", "RotationAngleMaximum"])
    if rot_max is not None:
        caps_kwargs["digital_rotation_max"] = Q_(float(rot_max), Units.DEG)

    # Frame Integration Limits
    fi_min = _pop_any(["frameIntegrationMinimum", "FrameIntegrationMinimum"])
    if fi_min is not None:
        caps_kwargs["frame_integration_min"] = int(fi_min)
    fi_max = _pop_any(["frameIntegrationMaximum", "FrameIntegrationMaximum"])
    if fi_max is not None:
        caps_kwargs["frame_integration_max"] = int(fi_max)

    # ROI Limits (Always use global Maximum for capabilities)
    roi_max_raw = _pop_any(["ImagingAreaMaximum", "imagingAreaMaximum"])
    roi_size_max = _parse_jeol_roi_limit(roi_max_raw)
    if roi_size_max:
        caps_kwargs["roi_size_max"] = roi_size_max

    # Pack Leftovers
    settings_extra_dict = {"is_stem_detector": is_stem}
    if is_stem:
        settings_extra_dict["dwell_time_us"] = float(exp_val)
        settings_extra_dict["active_roi_source"] = roi_source_key

    caps_extra_dict = {}
    out_info = p.get("OutputImageInformation")
    if isinstance(out_info, dict):
        ppm = out_info.get("PixelsPerMeter")
        if isinstance(ppm, dict):
            hz = ppm.get("Horizontal")
            if hz and float(hz) > 0:
                settings_extra_dict["calculated_pixel_size_nm"] = 1e9 / float(hz)

    for k, v in list(p.items()):
        if any(x in k for x in ["Max", "Min", "Can", "Information"]):
            caps_extra_dict[k] = v
        else:
            settings_extra_dict[k] = v

    settings = DetectorSettings(**settings_kwargs)
    if settings_extra_dict:
        settings.extra = _pack_vendor_extras(settings_extra_dict)

    caps = DetectorCapabilities(**caps_kwargs)
    if caps_extra_dict:
        caps.extra = _pack_vendor_extras(caps_extra_dict)

    return settings, caps


def to_jeol_detector_config(settings: DetectorSettings) -> Dict[str, Any]:
    """Convert to dict for Detector.set_detectorsetting()."""
    out = {}

    if settings.binning_index is not None:
        out["BinningIndex"] = int(settings.binning_index)

    if settings.exposure is not None:
        out["ExposureTimeValue"] = float(settings.exposure.to(Units.MS).magnitude)

    if settings.frame_integration is not None:
        out["frameIntegration"] = int(settings.frame_integration)

    if settings.gain_index is not None:
        out["GainIndex"] = int(settings.gain_index)

    if settings.offset_index is not None:
        out["OffsetIndex"] = int(settings.offset_index)

    if settings.digital_rotation is not None:
        out["DigitalRotation"] = float(settings.digital_rotation.to(Units.DEG).magnitude)

    # NOTE: ROI is not mapped here because it requires context (ScanMode)
    # The 'jeol_microscope' layer handles the key selection ('ImagingArea' vs 'AreaModeImagingArea')

    if settings.extra and settings.extra.vendor:
        jeol_extras = settings.extra.vendor.get("JEOL", {})
        out.update(jeol_extras)

    return out


# =============================================================================
# 2. Stage Adapters
# =============================================================================

def from_jeol_stage_position(pos_list: List[float], extra_flags: Optional[Dict] = None) -> StagePosition:
    """
    Convert PyJEM stage list [x, y, z, tx, ty] to StagePosition.

    Args:
        pos_list: [x, y, z, tx, ty] in nm/degrees.
        extra_flags: Optional dict of raw status codes (e.g. from GetStatus).
    """
    if not pos_list or len(pos_list) < 5:
        return StagePosition(_mode="lenient", extra=_pack_vendor_extras({"raw_input": pos_list}))

    # Map main axes
    sp = StagePosition(
        x=Q_(float(pos_list[0]), Units.NM),
        y=Q_(float(pos_list[1]), Units.NM),
        z=Q_(float(pos_list[2]), Units.NM),
        tilt_x=Q_(float(pos_list[3]), Units.DEG),
        tilt_y=Q_(float(pos_list[4]), Units.DEG),
        coordinate_system="raw_hardware",
        _mode="strict"
    )

    # If there are extra status flags (e.g. limit switch hits), pack them
    if extra_flags:
        sp.extra = _pack_vendor_extras(extra_flags)

    return sp

def to_jeol_stage_args(pos: StagePosition) -> Dict[str, float]:
    """Convert StagePosition to a flat dict of base units (nm, deg)."""
    out = {}
    if pos.x is not None: out['x'] = pos.x.to(Units.NM).magnitude
    if pos.y is not None: out['y'] = pos.y.to(Units.NM).magnitude
    if pos.z is not None: out['z'] = pos.z.to(Units.NM).magnitude
    if pos.tilt_x is not None: out['tx'] = pos.tilt_x.to(Units.DEG).magnitude
    if pos.tilt_y is not None: out['ty'] = pos.tilt_y.to(Units.DEG).magnitude
    if pos.r is not None: out['r'] = pos.r.to(Units.DEG).magnitude
    return out