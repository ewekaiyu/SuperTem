"""
supertem.vendor.JEOL.jeol_adapter

The Data Translation Layer for JEOL PyJEM.

This module is responsible for converting between:
1. Raw Python types returned by PyJEM (dicts, lists, int codes).
2. Strictly typed SuperTEM structures.

It isolates the 'Dirty' logic of parsing vendor keys/units from the 'Clean' logic of the microscope driver.
All unmapped data is preserved in the `extra.vendor` dictionary of the respective objects.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from supertem.structures.base import (
    DetectorSettings,
    DetectorCapabilities,
    ROI,
    StagePosition,
    Aperture,
    VacuumSettings,
    BeamSettings,
    ScanSettings,
    Extras,
    Q_,
    Units,
    Point
)

# =============================================================================
# 0. Helpers
# =============================================================================

def _pack_vendor_extras(data: Dict[str, Any]) -> Optional[Extras]:
    """Wraps a dictionary of raw vendor data into the standard Extras.vendor structure."""
    if not data:
        return None
    return Extras(vendor=data)

def _jeol_roi_to_struct(x: Any) -> Optional[ROI]:
    """Convert JEOL 'ImagingArea' dict to ROI object."""
    if not isinstance(x, dict):
        return None
    return ROI(
        x=int(x.get("X", 0)),
        y=int(x.get("Y", 0)),
        width=int(x.get("Width", 512)),
        height=int(x.get("Height", 512)),
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


# =============================================================================
# 1. Detector Adapters
# =============================================================================

def from_jeol_detector_response(payload: Dict[str, Any], detector_id: str) -> Tuple[DetectorSettings, DetectorCapabilities]:
    """
    Splits a single PyJEM detector configuration dictionary into Settings and Capabilities.
    Unmapped keys are sorted into extras.vendor for the appropriate object.
    """
    # Work on a copy to consume keys
    p = copy.deepcopy(payload)

    # --- 1. Extract Settings Fields ---
    # We pop keys that map directly to DetectorSettings
    roi_data = p.pop("ImagingArea", None)
    bin_size_data = p.pop("BinningSize", None)

    settings_kwargs = {
        "detector_id": detector_id,
        "binning_index": p.pop("BinningIndex", None),
        "frame_integration": p.pop("frameIntegration", None),
        "gain_index": p.pop("GainIndex", None),
        "offset_index": p.pop("OffsetIndex", None),
        "exposure": Q_(float(p.pop("ExposureTimeValue", 0.0)), Units.MS),
        "_mode": "lenient"
    }

    # Optional fields
    if "DigitalRotation" in p:
        settings_kwargs["digital_rotation"] = Q_(float(p.pop("DigitalRotation")), Units.DEG)

    if roi_data:
        settings_kwargs["roi"] = _jeol_roi_to_struct(roi_data)

    if bin_size_data:
        settings_kwargs["binning_xy"] = _jeol_bin_to_tuple(bin_size_data)

    # Remove redundant setting representations we don't need in 'vendor' extras
    p.pop("ExposureTimeIndex", None)
    p.pop("ExposureTimeString", None)

    # --- 2. Extract Capability Fields ---
    # We pop keys that map directly to DetectorCapabilities
    caps_kwargs = {
        "can_binning": bool(p.pop("CanBinning", False)),
        "binning_index_min": p.pop("BinningIndexMinimum", None),
        "binning_index_max": p.pop("BinningIndexMaximum", None),

        "can_gain": bool(p.pop("CanGain", False)),
        "gain_index_min": p.pop("GainIndexMinimum", None),
        "gain_index_max": p.pop("GainIndexMaximum", None),

        "can_offset": bool(p.pop("CanOffset", False)),
        "offset_index_min": p.pop("OffsetIndexMinimum", None),
        "offset_index_max": p.pop("OffsetIndexMaximum", None),

        "_mode": "lenient"
    }

    # Handle ROI Max
    roi_max_dict = p.pop("ImagingAreaMaximum", {})
    if roi_max_dict:
        caps_kwargs["roi_size_max"] = (int(roi_max_dict.get("Width", 0)), int(roi_max_dict.get("Height", 0)))

    # --- 3. Handle Extras ---
    # The remaining keys in `p` are unknown vendor-specific flags.
    # We heuristically split them: static "Max/Min" props go to Caps, others to Settings.

    caps_extra_dict = {}
    settings_extra_dict = {}

    for k, v in list(p.items()):
        if any(x in k for x in ["Max", "Min", "Can", "Information"]):
            caps_extra_dict[k] = v
        else:
            settings_extra_dict[k] = v

    # Construct Objects
    settings = DetectorSettings(**settings_kwargs)
    if settings_extra_dict:
        settings.extra = _pack_vendor_extras(settings_extra_dict)

    caps = DetectorCapabilities(**caps_kwargs)
    if caps_extra_dict:
        caps.extra = _pack_vendor_extras(caps_extra_dict)

    return settings, caps


def to_jeol_detector_config(settings: DetectorSettings) -> Dict[str, Any]:
    """
    Convert DetectorSettings back into a PyJEM-compatible dictionary.
    Includes any vendor-specific keys preserved in settings.extra.vendor.
    """
    out = {}

    # 1. Standard Fields
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

    if settings.roi is not None:
        out["ImagingArea"] = _struct_to_jeol_roi(settings.roi)

    # 2. Vendor Extras (Pass-through)
    if settings.extra and settings.extra.vendor:
        # We merge these back in so PyJEM receives keys we didn't touch
        out.update(settings.extra.vendor)

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
    return out


# =============================================================================
# 3. Vacuum Adapters
# =============================================================================

def from_jeol_vacuum_stats(
    p_values: List[float],
    valve_status_flags: Optional[Dict[str, int]] = None,
    raw_status_array: Optional[List[int]] = None
) -> VacuumSettings:
    """
    Convert raw vacuum readings to VacuumSettings.
    """
    # Pad list if short
    p = list(p_values) + [0.0] * (5 - len(p_values))

    # Map Valves
    valves_mapped = {}
    if valve_status_flags:
        for k, v in valve_status_flags.items():
            valves_mapped[k] = "OPEN" if v == 1 else "CLOSED"

    # Store raw inputs in vendor extras
    vendor_data = {
        "raw_pressures_pascals": p_values,
        "raw_valve_flags": valve_status_flags,
    }
    if raw_status_array:
        vendor_data["raw_status_array"] = raw_status_array

    return VacuumSettings(
        gun_pressure=Q_(p[0], Units.PA),
        column_pressure=Q_(p[1], Units.PA),
        chamber_pressure=Q_(p[2], Units.PA),
        camera_chamber_pressure=Q_(p[3], Units.PA),
        valves=valves_mapped,
        extra=_pack_vendor_extras(vendor_data),
        _mode="lenient"
    )


# =============================================================================
# 4. Beam Adapters
# =============================================================================

def from_jeol_beam_stats(
    voltage_v: float,
    current_ua: float,
    spot_size_idx: int,
    alpha_idx: int,
    beam_shift_dac: Optional[Tuple[int, int]] = None,
    raw_flags: Optional[Dict[str, Any]] = None
) -> BeamSettings:
    """
    Consolidate atomic beam stats into BeamSettings.
    """
    shift_pt = None
    if beam_shift_dac:
        shift_pt = Point(x=float(beam_shift_dac[0]), y=float(beam_shift_dac[1]))

    # Capture raw DACs and indices in extras
    vendor_data = {
        "raw_voltage_v": voltage_v,
        "raw_current_ua": current_ua,
        "spot_size_index": spot_size_idx,
        "alpha_index": alpha_idx,
        "beam_shift_dac": beam_shift_dac
    }
    if raw_flags:
        vendor_data.update(raw_flags)

    return BeamSettings(
        beam_on=True, # Inferred, or passed in raw_flags
        voltage=Q_(voltage_v, Units.V).to(Units.KV),
        current=Q_(current_ua, Units.UA).to(Units.NA),
        spot_size_index=spot_size_idx,
        convergence_angle_index=alpha_idx,
        shift=shift_pt,
        extra=_pack_vendor_extras(vendor_data),
        _mode="lenient"
    )


# =============================================================================
# 5. Scan Adapters
# =============================================================================

def from_jeol_scan_stats(
    rotation_deg: float,
    mag_correction: Optional[Tuple[float, float]] = None,
    scan_mode_int: Optional[int] = None
) -> ScanSettings:
    """
    Convert Scan3 return values to ScanSettings.
    """
    vendor_data = {}
    if mag_correction:
        vendor_data["MagCorrection"] = mag_correction
    if scan_mode_int is not None:
        vendor_data["ScanModeInt"] = scan_mode_int

    return ScanSettings(
        rotation=Q_(rotation_deg, Units.DEG),
        active=False,
        extra=_pack_vendor_extras(vendor_data) if vendor_data else None,
        _mode="lenient"
    )


# =============================================================================
# 6. Aperture Adapters
# =============================================================================

def from_jeol_aperture(
    aperture_id: str,
    size_index: int,
    pos_xy: List[int]
) -> Aperture:
    """
    Combine separate JEOL calls (GetExpSize, GetPosition) into an Aperture object.
    """
    is_inserted = (size_index > 0)

    point = None
    if pos_xy and len(pos_xy) >= 2:
        point = Point(x=float(pos_xy[0]), y=float(pos_xy[1]))

    vendor_data = {
        "raw_size_index": size_index,
        "raw_pos_dac": pos_xy
    }

    return Aperture(
        aperture_id=aperture_id,
        inserted=is_inserted,
        size_index=size_index,
        position=point,
        extra=_pack_vendor_extras(vendor_data),
        _mode="lenient"
    )