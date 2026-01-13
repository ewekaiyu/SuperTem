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


# =============================================================================
# 1. Detector Adapters
# =============================================================================

def from_jeol_detector_response(payload: Dict[str, Any], detector_id: str) -> Tuple[DetectorSettings, DetectorCapabilities]:
    """
    Parses dictionary from Detector.get_detectorsetting().
    """
    p = copy.deepcopy(payload)

    # 1. Robust Key Extraction (Handles PascalCase and camelCase variations)
    def _pop_any(keys, default=None):
        for k in keys:
            if k in p: return p.pop(k)
        return default

    # Extract Settings
    roi_data = _pop_any(["ImagingArea", "imagingArea"])
    bin_size_data = _pop_any(["BinningSize", "binningSize"])

    # Exposure key is notoriously inconsistent across versions
    exp_val = _pop_any(["ExposureTimeValue", "Exposure", "ExposureTime", "exposureTime"], 0.0)

    settings_kwargs = {
        "detector_id": detector_id,
        "binning_index": _pop_any(["BinningIndex", "binningIndex"]),
        "frame_integration": _pop_any(["frameIntegration", "AccumulationCount", "accumulationCount"]),
        "gain_index": _pop_any(["GainIndex", "gainIndex"]),
        "offset_index": _pop_any(["OffsetIndex", "offsetIndex"]),
        "exposure": Q_(float(exp_val), Units.MS), # Assuming MS based on typical PyJEM
        "_mode": "lenient"
    }

    rot_val = _pop_any(["DigitalRotation", "RotationAngle"])
    if rot_val is not None:
        settings_kwargs["digital_rotation"] = Q_(float(rot_val), Units.DEG)

    if roi_data:
        settings_kwargs["roi"] = _jeol_roi_to_struct(roi_data)

    if bin_size_data:
        settings_kwargs["binning_xy"] = _jeol_bin_to_tuple(bin_size_data)

    # Cleanup redundant keys often returned by hardware
    _pop_any(["ExposureTimeIndex", "ExposureTimeString"])

    # Extract Capabilities
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

    roi_max_dict = _pop_any(["ImagingAreaMaximum", "imagingAreaMaximum"], {})
    if roi_max_dict:
        caps_kwargs["roi_size_max"] = (int(roi_max_dict.get("Width", 0)), int(roi_max_dict.get("Height", 0)))

    # Pack leftovers into Extras
    caps_extra_dict = {}
    settings_extra_dict = {}

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
    """
    Convert to dict for Detector.set_detectorsetting().
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
    return out


# =============================================================================
# 3. Vacuum Adapters
# =============================================================================

def from_jeol_vacuum_stats(
    p_values: List[float],
    valve_status_flags: Optional[Dict[str, int]] = None,
) -> VacuumSettings:
    """
    Maps P1-P5 from vacuum3.py to semantic pressure fields.
    """
    p = list(p_values) + [0.0] * (5 - len(p_values))

    valves_mapped = {}
    if valve_status_flags:
        for k, v in valve_status_flags.items():
            valves_mapped[k] = "OPEN" if v == 1 else "CLOSED"

    return VacuumSettings(
        gun_pressure=Q_(p[0], Units.PA),      # P1
        column_pressure=Q_(p[1], Units.PA),   # P2
        # 'chamber_pressure' is not in base.py VacuumSettings, mapping P3 to extra or dropping?
        # base.py has: column, gun, buffer_tank.
        # JEOL P3 is usually Chamber. P4/P5 vary.
        # We will map P4 to buffer based on typical configs, or leave P3 in extras.
        buffer_tank_pressure=Q_(p[3], Units.PA), # Attempt mapping P4 to buffer

        column_valve_state=valves_mapped.get("V4") or valves_mapped.get("V7"), # Heuristic
        gun_valve_state=valves_mapped.get("V1"),

        extra=_pack_vendor_extras({
            "raw_pressures_P1_to_P5": p_values,
            "raw_valves": valve_status_flags,
            "chamber_pressure_P3_Pa": p[2]
        }),
        _mode="lenient"
    )


# =============================================================================
# 4. Beam Adapters
# =============================================================================

def from_jeol_beam_stats(
    voltage_val: float,
    current_ua: float,
    spot_size_idx: int,
    alpha_idx: int,
    beam_shift_dac: Optional[Tuple[int, int]] = None,
    raw_flags: Optional[Dict[str, Any]] = None
) -> BeamSettings:

    # Heuristic: PyJEM might return V or kV.
    # Offline ht3.py typically suggests V, but safe to check magnitude.
    if voltage_val > 5000:
        v_qty = Q_(voltage_val, "V").to(Units.KV)
    else:
        v_qty = Q_(voltage_val, Units.KV)

    shift_pt = None
    if beam_shift_dac:
        shift_pt = Point(x=float(beam_shift_dac[0]), y=float(beam_shift_dac[1]))

    vendor_data = {
        "raw_voltage": voltage_val,
        "raw_current_ua": current_ua,
        "spot_size_index": spot_size_idx,
        "alpha_index": alpha_idx,
        "beam_shift_dac": beam_shift_dac
    }
    if raw_flags:
        vendor_data.update(raw_flags)

    return BeamSettings(
        voltage=v_qty,
        beam_current=Q_(current_ua, Units.UA).to(Units.NA),
        spot_size=spot_size_idx,
        # COMPLIANCE FIX: base.py BeamSettings uses 'convergence_angle' (Quantity).
        # We cannot map an index (alpha_idx) to a Quantity without a table.
        # We store the index in extras and leave the quantity None.
        convergence_angle=None,
        beam_shift=shift_pt,
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
    Adapter for scan3.py.
    Scan3 only provides Rotation, MagCorrection, and Mode.
    """
    vendor_data = {}
    if mag_correction:
        vendor_data["MagCorrection"] = mag_correction
    if scan_mode_int is not None:
        vendor_data["ScanModeInt"] = scan_mode_int

    return ScanSettings(
        scan_rotation=Q_(rotation_deg, Units.DEG),
        scan_mode=str(scan_mode_int) if scan_mode_int is not None else None,
        # Fields not available in scan3.py:
        width_px=None,
        height_px=None,
        pixel_dwell_time=None,
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