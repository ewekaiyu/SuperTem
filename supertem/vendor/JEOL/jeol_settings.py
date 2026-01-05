import copy
from typing import Any, Dict, Optional, Tuple

from supertem.structures.base import TemDetectorSettings, ROI, DetectorCapabilities  # adjust import

JEOL_SETTING_KEYS = {
    "ImagingArea", "BinningSize",
    "ExposureTimeValue", "ExposureTimeIndex", "ExposureTimeString",
    "BinningIndex", "frameIntegration",
    "GainIndex", "OffsetIndex",
    "DigitalRotation",
}

# Keep capabilities as “extra” unless you really need them as typed fields
JEOL_CAPABILITIES = {
    "BinningIndexMaximum", "BinningIndexMinimum",
    "CanBinning", "CanGain", "CanOffset",
    "ExposureTimeIndexMaximum", "ExposureTimeIndexMinimum",
    "frameIntegrationMaximum", "frameIntegrationMinimum",
    "GainIndexMaximum", "GainIndexMinimum",
    "ImagingAreaMaximum", "ImagingAreaMinimum",
    "OffsetIndexMaximum", "OffsetIndexMinimum",
    "OutputImageInformation"
}

def _pop(d: Dict[str, Any], key: str) -> Any:
    return d.pop(key, None)

def _to_int(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except Exception:
        return None

def _to_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def _roi_from_dict(x: Any) -> Optional[ROI]:
    if not isinstance(x, dict):
        return None
    return ROI(
        x=int(x.get("X", 0)),
        y=int(x.get("Y", 0)),
        width=int(x.get("Width", 0)),
        height=int(x.get("Height", 0)),
    )

def _bin_xy_from_dict(x: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(x, dict):
        return None
    return (int(x.get("Width", 1)), int(x.get("Height", 1)))

def from_jeol_detector_setting(payload: Dict[str, Any], detector_id: Optional[str] = None) -> TemDetectorSettings:
    p = copy.deepcopy(payload)

    # pull settings
    roi_obj = _roi_from_dict(_pop(p, "ImagingArea"))
    binning_xy = _bin_xy_from_dict(_pop(p, "BinningSize"))

    exposure_val = _pop(p, "ExposureTimeValue")
    exposure_idx = _pop(p, "ExposureTimeIndex")
    _pop(p, "ExposureTimeString")

    settings = TemDetectorSettings(
        detector_id=detector_id,
        exposure_ms=_to_float(exposure_val) if exposure_val is not None else _to_float(exposure_idx),
        binning_index=_to_int(_pop(p, "BinningIndex")),
        binning_xy=binning_xy,
        roi=roi_obj,
        frame_integration=_to_int(_pop(p, "frameIntegration")),
        gain_index=_to_int(_pop(p, "GainIndex")),
        offset_index=_to_int(_pop(p, "OffsetIndex")),
        digital_rotation_deg=_to_float(_pop(p, "DigitalRotation")),
    )

    # split remaining fields into capabilities-ish vs misc extra
    cap_extra: Dict[str, Any] = {}
    misc_extra: Dict[str, Any] = {}

    for k, v in p.items():
        if any(h in k for h in JEOL_CAPABILITIES):
            cap_extra[k] = v
        else:
            misc_extra[k] = v

    settings.capabilities = DetectorCapabilities(extra=cap_extra) if cap_extra else None
    settings.extra = misc_extra
    return settings

def to_jeol_detector_setting(s: TemDetectorSettings, include_capabilities: bool = False, include_extra: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if s.binning_index is not None:
        out["BinningIndex"] = int(s.binning_index)

    if s.exposure_ms is not None:
        ms = float(s.exposure_ms)
        out["ExposureTimeValue"] = ms
        out["ExposureTimeIndex"] = int(round(ms))
        out["ExposureTimeString"] = str(round(ms)) + " msec"

    if s.frame_integration is not None:
        out["frameIntegration"] = int(s.frame_integration)

    if s.gain_index is not None:
        out["GainIndex"] = int(s.gain_index)

    if s.offset_index is not None:
        out["OffsetIndex"] = int(s.offset_index)

    if s.digital_rotation_deg is not None:
        out["DigitalRotation"] = float(s.digital_rotation_deg)

    if s.roi is not None:
        out["ImagingArea"] = {
            "X": int(s.roi.x),
            "Y": int(s.roi.y),
            "Width": int(s.roi.width),
            "Height": int(s.roi.height),
        }

    if include_capabilities:
        if s.capabilities.extra is not None:
            out.update(s.capabilities.extra)

    if include_extra:
        out.update(s.extra)

    return out
