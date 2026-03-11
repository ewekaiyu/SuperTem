"""
supertem.structures.jeol_structures

JEOL-specific hardware state and intent payloads.
These strictly typed classes ride inside the extra.vendor["JEOL"] dictionaries
of the canonical SuperTEM structures.

They use ParserExtras to prevent infinite recursion during serialization.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Tuple

# Import the core logic from the base file
from supertem.structures.base_structures import (
    ParseMode, FieldParser, Validator, ParserExtras, _auto_to_dict, _auto_from_dict, Quantity, Units
)

@dataclass
class JeolBeamExtras:
    """
    JEOL-specific hardware state/intent for the Illumination system.
    Rides inside BeamSettings.extra.vendor["JEOL"].
    """
    # Optics Controls
    alpha_index: Optional[int] = None
    brightness_value: Optional[int] = None
    condenser_lens_1: Optional[int] = None
    condenser_lens_2: Optional[int] = None
    condenser_lens_3: Optional[int] = None

    # Alignment Coils (Must be floats to preserve PyJEM telemetry precision)
    spot_alignment: Optional[Tuple[float, float]] = None
    condenser_alignment_1: Optional[Tuple[float, float]] = None
    condenser_alignment_2: Optional[Tuple[float, float]] = None
    gun_alignment_1: Optional[Tuple[float, float]] = None
    gun_alignment_2: Optional[Tuple[float, float]] = None

    # Source Control & Diagnostics
    feg_emission_state: Optional[str] = None
    gun_type_index: Optional[int] = None
    mds_mode: Optional[str] = None

    # Quantities require _UNITS mapping for JSON serialization
    gun_anode_1: Optional[Quantity] = None
    gun_anode_2: Optional[Quantity] = None
    gun_bias: Optional[Quantity] = None
    gun_filament: Optional[Quantity] = None

    # Recursion-Safe Data Plane Extra Bucket
    extra: ParserExtras = field(default_factory=ParserExtras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    # --- SERIALIZATION CONFIGURATION ---
    _UNITS = {
        "gun_anode_1": Units.KV,
        "gun_anode_2": Units.KV,
        "gun_bias": Units.UA,
        "gun_filament": "A"
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, self.__class__.__name__)
        self.alpha_index = p.int(self.alpha_index, "alpha_index")
        self.brightness_value = p.int(self.brightness_value, "brightness_value")
        self.condenser_lens_1 = p.int(self.condenser_lens_1, "condenser_lens_1")
        self.condenser_lens_2 = p.int(self.condenser_lens_2, "condenser_lens_2")
        self.condenser_lens_3 = p.int(self.condenser_lens_3, "condenser_lens_3")

        # FIX: Coils parsed as floats
        self.spot_alignment = p.pair_float(self.spot_alignment, "spot_alignment")
        self.condenser_alignment_1 = p.pair_float(self.condenser_alignment_1, "condenser_alignment_1")
        self.condenser_alignment_2 = p.pair_float(self.condenser_alignment_2, "condenser_alignment_2")
        self.gun_alignment_1 = p.pair_float(self.gun_alignment_1, "gun_alignment_1")
        self.gun_alignment_2 = p.pair_float(self.gun_alignment_2, "gun_alignment_2")

        self.feg_emission_state = p.str(self.feg_emission_state, "feg_emission_state")
        self.gun_type_index = p.int(self.gun_type_index, "gun_type_index")
        self.mds_mode = p.str(self.mds_mode, "mds_mode")

        self.gun_anode_1 = p.qty(self.gun_anode_1, "gun_anode_1", Units.KV)
        self.gun_anode_2 = p.qty(self.gun_anode_2, "gun_anode_2", Units.KV)
        self.gun_bias = p.qty(self.gun_bias, "gun_bias", Units.UA)
        self.gun_filament = p.qty(self.gun_filament, "gun_filament", "A")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        # JEOL hardware limits checking
        v.check(self.alpha_index is None or (0 <= self.alpha_index <= 8),
                "alpha_index", "must be between 0 and 8",
                heal=lambda: setattr(self, 'alpha_index', None))

        v.check(self.brightness_value is None or (0 <= self.brightness_value <= 65535),
                "brightness_value", "must be between 0 and 65535",
                heal=lambda: setattr(self, 'brightness_value', None))

        if self.feg_emission_state:
             v.check(self.feg_emission_state.upper() in {"ON", "OFF", "TRUE", "FALSE"}, "feg_emission_state",
                    "Must be ON or OFF", heal=lambda: setattr(self, 'feg_emission_state', None))

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "JeolBeamExtras":
        return _auto_from_dict(JeolBeamExtras, d, mode, alias_map={
            "gun_anode_1": "gun_anode_1_kv",
            "gun_anode_2": "gun_anode_2_kv",
            "gun_bias": "gun_bias_ua",
            "gun_filament": "gun_filament_a"
        })


@dataclass
class JeolProjectionExtras:
    """
    JEOL-specific hardware state/intent for the Imaging system.
    Rides inside ProjectionSettings.extra.vendor["JEOL"].
    """
    objective_lens_coarse: Optional[int] = None
    objective_lens_fine: Optional[int] = None
    objective_lens_superfine: Optional[int] = None
    objective_mini_lens_1: Optional[int] = None
    objective_mini_lens_2: Optional[int] = None
    focus_lens_coarse: Optional[int] = None
    focus_lens_fine: Optional[int] = None
    intermediate_lens_1: Optional[int] = None
    intermediate_lens_2: Optional[int] = None
    intermediate_lens_3: Optional[int] = None
    intermediate_lens_4: Optional[int] = None
    projector_lens_1: Optional[int] = None
    projector_lens_2: Optional[int] = None
    projector_lens_3: Optional[int] = None

    image_shift_2: Optional[Tuple[float, float]] = None # FIX: Coils parsed as floats
    diffraction_focus_index: Optional[int] = None

    extra: ParserExtras = field(default_factory=ParserExtras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, self.__class__.__name__)
        # Simplified parsing for repetitive lens fields
        for lens in [
            "objective_lens_coarse", "objective_lens_fine", "objective_lens_superfine",
            "objective_mini_lens_1", "objective_mini_lens_2",
            "focus_lens_coarse", "focus_lens_fine",
            "intermediate_lens_1", "intermediate_lens_2", "intermediate_lens_3", "intermediate_lens_4",
            "projector_lens_1", "projector_lens_2", "projector_lens_3",
            "diffraction_focus_index"
        ]:
            setattr(self, lens, p.int(getattr(self, lens), lens))

        self.image_shift_2 = p.pair_float(self.image_shift_2, "image_shift_2")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        return v.valid

    def to_dict(self) -> dict:
        # No Quantities here, so no unit_map needed
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "JeolProjectionExtras":
        return _auto_from_dict(JeolProjectionExtras, d, mode)


@dataclass
class JeolDetectorExtras:
    """
    JEOL-specific camera parameters.
    Populated by jeol_adapter.from_jeol_detector_response.
    Rides inside DetectorSettings.extra.vendor["JEOL"].
    """
    is_stem_detector: Optional[bool] = None

    dwell_time_us: Optional[float] = None
    active_roi_source: Optional[str] = None
    calculated_pixel_size_nm: Optional[float] = None

    extra: ParserExtras = field(default_factory=ParserExtras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, self.__class__.__name__)
        self.is_stem_detector = p.bool(self.is_stem_detector, "is_stem_detector")
        self.dwell_time_us = p.float(self.dwell_time_us, "dwell_time_us")
        self.active_roi_source = p.str(self.active_roi_source, "active_roi_source")
        self.calculated_pixel_size_nm = p.float(self.calculated_pixel_size_nm, "calculated_pixel_size_nm")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "JeolDetectorExtras":
        return _auto_from_dict(JeolDetectorExtras, d, mode)

# --- NEW: Stage Extras ---
@dataclass
class JeolStageExtras:
    holder_status: Optional[str] = None
    piezo_offset_x: Optional[Quantity] = None
    piezo_offset_y: Optional[Quantity] = None
    axis_status: Optional[dict] = None
    speed_mode: Optional[dict] = None

    extra: ParserExtras = field(default_factory=ParserExtras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "piezo_offset_x": Units.NM,
        "piezo_offset_y": Units.NM
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, self.__class__.__name__)
        self.holder_status = p.str(self.holder_status, "holder_status")
        self.piezo_offset_x = p.qty(self.piezo_offset_x, "piezo_offset_x", Units.NM)
        self.piezo_offset_y = p.qty(self.piezo_offset_y, "piezo_offset_y", Units.NM)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool: return Validator(self, mode).valid
    def to_dict(self) -> dict: return _auto_to_dict(self, unit_map=self._UNITS)
    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "JeolStageExtras":
        return _auto_from_dict(JeolStageExtras, d, mode, alias_map={
            "piezo_offset_x": "piezo_offset_x_nm", "piezo_offset_y": "piezo_offset_y_nm"
        })

# --- NEW: Vacuum Extras ---
@dataclass
class JeolVacuumExtras:
    column_ready_state: Optional[str] = None
    camera_ready_state: Optional[str] = None
    specimen_ready_state: Optional[str] = None
    column_air_state: Optional[str] = None
    camera_air_state: Optional[str] = None
    specimen_air_state: Optional[str] = None
    specimen_pre_evac_state: Optional[str] = None
    column_rough_pressure: Optional[Quantity] = None
    specimen_chamber_pressure: Optional[Quantity] = None
    detector_chamber_pressure: Optional[Quantity] = None

    extra: ParserExtras = field(default_factory=ParserExtras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "column_rough_pressure": Units.UA,
        "specimen_chamber_pressure": Units.UA,
        "detector_chamber_pressure": Units.UA
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, self.__class__.__name__)
        self.column_ready_state = p.str(self.column_ready_state, "column_ready_state")
        self.camera_ready_state = p.str(self.camera_ready_state, "camera_ready_state")
        self.specimen_ready_state = p.str(self.specimen_ready_state, "specimen_ready_state")
        self.column_air_state = p.str(self.column_air_state, "column_air_state")
        self.camera_air_state = p.str(self.camera_air_state, "camera_air_state")
        self.specimen_air_state = p.str(self.specimen_air_state, "specimen_air_state")
        self.specimen_pre_evac_state = p.str(self.specimen_pre_evac_state, "specimen_pre_evac_state")
        self.column_rough_pressure = p.qty(self.column_rough_pressure, "column_rough_pressure", Units.UA)
        self.specimen_chamber_pressure = p.qty(self.specimen_chamber_pressure, "specimen_chamber_pressure", Units.UA)
        self.detector_chamber_pressure = p.qty(self.detector_chamber_pressure, "detector_chamber_pressure", Units.UA)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool: return Validator(self, mode).valid
    def to_dict(self) -> dict: return _auto_to_dict(self)
    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "JeolVacuumExtras":
        return _auto_from_dict(JeolVacuumExtras, d, mode, alias_map={
            "column_rough_pressure": "column_rough_pressure_ua",
            "specimen_chamber_pressure": "specimen_chamber_pressure_ua",
            "detector_chamber_pressure": "detector_chamber_pressure_ua"
        })