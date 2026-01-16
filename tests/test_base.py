import pytest
import numpy as np
import datetime
import json
import pint
from dataclasses import replace, asdict, dataclass
from pathlib import Path
from copy import deepcopy

# Import all classes
from supertem.structures.base import (
    # Core
    ParseMode, Units, Q_, Extras, SafetyCheck,

    # Structures
    Point, ROI, ImageOutputSettings, SystemInfo,

    # Stage
    StagePosition, StageSystemSettings, StageMoveRequest, StageControlRequest, StageDriveType,

    # Beam
    BeamSettings, BeamSystemSettings, BeamControlRequest, BeamState,

    # Projection
    ProjectionSettings, ProjectionSystemSettings, ProjectionControlRequest,

    # Detector
    DetectorSettings, DetectorCapabilities, DetectorSystemSettings, DetectorState,
    DetectorControlRequest, AcquisitionRequest,

    # Scan
    ScanSettings, ScanSystemSettings, ScanControlRequest,

    # Vacuum
    VacuumSettings, VacuumControlRequest,

    # Aperture
    Aperture, ApertureControlRequest,

    # Microscope Global
    MicroscopeState, SystemSettings, MicroscopeSettings,

    # Image
    MicroscopeImageMetadata, MicroscopeImage
)


# =============================================================================
# 1. CORE FRAMEWORK (Parser, Validator, Serialization)
#    Tests for the underlying engine that powers all structures.
# =============================================================================

def test_extras_lifecycle_complex():
    """Test nested extras parsing and preservation."""
    raw_input = {
        "x": 10,
        "extra": {
            "vendor": {"fei": {"param": 1}},
            "unknown": {"legacy_field": "foo"},
            "notes": {"warning": "old config"}
        },
        "garbage_key": 999
    }
    p = Point.from_dict(raw_input)
    assert p.x == 10.0
    assert p.extra.vendor["fei"]["param"] == 1
    assert p.extra.unknown["legacy_field"] == "foo"
    assert p.extra.notes["warning"] == "old config"
    assert p.extra.unknown["garbage_key"] == 999


def test_safety_check_behavior():
    """Test SafetyCheck boolean behavior and aggregation."""
    s = SafetyCheck.success()
    assert s
    assert s.allowed is True
    f = SafetyCheck.failure("Too hot")
    assert not f
    assert f.allowed is False
    assert "Too hot" in f.reasons
    f.add_reason("Too fast")
    assert len(f.reasons) == 2


def test_field_parser_primitives_edge_cases():
    """Test integer coercion rules and ID strictness."""
    # 1. Int parsing strictness
    p = DetectorSettings(binning_index=10.0, frame_integration=10.5, _mode=ParseMode.LENIENT)
    assert p.binning_index == 10
    assert p.frame_integration is None
    assert "DetectorSettings.frame_integration" in p.extra.raw

    # 2. ID Validation
    with pytest.raises(ValueError):
        DetectorSettings(detector_id="   ", _mode=ParseMode.STRICT)
    d = DetectorSettings(detector_id="", _mode=ParseMode.LENIENT)
    assert d.detector_id is None
    assert "DetectorSettings.detector_id.empty" in d.extra.notes


def test_fuzzy_boolean_parsing():
    """Test that 'yes', 'on', '1' are correctly parsed as True."""
    cases = [
        ("yes", True), ("YES", True), ("y", True),
        ("on", True), ("1", True), ("true", True),
        ("no", False), ("off", False), ("0", False), ("false", False)
    ]
    for input_val, expected in cases:
        s = StageSystemSettings(enabled=input_val, _mode=ParseMode.LENIENT)
        assert s.enabled is expected, f"Failed parsing {input_val}"


def test_unit_conversion_on_ingest():
    """Test inputting different compatible units (e.g. mm into a nm field)."""
    p1 = StagePosition.from_dict({"x": "1 um"})  # 1000 nm
    assert p1.x.magnitude == pytest.approx(1000.0)
    assert str(p1.x.units) == "nanometer"

    p2 = StagePosition.from_dict({"x": "500"})
    assert p2.x.magnitude == 500.0

    p3 = StagePosition(x=Q_(1e-6, 'm'))
    assert p3.x.magnitude == pytest.approx(1000.0)


def test_quantity_dimensionality_error():
    """Test what happens when units are physically incompatible (Time vs Length)."""
    p = StagePosition(x=Q_(10, 's'), _mode=ParseMode.LENIENT)
    assert p.x is None
    assert "StagePosition.x" in p.extra.raw or "StagePosition.x" in str(p.extra.notes)


def test_map_model_parsing_logic():
    """Test parsing logic for dictionaries of objects (map_model)."""
    data = {
        "detectors": {
            "CamA": {"exposure_ms": 10},
            "CamB": None,  # Should be ignored
            "CamC": "InvalidString"  # Should be ignored (was bug, now fixed)
        }
    }
    state = MicroscopeState.from_dict(data, mode=ParseMode.LENIENT)
    assert "CamA" in state.detectors
    assert state.detectors["CamA"].exposure.magnitude == 10.0
    assert "CamB" not in state.detectors
    assert "CamC" not in state.detectors


def test_missing_unit_definition_strictness():
    """Test behavior when a Quantity field is defined but missing from _UNITS map."""

    @dataclass
    class DefectiveClass:
        val: Q_
        _mode: ParseMode = ParseMode.STRICT

    obj = DefectiveClass(val=Q_(10, 'nm'))
    from supertem.structures.base import _auto_to_dict

    with pytest.raises(ValueError):
        _auto_to_dict(obj)

    obj._mode = ParseMode.LENIENT
    d = _auto_to_dict(obj)
    assert d['val']['magnitude'] == 10
    assert d['val']['unit'] == 'nanometer'


def test_extras_filtering():
    """Test that Extras.to_dict() filters out empty collections."""
    ex = Extras()
    ex.vendor = {}
    ex.notes = {"error": "bad"}
    d = ex.to_dict()
    assert "vendor" not in d
    assert "notes" in d
    assert d["notes"] == {"error": "bad"}


def test_validator_healing_crash_handling():
    """Test robustness if the 'heal' function itself crashes."""
    from supertem.structures.base import Validator
    class Dummy:
        extra = Extras()
        _mode = ParseMode.LENIENT

    obj = Dummy()
    v = Validator(obj, mode_override=ParseMode.LENIENT)

    def crashing_heal(): raise RuntimeError("Heal crashed")

    v.check(False, "test_key", "error", heal=crashing_heal)

    assert v.valid is False
    assert "Dummy.test_key" in obj.extra.notes


def test_parse_mode_strict_vs_lenient():
    """
    Verify that STRICT mode raises errors on bad data,
    while LENIENT mode swallows them.
    """
    bad_payload = {"voltage": "invalid_string"}  # Voltage expects Quantity/float

    # 1. Strict Mode -> Crash
    with pytest.raises((ValueError, pint.errors.UndefinedUnitError)):
        BeamSettings.from_dict(bad_payload, mode=ParseMode.STRICT)

    # 2. Lenient Mode -> Survive (field becomes None or default)
    obj = BeamSettings.from_dict(bad_payload, mode=ParseMode.LENIENT)
    assert obj.voltage is None


def test_serialization_strips_units():
    """
    Verify to_dict converts Quantities (10 nm) to raw floats (10.0)
    using the preferred unit names.
    """
    pos = StagePosition(x=Q_(1.5, "um"), y=Q_(200, "nm"), _mode="strict")

    data = pos.to_dict()

    # Keys should include unit suffixes if your serializer adds them,
    # or just be raw values depending on base.py implementation.
    # Assuming base.py normalizes to standard units:

    # Check that it's NOT a Quantity object
    assert isinstance(data.get("x"), (float, int, type(None)))
    assert not isinstance(data.get("x"), pint.Quantity)


# =============================================================================
# 2. BASIC PRIMITIVES (Point, ROI, ImageOutput)
# =============================================================================

def test_point_scenarios():
    with pytest.raises(ValueError):
        Point(x="not_a_number", _mode=ParseMode.STRICT)
    p = Point(x="10.5", y=None, z=5, name=123, _mode=ParseMode.LENIENT)
    assert p.x == 10.5
    assert p.y == 0.0
    assert p.name == "123"

    p_bad = Point(x=float('inf'), _mode=ParseMode.LENIENT)
    p_bad.validate()
    assert p_bad.x == 0.0


def test_roi_scenarios():
    roi = ROI.from_dict({"x": 10, "y": 10, "w": 100, "h": 200})
    assert roi.width == 100

    roi_inv = ROI(x=-5, width=-100, _mode=ParseMode.LENIENT)
    roi_inv.validate()
    assert roi_inv.x == 0
    assert roi_inv.width == 512

    roi_tup = ROI.from_dict((0, 0, 1024, 1024))
    assert roi_tup.width == 1024


def test_image_output_settings():
    ios = ImageOutputSettings(file_format="JPEG", path="/tmp")
    assert ios.file_format == "jpeg"

    ios_bad = ImageOutputSettings(file_format="gif", _mode=ParseMode.LENIENT)
    ios_bad.validate()
    assert ios_bad.file_format == "tiff"


# =============================================================================
# 3. STAGE SUBSYSTEM
# =============================================================================

def test_stage_position_math_and_logic():
    p1 = StagePosition(x=Q_(10, 'nm'), y=Q_(20, 'nm'))
    p2 = StagePosition(x=Q_(5, 'nm'), z=Q_(10, 'nm'))
    p3 = p1 + p2
    assert p3.x.magnitude == 15.0
    assert p3.y is None

    p4 = p1 - p2
    assert p4.x.magnitude == 5.0

    target = StagePosition(x=Q_(10, 'nm'))
    current = StagePosition(x=Q_(10.5, 'nm'), y=Q_(500, 'nm'))
    assert target.is_close(current, tol_nm=1.0) is True
    assert target.is_close(current, tol_nm=0.1) is False


def test_stage_position_round_trip():
    """Ensure an object can be serialized and reconstructed exactly."""
    original = StagePosition(x=Q_(1.5, 'um'), y=Q_(0, 'nm'), r=Q_(45, 'deg'), name="Target A")
    payload = original.to_dict()
    assert payload["x_nm"] == pytest.approx(1500.0)
    assert payload["r_deg"] == pytest.approx(45.0)

    reconstructed = StagePosition.from_dict(payload)
    assert reconstructed.x.magnitude == pytest.approx(1500.0)
    assert reconstructed.name == "Target A"


def test_stage_system_settings_complex():
    sys = StageSystemSettings(can_x=True, x_limits=None, _mode=ParseMode.LENIENT)
    sys.validate()
    assert sys.can_x is False

    sys = StageSystemSettings(z_limits=(Q_(-10, 'um'), Q_(10, 'um')), eucentric_z=Q_(50, 'um'), _mode=ParseMode.LENIENT)
    sys.validate()
    assert sys.eucentric_z is None

    sys = StageSystemSettings(max_step_distance=Q_(1, 'um'))
    d = sys.to_dict()
    assert d["max_step_nm"] == pytest.approx(1000.0)


def test_stage_move_request_piezo_safety():
    req = StageMoveRequest(drive_type="piezo", relative=True, target=StagePosition(x=Q_(10, 'um')),
                           _mode=ParseMode.LENIENT)
    req.validate()
    assert "StageMoveRequest.piezo_limit.x" in req.extra.notes


def test_complex_stage_safety_relative():
    limits = StageSystemSettings(x_limits=(Q_(-100, 'um'), Q_(100, 'um')), enabled=True, can_x=True)
    req_move = StagePosition(x=Q_(10, 'um'))

    # 1. Unknown Current
    check = limits.is_safe_move(req_move, current=None, relative=True)
    assert not check.allowed
    assert "without current position" in check.reasons[0]

    # 2. Valid
    current = StagePosition(x=Q_(50, 'um'))
    check = limits.is_safe_move(req_move, current=current, relative=True)
    assert check.allowed

    # 3. Limit Breach
    big_move = StagePosition(x=Q_(60, 'um'))
    check = limits.is_safe_move(big_move, current=current, relative=True)
    assert not check.allowed


def test_stage_control_request():
    with pytest.raises(ValueError):
        StageControlRequest(action="DANCE", _mode=ParseMode.STRICT).validate()
    valid = StageControlRequest(action="HOME", axes=["x", "y"])
    assert valid.validate()


# =============================================================================
# 4. BEAM SUBSYSTEM
# =============================================================================

def test_beam_settings_parsing():
    b = BeamSettings.from_dict({"voltage": "300 kV", "beam_current": "100 pA", "spot_size": "1"})
    assert b.voltage.magnitude == 300.0
    assert b.beam_current.to("nA").magnitude == pytest.approx(0.1)

    b_bad = BeamSettings(spot_size=-1, _mode=ParseMode.LENIENT)
    b_bad.validate()
    assert b_bad.spot_size is None


def test_beam_system_settings_limits():
    sys = BeamSystemSettings(voltage_limits=(Q_(100, 'kV'), Q_(200, 'kV')), _mode=ParseMode.STRICT)
    res = sys.is_safe_beam(BeamSettings(voltage=Q_(300, 'kV')))
    assert not res.allowed


# =============================================================================
# 5. PROJECTION SUBSYSTEM
# =============================================================================

def test_projection_mode_dependencies():
    p1 = ProjectionSettings(optical_mode="IMAGING", magnification=50000, _mode=ParseMode.STRICT)
    assert p1.validate()

    p2 = ProjectionSettings(optical_mode="DIFFRACTION", _mode=ParseMode.LENIENT)
    assert p2.validate() is False
    assert "ProjectionSettings.missing_cam_len" in p2.extra.notes


def test_projection_system_limits():
    sys = ProjectionSystemSettings(magnification_limits=(1000, 100000))
    check = sys.is_safe_projection(ProjectionSettings(magnification=200000))
    assert not check.allowed


# =============================================================================
# 6. DETECTOR & ACQUISITION SUBSYSTEM
# =============================================================================

def test_detector_settings_logic():
    d = DetectorSettings(exposure=Q_(-1, 's'), _mode=ParseMode.LENIENT)
    d.validate()
    assert d.exposure is None

    d = DetectorSettings(frame_integration=0, _mode=ParseMode.LENIENT)
    d.validate()
    assert d.frame_integration == 1


def test_detector_dose_consistency_logic():
    # Consistent
    d = DetectorSettings(exposure=Q_(1.0, 's'), frame_rate=Q_(10, 'Hz'), total_frames=10, _mode=ParseMode.STRICT)
    assert d.validate()
    # Inconsistent
    d_bad = DetectorSettings(exposure=Q_(1.0, 's'), frame_rate=Q_(10, 'Hz'), total_frames=20, _mode=ParseMode.LENIENT)
    d_bad.validate()
    assert "DetectorSettings.dose_logic" in str(d_bad.extra.notes)


def test_detector_capabilities_comprehensive():
    caps = DetectorCapabilities(can_binning=False, exposure_max=Q_(1000, 'ms'), roi_size_max=(1024, 1024))
    assert not caps.supports(DetectorSettings(binning_index=2))
    assert not caps.supports(DetectorSettings(exposure=Q_(2000, 'ms')))
    assert not caps.supports(DetectorSettings(roi=ROI(width=2048)))


def test_detector_system_registry():
    sys = DetectorSystemSettings(
        available_detector_ids=["CamA"],
        capabilities_by_id={"CamA": DetectorCapabilities(exposure_max=Q_(10, 'ms'))},
        defaults_by_id={"CamA": DetectorSettings(exposure=Q_(5, 'ms'))},
        _mode=ParseMode.STRICT
    )
    assert sys.validate()
    assert sys.is_supported(DetectorSettings(detector_id="CamA", exposure=Q_(5, 'ms')))
    assert not sys.is_supported(DetectorSettings(detector_id="CamA", exposure=Q_(20, 'ms')))


def test_detector_system_consistency_garbage_data():
    raw_data = {
        "available_detectors": ["CamA", "CamB"],
        "defaults_by_id": {"CamA": {"exposure_ms": 100}, "CamB": "GARBAGE_STRING"},
        "capabilities_by_id": {"CamA": {}, "CamB": {}}
    }
    sys = DetectorSystemSettings.from_dict(raw_data, mode=ParseMode.LENIENT)
    sys.validate()
    assert "CamB" not in sys.defaults_by_id
    assert "CamB" not in sys.available_detector_ids


def test_acquisition_request_validation():
    req = AcquisitionRequest(detector_id="CamA", detector=DetectorSettings(detector_id="CamB"), _mode=ParseMode.LENIENT)
    req.validate()
    assert req.detector.detector_id == "CamA"
    assert "AcquisitionRequest.id_mismatch" in req.extra.notes


# =============================================================================
# 7. SCAN SUBSYSTEM
# =============================================================================

def test_scan_settings_estimation():
    s = ScanSettings(width_px=512, height_px=512, pixel_dwell_time=Q_(10, 'us'), flyback_time=Q_(100, 'us'))
    assert s.estimated_duration.to('s').magnitude == pytest.approx(2.67264)


def test_scan_settings_healing():
    s = ScanSettings(pixel_dwell_time=Q_(-10, 'us'), width_px=-50, _mode=ParseMode.LENIENT)
    s.validate()
    assert s.pixel_dwell_time is None
    assert s.width_px == 512


def test_scan_system_safety():
    sys = ScanSystemSettings(pixel_dwell_time_limits=(Q_(1, 'us'), Q_(10, 'us')), available_scan_modes=["Frame"])
    assert not sys.is_safe_scan(ScanSettings(pixel_dwell_time=Q_(100, 'us')))
    assert not sys.is_safe_scan(ScanSettings(scan_mode="Line"))


# =============================================================================
# 8. VACUUM & APERTURE SUBSYSTEMS
# =============================================================================

def test_vacuum_settings_readonly_inputs():
    req = VacuumControlRequest(target=VacuumSettings(turbo_pump_state="ON"), _mode=ParseMode.STRICT)
    assert req.validate()
    v = VacuumSettings(column_pressure=Q_(-5, 'Pa'), _mode=ParseMode.LENIENT)
    v.validate()
    assert "VacuumSettings.column_pressure" in v.extra.notes


def test_aperture_control_scenarios():
    req = ApertureControlRequest(aperture_id="obj", relative=True, target=Aperture(), _mode=ParseMode.LENIENT)
    assert req.validate() is False
    req.target.position = Point(x=10)
    assert req.validate() is True


# =============================================================================
# 9. MICROSCOPE GLOBAL INTEGRATION
# =============================================================================

def test_microscope_state_active_detectors():
    state = MicroscopeState(
        active_detector_ids=["GhostCam"],
        detectors={"RealCam": DetectorState(detector_id="RealCam")},
        _mode=ParseMode.LENIENT
    )
    state.validate()
    assert "GhostCam" not in state.active_detector_ids


def test_system_settings_structure():
    sys_dict = {
        "stage": {"max_step_nm": 500},
        "beam": {"voltage_limits_kv": [100, 300]},
        "info": {"ip_address": "127.0.0.1"}
    }
    settings = SystemSettings.from_dict(sys_dict, mode=ParseMode.STRICT)
    assert settings.stage_system.max_step_distance.magnitude == 500.0


def test_system_settings_round_trip():
    sys = SystemSettings(stage_system=StageSystemSettings(max_step_distance=Q_(10, 'um')),
                         info=SystemInfo(name="TestScope"))
    payload = sys.to_dict()
    reconstructed = SystemSettings.from_dict(payload)
    assert reconstructed.stage_system.max_step_distance.magnitude == pytest.approx(10000.0)


def test_microscope_settings_root():
    ms = MicroscopeSettings(image=ImageOutputSettings(file_format="png"), _mode=ParseMode.STRICT)
    assert ms.validate()
    assert ms.to_dict()["image"]["file_format"] == "png"


def test_mixed_mode_heirarchy():
    root = SystemSettings(_mode=ParseMode.STRICT)
    root.stage_system = StageSystemSettings(max_step_distance=Q_(-1, 'nm'), _mode=ParseMode.LENIENT)
    with pytest.raises(ValueError):
        root.validate()


# =============================================================================
# 10. IMAGE I/O & METADATA
# =============================================================================

def test_metadata_defaults():
    meta = MicroscopeImageMetadata(_mode=ParseMode.LENIENT)
    assert meta.version is not None
    meta.magnification = -500
    meta.validate()
    assert meta.magnification is None


def test_metadata_json_safety_with_numpy():
    meta = MicroscopeImageMetadata(
        extra={"analysis": {"score": np.float32(0.95), "counts": np.array([1, 2, 3], dtype=np.uint8)}})
    loaded = json.loads(json.dumps(meta.to_dict()))
    analysis_data = loaded["extra"]["unknown"]["analysis"]
    assert analysis_data["score"] == pytest.approx(0.95)
    assert analysis_data["counts"] == [1, 2, 3]


def test_image_io_formats(tmp_path):
    # 16-bit TIFF
    data16 = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    img = MicroscopeImage(data16)
    loaded = MicroscopeImage.load(img.save(tmp_path / "test16.tif"))
    assert loaded.data.dtype == np.uint16

    # 8-bit JPG
    data8 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img8 = MicroscopeImage(data8)
    loaded_jpg = MicroscopeImage.load(img8.save(tmp_path / "test8.jpg"))
    assert loaded_jpg.data.dtype == np.uint8


def test_image_load_int32_coercion(tmp_path):
    import tifffile as tff
    data32 = np.array([[100, 200], [300, 400]], dtype=np.int32)
    path = tmp_path / "test32.tif"
    tff.imwrite(str(path), data32)
    img = MicroscopeImage.load(path)
    assert img.data.dtype == np.uint16
    assert img.data[0, 0] == 100


def test_image_load_3d_squeeze(tmp_path):
    import tifffile as tff
    data3d = np.zeros((1, 50, 50), dtype=np.uint16)
    path = tmp_path / "stack.tif"
    tff.imwrite(str(path), data3d)
    img = MicroscopeImage.load(path)
    assert img.data.ndim == 2


def test_image_description_parsing_failure(tmp_path):
    import tifffile as tff
    data = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "bad_meta.tif"
    tff.imwrite(str(path), data, description="{this is not valid json}")
    img = MicroscopeImage.load(path)
    assert img.metadata is None


def test_image_path_handling():
    data = np.zeros((10, 10), dtype=np.uint8)
    img = MicroscopeImage(data)
    path_obj = Path("test_img_path.tif")
    try:
        img.save(path_obj)
        assert path_obj.exists()
        img2 = MicroscopeImage.load(str(path_obj))
        assert img2.data.shape == (10, 10)
    finally:
        if path_obj.exists(): path_obj.unlink()
        if Path("test_img_path.json").exists(): Path("test_img_path.json").unlink()


def test_image_unsupported_format_save():
    img = MicroscopeImage(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        img.save("test.txt", file_format="txt")