import pytest
from supertem.structures.base import (
    StagePosition,
    Aperture,
    ScanSettings,
    BeamSettings,
    VacuumSettings,
    DetectorSettings,
    DetectorCapabilities,
    ROI,
    Units,
    Q_,
    ParseMode
)
from supertem.vendor.JEOL import jeol_adapter


# =============================================================================
# 1. Stage Conversion Tests
# =============================================================================

def test_to_jeol_stage_args_handles_units_correctly():
    """
    Verify SuperTEM Objects (Quantities) -> JEOL Dict (Floats in nm/deg).
    JEOL requires: x,y,z in nm; tx,ty in degrees.
    """
    pos = StagePosition(
        x=Q_(2.5, Units.UM),  # 2500 nm
        y=Q_(100, Units.NM),  # 100 nm
        z=Q_(0, Units.NM),
        tilt_x=Q_(1.5, Units.DEG),
        tilt_y=Q_(0, Units.DEG),
        _mode=ParseMode.STRICT
    )

    jeol_args = jeol_adapter.to_jeol_stage_args(pos)

    assert jeol_args["x"] == pytest.approx(2500.0)
    assert jeol_args["y"] == 100.0
    assert jeol_args["z"] == 0.0
    assert jeol_args["tx"] == 1.5
    assert "ty" in jeol_args  # Should be present even if 0


def test_to_jeol_stage_args_ignores_none_fields():
    """
    If a SuperTEM field is None, it should NOT appear in the JEOL dict.
    This prevents overwriting hardware values with 0.0 accidentally.
    """
    pos = StagePosition(x=Q_(100, Units.NM))
    jeol_args = jeol_adapter.to_jeol_stage_args(pos)

    assert "x" in jeol_args
    assert "y" not in jeol_args
    assert "z" not in jeol_args
    assert "tx" not in jeol_args


def test_from_jeol_stage_position_parsing():
    """Verify list [x, y, z, tx, ty] conversion to StagePosition."""
    raw_list = [100.0, 200.0, 300.0, 1.5, -0.5]

    pos = jeol_adapter.from_jeol_stage_position(raw_list)

    assert pos.x.to(Units.NM).magnitude == 100.0
    assert pos.z.to(Units.NM).magnitude == 300.0
    assert pos.tilt_x.to(Units.DEG).magnitude == 1.5
    assert pos.tilt_y.to(Units.DEG).magnitude == -0.5
    assert pos.coordinate_system == "raw_hardware"


def test_from_jeol_stage_position_robustness():
    """Verify it survives incomplete lists."""
    bad_list = [10.0, 20.0]  # Missing Z/Tilts
    pos = jeol_adapter.from_jeol_stage_position(bad_list)

    # Should fallback to a lenient object with raw data in extras
    assert pos.x is None
    assert pos.extra.vendor["JEOL"]["raw_input"] == bad_list


# =============================================================================
# 2. Detector Adapter Tests
# =============================================================================

def test_from_jeol_detector_response_parsing():
    """
    Test parsing of chaotic PyJEM dictionary keys.
    Covers:
    1. Standard Settings (Exposure, Binning, ROI)
    2. Capabilities Limits (Integration, String ROI)
    3. Optional 'Bonus' Parsing (Frame Rate, Pixel Size)
    """
    mock_payload = {
        # --- Standard Settings ---
        "ExposureTimeValue": 200.0,
        "BinningIndex": 1,
        "GainIndex": 0,
        "ImagingArea": {"X": 10, "Y": 10, "Width": 512, "Height": 512},
        "BinningSize": {"Width": 2, "Height": 2},

        # --- Optional Feature 1: Frame Rate ---
        # PyJEM sometimes sends this as a string with units
        "FrameRate": "20.5 fps",

        # --- Optional Feature 2: Pixel Size Calculation ---
        "OutputImageInformation": {
            "PixelsPerMeter": {
                "Horizontal": 2000000.0,  # 2e6 px/m => 500 nm/px
                "Vertical": 2000000.0
            }
        },

        # --- Capabilities (Limits) ---
        "ExposureTimeMax": 10000.0,
        "BinningIndexMaximum": 4,
        "CanGain": 1,
        "frameIntegrationMaximum": 128,  # Previously missing
        "ImagingAreaMaximum": "1072, 1072",  # Previously crashed (String format)

        # --- Vendor Garbage (Pass-through) ---
        "UnknownHardwareFlag": 999
    }

    settings, caps = jeol_adapter.from_jeol_detector_response(mock_payload, "Det1")

    # 1. Check Standard Settings
    assert settings.detector_id == "Det1"
    assert settings.exposure.to(Units.MS).magnitude == 200.0
    assert settings.roi.width == 512

    # 2. Check Optional Feature: Frame Rate Parsing
    # Should strip "fps" and convert to Hz
    assert settings.frame_rate is not None
    assert settings.frame_rate.to(Units.HZ).magnitude == 20.5

    # 3. Check Optional Feature: Pixel Size Calculation
    # 1e9 / 2e6 = 500.0 nm
    assert "JEOL" in settings.extra.vendor
    assert settings.extra.vendor["JEOL"]["calculated_pixel_size_nm"] == 500.0

    # 4. Check Capabilities & Limits
    assert caps.exposure_max.to(Units.MS).magnitude == 10000.0
    assert caps.frame_integration_max == 128  # New limit support
    assert caps.roi_size_max == (1072, 1072)  # Crash fix verification

    # 5. Check Passthrough
    assert settings.extra.vendor["JEOL"]["UnknownHardwareFlag"] == 999


def test_to_jeol_detector_config():
    """Test serialization back to JEOL-compatible dict."""
    settings = DetectorSettings(
        exposure=Q_(0.5, Units.SEC),  # 500 ms
        roi=ROI(x=0, y=0, width=256, height=256),
        binning_index=2,
        _mode=ParseMode.STRICT
    )

    # Add vendor extras to ensure pass-through
    settings.extra.vendor["JEOL"] = {"CustomFlag": 123}

    config = jeol_adapter.to_jeol_detector_config(settings)

    assert config["ExposureTimeValue"] == 500.0  # Converted to MS
    assert config["BinningIndex"] == 2
    assert config["ImagingArea"] == {"X": 0, "Y": 0, "Width": 256, "Height": 256}
    assert config["CustomFlag"] == 123


# =============================================================================
# 3. Beam Adapter Tests
# =============================================================================

def test_from_jeol_beam_stats_voltage_heuristic():
    """
    Verify the voltage unit heuristic.
    PyJEM sometimes returns V (200000) and sometimes kV (200).
    """
    # Case A: Raw value > 5000 -> Treat as Volts
    b1 = jeol_adapter.from_jeol_beam_stats(
        voltage_val=200000.0,
        current_ua=1.0,
        spot_size_idx=1,
        alpha_idx=3
    )
    assert b1.voltage.units == "kilovolt"
    assert b1.voltage.magnitude == 200.0

    # Case B: Raw value <= 5000 -> Treat as kV
    b2 = jeol_adapter.from_jeol_beam_stats(
        voltage_val=300.0,
        current_ua=1.0,
        spot_size_idx=1,
        alpha_idx=3
    )
    assert b2.voltage.units == "kilovolt"
    assert b2.voltage.magnitude == 300.0


def test_from_jeol_beam_stats_extras():
    """Verify that Alpha Index (non-standard physics) goes to extras."""
    b = jeol_adapter.from_jeol_beam_stats(200, 10, 3, 5)  # 200kV, 10uA, Spot3, Alpha5

    # Standard physics
    assert b.beam_current.to(Units.NA).magnitude == pytest.approx(10000.0)  # 10uA -> 10000nA
    assert b.spot_size == 3

    # Non-standard
    assert b.convergence_angle is None  # Cannot map index to mrad blindly
    assert b.extra.vendor["JEOL"]["alpha_index"] == 5


# =============================================================================
# 4. Vacuum Adapter Tests
# =============================================================================

def test_from_jeol_vacuum_stats_mapping():
    """Verify mapping of pressure list P1..P5 and Valve bits."""
    # Simulating PyJEM: [P1, P2, P3, P4, P5]
    # P1=Gun, P2=Column
    raw_pressures = [2.5e-7, 4.0e-5, 0, 0, 0]

    # Simulating Valve Status Dict
    valves = {"V1": 1, "V4": 0}  # V1 Open, V4 Closed

    vac = jeol_adapter.from_jeol_vacuum_stats(raw_pressures, valves)

    assert vac.gun_pressure.magnitude == 2.5e-7
    assert vac.column_pressure.magnitude == 4.0e-5

    assert vac.gun_valve_state == "OPEN"
    assert vac.column_valve_state == "CLOSED"

    # Raw data preservation
    assert vac.extra.vendor["JEOL"]["raw_pressures_P1_to_P5"] == raw_pressures


# =============================================================================
# 5. Scan & Aperture Adapter Tests
# =============================================================================

def test_from_jeol_scan_stats():
    """Verify scan parameters mapping."""
    s = jeol_adapter.from_jeol_scan_stats(
        rotation_deg=45.0,
        mag_correction=(1.0, 1.0),
        scan_mode_int=1  # Spot
    )

    assert s.scan_rotation.magnitude == 45.0
    assert s.scan_mode == "1"  # Mapped to string
    # Extra fields
    assert s.extra.vendor["JEOL"]["MagCorrection"] == (1.0, 1.0)


def test_from_jeol_aperture_logic():
    """Verify conversion of JEOL's (index, list) to Aperture."""
    # Case A: Inserted
    apt = jeol_adapter.from_jeol_aperture("CLA", 1, [1000, 2000])
    assert apt.inserted is True
    assert apt.size_index == 1
    assert apt.position.x == 1000.0

    # Case B: Retracted (Size 0)
    apt_out = jeol_adapter.from_jeol_aperture("CLA", 0, [0, 0])
    assert apt_out.inserted is False
    assert apt_out.size_index == 0


# =============================================================================
# 6. Helper Tests
# =============================================================================

def test_jeol_roi_helpers():
    """Round trip ROI conversion."""
    jeol_dict = {"X": 10, "Y": 20, "Width": 100, "Height": 200}

    # To Struct
    roi = jeol_adapter._jeol_roi_to_struct(jeol_dict)
    assert roi.x == 10
    assert roi.height == 200

    # Back to JEOL
    out = jeol_adapter._struct_to_jeol_roi(roi)
    assert out == jeol_dict


def test_pack_vendor_extras():
    """Ensure data is strictly namespaced under 'JEOL'."""
    data = {"foo": "bar"}
    ex = jeol_adapter._pack_vendor_extras(data)

    assert ex is not None
    assert ex.vendor["JEOL"] == data
    assert ex.vendor["JEOL"]["foo"] == "bar"


def test_pack_vendor_extras_none():
    """Ensure empty input returns None (cleaner object graph)."""
    assert jeol_adapter._pack_vendor_extras({}) is None
    assert jeol_adapter._pack_vendor_extras(None) is None