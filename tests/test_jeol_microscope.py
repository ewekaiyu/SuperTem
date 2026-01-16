import pytest
import numpy as np
import itertools
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List

from supertem.microscope import MicroscopeSettings
from supertem.structures.base import (
    SystemSettings,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    AcquisitionRequest,
    DetectorSettings,
    DetectorControlRequest,
    Aperture,
    VacuumSettings,
    Q_,
    Units,
    Extras,
    Point,
    ROI
)

# Correct Import Path
from supertem.microscopes.jeol_microscope import JeolMicroscope


# =============================================================================
# 1. FIXTURES & MOCKS
# =============================================================================

@pytest.fixture
def mock_pyjem():
    """
    Patches PyJEM imports. Returns mocks for TEM3 and detector.
    """
    with patch("supertem.microscopes.jeol_microscope.TEM3") as mock_tem3, \
            patch("supertem.microscopes.jeol_microscope.detector") as mock_det:
        # --- Setup TEM3 Sub-modules ---
        mock_tem3.Stage3.return_value = MagicMock()
        mock_tem3.EOS3.return_value = MagicMock()
        mock_tem3.HT3.return_value = MagicMock()
        mock_tem3.Lens3.return_value = MagicMock()
        mock_tem3.Def3.return_value = MagicMock()
        mock_tem3.Apt3.return_value = MagicMock()
        mock_tem3.Scan3.return_value = MagicMock()
        mock_tem3.VACUUM3.return_value = MagicMock()
        mock_tem3.GUN3.return_value = MagicMock()
        mock_tem3.FEG3.return_value = MagicMock()
        mock_tem3.Detector3.return_value = MagicMock()

        # --- Setup Detector Module ---
        mock_det.get_attached_detector.return_value = ["Camera1"]
        mock_det.Detector.return_value = MagicMock()

        # Simulate 'function' submodule logic for compatibility
        mock_det.function = MagicMock()
        mock_det.function.get_attached_detector.return_value = ["Camera1"]

        yield mock_tem3, mock_det


@pytest.fixture
def jeol_scope(mock_pyjem):
    """
    Returns an instantiated and connected JeolMicroscope.
    """
    mock_tem3, mock_det = mock_pyjem

    # Initialize settings cleanly
    settings = MicroscopeSettings(system=SystemSettings())
    # Manually inject calibration attribute for test
    settings.defocus_scale = 0.5  # 1 nm = 0.5 DAC

    scope = JeolMicroscope(settings)
    scope.connect("localhost")
    return scope


# =============================================================================
# 2. CONNECTION & LIFECYCLE
# =============================================================================

def test_connect_success(mock_pyjem):
    mock_tem3, _ = mock_pyjem
    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    scope.connect("1.2.3.4")

    mock_tem3.connect.assert_called_once()
    assert scope.is_connected() is True


def test_disconnect(jeol_scope):
    jeol_scope.disconnect()
    assert jeol_scope.is_connected() is False


def test_get_instrument_info(jeol_scope):
    info = jeol_scope.get_instrument_info()
    assert info.manufacturer == "JEOL"
    assert "PyJEM" in info.software_version


def test_lazy_loading_safety():
    """Ensure instantiation is safe even if PyJEM imports fail."""
    try:
        scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
        assert scope is not None
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")


# =============================================================================
# 3. STAGE CONTROL
# =============================================================================

def test_get_stage_position(jeol_scope):
    # [x, y, z, tx, ty]
    jeol_scope.stage.GetPos.return_value = [1000.0, 2000.0, 500.0, 1.5, -0.5]
    pos = jeol_scope.get_stage_position()

    assert pos.x.magnitude == pytest.approx(1000.0)
    assert pos.y.magnitude == pytest.approx(2000.0)
    assert pos.tilt_x.magnitude == pytest.approx(1.5)


def test_move_stage_absolute_motor(jeol_scope):
    target = StagePosition(x=Q_(10, "um"))
    jeol_scope.stage.GetStatus.return_value = [0] * 5  # Idle
    jeol_scope.stage.GetPos.return_value = [10000.0, 0, 0, 0, 0]  # Arrived

    jeol_scope.move_stage_absolute(target, drive_type="motor")

    jeol_scope.stage.SelDrvMode.assert_called_with(0)
    jeol_scope.stage.SetX.assert_called_with(pytest.approx(10000.0))


def test_move_stage_piezo(jeol_scope):
    target = StagePosition(x=Q_(50, "nm"))
    jeol_scope.move_stage_absolute(target, drive_type="piezo")

    # Verify switch to piezo (1) then back to motor (0)
    assert jeol_scope.stage.SelDrvMode.call_args_list[0] == call(1)
    assert jeol_scope.stage.SelDrvMode.call_args_list[-1] == call(0)
    jeol_scope.stage.SetX.assert_called_with(pytest.approx(50.0))


def test_move_stage_hysteresis_retry(jeol_scope):
    target = StagePosition(x=Q_(1000, "nm"))
    jeol_scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]

    # 1. Check: 900nm (Fail), 2. Check: 1000nm (Success)
    jeol_scope.stage.GetPos.side_effect = [
        [900.0, 0, 0, 0, 0],
        [1000.0, 0, 0, 0, 0],
    ]

    jeol_scope.move_stage_absolute(target, tolerance_nm=10.0, max_retries=3)
    assert jeol_scope.stage.SetX.call_count == 2


def test_stop_stage(jeol_scope):
    jeol_scope.stop_stage()
    jeol_scope.stage.Stop.assert_called_once()


def test_home_stage(jeol_scope):
    jeol_scope.home_stage()
    jeol_scope.stage.SetOrg.assert_called_once()


def test_stage_timeout(jeol_scope):
    """Wait timeout if stage never reports idle."""
    target = StagePosition(x=Q_(10, "um"))
    jeol_scope.stage.GetStatus.return_value = [1, 0, 0, 0, 0]  # Always Busy

    # We must patch sleep to avoid actually waiting, and time to control the loop
    with patch("time.time") as mock_time, patch("time.sleep"):
        # Infinite generator starting at 0, stepping by 10s
        # This prevents StopIteration if the retry loop runs extra times
        mock_time.side_effect = itertools.count(start=0, step=10)

        # Ensure GetPos returns a valid structure so 'from_jeol' works during retries
        jeol_scope.stage.GetPos.return_value = [0.0, 0.0, 0.0, 0.0, 0.0]

        jeol_scope.move_stage_absolute(target, wait=True)

        # Should verify position at least once before quitting
        assert jeol_scope.stage.GetStatus.called


def test_stage_disconnected_error(jeol_scope):
    jeol_scope.stage = None
    with pytest.raises(RuntimeError):
        jeol_scope.move_stage_absolute(StagePosition(x=Q_(0, 'nm')))


def test_stage_hysteresis_retry_loop(mock_pyjem):
    """
    Verify move_stage_absolute retries if the stage reports IDLE
    but position is not yet within tolerance.
    """
    # Setup
    scope = JeolMicroscope(MicroscopeSettings())
    scope.connect("localhost")

    # Mock Stage behavior
    # We want 'GetStatus' to always return [0,0,0,0,0] (Idle)
    # But 'GetPos' to return incorrect values initially (simulating drift/lag)
    scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]

    # Sequence of GetPos results:
    # 1. Initial check (during move) -> Not used by wait logic directly but needed
    # 2. Verification 1: 900nm (Target 1000nm) -> Fail
    # 3. Verification 2: 950nm -> Fail
    # 4. Verification 3: 1000nm -> Success
    scope.stage.GetPos.side_effect = [
        {"x": 900, "y": 0},  # Attempt 0 check
        {"x": 950, "y": 0},  # Attempt 1 check
        {"x": 1000, "y": 0}  # Attempt 2 check
    ]

    # Mock SetX to avoid actual IO
    scope.stage.SetX = MagicMock()

    # Mock time.sleep to speed up test
    with patch("time.sleep", return_value=None):
        target = StagePosition(x=Q_(1000, "nm"), y=Q_(0, "nm"))
        scope.move_stage_absolute(target, tolerance_nm=10.0, max_retries=3)

    # Assert
    # Logic should have called SetX multiple times (Initial + Retries)
    # Since we failed verification twice, we expect retries.
    assert scope.stage.SetX.call_count >= 2


# =============================================================================
# 4. BEAM CONTROL
# =============================================================================

def test_get_beam_properties(jeol_scope):
    # Voltage
    jeol_scope.ht.GetHtValue.return_value = 200000.0
    assert jeol_scope.get_acceleration_voltage().magnitude == pytest.approx(200.0)

    # Current (Gun)
    jeol_scope.gun.GetEmissionCurrent.return_value = 150.0  # uA
    cur = jeol_scope.get_beam_current()
    assert cur.to("nA").magnitude == pytest.approx(150000.0)

    # Spot Size (Int)
    jeol_scope.eos.GetSpotSize.return_value = 1
    assert jeol_scope.get_spot_size() == 1


def test_set_beam_properties(jeol_scope):
    jeol_scope.set_acceleration_voltage(Q_(300, "kV"))
    jeol_scope.ht.SetHtValue.assert_called_with(pytest.approx(300000.0))

    jeol_scope.set_spot_size(3)
    jeol_scope.eos.SelectSpotSize.assert_called_with(3)


def test_beam_alignments_getters(jeol_scope):
    # Shift (CLA1)
    jeol_scope.def_.GetCLA1.return_value = [100.1, 200.2]
    assert jeol_scope.get_beam_shift() == pytest.approx((100.1, 200.2))

    # Stigmation (CLs)
    jeol_scope.def_.GetCLs.return_value = [5.5, -5.5]
    assert jeol_scope.get_condenser_stigmation() == pytest.approx((5.5, -5.5))

    # Tilt (AngBal)
    jeol_scope.def_.GetAngBal.return_value = [10.0, 10.0]
    assert jeol_scope.get_gun_tilt() == pytest.approx((10.0, 10.0))


def test_beam_alignments_setters(jeol_scope):
    jeol_scope.set_beam_shift(50, 60)
    jeol_scope.def_.SetCLA1.assert_called_with(50, 60)

    jeol_scope.set_condenser_stigmation(1, 2)
    jeol_scope.def_.SetCLs.assert_called_with(1, 2)

    jeol_scope.set_gun_tilt(3, 4)
    jeol_scope.def_.SetAngBal.assert_called_with(3, 4)


def test_beam_blank(jeol_scope):
    jeol_scope.set_beam_blank(True)
    jeol_scope.def_.SetBeamBlank.assert_called_with(1)

    jeol_scope.def_.GetBeamBlank.return_value = 1
    assert jeol_scope.get_beam_blank() is True


def test_set_beam_current_unsupported(jeol_scope):
    with pytest.raises(NotImplementedError):
        jeol_scope.set_beam_current(Q_(1, "nA"))


# --- BEAM VENDOR EXTRAS ---

def test_get_beam_settings_vendor_extras(jeol_scope):
    jeol_scope.eos.GetAlpha.return_value = 3
    jeol_scope.ht.GetHtValue.return_value = 200000.0
    beam = jeol_scope.get_beam_settings()
    assert beam.extra.vendor["JEOL"]["alpha_index"] == 3


def test_apply_beam_settings_alpha_limit(jeol_scope):
    # Valid
    jeol_scope.apply_beam_settings(BeamSettings(
        extra=Extras(vendor={"JEOL": {"alpha_index": 5}})
    ))
    jeol_scope.eos.SetAlphaSelector.assert_called_with(5)

    # Invalid
    with pytest.raises(ValueError, match="out of bounds"):
        jeol_scope.apply_beam_settings(BeamSettings(
            extra=Extras(vendor={"JEOL": {"alpha_index": 99}})
        ))


def test_apply_beam_convergence_angle_error(jeol_scope):
    with pytest.raises(ValueError, match="alpha_index"):
        jeol_scope.apply_beam_settings(BeamSettings(
            convergence_angle=Q_(10, "mrad")
        ))


# =============================================================================
# 5. MODE CONTROL & EOS
# =============================================================================

def test_set_mode_stem(jeol_scope):
    jeol_scope.set_mode("STEM")
    jeol_scope.eos.SelectTemStem.assert_called_with(1)  # 1 = STEM


def test_set_mode_tem(jeol_scope):
    jeol_scope.set_mode("TEM")
    jeol_scope.eos.SelectTemStem.assert_called_with(0)  # 0 = TEM


def test_set_projection_mode_diffraction(jeol_scope):
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.set_projection_mode("DIFFRACTION")
    # TEM:DIFF
    jeol_scope.eos.SelectFunctionMode.assert_called_with(4)


def test_get_camera_length_tem(jeol_scope):
    # TEM + DIFF
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [4]
    jeol_scope.eos.GetMagValue.return_value = [150.0, "cm", "L"]

    cl = jeol_scope.get_camera_length()
    assert cl.to("mm").magnitude == pytest.approx(1500.0)


def test_get_camera_length_stem(jeol_scope):
    # STEM + UUDIFF
    jeol_scope.eos.GetTemStemMode.return_value = 1
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    jeol_scope.eos.GetStemCamValue.return_value = [80.0, "mm", "L"]

    cl = jeol_scope.get_camera_length()
    assert cl.to("mm").magnitude == pytest.approx(80.0)


# =============================================================================
# 6. PROJECTION (MAG & DEFOCUS)
# =============================================================================

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_magnification_logic(mock_get_list, jeol_scope):
    """Verify fallback to table lookup if direct hardware read fails."""
    # Setup Table
    mock_get_list.return_value = [("2000.0", "x", "2k"), ("5000.0", "x", "5k")]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]

    # Force fallback to Method 2 (Table)
    jeol_scope.eos.GetMagValue.side_effect = Exception("HW Fail")
    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = Exception("Not supported")

    # Test Set
    jeol_scope.set_magnification(5000)
    jeol_scope.eos.SetSelector.assert_called_with(2)

    # Test Get
    jeol_scope.eos.GetSelector.return_value = 2
    mag = jeol_scope.get_magnification()
    assert mag == 5000


def test_defocus_calibration_logic(jeol_scope):
    target_defocus = Q_(100, "nm")
    jeol_scope.set_defocus(target_defocus)
    # 100 * 0.5 = 50
    jeol_scope.lens.SetOLc.assert_called_with(50)


def test_defocus_uncalibrated_access(jeol_scope):
    jeol_scope._has_defocus_calibration = False

    jeol_scope.lens.GetOLc.return_value = 32768
    ps = jeol_scope.get_projection_settings()

    assert ps.defocus is None
    assert ps.extra.vendor["JEOL"]["defocus_olc_dac"] == 32768

    req = ProjectionSettings(extra=Extras(vendor={"JEOL": {"defocus_olc_dac": 12345}}))
    jeol_scope.apply_projection_settings(req)
    jeol_scope.lens.SetOLc.assert_called_with(12345)


def test_set_camera_length_wrong_mode(jeol_scope):
    """
    Ensure setting Camera Length fails if we are in TEM:MAG mode.
    This tests the STRICT SAFETY check implemented in jeol_microscope.py.
    """
    # 1. Simulate TEM Mag Mode (The "Dangerous" state)
    jeol_scope.eos.GetTemStemMode.return_value = 0  # TEM
    jeol_scope.eos.GetFunctionMode.return_value = [0]  # MAG

    # 2. Attempt to set Camera Length (e.g., 100 cm)
    with pytest.raises(RuntimeError) as exc:
        jeol_scope.set_camera_length(Q_(100, "cm"))

    # 3. Verify the error message matches the new safety check
    assert "Cannot set Camera Length in mode TEM" in str(exc.value)


def test_set_magnification_wrong_mode(jeol_scope):
    """Ensure setting Magnification fails if we are in TEM:DIFF mode."""
    # 1. Simulate TEM Diff Mode
    jeol_scope.eos.GetTemStemMode.return_value = 0  # TEM
    jeol_scope.eos.GetFunctionMode.return_value = [4]  # DIFF

    # 2. Attempt to set Magnification
    with pytest.raises(RuntimeError) as exc:
        jeol_scope.set_magnification(5000)

    assert "Magnification table not found for mode TEM:DIFF" in str(exc.value)


def test_objective_alignments(jeol_scope):
    jeol_scope.def_.GetOLs.return_value = [1.1, 2.2]
    assert jeol_scope.get_objective_stigmation() == pytest.approx((1.1, 2.2))

    jeol_scope.def_.GetIS1.return_value = [10.5, 20.5]
    assert jeol_scope.get_image_shift() == pytest.approx((10.5, 20.5))

    jeol_scope.def_.GetPLA.return_value = [5.1, 6.1]
    assert jeol_scope.get_diffraction_shift() == pytest.approx((5.1, 6.1))


# =============================================================================
# 7. SCAN CONTROL
# =============================================================================

def test_get_scan_mode(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    det.get_detectorsetting.return_value = {"ScanMode": 3}
    assert jeol_scope.get_scan_mode() == "Area"


def test_set_scan_mode(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    jeol_scope.set_scan_mode("Spot")
    det.set_scanmode.assert_called_with(1)


def test_scan_dimensions(jeol_scope):
    det = jeol_scope._get_detector("Camera1")

    # Get
    det.get_detectorsetting.return_value = {"Width": 1024, "Height": 1024}
    assert jeol_scope.get_scan_width() == 1024

    # Set
    jeol_scope.set_scan_width(512)
    det.set_imaging_area.assert_called()
    args = det.set_imaging_area.call_args[0]
    assert args[0] == 512


def test_scan_active_fallback(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    det.livestart.side_effect = Exception("Det fail")
    jeol_scope.set_scan_active(True)
    jeol_scope.scan.SetExtScanMode.assert_called_with(1)


def test_set_scan_rotation_fallback(jeol_scope):
    det = jeol_scope._get_detector("Camera1")

    # Case 1: Detector
    det.set_scanrotation = MagicMock()
    jeol_scope.set_scan_rotation(Q_(45, "deg"))
    det.set_scanrotation.assert_called_with(pytest.approx(45.0))

    # Case 2: Hardware Fallback
    det.set_scanrotation.side_effect = Exception("Not supported")
    jeol_scope.scan.SetRotationAngleEx = MagicMock()
    jeol_scope.set_scan_rotation(Q_(90, "deg"))
    jeol_scope.scan.SetRotationAngleEx.assert_called_with(pytest.approx(90.0))


def test_scan_rotation_hardware_fallback(mock_pyjem):
    """
    Verify that if the Detector fails to set rotation, the driver
    falls back to the Scan Coils.
    """
    scope = JeolMicroscope(MicroscopeSettings())
    scope.connect("localhost")

    # 1. Setup Detector to FAIL
    mock_det = MagicMock()
    mock_det.set_scanrotation.side_effect = RuntimeError("Det Fail")
    scope._active_detectors["Det1"] = mock_det
    scope._primary_detector_id = "Det1"

    # 2. Setup Scan Coils to SUCCEED
    scope.scan.SetRotationAngleEx = MagicMock()

    # 3. Execute
    scope.set_scan_rotation(Q_(90, "deg"))

    # 4. Assert
    # Detector was tried
    mock_det.set_scanrotation.assert_called_once()
    # Scan coils were called as fallback
    scope.scan.SetRotationAngleEx.assert_called_once_with(90.0)


def test_unsupported_scan_calls(jeol_scope):
    with pytest.raises(NotImplementedError):
        jeol_scope.set_scan_pixel_dwell(Q_(1, "us"))


# --- FAILURE MODES FOR SCAN ROTATION ---

def test_set_scan_rotation_hardware_failure(jeol_scope):
    """
    Ensure we raise the hardware exception directly (Fail Loudly)
    if the hardware method exists but fails.
    """
    det = jeol_scope._get_detector("Camera1")
    det.set_scanrotation.side_effect = Exception("Det Not Supported")

    # Mock hardware to exist but fail
    jeol_scope.scan.SetRotationAngleEx.side_effect = Exception("HW Critical Fail")

    with pytest.raises(Exception) as exc:
        jeol_scope.set_scan_rotation(Q_(90, "deg"))

    assert "HW Critical Fail" in str(exc.value)


def test_set_scan_rotation_no_capability(jeol_scope):
    """Ensure we raise RuntimeError if NO hardware methods are found."""
    det = jeol_scope._get_detector("Camera1")
    # Detector fails
    det.set_scanrotation.side_effect = Exception("Det Not Supported")

    # Hardware methods missing (simulate by deletion or Mock spec)
    del jeol_scope.scan.SetRotationAngleEx
    del jeol_scope.scan.SetRotationAngle

    with pytest.raises(RuntimeError) as exc:
        jeol_scope.set_scan_rotation(Q_(90, "deg"))

    assert "SetRotation failed on both" in str(exc.value)


# =============================================================================
# 8. VACUUM & APERTURE
# =============================================================================

def test_vacuum_gauges(jeol_scope):
    jeol_scope.vac.GetPigInfo.return_value = [1.2e-5]
    p = jeol_scope.get_pressure("P1")
    assert p.magnitude == pytest.approx(1.2e-5)


def test_get_valve_state_bitmask(jeol_scope):
    # [count, bitfield]. Bit 0=Col(Open), Bit 1=Turbo(Closed) -> ...01 -> 1
    jeol_scope.vac.GetValveStatus.return_value = [10, 1]
    assert jeol_scope.get_valve_state("column") == "OPEN"
    assert jeol_scope.get_valve_state("turbo") == "CLOSED"


def test_set_valve_state_gun(jeol_scope):
    jeol_scope.set_valve_state("gun", "OPEN")
    jeol_scope.gun.SetBeamValve.assert_called_with(1)


def test_aperture_get_set(jeol_scope):
    jeol_scope.apt.GetExpSize.return_value = 2
    jeol_scope.apt.GetPosition.return_value = [1000, -1000]

    apt = jeol_scope.get_aperture("CLA")
    assert apt.size_index == 2
    assert apt.position.x == pytest.approx(1000.0)

    jeol_scope.set_aperture("CLA", Aperture(size_index=1, position=Point(x=0, y=0)))
    jeol_scope.apt.SetExpSize.assert_called_with(1, 1)  # Kind 1, Size 1
    jeol_scope.apt.SetPosition.assert_called_with(0, 0)


def test_set_aperture_invalid_id(jeol_scope):
    with pytest.raises(ValueError):
        jeol_scope.set_aperture("BAD_APT", Aperture(size_index=1))


# =============================================================================
# 9. DETECTOR & ACQUISITION
# =============================================================================

def test_detector_getters(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    det.get_detectorsetting.return_value = {
        "BinningIndex": 2,
        "frameIntegration": 4,
        "ImagingArea": {"Width": 256, "Height": 256, "X": 128, "Y": 128}
    }
    assert jeol_scope.get_detector_binning("Camera1") == 2
    assert jeol_scope.get_detector_integration("Camera1") == 4


def test_detector_setters(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    jeol_scope.set_detector_exposure("Camera1", Q_(100, "ms"))
    det.set_exposuretime_value.assert_called_with(100000)


def test_detector_insertion_status(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    det.get_insert_state.return_value = {"Status": "IN"}
    assert jeol_scope.get_detector_inserted("Camera1") is True


def test_acquire_image_flow(jeol_scope):
    det = jeol_scope._get_detector("Camera1")

    # Mock settings
    det.get_detectorsetting.return_value = {
        "BinningIndex": 1,
        "ExposureTimeValue": 100.0,
        "ImagingArea": {"Width": 512, "Height": 512}
    }

    # Mock Data: 512x512
    det.snapshot_rawdata.return_value = np.zeros((512, 512), dtype=np.uint16).tolist()

    req = AcquisitionRequest(detector_id="Camera1")
    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (512, 512)
    assert img.metadata.exposure_ms == pytest.approx(100.0)


def test_acquire_image_reshaping(jeol_scope):
    """Test reshaping of flat 1D arrays from PyJEM."""
    det = jeol_scope._get_detector("Camera1")

    # Mock FLAT array (100 elements)
    det.snapshot_rawdata.return_value = np.zeros(100, dtype=np.uint16).tolist()

    # ROI: 10x10
    roi = ROI(x=0, y=0, width=10, height=10)
    det.get_detectorsetting.return_value = {"ImagingArea": {"Width": 10, "Height": 10}}

    req = AcquisitionRequest(detector_id="Camera1", detector=DetectorSettings(roi=roi))
    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (10, 10)


def test_acquire_image_with_roi(jeol_scope):
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = np.zeros((10, 10)).tolist()

    req = AcquisitionRequest(
        detector_id="Camera1",
        detector=DetectorSettings(roi=ROI(x=0, y=0, width=10, height=10))
    )
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (10, 10)
    det.set_areamode_imagingarea.assert_called()


def test_acquire_image_metadata_failure_recovery(jeol_scope):
    """
    If getting the full microscope state fails (e.g. Stage disconnected during capture),
    the driver must still return the image data, with the error logged in metadata.
    """
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = np.zeros((128, 128), dtype=np.uint16).tolist()
    det.get_detectorsetting.return_value = {}

    # FORCE FAILURE: Make get_full_state crash
    # Patch the method on the instance to simulate a deep subsystem failure
    with patch.object(jeol_scope, 'get_full_state', side_effect=RuntimeError("Stage Timeout")):
        req = AcquisitionRequest(detector_id="Camera1")
        img = jeol_scope.acquire_image(req)

    # 1. We still got the image
    assert img.data.shape == (128, 128)

    # 2. The error was captured in the notes
    notes = img.metadata.extra.notes
    assert "MicroscopeImageMetadata.state_capture_failed" in notes
    assert "Stage Timeout" in notes["MicroscopeImageMetadata.state_capture_failed"]


def test_refresh_detectors_empty(jeol_scope):
    """Test behavior when no detectors are attached."""
    # Mock the detector module to return an empty list
    with patch("supertem.microscopes.jeol_microscope.detector") as mock_det_mod:
        mock_det_mod.get_attached_detector.return_value = []
        mock_det_mod.function = None

        # Trigger refresh manually
        jeol_scope._refresh_detectors()

        assert jeol_scope.list_detectors() == []
        assert jeol_scope.get_primary_detector_id() is None

        # Verify asking for a detector raises/handles safely
        with pytest.raises(RuntimeError):
            jeol_scope.acquire_image(AcquisitionRequest(detector_id=None))


# =============================================================================
# 10. ERROR SAFETY & UTILS
# =============================================================================

def test_fail_loudly_setters(jeol_scope):
    jeol_scope.def_ = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_beam_shift(0, 0)

    jeol_scope.eos = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_magnification(1000)


def test_null_getters_when_disconnected(jeol_scope):
    jeol_scope.def_ = None
    assert jeol_scope.get_beam_shift() == (None, None)

    jeol_scope.vac = None
    assert jeol_scope.get_pressure("P1") is None


def test_invalid_mode_key_normalization(jeol_scope):
    assert jeol_scope._normalize_eos_key(None) is None
    assert jeol_scope._normalize_eos_key("GARBAGE_MODE") is None