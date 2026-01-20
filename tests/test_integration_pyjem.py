"""
tests.test_integration_pyjem

Integration test using the REAL PyJEM.offline library.
Requires PyJEM to be installed in the environment.
"""
import pytest
import yaml
import numpy as np
import tifffile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Try to import the Real Offline Library
try:
    from PyJEM.offline import TEM3 as OfflineTEM3
    from PyJEM.offline import detector as OfflineDetector
    PYJEM_AVAILABLE = True
except ImportError:
    PYJEM_AVAILABLE = False

from supertem.utils import setup_session
from supertem.structures.base import (
    StagePosition,
    StageMoveRequest,
    AcquisitionRequest,
    DetectorSettings,
    BeamControlRequest,
    BeamSettings,
    ProjectionControlRequest,
    ProjectionSettings,
    ScanControlRequest,
    ScanSettings,
    Q_
)

# Skip entire file if PyJEM is not installed
pytestmark = pytest.mark.skipif(not PYJEM_AVAILABLE, reason="PyJEM library not found")

@pytest.fixture
def offline_hardware_stack():
    """
    Injects PyJEM.offline modules and patches specific methods to ensure stateful stability.
    This creates a 'Virtual Microscope' that remembers what you set.
    """
    # --- 1. Stage Mock (Stateful) ---
    stage = OfflineTEM3.Stage3()
    stage_state = {"x": 0.0, "y": 0.0, "z": 0.0}
    stage.SetX = MagicMock(side_effect=lambda v: stage_state.update({"x": float(v)}))
    stage.SetY = MagicMock(side_effect=lambda v: stage_state.update({"y": float(v)}))
    stage.SetZ = MagicMock(side_effect=lambda v: stage_state.update({"z": float(v)}))
    stage.GetPos = MagicMock(side_effect=lambda: [stage_state["x"], stage_state["y"], stage_state["z"], 0.0, 0.0])
    stage.GetStatus = MagicMock(return_value=[0, 0, 0, 0, 0])

    # --- 2. Beam/HT Mock (Stateful) ---
    ht = OfflineTEM3.HT3()
    beam_state = {"voltage": 200000.0} # Volts
    ht.SetHtValue = MagicMock(side_effect=lambda v: beam_state.update({"voltage": float(v)}))
    ht.GetHtValue = MagicMock(side_effect=lambda: beam_state["voltage"])

    # --- 3. EOS/Lens Mock (Stateful) ---
    eos = OfflineTEM3.EOS3()
    eos_state = {"spot": 1, "alpha": 3, "mode": 0} # 0=TEM
    eos.SelectSpotSize = MagicMock(side_effect=lambda v: eos_state.update({"spot": int(v)}))
    eos.GetSpotSize = MagicMock(side_effect=lambda: eos_state["spot"])
    eos.SetAlphaSelector = MagicMock(side_effect=lambda v: eos_state.update({"alpha": int(v)}))
    eos.GetAlpha = MagicMock(side_effect=lambda: eos_state["alpha"])
    # Mode logic
    eos.SelectTemStem = MagicMock(side_effect=lambda v: eos_state.update({"mode": int(v)}))
    eos.GetTemStemMode = MagicMock(side_effect=lambda: eos_state["mode"])
    # Mag logic (Simple passthrough for integration test)
    eos.SetSelector = MagicMock()
    eos.GetMagValue = MagicMock(return_value=[50000.0, "X", "x50k"])

    # --- 4. Scan Mock ---
    scan = OfflineTEM3.Scan3()
    scan.SetRotationAngleEx = MagicMock() # Capture calls

    # --- 5. Detector Mock ---
    real_detector_class = OfflineDetector.Detector
    class StableOfflineDetector(real_detector_class):
        def snapshot_rawdata(self):
            return np.random.randint(0, 1000, (512, 512), dtype=np.uint16).tobytes()
        def get_image_cache(self):
            return self.snapshot_rawdata()
        def get_detectorsetting(self):
            # Return plausible JEOL dict
            return {"BinningIndex": 1, "ExposureTimeValue": 100, "ImagingArea": {"Width": 512, "Height": 512}}

    # Inject into the driver via Patch
    with patch("supertem.microscopes.jeol_microscope.TEM3") as m_tem_module, \
         patch("supertem.microscopes.jeol_microscope.detector") as m_det_mod:

        m_tem_module.Stage3.return_value = stage
        m_tem_module.HT3.return_value = ht
        m_tem_module.EOS3.return_value = eos
        m_tem_module.Scan3.return_value = scan

        # Pass through others as default offline
        m_tem_module.Def3.side_effect = OfflineTEM3.Def3
        m_tem_module.Lens3.side_effect = OfflineTEM3.Lens3
        m_tem_module.Apt3.side_effect = OfflineTEM3.Apt3
        m_tem_module.VACUUM3.side_effect = OfflineTEM3.VACUUM3
        m_tem_module.FEG3.side_effect = OfflineTEM3.FEG3
        m_tem_module.GUN3.side_effect = OfflineTEM3.GUN3

        # Detector Injection
        m_det_mod.Detector.side_effect = StableOfflineDetector
        if hasattr(OfflineDetector, "get_attached_detector"):
             m_det_mod.get_attached_detector = MagicMock(return_value=["OfflineCam"])

        yield {
            "stage": stage,
            "ht": ht,
            "eos": eos,
            "scan": scan
        }

def setup_mock_session(context):
    """Helper to bootstrap the session with a basic config."""
    config_data = {
        "system": {
            "stage_system": {
                "max_step_nm": 1000000.0,
                "enabled": True,
                "can_x": True,
                "can_y": True,
                "x_limits": ["-2 mm", "2 mm"],
                "y_limits": ["-2 mm", "2 mm"]
            },
            "beam_system": {
                "voltage_limits_kv": [60, 300],
                "spot_size_limits": [1, 5]
            }
        },
        "image": {"file_format": "tiff", "path": "images/"}
    }
    context.config_path.mkdir(parents=True, exist_ok=True)
    (context.config_path / "microscope.yaml").write_text(yaml.safe_dump(config_data), encoding="utf-8")

    return setup_session(
        context=context,
        manufacturer="JEOL",
        config_path=context.config_path / "microscope.yaml",
        setup_logging=True
    )

# =============================================================================
# TESTS
# =============================================================================

def test_pyjem_offline_workflow_basic(mock_context, offline_hardware_stack):
    """
    Original test: Moves stage and captures image.
    """
    scope, settings = setup_mock_session(mock_context)

    # Move
    target = StagePosition(x=Q_(1.5, "um"), y=Q_(0.5, "um"))
    scope.execute_stage_move(StageMoveRequest(target=target))

    # Verify via Mock
    assert offline_hardware_stack["stage"].SetX.called
    assert scope.get_stage_position().x.to("nm").magnitude == pytest.approx(1500.0)

def test_beam_control_stateful(mock_context, offline_hardware_stack):
    """
    Tests Beam Control logic: Voltage, Spot Size, and Vendor Extras (Alpha).
    """
    scope, settings = setup_mock_session(mock_context)

    # 1. Change Voltage (Standard Physics)
    # Request: 200kV -> 80kV
    req = BeamControlRequest(target=BeamSettings(voltage=Q_(80, "kV")))
    scope.execute_beam_control(req)

    # Verify Hardware call
    offline_hardware_stack["ht"].SetHtValue.assert_called_with(80000.0)
    # Verify Stateful Readback
    assert scope.get_acceleration_voltage().magnitude == 80.0

    # 2. Change Vendor Specific (Alpha Index) via Extras
    # Note: Base BeamSettings doesn't have 'alpha_index', so we pass it in extras
    req_vendor = BeamControlRequest(
        target=BeamSettings(
            spot_size=3,
            extra={"vendor": {"JEOL": {"alpha_index": 5}}}
        )
    )
    scope.execute_beam_control(req_vendor)

    # Verify
    offline_hardware_stack["eos"].SelectSpotSize.assert_called_with(3)
    offline_hardware_stack["eos"].SetAlphaSelector.assert_called_with(5)

    # Readback
    assert scope.get_spot_size() == 3
    assert scope.get_alpha_index() == 5

def test_stem_mode_switch_and_scan(mock_context, offline_hardware_stack):
    """
    Tests switching optical modes (TEM->STEM) and configuring Scan parameters.
    """
    scope, settings = setup_mock_session(mock_context)

    # 1. Switch to STEM
    # In JEOL driver, Projection Mode "STEM:..." triggers mode switch
    scope.set_projection_mode("STEM:MAG")

    # Verify hardware call (1 = STEM in PyJEM)
    offline_hardware_stack["eos"].SelectTemStem.assert_called_with(1)
    assert scope.get_mode() == "STEM"

    # 2. Configure Scan Rotation
    scan_req = ScanControlRequest(
        action="START",
        target=ScanSettings(scan_rotation=Q_(45.0, "deg"))
    )
    scope.execute_scan_control(scan_req)

    # Verify hardware call
    offline_hardware_stack["scan"].SetRotationAngleEx.assert_called_with(45.0)

def test_safety_guardrails(mock_context, offline_hardware_stack):
    """
    Tests that the Orchestrator layer prevents unsafe moves BEFORE they reach PyJEM.
    """
    scope, settings = setup_mock_session(mock_context)

    # 1. Stage Safety (System Config Limit is +/- 2mm)
    unsafe_pos = StagePosition(x=Q_(5, "mm")) # 5mm > 2mm limit

    with pytest.raises(RuntimeError) as excinfo:
        scope.execute_stage_move(StageMoveRequest(target=unsafe_pos))

    assert "outside limits" in str(excinfo.value)
    # Ensure hardware was NOT called
    offline_hardware_stack["stage"].SetX.assert_not_called()

    # 2. Vendor Specific Safety (Driver Level)
    # JEOL Alpha selector valid range is 0-8. Try setting 9.
    unsafe_beam = BeamControlRequest(
        target=BeamSettings(
            extra={"vendor": {"JEOL": {"alpha_index": 9}}}
        )
    )

    with pytest.raises(ValueError) as excinfo:
        scope.execute_beam_control(unsafe_beam)

    assert "out of bounds" in str(excinfo.value)
    assert "no parameters set" not in str(excinfo.value)

def test_metadata_fidelity(mock_context, offline_hardware_stack):
    """
    Ensures that a captured image contains the full state of the microscope
    in its metadata (Voltage, Stage, Mag).
    """
    scope, settings = setup_mock_session(mock_context)

    # Setup State
    scope.set_acceleration_voltage(Q_(200, "kV"))
    scope.move_stage_absolute(StagePosition(x=Q_(10, "um")))
    
    # Acquire
    img = scope.acquire_image(AcquisitionRequest(detector_id="OfflineCam"))

    # Assertions on Image Object
    assert img.metadata is not None
    assert img.metadata.accelerating_voltage_kv == 200.0
    
    # Check deeply nested state
    saved_stage = img.metadata.microscope_state.stage_position
    assert saved_stage.x.to("um").magnitude == pytest.approx(10.0)

    # Save and Check File
    out_path = Path(settings.image.path) / "metadata_test.tif"
    img.save(out_path)
    
    with tifffile.TiffFile(out_path) as tf:
        desc = tf.pages[0].tags["ImageDescription"].value
        meta_json = json.loads(desc)
        assert meta_json["accelerating_voltage_kv"] == 200.0