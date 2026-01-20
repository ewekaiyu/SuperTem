"""
tests.test_integration

End-to-End (E2E) Integration Tests.
"""
import pytest
import yaml
import numpy as np
import tifffile
from unittest.mock import patch
from pathlib import Path

from supertem.utils import setup_session
from supertem.structures.base import (
    StagePosition,
    StageMoveRequest,
    AcquisitionRequest,
    DetectorSettings,
    Q_,
    Units
)

# Patch paths matching your error logs
PATCH_PATH_TEM3 = "supertem.microscopes.jeol_microscope.TEM3"
PATCH_PATH_DET = "supertem.microscopes.jeol_microscope.detector"

@pytest.fixture
def mock_hardware_stack():
    """
    Creates a 'Stateful' mock hardware layer.
    """
    with patch(PATCH_PATH_TEM3) as m_tem, \
         patch(PATCH_PATH_DET) as m_det:

        # --- 1. DETECTOR MOCK ---
        # Fix for list_detectors / get_primary_detector_id
        m_det.function.get_attached_detector.return_value = ["Det1"]
        m_det.get_attached_detector.return_value = ["Det1"]

        # Setup Image data
        # We use a distinct value (100) to verify we aren't getting zeros
        dummy_image = np.ones((512, 512), dtype=np.uint16) * 100

        det_instance = m_det.Detector.return_value

        # --- CRITICAL FIX BASED ON YOUR SOURCE CODE ---
        # Your code checks `snapshot_rawdata` FIRST (Line 1319).
        # We must mock this specifically to return the array.
        det_instance.snapshot_rawdata.return_value = dummy_image

        # Fallbacks (just in case logic changes)
        det_instance.get_image_cache.return_value = dummy_image
        det_instance.livesnapshot.return_value = dummy_image

        # Mock settings so get_detector_roi / binning works
        det_instance.get_detectorsetting.return_value = {
            "Width": 512, "Height": 512,
            "Binning": 1, "ScanMode": 1
        }

        # --- 2. STAGE MOCK (Stateful) ---
        stage_state = [0.0, 0.0, 0.0, 0.0, 0.0] # X, Y, Z, TX, TY

        def side_effect_get_pos(): return list(stage_state)
        def side_effect_set_x(val): stage_state[0] = val
        def side_effect_set_y(val): stage_state[1] = val

        m_tem.Stage3.return_value.GetPos.side_effect = side_effect_get_pos
        m_tem.Stage3.return_value.SetX.side_effect = side_effect_set_x
        m_tem.Stage3.return_value.SetY.side_effect = side_effect_set_y
        m_tem.Stage3.return_value.GetStatus.return_value = [0, 0, 0, 0, 0]

        # --- 3. OTHER MOCKS ---
        m_tem.HT3.return_value.GetHtValue.return_value = 200000.0
        m_tem.EOS3.return_value.GetMagValue.return_value = [50000, "X"]

        yield m_tem, m_det

def test_full_acquisition_workflow(mock_context, mock_hardware_stack):
    # 1. SETUP: Create config
    config_data = {
        "system": {
            "stage_system": {
                "max_step_nm": 1000.0,
                "enabled": True,
                "can_x": True, "can_y": True,
                "x_limits": ["-1 mm", "1 mm"],
                "y_limits": ["-1 mm", "1 mm"]
            },
            "beam_system": {"voltage_limits_kv": [80, 300]}
        },
        "image": {
            "file_format": "tiff",
            "path": "images/session_{date}"
        },
        "defocus_scale": 0.5
    }

    mock_context.config_path.mkdir(parents=True, exist_ok=True)
    (mock_context.config_path / "microscope.yaml").write_text(yaml.safe_dump(config_data), encoding="utf-8")

    # 2. INITIALIZATION
    scope, settings = setup_session(
        context=mock_context,
        manufacturer="JEOL",
        config_path=mock_context.config_path / "microscope.yaml",
        setup_logging=True
    )

    assert scope.is_connected()

    # 3. EXECUTION: Run "Script"

    # A. Move Stage
    target = StagePosition(x=Q_(10.0, "um"), y=Q_(5.0, "um"))
    move_req = StageMoveRequest(target=target, wait_for_settle=True)
    scope.execute_stage_move(move_req)

    # Verify Stage Mock moved
    final_pos = scope.get_stage_position()
    assert final_pos.x.to("um").magnitude == pytest.approx(10.0)

    # B. Acquire Image
    # This will now use the Real Driver Code -> Mocked Hardware (snapshot_rawdata)
    acq_req = AcquisitionRequest(
        detector=DetectorSettings(exposure=Q_(0.5, "s"))
    )
    image_obj = scope.acquire_image(acq_req)

    # C. Save to Disk
    output_dir = Path(settings.image.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "test_image_001.tif"

    saved_file_path = image_obj.save(save_path)

    # 4. VERIFICATION
    assert Path(saved_file_path).exists()

    with tifffile.TiffFile(saved_file_path) as tf:
        data = tf.asarray()

        # Assertions
        assert data.shape == (512, 512)
        assert data.dtype == np.uint16
        assert data[0, 0] == 100 # Verify we got our dummy data, not empty zeros

        # Check Metadata Persistence
        sidecar_path = Path(saved_file_path).parent / (Path(saved_file_path).stem + ".json")
        if sidecar_path.exists():
            import json
            meta = json.loads(sidecar_path.read_text())
            x_val = meta.get("microscope_state", {}).get("stage_position", {}).get("x_nm")
            # Verify the 10um (10000nm) move was captured in the file metadata
            assert x_val == pytest.approx(10000.0)


def test_session_directory_structure(mock_context, mock_hardware_stack):
    """
    Verify that setup_session creates the organized folder structure expected by users.
    """
    scope, settings = setup_session(mock_context, manufacturer="JEOL", setup_logging=False)

    # Check Context structure
    assert mock_context.log_path.exists()
    assert mock_context.data_path.exists()

    # Check that settings pointed to the data path
    assert str(mock_context.base_path) in str(settings.image.path)


def test_setup_session_demo_driver_fallback(mock_context):
    """Requesting manufacturer='DEMO' should load the DemoMicroscope and set manufacturer."""
    with patch("supertem.microscopes.demo_microscope.DemoMicroscope") as MockDemoClass:
        scope, settings = setup_session(
            context=mock_context,
            manufacturer="DEMO",
            setup_logging=False,
        )

    MockDemoClass.assert_called_once()
    assert settings.system.info.manufacturer == "DEMO"


def test_setup_session_creates_log_file_when_enabled(mock_context):
    """When setup_logging=True, setup_session should create a .log file in the session folder."""
    with patch("supertem.microscopes.demo_microscope.DemoMicroscope"):
        _, settings = setup_session(
            context=mock_context,
            manufacturer="DEMO",
            setup_logging=True,
        )

    session_dir = Path(settings.image.path).parent
    assert any(session_dir.glob("*.log"))


def test_setup_session_ip_address_override_is_applied(mock_context):
    """If ip_address is provided, it should be written into settings.system.info.ip_address."""
    with patch("supertem.microscopes.demo_microscope.DemoMicroscope"):
        _, settings = setup_session(
            context=mock_context,
            manufacturer="DEMO",
            ip_address="127.0.0.1",
            setup_logging=False,
        )

    assert settings.system.info.ip_address == "127.0.0.1"


def test_setup_session_strict_validation_failure(mock_context, mock_registry):
    """Invalid config types should cause strict parsing/validation to raise."""
    config_path = mock_registry.default_microscope_config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Inject an invalid type where a numeric is expected.
    data["system"]["beam_system"]["voltage_limits_kv"] = ["NOT_A_NUMBER", "INVALID"]
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with patch("supertem.microscopes.jeol_microscope.JeolMicroscope"):
        with pytest.raises(Exception):
            setup_session(context=mock_context, manufacturer="JEOL", setup_logging=False)


def test_setup_session_instantiates_jeol_driver(mock_context):
    """setup_session should instantiate JeolMicroscope and call connect() exactly once."""
    with patch("supertem.microscopes.jeol_microscope.JeolMicroscope") as MockJeolClass:
        mock_instance = MockJeolClass.return_value

        setup_session(
            context=mock_context,
            manufacturer="JEOL",
            setup_logging=False,
        )

    MockJeolClass.assert_called_once()
    mock_instance.connect.assert_called_once()