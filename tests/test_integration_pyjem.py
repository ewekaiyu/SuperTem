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
    Q_
)

# Skip entire file if PyJEM is not installed
pytestmark = pytest.mark.skipif(not PYJEM_AVAILABLE, reason="PyJEM library not found")

@pytest.fixture
def offline_hardware_stack():
    """
    Injects PyJEM.offline modules and patches specific methods to ensure stability.
    """
    # 1. Instantiate Stage (Stateful Wrapper)
    # OfflineTEM3 is a module, so we instantiate the class inside it
    stage = OfflineTEM3.Stage3()
    state = {"x": 0.0, "y": 0.0}

    def set_x(v): state["x"] = float(v)
    def set_y(v): state["y"] = float(v)
    def get_pos(): return [state["x"], state["y"], 0.0, 0.0, 0.0]
    def get_status(): return [0, 0, 0, 0, 0]

    stage.SetX = MagicMock(side_effect=set_x)
    stage.SetY = MagicMock(side_effect=set_y)
    stage.GetPos = MagicMock(side_effect=get_pos)
    stage.GetStatus = MagicMock(side_effect=get_status)

    # 2. Instantiate Detector (Synthetic Image Wrapper)
    real_detector_class = OfflineDetector.Detector

    class StableOfflineDetector(real_detector_class):
        def snapshot_rawdata(self):
            # Return a 512x512 uint16 noise array as raw bytes
            arr = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)
            return arr.tobytes()

        def get_image_cache(self):
            return self.snapshot_rawdata()

    # Inject into the driver
    with patch("supertem.microscopes.jeol_microscope.TEM3") as m_tem_module, \
         patch("supertem.microscopes.jeol_microscope.detector") as m_det_mod:

        m_tem_module.Stage3.return_value = stage

        # Pass through other modules
        m_tem_module.EOS3.side_effect = OfflineTEM3.EOS3
        m_tem_module.HT3.side_effect = OfflineTEM3.HT3
        m_tem_module.Def3.side_effect = OfflineTEM3.Def3
        m_tem_module.Apt3.side_effect = OfflineTEM3.Apt3
        m_tem_module.Lens3.side_effect = OfflineTEM3.Lens3
        m_tem_module.Scan3.side_effect = OfflineTEM3.Scan3
        m_tem_module.VACUUM3.side_effect = OfflineTEM3.VACUUM3
        m_tem_module.FEG3.side_effect = OfflineTEM3.FEG3
        m_tem_module.GUN3.side_effect = OfflineTEM3.GUN3

        # Use our Safe Detector Class
        m_det_mod.Detector.side_effect = StableOfflineDetector
        m_det_mod.function = OfflineDetector.function

        # Force detector discovery
        if hasattr(OfflineDetector, "get_attached_detector"):
             m_det_mod.get_attached_detector = MagicMock(return_value=["OfflineCam"])
             if hasattr(m_det_mod.function, "get_attached_detector"):
                 m_det_mod.function.get_attached_detector = MagicMock(return_value=["OfflineCam"])

        yield

def test_pyjem_offline_workflow(mock_context, offline_hardware_stack):
    """
    Runs the full workflow using PyJEM.offline classes.
    """
    # 1. SETUP
    config_data = {
        "system": {
            "stage_system": {
                "max_step_nm": 50000.0,
                "enabled": True,
                "can_x": True, "can_y": True,
                "x_limits": ["-2 mm", "2 mm"],
                "y_limits": ["-2 mm", "2 mm"]
            },
            "beam_system": {"voltage_limits_kv": [80, 300]}
        },
        "image": {
            "file_format": "tiff",
            "path": "images/session_{date}"
        },
        "defocus_scale": 1.0
    }

    mock_context.config_path.mkdir(parents=True, exist_ok=True)
    (mock_context.config_path / "microscope.yaml").write_text(yaml.safe_dump(config_data), encoding="utf-8")

    # 2. INITIALIZE
    scope, settings = setup_session(
        context=mock_context,
        manufacturer="JEOL",
        config_path=mock_context.config_path / "microscope.yaml",
        setup_logging=True
    )

    assert scope.is_connected()

    # 3. EXECUTE

    # A. Move Stage
    target = StagePosition(x=Q_(1.5, "um"), y=Q_(0.5, "um"))
    move_req = StageMoveRequest(target=target, wait_for_settle=True)
    scope.execute_stage_move(move_req)

    # Verify State
    final_pos = scope.get_stage_position()
    assert final_pos.x.to("nm").magnitude == pytest.approx(1500.0)

    # B. Acquire Image
    acq_req = AcquisitionRequest(
        detector=DetectorSettings(exposure=Q_(0.1, "s"))
    )
    image_obj = scope.acquire_image(acq_req)

    # C. Save
    out_dir = Path(settings.image.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pyjem_offline_test.tif"

    image_obj.save(out_path)

    assert out_path.exists()

    # 4. VERIFY METADATA
    with tifffile.TiffFile(out_path) as tf:
        # 1. Check Data Presence
        assert tf.asarray().size > 0

        # 2. Check Metadata Extraction
        # Strategy: Look in TIFF tags first (Standard), then Sidecar (Legacy)
        meta_dict = {}

        # Try TIFF Tags
        try:
            page = tf.pages[0]
            if "ImageDescription" in page.tags:
                desc_str = page.tags["ImageDescription"].value
                meta_dict = json.loads(desc_str)
        except Exception:
            pass

        # Try Sidecar if Tags failed
        if not meta_dict:
            sidecar = out_path.parent / (out_path.stem + ".json")
            if sidecar.exists():
                meta_dict = json.loads(sidecar.read_text())

        # ASSERTIONS
        # We must find the stage position we moved to (1500nm)
        assert meta_dict, "No metadata found in TIFF Header or Sidecar JSON"

        x_val = meta_dict.get("microscope_state", {}).get("stage_position", {}).get("x_nm")
        assert x_val == pytest.approx(1500.0), f"Metadata missing updated stage position. Found: {x_val}"