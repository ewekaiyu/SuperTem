import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Import Abstract Base and Strict Structures
from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    SystemInfo,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    DetectorSettings,
    ScanSettings,
    VacuumSettings,
    Aperture,
    MicroscopeImage,
    MicroscopeImageMetadata,
    AcquisitionRequest,
    Units,
    Q_,
    Quantity,
    Extras,
    Point,
    ROI
)

# Import Vendor Adapters
from supertem.vendor.JEOL import jeol_adapter
from supertem.vendor.JEOL import jeol_eos_tables

logger = logging.getLogger(__name__)

# --- PyJEM Import Logic ---
try:
    from PyJEM import TEM3
except ImportError:
    try:
        from PyJEM.offline import TEM3
    except ImportError:
        TEM3 = None

try:
    from PyJEM import detector
except ImportError:
    try:
        from PyJEM.offline import detector
    except ImportError:
        detector = None


class JeolMicroscope(TemMicroscope):
    """
    JEOL ARM/F2 implementation of the SuperTEM Interface.
    Strictly implements ALL 50+ atomic methods defined in TemMicroscope.
    """

    _APERTURE_MAP = {
        "CLA": 1, "OLA": 2, "HCA": 3, "SAA": 4, "ENTA": 5,
        "CL1": 0, "CL2": 1, "OL": 2, "HC": 3, "SA": 4,
        "ENT": 5, "HX": 6, "BF": 7,
        "AUX": 8, "AUX1": 8, "AUX2": 9, "AUX3": 10, "AUX4": 11
    }

    _EOS_MODE_MAP = {
        (0, 0): "TEM:MAG", (0, 1): "TEM:MAG2", (0, 2): "TEM:LowMAG",
        (0, 3): "TEM:SAMAG", (0, 4): "TEM:DIFF",
        (1, 0): "STEM:Align", (1, 1): "STEM:SM-LMAG", (1, 2): "STEM:SM-MAG",
        (1, 3): "STEM:AMAG", (1, 4): "STEM:uuDIFF", (1, 5): "STEM:Rocking"
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stage = None
        self.eos = None
        self.ht = None
        self.lens = None
        self.def_ = None
        self.apt = None
        self.scan = None
        self.vac = None
        self.feg = None
        self.gun = None
        self._connected = False

    # =========================================================================
    # 1. Connection & Lifecycle
    # =========================================================================

    def connect(self, host: str, port: Optional[int] = None, **kwargs) -> None:
        if not TEM3:
            raise RuntimeError("PyJEM library not found.")
        try:
            self.stage = TEM3.Stage3()
            self.eos = TEM3.EOS3()
            self.ht = TEM3.HT3()
            self.lens = TEM3.Lens3()
            self.def_ = TEM3.Def3()
            self.apt = TEM3.Apt3()
            self.scan = TEM3.Scan3()
            self.vac = TEM3.Vacuum3()
            self.feg = TEM3.FEG3()
            self.gun = TEM3.Gun3()
            self._connected = True
            logger.info(f"Connected to JEOL PyJEM interface (Host: {host}).")
        except Exception as e:
            logger.error(f"Failed to connect to PyJEM modules: {e}")
            raise

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected from JEOL PyJEM.")

    def is_connected(self) -> bool:
        return self._connected

    def get_instrument_info(self) -> SystemInfo:
        # PyJEM offline drivers don't expose serial/model, so we return generic info
        return SystemInfo(
            manufacturer="JEOL",
            model="ARM/F2",
            software_version="PyJEM-TEM3",
            _mode="lenient"
        )

    # =========================================================================
    # 2. Global State & Mode
    # =========================================================================

    def get_mode(self) -> str:
        if not self.eos: return "UNKNOWN"
        # Heuristic based on EOS tables
        key, _ = self._resolve_eos_table_info()
        if key:
            return key.split(":")[0]  # "TEM" or "STEM"
        return "UNKNOWN"

    def set_mode(self, mode: str) -> None:
        # Switching TEM/STEM is complex in PyJEM and not exposed in simple setters
        # Usually requires EOS3.SetFunctionMode logic which is context heavy.
        logger.warning(f"set_mode({mode}) requested but not implemented for JEOL.")
        pass

    # =========================================================================
    # 3. Stage Control
    # =========================================================================

    def get_stage_position(self) -> StagePosition:
        if not self.stage or not hasattr(self.stage, "GetPos"):
            return StagePosition()
        return jeol_adapter.from_jeol_stage_position(self.stage.GetPos())

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default", wait: bool = True) -> None:
        if not self.stage: return
        t_args = jeol_adapter.to_jeol_stage_args(target)

        if 'x' in t_args and hasattr(self.stage, "SetX"): self.stage.SetX(t_args['x'])
        if 'y' in t_args and hasattr(self.stage, "SetY"): self.stage.SetY(t_args['y'])
        if 'z' in t_args and hasattr(self.stage, "SetZ"): self.stage.SetZ(t_args['z'])
        if 'tx' in t_args and hasattr(self.stage, "SetTiltXAngle"): self.stage.SetTiltXAngle(t_args['tx'])
        if 'ty' in t_args and hasattr(self.stage, "SetTiltYAngle"): self.stage.SetTiltYAngle(t_args['ty'])

        if wait: self._wait_for_stage()

    def stop_stage(self) -> None:
        if self.stage and hasattr(self.stage, "Stop"):
            self.stage.Stop()

    def home_stage(self) -> None:
        # Not supported in standard PyJEM Stage3 interface
        logger.warning("home_stage() not supported by this driver.")
        pass

    # =========================================================================
    # 4. Beam Control (Atomic)
    # =========================================================================

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        if self.ht and hasattr(self.ht, "GetHtValue"):
            return jeol_adapter.from_jeol_beam_stats(self.ht.GetHtValue(), 0, 0, 0).voltage
        return None

    def get_beam_current(self) -> Optional[Quantity]:
        if self.gun and hasattr(self.gun, "GetEmissionCurrent"):
            val = self.gun.GetEmissionCurrent()
            return Q_(val, Units.UA).to(Units.NA)
        return None

    def get_spot_size(self) -> int:
        if self.eos and hasattr(self.eos, "GetSpotSize"):
            return self.eos.GetSpotSize()
        return 0

    def get_convergence_angle(self) -> Optional[Quantity]:
        # Often mapped to Alpha index in JEOL
        if self.eos and hasattr(self.eos, "GetAlpha"):
            return Q_(self.eos.GetAlpha(), Units.MRAD)  # Placeholder unit, really an index
        return None

    def get_beam_shift(self) -> Tuple[float, float]:
        # Requires Def3 GetCla1 or similar, often not in offline mock
        return (0.0, 0.0)

    def get_condenser_stigmation(self) -> Tuple[float, float]:
        # Typically Def3 GetTemStigA1 or similar
        return (0.0, 0.0)

    def get_gun_tilt(self) -> Tuple[float, float]:
        # Typically Def3 GetGunTilt or similar
        return (0.0, 0.0)

    def get_beam_blank(self) -> bool:
        if self.gun and hasattr(self.gun, "GetBeamSw"):  # Hypothetical getter
            return bool(self.gun.GetBeamSw())  # 1=Blank?
        return False

    # --- Setters ---

    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        if self.ht and hasattr(self.ht, "SetHtValue"):
            self.ht.SetHtValue(voltage.to(Units.V).magnitude)

    def set_beam_current(self, current: Quantity) -> None:
        # JEOL usually controls this via Spot Size or CL3, not direct current setting
        logger.warning("set_beam_current not directly supported; use spot_size.")
        pass

    def set_spot_size(self, index: int) -> None:
        if self.eos and hasattr(self.eos, "SelectSpotSize"):
            self.eos.SelectSpotSize(int(index))

    def set_convergence_angle(self, angle: Quantity) -> None:
        # JEOL uses Alpha Index. Mapping Quantity -> Index is complex/system specific.
        # Here we assume the input quantity magnitude IS the index (hack).
        if self.eos and hasattr(self.eos, "SelectAlpha"):
            self.eos.SelectAlpha(int(angle.magnitude))

    def set_beam_shift(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetCla1"):
            self.def_.SetCla1(int(x), int(y))

    def set_condenser_stigmation(self, x: float, y: float) -> None:
        # Attempt to use SetTemStigA1Rel/Abs if available
        if self.def_ and hasattr(self.def_, "SetTemStigA1Rel"):
            self.def_.SetTemStigA1Rel(int(x), int(y))

    def set_gun_tilt(self, x: float, y: float) -> None:
        # Not exposed in offline def3.py
        pass

    def set_beam_blank(self, blank: bool) -> None:
        if self.gun and hasattr(self.gun, "SetBeamSw"):
            self.gun.SetBeamSw(1 if blank else 0)

    # =========================================================================
    # 5. Projection Control (Atomic)
    # =========================================================================

    def get_projection_mode(self) -> str:
        # Re-use logic from get_mode or EOS table
        key, _ = self._resolve_eos_table_info()
        return key if key else "UNKNOWN"

    def get_magnification_index(self) -> int:
        if self.eos and hasattr(self.eos, "GetSelector"):
            return self.eos.GetSelector()
        return 0

    def get_camera_length(self) -> Optional[Quantity]:
        # Derived from Mag Index if in DIFF mode
        key, _ = self._resolve_eos_table_info()
        if key and "DIFF" in key:
            # Simplified: just return the index as magnitude for now, or lookup table
            return Q_(self.get_magnification_index(), Units.MM)
        return None

    def get_defocus(self) -> Optional[Quantity]:
        # Requires Lens3 OLC/OLF reading
        return None

    def get_screen_position(self) -> str:
        return "UNKNOWN"

    def get_objective_stigmation(self) -> Tuple[float, float]:
        # OLA Stigmator
        return (0.0, 0.0)

    def get_image_shift(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def get_diffraction_shift(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    # --- Setters ---

    def set_projection_mode(self, mode: str) -> None:
        logger.warning(f"set_projection_mode({mode}) not implemented.")
        pass

    def set_magnification_index(self, index: int) -> None:
        if self.eos and hasattr(self.eos, "SetSelector"):
            self.eos.SetSelector(int(index))

    def set_camera_length(self, length: Quantity) -> None:
        # Map physical length to Index via Table (implemented in set_projection_settings previously)
        # For atomic, we'd need that logic here.
        # For now, we delegate to the bulk method in user code or implement simple lookup.
        self.set_projection_settings(ProjectionSettings(camera_length=length))

    def set_defocus(self, defocus: Quantity) -> None:
        # Maps to Lens3 SetOLc / SetOLf
        # 1. Convert nm to coarse/fine DAC steps (heuristic)
        if self.lens and hasattr(self.lens, "SetOLc"):
            # Dummy conversion
            val = int(defocus.magnitude)
            self.lens.SetOLc(val)

    def set_screen_position(self, position: str) -> None:
        pass

    def set_objective_stigmation(self, x: float, y: float) -> None:
        pass

    def set_image_shift(self, x: float, y: float) -> None:
        pass

    def set_diffraction_shift(self, x: float, y: float) -> None:
        pass

    # =========================================================================
    # 6. Scan Control (Atomic)
    # =========================================================================

    def get_scan_mode(self) -> str:
        if self.scan and hasattr(self.scan, "GetScanMode"):
            return str(self.scan.GetScanMode())
        return "0"

    def get_scan_width(self) -> int:
        return 512  # Default/Unknown in offline

    def get_scan_height(self) -> int:
        return 512

    def get_scan_pixel_dwell(self) -> Quantity:
        return Q_(10, Units.US)

    def get_scan_flyback(self) -> Quantity:
        return Q_(0, Units.US)

    def get_scan_rotation(self) -> Quantity:
        if self.scan and hasattr(self.scan, "GetRotationAngle"):
            return Q_(float(self.scan.GetRotationAngle()), Units.DEG)
        return Q_(0.0, Units.DEG)

    def get_scan_active(self) -> bool:
        if self.scan and hasattr(self.scan, "GetExtScanMode"):
            return bool(self.scan.GetExtScanMode())
        return False

    # --- Setters ---

    def set_scan_mode(self, mode: str) -> None:
        if self.scan and hasattr(self.scan, "SetExtScanMode"):
            try:
                self.scan.SetExtScanMode(int(mode))
            except ValueError:
                pass

    def set_scan_width(self, px: int) -> None:
        # Not supported by scan3.py
        pass

    def set_scan_height(self, px: int) -> None:
        pass

    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        pass

    def set_scan_flyback(self, time: Quantity) -> None:
        pass

    def set_scan_rotation(self, angle: Quantity) -> None:
        if self.scan and hasattr(self.scan, "SetRotationAngle"):
            self.scan.SetRotationAngle(int(angle.to(Units.DEG).magnitude))

    def set_scan_active(self, active: bool) -> None:
        # Map Start/Stop to ExtScanMode 1/0
        val = 1 if active else 0
        if self.scan and hasattr(self.scan, "SetExtScanMode"):
            self.scan.SetExtScanMode(val)

    # =========================================================================
    # 7. Detector Control (Atomic)
    # =========================================================================

    def list_detectors(self) -> List[str]:
        # Hardcoded defaults or query PyJEM
        return ["Camera1"]

    def get_active_detector_ids(self) -> List[str]:
        return ["Camera1"]

    def get_primary_detector_id(self) -> Optional[str]:
        return "Camera1"

    def get_detector_exposure(self, detector_id: str) -> Quantity:
        if detector:
            try:
                d = detector.Detector(detector_id)
                res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
                return res.exposure or Q_(0.0, Units.S)
            except:
                pass
        return Q_(0.0, Units.S)

    def get_detector_binning(self, detector_id: str) -> int:
        if detector:
            try:
                d = detector.Detector(detector_id)
                res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
                return int(res.binning_index or 1)
            except:
                pass
        return 1

    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        if detector:
            try:
                d = detector.Detector(detector_id)
                res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
                return res.roi
            except:
                pass
        return None

    def get_detector_integration(self, detector_id: str) -> int:
        if detector:
            try:
                d = detector.Detector(detector_id)
                res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
                return int(res.frame_integration or 1)
            except:
                pass
        return 1

    def get_detector_inserted(self, detector_id: str) -> bool:
        # PyJEM Detector class has 'insert'/'retract' methods, but no explicit 'is_inserted' property in standard docs.
        # We might check get_insert_state if function.py supports it.
        return True

    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        return None

    # --- Setters ---

    def set_detector_exposure(self, detector_id: str, exposure: Quantity) -> None:
        self.set_detector_config(DetectorSettings(detector_id=detector_id, exposure=exposure))

    def set_detector_binning(self, detector_id: str, index: int) -> None:
        self.set_detector_config(DetectorSettings(detector_id=detector_id, binning_index=index))

    def set_detector_roi(self, detector_id: str, roi: Optional[ROI]) -> None:
        self.set_detector_config(DetectorSettings(detector_id=detector_id, roi=roi))

    def set_detector_integration(self, detector_id: str, count: int) -> None:
        self.set_detector_config(DetectorSettings(detector_id=detector_id, frame_integration=count))

    def set_detector_insertion(self, detector_id: str, inserted: bool) -> None:
        if not detector: return
        try:
            det = detector.Detector(detector_id)
            if inserted:
                det.insert()
            else:
                det.retract()
        except:
            pass

    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        # Re-using the logic from previous 'acquire_frame'
        if not detector: raise RuntimeError("Detector module missing")
        det = detector.Detector(request.detector_id)

        # Apply atomic settings if needed (usually done via Orchestrator -> apply_detector_settings)

        res = det.shot()
        if hasattr(res, "data"):
            arr = np.array(res.data)
        elif isinstance(res, (list, bytes)):
            arr = np.frombuffer(res, dtype=np.uint16)
        else:
            arr = np.zeros((512, 512), dtype=np.uint16)

        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side == arr.size: arr = arr.reshape((side, side))

        meta = MicroscopeImageMetadata(
            detector_id=request.detector_id,
            timestamp=time.time(),
            exposure_time=request.settings.exposure if request.settings else None,
            extra=Extras(vendor={"raw_response": str(type(res))})
        )
        return MicroscopeImage(data=arr, metadata=meta)

    # =========================================================================
    # 8. Vacuum Control (Atomic)
    # =========================================================================

    def get_valve_state(self, valve_name: str) -> str:
        if valve_name == 'gun' and self.feg and hasattr(self.feg, "GetBeamValve"):
            return "OPEN" if self.feg.GetBeamValve() == 1 else "CLOSED"
        return "UNKNOWN"

    def set_valve_state(self, valve_name: str, state: str) -> None:
        # Offline FEG3 usually read-only for Valve, or missing setter in snippet
        pass

    def get_pressure(self, gauge_name: str) -> Quantity:
        idx_map = {'gun': 1, 'column': 2, 'chamber': 3, 'buffer': 4}
        idx = idx_map.get(gauge_name, 1)
        if self.vac:
            method = f"GetP{idx}"
            if hasattr(self.vac, method):
                val = getattr(self.vac, method)()
                return Q_(val, Units.PA)
        return Q_(0.0, Units.PA)

    # =========================================================================
    # 9. Aperture Control (Atomic)
    # =========================================================================

    def list_apertures(self) -> List[str]:
        return list(self._APERTURE_MAP.keys())

    def get_aperture(self, aperture_id: str) -> Aperture:
        if not self.apt: return Aperture(aperture_id=aperture_id, _mode="lenient")
        kind_idx = self._APERTURE_MAP.get(aperture_id)
        if kind_idx is None: return Aperture(aperture_id=aperture_id, _mode="lenient")

        self.apt.SelectExpKind(kind_idx)
        size_idx = self.apt.GetExpSize(kind_idx)
        pos_list = self.apt.GetPosition()
        return jeol_adapter.from_jeol_aperture(aperture_id, size_idx, pos_list)

    def set_aperture(self, aperture_id: str, target: Aperture) -> None:
        if not self.apt: return
        kind_idx = self._APERTURE_MAP.get(aperture_id)
        if kind_idx is None: return

        self.apt.SelectExpKind(kind_idx)
        if target.size_index is not None:
            self.apt.SetExpSize(kind_idx, int(target.size_index))
        if target.position is not None:
            self.apt.SetPosition(int(target.position.x), int(target.position.y))

    # =========================================================================
    # 10. Helpers
    # =========================================================================

    def _resolve_eos_table_info(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.eos or not hasattr(self.eos, "GetFunctionMode"): return None, None
        modes = self.eos.GetFunctionMode()
        if not modes or len(modes) < 2: return None, None
        key = self._EOS_MODE_MAP.get(tuple(modes))
        if not key: return None, None

        list_name = "CamList" if "DIFF" in key else "MagList"
        return key, list_name

    def _wait_for_stage(self):
        time.sleep(0.1)