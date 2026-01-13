import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Import Abstract Base and Strict Structures
from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    MicroscopeSettings,
    SystemInfo,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    DetectorSettings,
    DetectorCapabilities,
    ScanSettings,
    VacuumSettings,
    Aperture,
    MicroscopeImage,
    MicroscopeImageMetadata,
    Point,
    ROI,
    AcquisitionRequest,
    Units,
    Q_,
    Quantity,
    Extras
)

# Import Vendor Adapters
from supertem.vendor.JEOL import jeol_adapter
from supertem.vendor.JEOL.jeol_eos_tables import get_list

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
    """

    # Corrected Aperture Mapping
    # 0=CL1, 1=CL2, 2=OL, 3=HC, 4=SA, 5=ENT, 6=HX, 7=BF, 8-11=AUX
    _APERTURE_MAP = {
        # Hardware Indices
        "CL1": 0, "CL2": 1, "OL": 2, "HC": 3, "SA": 4,
        "ENT": 5, "HX": 6, "BF": 7,
        "AUX1": 8, "AUX2": 9, "AUX3": 10, "AUX4": 11,
        # Common Aliases
        "CLA": 1,  # Main Condenser (CL2)
        "OLA": 2,  # Objective
        "HCA": 3,  # High Contrast / Lower OL
        "SAA": 4,  # Selected Area
        "ENTA": 5,
        "EDS": 6
    }

    # Mapping Projection Modes (Ref: eos3.py)
    _TEM_FUNC_MAP = {
        "MAG": 0, "MAG2": 1, "LOWMAG": 2, "SAMAG": 3, "DIFF": 4
    }

    def __init__(self, settings: Optional[MicroscopeSettings] = None):
        super().__init__(settings)

        # Hardware Interface Placeholders
        self.apt = None
        self.deflector = None
        self.eos = None
        self.feg = None
        self.gun = None
        self.ht = None
        self.lens = None
        self.stage = None
        self.scan = None
        self.vac = None

        self._active_detector_obj = None
        self._active_detector_id: Optional[str] = None

        # State Caches
        self._defocus_cache = Q_(0, Units.NM)
        self._connected = False

    # =========================================================================
    # 1. Connection & Lifecycle
    # =========================================================================

    def connect(self, host: str = "localhost", port: Optional[int] = 8088, **kwargs) -> None:
        if TEM3 is None:
            raise RuntimeError("PyJEM library not found.")

        try:
            self.apt = TEM3.Apt3()
            self.deflector = TEM3.Def3()
            self.eos = TEM3.EOS3()
            self.feg = TEM3.FEG3()
            self.gun = TEM3.GUN3()
            self.ht = TEM3.HT3()
            self.lens = TEM3.Lens3()
            self.scan = TEM3.Scan3() if hasattr(TEM3, 'Scan3') else None
            self.stage = TEM3.Stage3()
            self.vac = TEM3.VACUUM3() if hasattr(TEM3, 'VACUUM3') else None

            self._connected = True
            logger.info(f"Connected to JEOL Microscope at {host}")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to PyJEM: {e}")
            raise e

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_instrument_info(self) -> SystemInfo:
        return SystemInfo(manufacturer="JEOL", model="ARM/F2 Series", software_version="PyJEM TEM3")

    # =========================================================================
    # 2. Global State & EOS Tables
    # =========================================================================

    def get_mode(self) -> str:
        if not self.eos: return "UNKNOWN"
        idx = self.eos.GetTemStemMode()  # 0=TEM, 1=STEM
        return "STEM" if idx == 1 else "TEM"

    def set_mode(self, mode: str) -> None:
        if not self.eos: return
        m = mode.upper()
        if m == "TEM":
            self.eos.SelectTemStem(0)
        elif m == "STEM":
            self.eos.SelectTemStem(1)
        else:
            logger.warning(f"Unsupported mode '{mode}', defaulting to TEM")

    def _get_current_eos_table(self) -> List[Tuple[float, str, str]]:
        if not self.eos: return []

        mode_idx = self.eos.GetTemStemMode()
        mode_str = "STEM" if mode_idx == 1 else "TEM"

        func_ret = self.eos.GetFunctionMode()
        if not func_ret or len(func_ret) < 2:
            return []

        func_name = str(func_ret[1]).upper()
        table_key = f"{mode_str}:{func_name}"

        try:
            return get_list(table_key, "MagList")
        except KeyError:
            try:
                return get_list(table_key, "StemCamList")
            except KeyError:
                return []

    # =========================================================================
    # 3. ATOMIC METHODS (Implementation of Abstract Base Class)
    # =========================================================================
    # These methods provide direct, unbuffered access to hardware controls.

    # --- Beam Atomic ---

    def set_voltage(self, voltage: Quantity) -> None:
        if self.ht:
            self.ht.SetHtValue(voltage.to(Units.V).magnitude)

    def set_spot_size(self, index: int) -> None:
        if self.eos:
            self.eos.SelectSpotSize(int(index))

    def set_convergence_angle(self, index: int) -> None:
        # JEOL uses Alpha Index (1-9)
        if self.eos:
            self.eos.SelectAlpha(int(index))

    def set_beam_shift(self, x: float, y: float) -> None:
        if self.deflector:
            self.deflector.SetCLA1(int(x), int(y))

    def set_condenser_stigmator(self, x: float, y: float) -> None:
        if self.deflector:
            # JEOL CLs Deflector
            self.deflector.SetCLs(int(x), int(y))

    def set_beam_blank(self, blank: bool) -> None:
        if self.deflector:
            # 1=ON (Blank), 0=OFF
            self.deflector.SetBeamBlank(1 if blank else 0)

    def set_gun_tilt(self, x: float, y: float) -> None:
        if self.deflector:
            self.deflector.SetGunA1(int(x), int(y))

    # --- Projection Atomic ---

    def set_magnification_index(self, index: int) -> None:
        if self.eos:
            self.eos.SetSelector(int(index))

    def set_image_shift(self, x: float, y: float) -> None:
        if self.deflector:
            self.deflector.SetIS1(int(x), int(y))

    def set_objective_stigmator(self, x: float, y: float) -> None:
        if self.deflector:
            self.deflector.SetOLs(int(x), int(y))

    def set_diffraction_shift(self, x: float, y: float) -> None:
        if self.deflector:
            self.deflector.SetPLA(int(x), int(y))

    def set_defocus(self, defocus: Quantity) -> None:
        # Hardware GetDefocus not available; SetDefocus is relative/unsafe.
        # We only update the software cache here.
        self._defocus_cache = defocus

    # --- Scan Atomic ---

    def set_scan_rotation(self, rotation: Quantity) -> None:
        if self.scan:
            deg = rotation.to(Units.DEG).magnitude
            self.scan.SetRotationAngle(int(deg))

    def set_scan_pixel_dwell(self, dwell: Quantity) -> None:
        pass  # Not supported in Scan3

    def set_scan_flyback(self, time: Quantity) -> None:
        pass  # Not supported in Scan3

    # =========================================================================
    # 4. Helper Methods (Bulk Getters/Setters)
    # =========================================================================

    def get_stage_position(self) -> StagePosition:
        if not self.stage: return StagePosition()
        raw_pos = self.stage.GetPos()
        return jeol_adapter.from_jeol_stage_position(raw_pos)

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default", wait: bool = True) -> None:
        if not self.stage: return

        t_args = jeol_adapter.to_jeol_stage_args(target)

        if 'x' in t_args: self.stage.SetX(t_args['x'])
        if 'y' in t_args: self.stage.SetY(t_args['y'])
        if 'z' in t_args: self.stage.SetZ(t_args['z'])
        if 'tx' in t_args: self.stage.SetTiltXAngle(t_args['tx'])
        if 'ty' in t_args: self.stage.SetTiltYAngle(t_args['ty'])

        if wait:
            self._wait_for_stage()

    def _wait_for_stage(self, timeout: float = 30.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                status = self.stage.GetStatus()
                if isinstance(status, list) and all(s == 0 for s in status):
                    return
            except Exception:
                pass
            time.sleep(0.1)

    def stop_stage(self) -> None:
        if self.stage: self.stage.Stop()

    def get_beam_settings(self) -> BeamSettings:
        v_val = self.ht.GetHtValue() if self.ht else 0.0
        c_val = self.gun.GetEmissionCurrentValue() if self.gun else 0.0
        spot_idx = self.eos.GetSpotSize() if self.eos else 0
        alpha_idx = self.eos.GetAlpha() if self.eos else 0
        shift_dac = self.deflector.GetCLA1() if self.deflector else (0, 0)

        return jeol_adapter.from_jeol_beam_stats(
            voltage_v=v_val,
            current_ua=c_val,
            spot_size_idx=spot_idx,
            alpha_idx=alpha_idx,
            beam_shift_dac=shift_dac
        )

    def set_beam_settings(self, settings: BeamSettings) -> None:
        if settings.voltage is not None: self.set_voltage(settings.voltage)
        if settings.spot_size_index is not None: self.set_spot_size(settings.spot_size_index)
        if settings.convergence_angle_index is not None: self.set_convergence_angle(settings.convergence_angle_index)
        if settings.shift is not None: self.set_beam_shift(settings.shift.x, settings.shift.y)
        if settings.beam_on is not None: self.set_beam_blank(not settings.beam_on)

    def get_projection_settings(self) -> ProjectionSettings:
        mode_str = "UNKNOWN"
        if self.eos:
            ret = self.eos.GetFunctionMode()
            if ret and len(ret) > 1:
                mode_str = str(ret[1]).upper()

        mag_idx = self.eos.GetCurrentMagSelectorID() if self.eos else 0
        mag_val = None
        cam_len = None

        try:
            table = self._get_current_eos_table()
            if 0 <= mag_idx < len(table):
                val, unit, label = table[mag_idx]
                if unit.upper() == "X":
                    mag_val = val
                elif unit.lower() in ["cm", "mm"]:
                    cam_len = Q_(val, unit.lower())
        except Exception as e:
            logger.warning(f"Could not resolve mag table: {e}")

        defocus_val = self._defocus_cache
        ishift = self.deflector.GetIS1() if self.deflector else (0, 0)
        stig = self.deflector.GetOLs() if self.deflector else (0, 0)

        return ProjectionSettings(
            mode=mode_str,
            magnification=mag_val,
            magnification_index=mag_idx,
            camera_length=cam_len,
            defocus=defocus_val,
            image_shift=Point(x=ishift[0], y=ishift[1]),
            objective_stigmator=Point(x=stig[0], y=stig[1]),
            _mode="lenient"
        )

    def set_projection_settings(self, settings: ProjectionSettings) -> None:
        if not self.eos or not self.deflector: return

        if settings.mode:
            idx = self._TEM_FUNC_MAP.get(settings.mode.upper())
            if idx is not None:
                self.eos.SelectFunctionMode(idx)
                time.sleep(0.5)

        if settings.magnification_index is not None:
            self.set_magnification_index(settings.magnification_index)
        elif settings.magnification is not None or settings.camera_length is not None:
            self._set_mag_from_value(settings.magnification, settings.camera_length)

        if settings.defocus is not None:
            self.set_defocus(settings.defocus)

        if settings.image_shift:
            self.set_image_shift(settings.image_shift.x, settings.image_shift.y)

        if settings.objective_stigmator:
            self.set_objective_stigmator(settings.objective_stigmator.x, settings.objective_stigmator.y)

    def _set_mag_from_value(self, mag_target: Optional[float], cam_target: Optional[Quantity]):
        table = self._get_current_eos_table()
        if not table: return

        best_idx = -1
        min_diff = float('inf')

        for i, (val, unit, label) in enumerate(table):
            if mag_target is not None and unit.upper() == "X":
                diff = abs(val - mag_target)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            elif cam_target is not None and unit.lower() in ["cm", "mm"]:
                try:
                    target_val = cam_target.to(unit.lower()).magnitude
                    diff = abs(val - target_val)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                except:
                    pass

        if best_idx >= 0:
            self.set_magnification_index(best_idx)

    # =========================================================================
    # 5. Detector Control
    # =========================================================================

    def _ensure_detector(self, detector_id: Optional[str]):
        if not detector_id:
            detector_id = self.get_primary_detector_id()

        if self._active_detector_id != detector_id:
            if detector:
                self._active_detector_obj = detector.Detector(detector_id)
                self._active_detector_id = detector_id
            else:
                raise RuntimeError("PyJEM.detector module not available")
        return self._active_detector_obj

    def list_detectors(self) -> List[str]:
        if detector:
            return detector.get_attached_detector()
        return []

    def get_primary_detector_id(self) -> Optional[str]:
        dets = self.list_detectors()
        return dets[0] if dets else None

    def get_detector_settings(self, detector_id: str) -> DetectorSettings:
        det = self._ensure_detector(detector_id)
        raw_conf = det.get_detectorsetting()
        settings, _ = jeol_adapter.from_jeol_detector_response(raw_conf, detector_id)
        return settings

    def get_detector_capabilities(self, detector_id: str) -> DetectorCapabilities:
        det = self._ensure_detector(detector_id)
        raw_conf = det.get_detectorsetting()
        _, caps = jeol_adapter.from_jeol_detector_response(raw_conf, detector_id)
        return caps

    def set_detector_settings(self, settings: DetectorSettings) -> None:
        det_id = settings.detector_id or self._active_detector_id
        if not det_id: return
        det = self._ensure_detector(det_id)

        jeol_conf = jeol_adapter.to_jeol_detector_config(settings)
        if jeol_conf:
            det.set_detectorsetting(jeol_conf)

    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        det_id = request.detector_id or self.get_primary_detector_id()
        if not det_id: raise ValueError("No detector selected.")

        if request.detector:
            request.detector.detector_id = det_id
            self.set_detector_settings(request.detector)

        det = self._ensure_detector(det_id)

        raw_bytes = det.snapshot(ext="tiff")

        try:
            import tifffile
            from io import BytesIO
            image_data = tifffile.imread(BytesIO(raw_bytes))
        except ImportError:
            logger.warning("tifffile missing. Returning empty.")
            image_data = np.zeros((512, 512), dtype=np.uint8)

        meta = MicroscopeImageMetadata(
            microscope_state=self.get_full_state()
        )
        return MicroscopeImage(data=image_data, metadata=meta)

    # =========================================================================
    # 6. Aperture Control
    # =========================================================================

    def list_apertures(self) -> List[str]:
        return list(self._APERTURE_MAP.keys())

    def get_aperture(self, aperture_id: str) -> Aperture:
        kind_idx = self._APERTURE_MAP.get(aperture_id)
        if kind_idx is None or not self.apt:
            return Aperture(aperture_id=aperture_id, _mode="lenient")

        self.apt.SelectExpKind(kind_idx)
        size = self.apt.GetExpSize(kind_idx)
        pos = self.apt.GetPosition()

        return jeol_adapter.from_jeol_aperture(aperture_id, size, pos)

    def set_aperture(self, aperture_id: str, target: Aperture) -> None:
        kind_idx = self._APERTURE_MAP.get(aperture_id)
        if kind_idx is None or not self.apt: return

        self.apt.SelectExpKind(kind_idx)
        if target.size_index is not None:
            self.apt.SetExpSize(kind_idx, int(target.size_index))

        if target.position:
            self.apt.SetPosition(int(target.position.x), int(target.position.y))

    # =========================================================================
    # 7. Vacuum & Scan
    # =========================================================================

    def get_vacuum_status(self) -> VacuumSettings:
        if not self.vac: return VacuumSettings()
        pressures = []
        for i in range(1, 6):
            m = f"GetP{i}"
            if hasattr(self.vac, m):
                pressures.append(getattr(self.vac, m)())
            else:
                pressures.append(0.0)

        valve_flags = {}
        if self.feg:
            valve_flags['gun'] = self.feg.GetBeamValve()

        return jeol_adapter.from_jeol_vacuum_stats(pressures, valve_flags)

    def get_scan_settings(self) -> ScanSettings:
        if not self.scan: return ScanSettings()
        rot = self.scan.GetRotationAngle()
        return jeol_adapter.from_jeol_scan_stats(rotation_deg=float(rot))

    def set_scan_settings(self, settings: ScanSettings) -> None:
        if settings.rotation is not None:
            self.set_scan_rotation(settings.rotation)