import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from io import BytesIO
from pint import Quantity
from dataclasses import asdict

from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    ImageSettings,
    MicroscopeSettings,
    Q_,
    ensure_quantity,
    magnitude,
    TemImage,
    TemImageMetadataRefined,
    TemStagePosition,
    TemDetectorSettings
)
from supertem.vendor.JEOL.jeol_eos_tables import get_list, list_unit
from supertem.vendor.JEOL.jeol_settings import from_jeol_detector_setting, to_jeol_detector_setting

logger = logging.getLogger(__name__)
try:
    from PyJEM import TEM3  # type: ignore
except Exception: # pragma: no cover
    logger.warning("TEM3 is not available; trying offline version.")
    try:
        from PyJEM.offline import TEM3
        logger.info("offline.TEM3 is available.")
    except Exception:
        TEM3 = None
        logger.error("TEM3 is None.")

try:
    from PyJEM import detector
except Exception:
    logger.warning("Detector is not available, trying offline version")
    try:
        from PyJEM.offline import detector
        logger.info("offline.detector is available")
    except Exception:
        detector = None
        logger.error("detector is None.")


class JeolMicroscope(TemMicroscope):
    """
        JEOL implementation using PyJEM TEM3 API.

        Important reality check:
          - PyJEM exposes many controls in "device units" (knob steps, I/O values).
          - Some high-level features (camera acquisition, live view) depend on the camera system
            and may NOT be provided by TEM3 alone. Those methods raise NotImplementedError by default.
        """

    # Function mode maps (per EOS3.SelectFunctionMode docstring)
    _TEM_FUNCTION_MAP = {
        "mag": 0,
        "mag2": 1,
        "lowmag": 2,
        "samag": 3,
        "diff": 4,
    }
    _STEM_FUNCTION_MAP = {
        "align": 0,
        "sm-lmag": 1,
        "sm-mag": 2,
        "amag": 3,
        "uudiff": 4,
        "rocking": 5,
    }

    def __init__(self, settings: Optional[MicroscopeSettings] = None, logger_: Optional[logging.Logger] = None):
        if TEM3 is None:
            raise ImportError("PyJEM TEM3 is not available. Install PyJEM on the microscope control PC.")

        if settings is None:
            settings = MicroscopeSettings()
        self.settings = settings
        self._system_settings = settings.system
        self._image_settings = settings.image
        self.logger = logger_ or logging.getLogger(__name__)



        # Cached "software state" for features TEM3 doesn't report back reliably
        self._selected_aperture: Optional[str] = None
        self._selected_detector: Optional[Any] = None
        self._defocus_cache: Optional[Quantity] = Q_(0, "nanometer")  # cached defocus (nm by default)
        self._stigmation_cache: Tuple[float, float] = (0.0, 0.0)
        self.aperture_dict = dict(CL1 = 0, CL2 = 1, OL_Upper = 2, Ol_Lower = 3, SA = 4, ENT = 5, HX = 6, BF = 7, AUX1 = 8, AUX2 = 9, AUX3 = 10, AUX4 = 11)

    def _log_event(self, msg: str) -> None:
        try:
            self.logger.info(msg)
        except Exception:
            pass

    def connect_to_microscope(self, ip_address: str = "0.0.0.0", port: int = 0, timeout_s: float = 5.0) -> None:
        try:
            TEM3.connect()
            self._log_event("Connected via TEM3.connect()")
            # TEM3 controllers
            self.apt = TEM3.Apt3()
            self.deflector = TEM3.Def3()
            self.detector = TEM3.Detector3()
            self.eos = TEM3.EOS3()
            self.feg = TEM3.FEG3()
            self.gun = TEM3.GUN3()
            self.ht = TEM3.HT3()
            self.lens = TEM3.Lens3()
            self.stage = TEM3.Stage3()
            self.vac = TEM3.VACUUM3()
        except Exception:
            self._log_event("Unable to connect via TEM3.connect()")

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        try:
            return bool(TEM3.is_connect())
        except Exception:
            try:
                return bool(TEM3.get_connect_status())
            except Exception:
                return False

    # -----------------------
    # Status / Info
    # -----------------------

    def get_instrument_info(self) -> Dict[str, Any]:
        return {
            "vendor": "JEOL",
            "api": "PyJEM TEM3",
        }

    def get_status(self) -> Dict[str, Any]:
        temstem = self.eos.GetTemStemMode()
        func = self.eos.GetFunctionMode()  # [index, name]
        mag = self.get_magnification()
        stage_pos = self.get_stage_position()

        return {
            "connected": self.is_connected(),
            "mode": ("TEM" if temstem == 0 else "STEM"),
            "function_mode": {"index": func[0], "name": func[1]} if isinstance(func, list) else func,
            "magnification_or_cam_length": mag,
            "beam_blank": self.get_beam_blank(),
            "ht_kv": magnitude(self.get_acceleration_voltage(), "kilovolt"),
            "emission_current": magnitude(self.get_emission_current(), "microampere"),
            "vacuum": {
                "column_ready": self.vac.GetColumnReady(),
                "camera_ready": self.vac.GetCameraReady(),
                "specimen_ready": self.vac.GetSpecimenReady(),
                "valves": self.vac.GetValveStatus(),
            },
            "stage": {
                "position": stage_pos.to_dict(),
                "status": self.get_stage_status(),
                "holder": self.stage.GetHolderStts(),
            },
        }

    # -----------------------
    # Mode
    # -----------------------

    def get_mode(self) -> str:
        temstem = self.eos.GetTemStemMode()
        func = self.eos.GetFunctionMode()  # [index, name]
        prefix = "TEM" if temstem == 0 else "STEM"
        if isinstance(func, list) and len(func) >= 2:
            if temstem == 0:
                function = [k for k, v in self._TEM_FUNCTION_MAP.items() if v == func[0]][0]
            else:
                function = [k for k, v in self._STEM_FUNCTION_MAP.items() if v == func[0]][0]
            return f"{prefix}:{str(function).upper()}"
        return prefix

    def set_mode(self, mode: str) -> None:
        """Set observation + EOS function mode.

        Format: "TEM:function" or "STEM:function", e.g. "TEM:MAG", "TEM:DIFF", "STEM:SM-MAG".
        """
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("mode must be a non-empty string")

        raw = mode.strip()
        parts = raw.split(":", 1)
        obs = parts[0].strip().upper()
        func = parts[1].strip() if len(parts) == 2 else ""

        if obs == "TEM":
            self.eos.SelectTemStem(0)
            if func:
                idx = self._TEM_FUNCTION_MAP.get(func.replace(" ", "").lower())
                if idx is None:
                    raise ValueError(f"Unknown TEM function mode: {func}")
                self.eos.SelectFunctionMode(idx)

        elif obs == "STEM":
            self.eos.SelectTemStem(1)
            if func:
                key = func.replace(" ", "").lower()
                idx = self._STEM_FUNCTION_MAP.get(key)
                if idx is None:
                    raise ValueError(f"Unknown STEM function mode: {func}")
                self.eos.SelectFunctionMode(idx)

        else:
            raise ValueError(f"Unknown observation mode: {obs}")

        self._log_event(f"Set mode -> {self.get_mode()}")

    # -----------------------
    # Beam / HT
    # -----------------------

    def set_acceleration_voltage(self, voltage: Optional[Quantity]) -> None:
        v = ensure_quantity(voltage, "volt")
        self.ht.SetHtValue(float(v.magnitude))
        self._log_event(f"Set acceleration voltage -> {v.to('kilovolt')}")

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        return Q_(float(self.ht.GetHtValue()), "volt").to("kilovolt")

    def get_emission_current(self) -> Optional[Quantity]:
        # PyJEM Gun emission current value is typically in µA, but treat as vendor-defined.
        return Q_(float(self.gun.GetEmissionCurrentValue()), "microampere")

    def set_beam_blank(self, blank: bool) -> None:
        self.deflector.SetBeamBlank(1 if blank else 0)
        self._log_event(f"Beam blank -> {bool(blank)}")

    def get_beam_blank(self) -> bool:
        return bool(self.deflector.GetBeamBlank())

    # -----------------------
    # Mag / Spot
    # -----------------------

    def set_magnification(self, mag: float) -> None:
        """Set magnification (unitless, in X) using the current EOS mode's MagList."""
        mag_list = get_list(self.get_mode(), "MagList")
        unit = list_unit(mag_list).strip().upper()

        if unit != "X":
            raise ValueError(
                f"Current mode '{self.get_mode()}' does not use magnification selectors (MagList unit='{unit}')."
            )

        target = float(mag)
        values = [v for v, _, _ in mag_list]
        idx = max((i for i, v in enumerate(values) if v <= target), default=0)

        self.eos.SetSelector(idx)
        self._log_event(f"Set magnification -> {self.get_magnification()} X")

    def get_magnification(self) -> float:
        """Get magnification (unitless, in X).

        In TEM DIFF (and some special function modes), JEOL reports a *length* here instead.
        We guard against that by checking the returned unit string.
        """
        val = self.eos.GetMagValue()  # [value, unit, label]
        if isinstance(val, list) and len(val) >= 2:
            v = float(val[0])
            unit = str(val[1]).strip().upper()
            if unit == "X":
                return v
            raise ValueError(
                f"EOS reports unit '{unit}' for GetMagValue() in mode '{self.get_mode()}'. "
                f"That's not magnification; use get_camera_length() instead."
            )
        return float(val)

    def set_camera_length(self, camera_length: Optional[Quantity]) -> None:
        """Set camera length for the current mode.

        - TEM:DIFF uses EOS selector (MagList contains lengths, e.g. cm/mm).
        - STEM:* uses EOS stem camera selector (StemCamList contains lengths).
        """
        camera_length = ensure_quantity(camera_length, "cm")
        temstem = self.eos.GetTemStemMode()
        key = self.get_mode()

        if temstem == 0:
            lst = get_list(key, "MagList")  # in DIFF this is length
            unit = list_unit(lst).strip()
            if unit.lower() not in {"m", "cm", "mm", "um", "nm"}:
                raise ValueError(f"Mode '{self.get_mode()}' does not expose camera length via MagList (unit='{unit}').")
            target = magnitude(camera_length, unit)
            values = [v for v, _, _ in lst]
            idx = max((i for i, v in enumerate(values) if v <= target), default=0)
            self.eos.SetSelector(idx)
        else:
            lst = get_list(key, "StemCamList")
            unit = list_unit(lst).strip()
            if unit.lower() not in {"m", "cm", "mm", "um", "nm"}:
                raise ValueError(
                    f"Mode '{self.get_mode()}' does not expose STEM camera length via StemCamList (unit='{unit}')."
                )
            target = magnitude(camera_length, unit)
            values = [v for v, _, _ in lst]
            idx = max((i for i, v in enumerate(values) if v <= target), default=0)
            self.eos.SetStemCamSelector(idx)

        self._log_event(f"Set camera length -> {self.get_camera_length()}")

    def get_camera_length(self) -> Optional[Quantity]:
        """Get camera length as a Quantity (or None if not applicable in the current mode)."""
        temstem = self.eos.GetTemStemMode()

        if temstem == 0:
            val = self.eos.GetMagValue()  # in DIFF this is length
        else:
            val = self.eos.GetStemCamValue()

        if isinstance(val, list) and len(val) >= 2:
            v = float(val[0])
            unit = str(val[1]).strip()
            if unit.lower() in {"m", "cm", "mm", "um", "nm"}:
                return Q_(v, unit)
        return None

    def set_spot_size(self, index: int) -> None:
        self.eos.SelectSpotSize(int(index))
        self._log_event(f"Spot size -> {index}")

    def get_spot_size(self) -> int:
        return int(self.eos.GetSpotSize())

    # -----------------------
    # Focus / Stig (device units)
    # -----------------------
    def set_defocus(self, defocus: Optional[Quantity]) -> None:
        #need fix: still don't know current defocus, can't directly assign defocus value
        """
        JEOL PyJEM does not provide a universal defocus-in-nm API in TEM3.

        Here we cache `defocus` locally as a Quantity.
        If you calibrate a nm<->steps mapping for your instrument, apply it before calling.
        If you calibrate a nm<->steps mapping for your instrument, apply it before calling.
        """
        try:
            self._defocus_cache = ensure_quantity(defocus, "nanometer")
        except Exception:
            # If caller passes a non-length Quantity (e.g. dimensionless steps), keep magnitude as-is.
            self._defocus_cache = Q_(float(getattr(defocus, "magnitude", defocus)), "dimensionless")
        return
        try:
            self.eos.SetObjFocus(int(round(float(getattr(self._defocus_cache, "magnitude", 0.0)))))
        except Exception:
            pass

    def get_defocus(self) -> Optional[Quantity]:
        return self._defocus_cache

    def set_stigmation(self, x: float, y: float) -> None:
        self._stigmation_cache = (float(x), float(y))
        temstem = self.eos.GetTemStemMode()
        try:
            if temstem == 0:
                self.deflector.SetTemStigA1Rel(int(round(x)), int(round(y)))
            else:
                self.deflector.SetStemStigA1Rel(int(round(x)), int(round(y)))
            self._log_event(f"Stigmation (relative) -> ({x}, {y})")
        except Exception:
            pass

    def get_stigmation(self) -> Tuple[float, float]:
        return self._stigmation_cache

    def align_beam(self) -> None:
        raise NotImplementedError(
            "Beam alignment is instrument/procedure specific; implement with your lab's recipe.")

    # -----------------------
    # Apertures
    # -----------------------

    def list_apertures(self) -> List[str]:
        return list(self.aperture_dict.keys())

    def get_aperture_status(self) -> Dict[str, Any]:
        status = {}
        for aperture, idx in self.aperture_dict.items():
            # PyJEM GetExpSize only retrieves the size of the selected aperture from SelectExpKind, it doesn't use the value of argument 'kind'. Need to first select the aperture.
            self.select_aperture(aperture)
            status[aperture] = self.apt.GetExpSize(idx)
        return status

    def select_aperture(self, kind: str) -> int:
        if kind not in self.aperture_dict:
            raise ValueError(f"Unknown aperture kind: {kind}. Known: {list(self.aperture_dict)}")
        idx = self.aperture_dict[kind]
        self.apt.SelectExpKind(idx)
        self._selected_aperture = kind
        return idx

    def insert_aperture(self, kind: str, size: Optional[int]) -> None:
        if size == 0:
            raise ValueError("Size cannot be zero, use retract_aperture() instead")
        idx = self.select_aperture(kind=kind)
        self._log_event(f"Selected Aperture: {self._selected_aperture}")
        self.apt.SetExpSize(kind = idx, size = size)
        self._log_event(f"Inserting {kind}")
        time.sleep(5)
        self._log_event(f"Aperture: {kind} -> {size}")

    def retract_aperture(self, kind: str) -> None:
        idx = self.select_aperture(kind=kind)
        self._log_event(f"Selected Aperture: {self._selected_aperture}")
        self.apt.SetExpSize(kind = idx, size = 0)
        self._log_event(f"Retracting {kind}")
        time.sleep(5)
        self._log_event(f"Aperture: {kind} -> {0}")

    # -----------------------
    # Detectors
    # -----------------------

    def list_detectors(self) -> List[str]:
        return detector.get_attached_detector()

    def select_detector(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Detector name must be a non-empty string")
        if name not in detector.get_attached_detector():
            raise ValueError("Detector name does not exist")
        self._selected_detector = detector.Detector(name)
        self._log_event(f"Selected detector -> {name}")


    def get_detector_settings(self) -> TemDetectorSettings:
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        return from_jeol_detector_setting(payload=self._selected_detector.get_detectorsetting(), detector_id=self._selected_detector.detector)

    def set_detector_settings(self, settings: Optional[TemDetectorSettings]) -> None:
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        setting = to_jeol_detector_setting(s=settings)
        success = {}
        fail = {}
        for k, v in setting.items():
            try:
                self._selected_detector.set_detectorsetting(dict(k=v))
                success[k] = v
            except Exception:
                fail[k] = v
        print(f"Settings that updated successfully: {success}"
              f", Settings that failed to update: {fail}")
        if len(success) != 0:
            self._log_event(f"{self._selected_detector.detector} settings updated -> {success}")

    @staticmethod
    def _decode_image_bytes(data: bytes, ext: str) -> np.ndarray:
        ext_l = (ext or "tiff").lower().strip(".")
        if ext_l in {"tif", "tiff"}:
            import tifffile as tff
            return tff.imread(BytesIO(data))
        return np.frombuffer(data, dtype=np.uint8)

    def acquire_image(self, settings: Optional[ImageSettings] = None) -> TemImage:
        if settings is None:
            settings = self._image_settings
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        detector_setting = TemDetectorSettings.from_dict(settings.to_dict())
        self.set_detector_settings(settings=detector_setting)
        ext = (settings.file_format or "tif").lower().strip(".")
        if ext == 'tiff':
            ext = 'tif'
        raw = self._selected_detector.snapshot(ext, save=False, filename=None, show=False)
        arr = self._decode_image_bytes(raw, ext)

        try:
            stage = self.get_stage_position()
        except Exception:
            stage = TemStagePosition()

        meta = TemImageMetadataRefined(
            device="JEOL",
            date=datetime.now().date().isoformat(),
            time=datetime.now().time().isoformat(timespec="seconds"),
            magnification=self.get_magnification(),
            accelerating_voltage=magnitude(self.get_acceleration_voltage(), "kilovolt"),
            emission_current=magnitude(self.get_emission_current(), "microampere"),
            dwell_time=magnitude(settings.dwell_us, "microsecond"),
            stage_x=magnitude(stage.x, "nanometer"),
            stage_y=magnitude(stage.y, "nanometer"),
            stage_z=magnitude(stage.z, "nanometer"),
            extra={
                "mode": self.get_mode(),
                "detector": self._selected_detector.detectorname,
                "imaging_area": asdict(settings.roi),
                "binning": settings.binning,
                "exposure_ms": magnitude(settings.exposure_ms, "millisecond"),
            },
        )

        self._log_event(f"Acquire image -> detector={self._selected_detector.detectorname}, format={ext}, shape={getattr(arr, 'shape', None)}")
        return TemImage(data=arr, metadata=meta)


    def start_live(self, settings: Optional[ImageSettings] = None) -> None:
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        self._selected_detector.livestart()
        self._log_event(f"{self._selected_detector.detectorname} live-started")

    def stop_live(self) -> None:
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        self._selected_detector.livestop()
        self._log_event(f"{self._selected_detector.detectorname} live-stopped")

    def get_live_frame(self) -> Optional[bytes]:
        if self._selected_detector is None:
            raise RuntimeError("Detector is not selected (call select_detector first)")
        try:
            raw = self._selected_detector.get_image_cache()
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)

        except Exception:
            # Fallback: a regular snapshot while in live
            ext = "tiff"
            try:
                raw = self._selected_detector.livesnapshot(ext, save=False, filename=None, show=False)
                return bytes(raw)
            except Exception as e:
                raise RuntimeError(f"Failed to fetch live frame: {e}")

    # def get_raw_image_data(self, detector: Optional[str] = None) -> bytes:
    #     if self._selected_detector is None:
    #         raise RuntimeError("Detector is not selected (call select_detector first)")

    # def set_imaging_area(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
    #     self._image_settings.width = int(width)
    #     self._image_settings.height = int(height)
    #     self._image_settings.x = int(x)
    #     self._image_settings.y = int(y)
    #
    # def set_binning(self, binning: Union[int, Tuple[int, int]]) -> None:
    #     # Abstract uses int; accept tuple for backward compatibility.
    #     if isinstance(binning, tuple):
    #         # take the smaller (or assume square binning)
    #         b = int(min(binning))
    #     else:
    #         b = int(binning)
    #     self._image_settings.binning = b
    #
    # def set_exposure_time(self, ms: float) -> None:
    #     self._image_settings.exposure_ms = float(ms)
    #
    # def set_dwell_time(self, us: float) -> None:
    #     self._image_settings.dwell_us = float(us)

    # -----------------------
    # Stage
    # -----------------------

    def get_stage_position(self) -> TemStagePosition:
        pos = self.stage.GetPos()  # [x, y, z, tx, ty] in nm / degrees
        return TemStagePosition(
            x=Q_(float(pos[0]), "nanometer"),
            y=Q_(float(pos[1]), "nanometer"),
            z=Q_(float(pos[2]), "nanometer"),
            tilt_x=Q_(float(pos[3]), "degree"),
            tilt_y=Q_(float(pos[4]), "degree"),
            coordinate_system="stage",
        )

    def move_stage_absolute(self, pos: TemStagePosition, exact: bool = True, tolerance_length: Optional[Quantity] = None,
                            tolerance_angle: Optional[Quantity] = None) -> None:
        if tolerance_length is None:
            tolerance_length = Q_(10, "nanometer")
        if tolerance_angle is None:
            tolerance_angle = Q_(0.1, "degree")

        def ensure_stage_rest(axis: str, timeout_s: float = 30.0, poll_s: float = 0.1) -> None:
            deadline = time.monotonic() + timeout_s
            while True:
                status = self.get_stage_status()
                if status.get(axis, 1) == 0:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Stage axis '{axis}' did not settle within {timeout_s}s. status={status}")
                time.sleep(poll_s)

        def exceed_tolerance(cur: TemStagePosition) -> Optional[TemStagePosition]:
            exceed = False
            new_tgt = TemStagePosition()

            tol_nm = magnitude(tolerance_length, "nanometer")
            tol_deg = magnitude(tolerance_angle, "degree")

            if pos.x is not None:
                if abs(magnitude(pos.x, "nanometer") - magnitude(cur.x, "nanometer")) > tol_nm:
                    new_tgt.x = pos.x
                    exceed = True

            if pos.y is not None:
                if abs(magnitude(pos.y, "nanometer") - magnitude(cur.y, "nanometer")) > tol_nm:
                    new_tgt.y = pos.y
                    exceed = True

            if pos.z is not None:
                if abs(magnitude(pos.z, "nanometer") - magnitude(cur.z, "nanometer")) > tol_nm:
                    new_tgt.z = pos.z
                    exceed = True

            if pos.tilt_x is not None:
                if abs(magnitude(pos.tilt_x, "degree") - magnitude(cur.tilt_x, "degree")) > tol_deg:
                    new_tgt.tilt_x = pos.tilt_x
                    exceed = True

            if pos.tilt_y is not None:
                if abs(magnitude(pos.tilt_y, "degree") - magnitude(cur.tilt_y, "degree")) > tol_deg:
                    new_tgt.tilt_y = pos.tilt_y
                    exceed = True

            return new_tgt if exceed else None

        if pos.x is not None:
            self.stage.SetX(float(magnitude(pos.x, "nanometer")))
            ensure_stage_rest("x", timeout_s=30.0)

        if pos.y is not None:
            self.stage.SetY(float(magnitude(pos.y, "nanometer")))
            ensure_stage_rest("y", timeout_s=30.0)

        if pos.z is not None:
            self.stage.SetZ(float(magnitude(pos.z, "nanometer")))
            ensure_stage_rest("z", timeout_s=30.0)

        if pos.tilt_x is not None:
            self.stage.SetTiltXAngle(float(magnitude(pos.tilt_x, "degree")))
            ensure_stage_rest("tilt_x", timeout_s=30.0)

        if pos.tilt_y is not None:
            self.stage.SetTiltYAngle(float(magnitude(pos.tilt_y, "degree")))
            ensure_stage_rest("tilt_y", timeout_s=30.0)

        current_pos = self.get_stage_position()
        new_target = exceed_tolerance(current_pos)
        if not exact or new_target is None:
            return

        max_retries = 5
        for _ in range(max_retries):
            self.move_stage_absolute(pos=new_target, exact=False,
                                     tolerance_length=tolerance_length,
                                     tolerance_angle=tolerance_angle)
            cur_pos = self.get_stage_position()
            new_target = exceed_tolerance(cur_pos)
            if new_target is None:
                break
        else:
            final_pos = self.get_stage_position()
            raise TimeoutError(
                f"Stage failed to reach target within tolerance after {max_retries} retries. "
                f"target={pos} final={final_pos} remaining_axes={new_target}"
            )


    def move_stage_relative(self, pos: TemStagePosition, exact: bool = False, tolerance_length: Optional[Quantity] = None,
                            tolerance_angle: Optional[Quantity] = None) -> None:
        cur = self.get_stage_position()
        pos.x = ensure_quantity(pos.x, "nanometer")
        pos.y = ensure_quantity(pos.y, "nanometer")
        pos.z = ensure_quantity(pos.z, "nanometer")
        pos.tilt_x = ensure_quantity(pos.tilt_x, "degree")
        pos.tilt_y = ensure_quantity(pos.tilt_y, "degree")
        target = TemStagePosition(
            x=(cur.x + pos.x) if pos.x is not None else None,
            y=(cur.y + pos.y) if pos.y is not None else None,
            z=(cur.z + pos.z) if pos.z is not None else None,
            tilt_x=(cur.tilt_x + pos.tilt_x) if pos.tilt_x is not None else None,
            tilt_y=(cur.tilt_y + pos.tilt_y) if pos.tilt_y is not None else None,
            coordinate_system=cur.coordinate_system,
        )
        self.move_stage_absolute(pos=target, exact=exact, tolerance_length=tolerance_length, tolerance_angle=tolerance_angle)


    def set_stage_drive_mode(self, mode: str) -> None:
        key = mode.strip().lower()
        if key in {"motor", "m"}:
            self.stage.SelDrvMode(0)
            self._log_event("Stage drive mode set to motor")
        elif key in {"piezo", "p"}:
            self.stage.SelDrvMode(1)
            self._log_event("Stage drive mode set to piezo")
        else:
            raise ValueError("mode must be 'motor' / 'm' or 'piezo' / 'p'")

    def stop_stage(self) -> None:
        self.stage.Stop()
        self._log_event("Stage stopped")
        current_pos = self.get_stage_position()
        self._log_event(f"Current stage position: x = {current_pos.x}, y = {current_pos.y}, z = {current_pos.z}, tilt x = {current_pos.tilt_x}, tilt y = {current_pos.tilt_y}:")

    def get_stage_status(self) -> Dict[str, Any]:
        st = self.stage.GetStatus()
        return {"x": st[0], "y": st[1], "z": st[2], "tilt_x": st[3], "tilt_y": st[4]}

    def insert_holder(self) -> None:
        raise NotImplementedError("Holder insert is not exposed by Stage3 in this TEM3 interface.")

    def retract_holder(self) -> None:
        raise NotImplementedError("Holder retract is not exposed by Stage3 in this TEM3 interface.")

    def run_autofunction(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._selected_detector is None:
            raise RuntimeError(
                "Autofunction requires a selected camera detector (select_detector) and detector REST support.")

        key = (name or "").strip().lower()
        params = params or {}

        # Common mappings to PyJEM detector REST endpoints
        if key in {"autofocus", "focus"}:
            return dict(self._selected_detector.AutoFocus())
        if key in {"autostig", "autostigmator", "stigmator"}:
            return dict(self._selected_detector.AutoStigmator())
        if key in {"autocb", "autocontrast", "autocontrastbrightness"}:
            return dict(self._selected_detector.AutoContrastBrightness())
        if key in {"autoz"}:
            return dict(self._selected_detector.AutoZ())
        if key in {"autoorientation", "orientation"}:
            return dict(self._selected_detector.AutoOrientation())

        raise ValueError(
            f"Unknown autofunction '{name}'. Supported: autofocus, autostig, autocb, autoz, autoorientation")

    def discover_capabilities(self) -> Dict[str, Any]:
        return {
            "vendor": "JEOL",
            "modes": ["TEM", "STEM"],
            "tem_function_modes": list(self._TEM_FUNCTION_MAP.keys()),
            "stem_function_modes": list(self._STEM_FUNCTION_MAP.keys()),
            "apertures": self.list_apertures(),
            "detectors": self.list_detectors(),
            "stage_axes": ["x", "y", "z", "tilt_x", "tilt_y"],
            }

    def get_log(self, n: int = 100) -> List[str]:
        return self._log[-int(n):]

    def send_raw_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Minimal "raw" dispatcher: command like "eos.GetMagValue" or "stage.SetX".
        """
        params = params or {}
        if "." not in command:
            raise ValueError("command must be like '<module>.<method>'")

        mod_name, meth_name = command.split(".", 1)
        mod = getattr(self, mod_name, None)
        if mod is None:
            raise ValueError(f"Unknown module: {mod_name}")

        meth = getattr(mod, meth_name, None)
        if meth is None:
            raise ValueError(f"Unknown method: {mod_name}.{meth_name}")

        args = params.get("args", [])
        kwargs = params.get("kwargs", {})
        output = meth(*args, **kwargs)
        self._log_event(f"Raw command sent: {command}, args = {args}, kwargs = {kwargs}, output = {output}")
        return output

    def safe_move_stage(
            self,
            pos: TemStagePosition,
            max_step: Optional[Quantity] = None,
            *,
            exact: bool = True,
            tolerance_length: Optional[Quantity] = None,
            tolerance_angle: Optional[Quantity] = None,
    ) -> None:
        return super().safe_move_stage(
            pos=pos,
            max_step=max_step,
            exact=exact,
            tolerance_length=tolerance_length,
            tolerance_angle=tolerance_angle,
        )