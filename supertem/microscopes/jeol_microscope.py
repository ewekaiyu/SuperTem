import time
from typing import Dict, Any, Optional, List, Tuple, Union

from supertem.microscope import TemMicroscope
from supertem.structures.base import ImageSettings, TemImage, TemStagePosition

try:
    from PyJEM import TEM3  # type: ignore
except Exception:  # pragma: no cover
    TEM3 = None


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

    def __init__(self):
        if TEM3 is None:
            raise ImportError("PyJEM TEM3 is not available. Install PyJEM on the microscope control PC.")

        # TEM3 controllers
        self.apt = TEM3.Apt3()
        self.deflector = TEM3.Def()
        self.detector = TEM3.Detector3()
        self.eos = TEM3.EOS3()
        self.feg = TEM3.FEG3()
        self.gun = TEM3.GUN3()
        self.ht = TEM3.HT3()
        self.lens = TEM3.Lens3()
        self.stage = TEM3.Stage3()
        self.vac = TEM3.VACUUM3()

        # Cached "software state" for features TEM3 doesn't report back reliably
        self._selected_aperture: Optional[str] = None
        self._selected_detector: str = "0"  # detector ID as string by default
        self._defocus_cache: float = 0.0  # device units unless you calibrate nm mapping
        self._stigmation_cache: Tuple[float, float] = (0.0, 0.0)
        self._image_settings: ImageSettings = ImageSettings()  # for camera integration later


        self.magnification_list = [6000, 8000, 10000, 20000, 30000, 40000, 50000, 80000, 100000, 120000, 150000]
        self.camera_length_list = []
        self.aperture_dict = dict(CL1 = 0, CL2 = 1, OL_Upper = 2, Ol_Lower = 3, SA = 4, ENT = 5, HX = 6, BF = 7, AUX1 = 8, AUX2 = 9, AUX3 = 10, AUX4 = 11)

    # -----------------------
    # Connection
    # -----------------------

    def connect_to_microscope(self, ip_address: str = "0.0.0.0", port: int = 0, timeout_s: float = 5.0) -> None:
        try:
            t0 = time.time()
            TEM3.connect()
            while time.time() - t0 < timeout_s:
                try:
                    if self.is_connected():
                        return
                except Exception:
                    pass
                time.sleep(0.1)
        except Exception:
            pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return TEM3.is_connect()

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
            "ht_kv": self.get_acceleration_voltage(),
            "emission_current": self.get_emission_current(),
            "vacuum": {
                "column_ready": self.vac.GetColumnReady(),
                "camera_ready": self.vac.GetCameraReady(),
                "specimen_ready": self.vac.GetSpecimenReady(),
                "valves": self.vac.GetValveStatus(),
            },
            "stage": {
                "position": stage_pos.__dict__,
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
            return f"{prefix}:{str(func[1])}"
        return prefix

    def set_mode(self, mode: str):
        #Mode should be "TEM/STEM:function", for example "TEM:MAG" or "STEM:SM-MAG".
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("mode must be a non-empty string")

        raw = mode.strip()
        parts = raw.split(":", 1)
        obs = parts[0].strip().upper()
        func = parts[1].strip() if len(parts) == 2 else ""

        if obs in {"TEM"}:
            self.eos.SelectTemStem(0)
            if func:
                idx = self._TEM_FUNCTION_MAP.get(func.replace(" ", "").lower())
                if idx is None:
                    raise ValueError(f"Unknown TEM function mode: {func}")
                self.eos.SelectFunctionMode(idx)
            return

        if obs in {"STEM", "ASID"}:
            self.eos.SelectTemStem(1)
            if func:
                key = func.replace(" ", "").lower()
                idx = self._STEM_FUNCTION_MAP.get(key)
                if idx is None:
                    raise ValueError(f"Unknown STEM function mode: {func}")
                self.eos.SelectFunctionMode(idx)
            return

        raise ValueError(f"Unknown observation mode: {mode}")

    # -----------------------
    # Beam / HT
    # -----------------------

    def set_acceleration_voltage(self, kv: float) -> None:
        self.ht.SetHtValue(float(kv) * 1000.0)

    def get_acceleration_voltage(self) -> float:
        return float(self.ht.GetHtValue()) / 1000.0

    def get_emission_current(self) -> Optional[float]:
        # PyJEM Gun emission current value is typically in µA, but treat as vendor-defined.
        return float(self.gun.GetEmissionCurrentValue())

    def set_beam_blank(self, blank: bool) -> None:
        self.deflector.SetBeamBlank(1 if blank else 0)

    def get_beam_blank(self) -> bool:
        return bool(self.deflector.GetBeamBlank())

    # -----------------------
    # Mag / Spot
    # -----------------------

    def set_magnification(self, mag: float) -> None:
        """
        TEM: select the closest lower/equal magnification from `self.magnification_list`
             and set EOS selector index.
        STEM: uses `self.camera_length_list` and EOS stem camera selector if provided.
        """
        temstem = self.eos.GetTemStemMode()
        if temstem == 0:
            if not self.magnification_list:
                raise RuntimeError("magnification_list is empty.")
            # Choose closest <= target; fallback to smallest if target is below range.
            candidates = [m for m in self.magnification_list if m <= mag]
            chosen = candidates[-1] if candidates else self.magnification_list[0]
            idx = self.magnification_list.index(chosen)
            self.eos.SetSelector(idx)
            return

        # STEM camera length selector
        if not self.camera_length_list:
            raise RuntimeError("camera_length_list is empty. Populate it for STEM camera length control.")
        candidates = [cl for cl in self.camera_length_list if cl <= mag]
        chosen = candidates[-1] if candidates else self.camera_length_list[0]
        idx = self.camera_length_list.index(chosen)
        self.eos.SetStemCamSelector(idx)

    def get_magnification(self) -> float:
        temstem = self.eos.GetTemStemMode()
        if temstem == 0:
            val = self.eos.GetMagValue()
        else:
            val = self.eos.GetStemCamValue()
        # TEM3 returns [value, unit, string]
        if isinstance(val, list) and len(val) > 0:
            return float(val[0])
        return float(val)

    def set_spot_size(self, index: int) -> None:
        self.eos.SelectSpotSize(int(index))

    def get_spot_size(self) -> int:
        return int(self.eos.GetSpotSize())

    # -----------------------
    # Focus / Stig (device units)
    # -----------------------
    def set_defocus(self, defocus_nm: float) -> None:
        #need fix: still don't know current defocus, can't directly assign defocus value
        """
        JEOL PyJEM does not provide a universal defocus-in-nm API in TEM3.

        Here we interpret `defocus_nm` as **OBJ focus knob units** (integer steps) by default.
        If you calibrate a nm<->steps mapping for your instrument, apply it before calling.
        """
        self._defocus_cache = float(defocus_nm)
        return
        try:
            self.eos.SetObjFocus(int(round(defocus_nm)))
        except Exception:
            pass

    def get_defocus(self) -> float:
        return float(self._defocus_cache)

    def set_stigmation(self, x: float, y: float) -> None:
        self._stigmation_cache = (float(x), float(y))
        temstem = self.eos.GetTemStemMode()
        try:
            if temstem == 0:
                self.deflector.SetTemStigA1Rel(int(round(x)), int(round(y)))
            else:
                self.deflector.SetStemStigA1Rel(int(round(x)), int(round(y)))
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

    def set_aperture(self, kind: str) -> None:
        if kind not in self.aperture_dict:
            raise ValueError(f"Unknown aperture kind: {kind}. Known: {list(self.aperture_dict)}")
        self.apt.SelectExpKind(self.aperture_dict[kind])
        self._selected_aperture = kind

    def insert_aperture(self, size: Optional[int]) -> None:
        self.apt.SetExpSize(kind = self._selected_aperture, size = size)

    def retract_aperture(self) -> None:
        self.apt.SetExpSize(kind = self._selected_aperture, size = 0)

    # -----------------------
    # Detectors
    # -----------------------

    def list_detectors(self) -> List[str]:
        #need fix
        pass

    def select_detector(self, detector: str) -> None:
        #need fix
        pass

    def get_detector_settings(self) -> Dict[str, Any]:
        #need fix
        if self._selected_detector == "screen":
            return {"screen": int(self.detector.GetScreen())}

        did = int(self._selected_detector)
        return {
            "id": did,
            "brightness": int(self.detector.GetBrt(did)),
            "contrast": int(self.detector.GetCont(did)),
            "image_sw": int(self.detector.GetImageSw(did)),
            "position": int(self.detector.GetPosition(did)),
        }

    def set_detector_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        #need fix
        if not isinstance(settings, dict):
            raise ValueError("settings must be a dict")

        if self._selected_detector == "screen":
            if "screen" in settings:
                self.detector.SetScreen(int(settings["screen"]))
            return

        did = int(self._selected_detector)
        if "brightness" in settings:
            self.detector.SetBrt(did, int(settings["brightness"]))
        if "contrast" in settings:
            self.detector.SetCont(did, int(settings["contrast"]))
        if "image_sw" in settings:
            self.detector.SetImageSw(did, int(settings["image_sw"]))
        if "position" in settings:
            self.detector.SetPosition(did, int(settings["position"]))

    def acquire_image(self, settings: Optional[ImageSettings] = None) -> TemImage:
        #need fix
        pass

    def start_live(self, settings: Optional[ImageSettings] = None) -> None:
        raise NotImplementedError("Live view is camera-dependent; integrate your camera SDK here.")

    def stop_live(self) -> None:
        raise NotImplementedError("Live view is camera-dependent; integrate your camera SDK here.")

    def get_live_frame(self) -> bytes:
        raise NotImplementedError("Live view is camera-dependent; integrate your camera SDK here.")

    def get_raw_image_data(self, detector: Optional[str] = None) -> bytes:
        raise NotImplementedError("Raw image data is camera-dependent; integrate your camera SDK here.")

    def set_imaging_area(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
        self._image_settings.width = int(width)
        self._image_settings.height = int(height)
        self._image_settings.x = int(x)
        self._image_settings.y = int(y)

    def set_binning(self, binning: Union[int, Tuple[int, int]]) -> None:
        # Abstract uses int; accept tuple for backward compatibility.
        if isinstance(binning, tuple):
            # take the smaller (or assume square binning)
            b = int(min(binning))
        else:
            b = int(binning)
        self._image_settings.binning = b

    def set_exposure_time(self, ms: float) -> None:
        self._image_settings.exposure_ms = float(ms)

    def set_dwell_time(self, us: float) -> None:
        self._image_settings.dwell_us = float(us)

    # -----------------------
    # Stage
    # -----------------------

    def get_stage_position(self) -> TemStagePosition:
        pos = self.stage.GetPos()  # [x, y, z, tx, ty] in nm / degrees
        return TemStagePosition(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            tilt_x=float(pos[3]),
            tilt_y=float(pos[4]),
            coordinate_system="stage",
        )

    def move_stage_absolute(self, pos: TemStagePosition, wait: bool = True, tolerance_nm: float = 10.0) -> None:
        # Only set axes that are not None
        if pos.x is not None:
            self.stage.SetX(float(pos.x))
        if pos.y is not None:
            self.stage.SetY(float(pos.y))
        if pos.z is not None:
            self.stage.SetZ(float(pos.z))
        if pos.tilt_x is not None:
            self.stage.SetTiltXAngle(float(pos.tilt_x))
        if pos.tilt_y is not None:
            self.stage.SetTiltYAngle(float(pos.tilt_y))

        if wait:
            self._wait_stage(pos, tolerance_nm=tolerance_nm)

    def move_stage_relative(self, dx: float, dy: float, dz: float, wait: bool = True,
                            tolerance_nm: float = 10.0) -> None:
        if dx:
            self.stage.SetXRel(float(dx))
        if dy:
            self.stage.SetYRel(float(dy))
        if dz:
            self.stage.SetZRel(float(dz))
        if wait:
            # Relative: just poll until rest; or approximate by checking delta near 0 from target.
            time.sleep(0.05)

    def _wait_stage(self, target: TemStagePosition, tolerance_nm: float, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = self.get_stage_position()
            ok = True
            if target.x is not None:
                ok &= abs(cur.x - target.x) <= tolerance_nm
            if target.y is not None:
                ok &= abs(cur.y - target.y) <= tolerance_nm
            if target.z is not None:
                ok &= abs(cur.z - target.z) <= tolerance_nm
            if ok:
                return
            time.sleep(0.05)

    def set_stage_drive_mode(self, mode: str) -> None:
        key = mode.strip().lower()
        if key in {"motor", "m"}:
            self.stage.SelDrvMode(0)
        elif key in {"piezo", "p"}:
            self.stage.SelDrvMode(1)
        else:
            raise ValueError("mode must be 'motor' / 'm' or 'piezo' / 'p'")

    def stop_stage(self) -> None:
        self.stage.Stop()

    def get_stage_status(self) -> Dict[str, Any]:
        #need fix return TemStagePosition
        st = self.stage.GetStatus()
        return {"x": st[0], "y": st[1], "z": st[2], "tilt_x": st[3], "tilt_y": st[4]}

    def insert_holder(self) -> None:
        raise NotImplementedError("Holder insert is not exposed by Stage3 in this TEM3 interface.")

    def retract_holder(self) -> None:
        raise NotImplementedError("Holder retract is not exposed by Stage3 in this TEM3 interface.")

    def run_autofunction(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        #need fix
        pass

    def discover_capabilities(self) -> Dict[str, Any]:
        return {
            "vendor": "JEOL",
            "modes": ["TEM", "STEM"],
            "tem_function_modes": list(self._TEM_FUNCTION_MAP.keys()),
            "stem_function_modes": list(self._STEM_FUNCTION_MAP.keys()),
            "apertures": self.list_apertures(),
            "detectors": self.list_detectors(),
            "has_beam_blank": True,
            "has_camera_acquisition": False,
        }

    def get_log(self, n: int = 100) -> List[str]:
        #need logging
        pass

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
        return meth(*args, **kwargs)

    def safe_move_stage(self, pos: TemStagePosition, max_step_nm: float = 50000.0) -> None:
        return super().safe_move_stage(pos, max_step_nm=max_step_nm)