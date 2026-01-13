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
from supertem.vendor.JEOL.jeol_eos_tables import EOS_MODE_TABLES, get_list

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
        (0, 0): "TEM:MAG",
        (0, 1): "TEM:MAG2",
        (0, 2): "TEM:LOWMAG",
        (0, 3): "TEM:SAMAG",
        (0, 4): "TEM:DIFF",
        (1, 0): "STEM:ALIGN",
        (1, 1): "STEM:SM-LMAG",
        (1, 2): "STEM:SM-MAG",
        (1, 3): "STEM:AMAG",
        (1, 4): "STEM:UUDIFF",
        (1, 5): "STEM:ROCKING",
    }

    def __init__(self, config: MicroscopeSettings):
        super().__init__(config)
        self.stage = None
        self.eos = None
        self.ht = None
        self.lens = None
        self.def_ = None
        self.apt = None
        self.scan = None
        self.vac = None
        self.gun = None
        self.feg = None  # legacy alias for gun
        self.det3 = None
        self._connected = False

        self._active_detectors: Dict[str, Any] = {}
        self._primary_detector_id: Optional[str] = None

        # Mapping scale factors (no calibration layer yet; treat quantities as raw DAC by default)
        cfg = config if isinstance(config, dict) else {}
        self.defocus_scale: float = float(getattr(config, 'defocus_scale', cfg.get('defocus_scale', 1.0)))
    # =========================================================================
    # 1. Connection & Lifecycle
    # =========================================================================

    def connect(self, host: str, port: Optional[int] = None, **kwargs) -> None:
        if not TEM3:
            raise RuntimeError("PyJEM library not found.")
        try:
            TEM3.connect()
            self.stage = TEM3.Stage3()
            self.eos = TEM3.EOS3()
            self.ht = TEM3.HT3()
            self.lens = TEM3.Lens3()
            self.def_ = TEM3.Def3()
            self.apt = TEM3.Apt3()
            self.scan = TEM3.Scan3()
            self.vac = TEM3.VACUUM3()
            self.feg = TEM3.FEG3()
            self.gun = TEM3.GUN3()
            self.det3 = TEM3.Detector3()
            self._connected = True
            self._refresh_detectors()
            logger.info(f"Connected to JEOL PyJEM interface (Host: {host}).")
        except Exception as e:
            logger.error(f"Failed to connect to PyJEM modules: {e}")
            raise


    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Microscope is not connected.")

    def _get_detector_function_module(self):
        """Return the PyJEM detector.function module if available."""
        if detector is None:
            return None
        fn_mod = getattr(detector, "function", None)
        return fn_mod if fn_mod is not None else detector

    def _refresh_detectors(self) -> None:
        """Discover detectors once and cache Detector instances to avoid repeated IPC."""
        self._active_detectors = {}
        self._primary_detector_id = None
        self._scan_cfg = {
            'pixel_dwell_us': 10.0,
            'flyback_us': 0.0,
            'width_px': 512,
            'height_px': 512,
        }

        if detector is None:
            return

        fn_mod = self._get_detector_function_module()
        getter = getattr(fn_mod, "get_attached_detector", None)
        if not callable(getter):
            return

        try:
            ids = list(getter())
        except Exception:
            return

        for det_id in ids:
            try:
                self._active_detectors[det_id] = detector.Detector(det_id)
            except Exception:
                continue

        if ids:
            self._primary_detector_id = ids[0]

    def _get_detector(self, detector_id: str):
        """Get a cached Detector instance (creates + caches if missing)."""
        if detector is None:
            raise RuntimeError("PyJEM detector module missing")

        d = self._active_detectors.get(detector_id)
        if d is None:
            d = detector.Detector(detector_id)
            self._active_detectors[detector_id] = d
            if self._primary_detector_id is None:
                self._primary_detector_id = detector_id
        return d

    def _coerce_xy(self, xy: Any) -> Tuple[float, float]:
        """Coerce PyJEM (x,y) returns (tuple/list) into float pair."""
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                return float(xy[0]), float(xy[1])
            except Exception:
                return 0.0, 0.0
        return 0.0, 0.0

# --- EOS Helpers ---------------------------------------------------------

    def _get_eos_mode_key(self) -> Optional[str]:
        """Return EOS mode key like 'TEM:MAG' or 'STEM:SM-MAG' matching jeol_eos_tables."""
        if not self.eos or not hasattr(self.eos, "GetFunctionMode"):
            return None
        try:
            modes = self.eos.GetFunctionMode()
        except Exception:
            return None
        if not modes or len(modes) < 2:
            return None
        key = self._EOS_MODE_MAP.get((int(modes[0]), int(modes[1])))
        if key:
            return key
        # Fallback: best-effort string
        obs = "TEM" if int(modes[0]) == 0 else "STEM"
        return f"{obs}:{int(modes[1])}"

    def _normalize_eos_key(self, key: str) -> Optional[str]:
        """Case-insensitive match against EOS_MODE_TABLES keys."""
        if not key:
            return None
        if key in EOS_MODE_TABLES:
            return key
        up = key.upper()
        if up in EOS_MODE_TABLES:
            return up
        # Last resort: case-insensitive scan (tables are small)
        for k in EOS_MODE_TABLES.keys():
            if k.upper() == up:
                return k
        return None

    def _select_eos_mode_key(self, key: str) -> None:
        """Select EOS mode by key (TEM:DIFF, STEM:SM-MAG, etc.)."""
        if not self.eos:
            return
        norm = self._normalize_eos_key(key) or key
        # Decode TEM/STEM + function string
        if ":" not in norm:
            raise ValueError(f"Invalid EOS mode key: {key!r}")
        obs, func = norm.split(":", 1)
        obs = obs.strip().upper()
        func = func.strip().upper()

        tem_funcs = {"MAG": 0, "MAG2": 1, "LOWMAG": 2, "SAMAG": 3, "DIFF": 4}
        stem_funcs = {"ALIGN": 0, "SM-LMAG": 1, "SM-MAG": 2, "AMAG": 3, "UUDIFF": 4, "ROCKING": 5}

        if obs == "TEM":
            if hasattr(self.eos, "SelectTemStem"):
                self.eos.SelectTemStem(0)
            if hasattr(self.eos, "SelectFunctionMode"):
                self.eos.SelectFunctionMode(int(tem_funcs.get(func, 0)))
        elif obs == "STEM":
            if hasattr(self.eos, "SelectTemStem"):
                self.eos.SelectTemStem(1)
            if hasattr(self.eos, "SelectFunctionMode"):
                self.eos.SelectFunctionMode(int(stem_funcs.get(func, 2)))
        else:
            raise ValueError(f"Unknown EOS observation mode: {obs!r}")


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
        """Return 'TEM' or 'STEM' when available."""
        if not self.eos or not hasattr(self.eos, "GetTemStemMode"):
            return "UNKNOWN"
        try:
            mode = int(self.eos.GetTemStemMode())
        except Exception:
            return "UNKNOWN"
        return "TEM" if mode == 0 else "STEM"


    def set_mode(self, mode: str) -> None:
        """Set microscope observation mode: 'TEM' or 'STEM'."""
        if not self.eos or not hasattr(self.eos, "SelectTemStem"):
            return
        m = (mode or "").strip().upper()
        if m not in {"TEM", "STEM"}:
            return
        try:
            self.eos.SelectTemStem(0 if m == "TEM" else 1)
        except Exception:
            return

    # =========================================================================
    # 3. Stage Control
    # =========================================================================

    def get_stage_position(self) -> StagePosition:
        if not self.stage or not hasattr(self.stage, "GetPos"):
            return StagePosition()
        return jeol_adapter.from_jeol_stage_position(self.stage.GetPos())

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default", wait: bool = True) -> None:
        # TODO: include piezo /motor
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
        # TODO: Move to 0,0,0,0,0?
        # Not supported in standard PyJEM Stage3 interface
        logger.warning("home_stage() not supported by this driver.")
        pass

    # =========================================================================
    # 4. Beam Control (Atomic)
    # =========================================================================

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        if not self.ht or not hasattr(self.ht, "GetHtValue"):
            return None
        try:
            v = float(self.ht.GetHtValue())  # Volts
        except Exception:
            return None
        return Q_(v, "V").to(Units.KV)


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
        """Return alpha selector index as a dimensionless quantity.

        PyJEM exposes alpha as a discrete selector number (0-8), not a calibrated angle.
        """
        if not self.eos or not hasattr(self.eos, "GetAlpha"):
            return None
        try:
            idx = int(self.eos.GetAlpha())
        except Exception:
            return None
        return Q_(idx, "")  # dimensionless


    def get_beam_shift(self) -> Tuple[float, float]:
        """Beam shift (CLA1) in raw JEOL DAC units."""
        if self.def_ and hasattr(self.def_, "GetCLA1"):
            return self._coerce_xy(self.def_.GetCLA1())
        return (0.0, 0.0)

    def get_condenser_stigmation(self) -> Tuple[float, float]:
        """Condenser stigmation (CLs) in raw JEOL DAC units."""
        if self.def_ and hasattr(self.def_, "GetCLs"):
            return self._coerce_xy(self.def_.GetCLs())
        return (0.0, 0.0)

    def get_gun_tilt(self) -> Tuple[float, float]:
        """
        Gun/beam tilt proxy.
        PyJEM TEM3 does not expose a literal 'GunTilt' on Def3; AngleBalance is the closest
        generic (x,y) beam-angle control on many JEOL systems.
        """
        if self.def_ and hasattr(self.def_, "GetAngBal"):
            return self._coerce_xy(self.def_.GetAngBal())
        return (0.0, 0.0)

    def get_beam_blank(self) -> bool:
        if self.def_ and hasattr(self.def_, "GetBeamBlank"):
            # 0=OFF, 1=ON
            return bool(self.def_.GetBeamBlank())
        return False

    # --- Setters ---

    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        if not self.ht or not hasattr(self.ht, "SetHtValue"):
            return
        try:
            v = float(voltage.to("V").magnitude)
        except Exception:
            v = float(voltage.magnitude)
        try:
            self.ht.SetHtValue(v)
        except Exception:
            return


    def set_beam_current(self, current: Quantity) -> None:
        # JEOL usually controls this via Spot Size or CL3, not direct current setting
        logger.warning("set_beam_current not directly supported; use spot_size.")
        pass

    def set_spot_size(self, index: int) -> None:
        if self.eos and hasattr(self.eos, "SelectSpotSize"):
            self.eos.SelectSpotSize(int(index))

    def set_convergence_angle(self, angle: Quantity) -> None:
        """Set alpha selector index (0-8).

        Until a calibration layer exists, we treat the input quantity's magnitude as the raw index.
        """
        if not self.eos or not hasattr(self.eos, "SetAlphaSelector"):
            return
        try:
            idx = int(float(angle.to("").magnitude))
        except Exception:
            idx = int(float(angle.magnitude))
        idx = max(0, min(8, idx))
        try:
            self.eos.SetAlphaSelector(idx)
        except Exception:
            return


    def set_beam_shift(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetCLA1"):
            self.def_.SetCLA1(int(x), int(y))

    def set_condenser_stigmation(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetCLs"):
            self.def_.SetCLs(int(x), int(y))

    def set_gun_tilt(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetAngBal"):
            self.def_.SetAngBal(int(x), int(y))

    def set_beam_blank(self, blank: bool) -> None:
        if self.def_ and hasattr(self.def_, "SetBeamBlank"):
            self.def_.SetBeamBlank(1 if blank else 0)

    # =========================================================================
    # 5. Projection Control (Atomic)
    # =========================================================================

    def get_projection_mode(self) -> str:
        # Re-use logic from get_mode or EOS table
        key, _ = self._resolve_eos_table_info()
        return key if key else "UNKNOWN"

    def get_magnification_index(self) -> int:
        """Return the *magnification value* (e.g. 100000), not the selector index.

        Notes:
            - In JEOL TEM:DIFF, EOS3.GetMagValue() reports *camera length* (units like cm/mm),
              so this method returns 0 in diffraction-like modes.
            - When EOS reports magnification as unit 'X', we return that value.
            - Fallback: if EOS can't report magnification directly, we map the current selector
              through jeol_eos_tables.MagList (when that list uses unit 'X').
        """
        if not self.eos:
            return 0

        # Preferred: ask EOS for the current value (works across many function modes).
        if hasattr(self.eos, "GetMagValue"):
            try:
                val = self.eos.GetMagValue()  # [value, unit, label] or scalar
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    v = float(val[0])
                    unit = str(val[1]).strip().upper()
                    if unit == "X":
                        return int(round(v))
                    # Not magnification (e.g. TEM:DIFF reports camera length here).
                    return 0
                # Scalar fallback (rare)
                return int(round(float(val)))
            except Exception:
                pass

        # Fallback: map selector -> EOS table (only valid when MagList unit is 'X').
        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key:
            return 0

        try:
            lst = get_list(key, "MagList") or []
        except Exception:
            return 0

        if not lst:
            return 0

        if str(lst[0][1]).strip().upper() != "X":
            return 0

        # Try to read selector id (0-based/1-based varies by install/offline stubs).
        sel = None
        if hasattr(self.eos, "GetCurrentMagSelectorID"):
            try:
                sel = int(self.eos.GetCurrentMagSelectorID())
            except Exception:
                sel = None
        if sel is None and hasattr(self.eos, "GetSelector"):
            try:
                sel = int(self.eos.GetSelector())
            except Exception:
                sel = None
        if sel is None:
            return 0

        # Tolerate both 0-based and 1-based returns by probing nearby indices.
        for idx in (sel - 1, sel, sel + 1):
            if 0 <= idx < len(lst):
                try:
                    return int(round(float(lst[idx][0])))
                except Exception:
                    continue

        return 0

    def get_camera_length(self) -> Optional[Quantity]:
        """Return camera length if we are in a diffraction-like mode.

        JEOL/PyJEM:
            * TEM diffraction camera length is exposed via EOS3.GetMagValue()
              when the current EOS function mode is TEM:DIFF.
            * STEM camera length is exposed via EOS3.GetStemCamValue().
        """
        if not self.eos:
            return None

        key = self._get_eos_mode_key() or ""
        key_u = key.upper()

        try:
            if key_u.startswith("STEM:"):
                if hasattr(self.eos, "GetStemCamValue"):
                    val, unit, _name = self.eos.GetStemCamValue()
                    # unit is typically 'cm' on JEOL tables
                    return Q_(float(val), str(unit)).to(Units.MM)
                return None

            # TEM: camera length is only meaningful in DIFF mode
            if "DIFF" in key_u and hasattr(self.eos, "GetMagValue"):
                val, unit, _name = self.eos.GetMagValue()
                return Q_(float(val), str(unit)).to(Units.MM)

        except Exception:
            return None

        return None

    def get_defocus(self) -> Optional[Quantity]:
        # NOTE: Without a calibration layer, we treat the returned value as a raw OLc DAC value.
        if self.lens and hasattr(self.lens, "GetOLc"):
            try:
                val = float(self.lens.GetOLc())
                return Q_(val / (self.defocus_scale or 1.0), Units.NM)
            except Exception:
                return None
        return None

    def get_screen_position(self) -> str:
        """Get the fluorescent screen position.

        PyJEM exposes screen angle index via Detector3.GetScreen/SetScreen:
            0=0deg, 1=45deg, 2=90deg.

        We map:
            * 2 (90deg) -> 'DOWN'
            * everything else -> 'UP'
        """
        if not self.det3:
            return "UNKNOWN"
        try:
            if hasattr(self.det3, "GetScreen"):
                idx = int(self.det3.GetScreen())
                return "DOWN" if idx == 2 else "UP"
        except Exception:
            return "UNKNOWN"
        return "UNKNOWN"


    def get_objective_stigmation(self) -> Tuple[float, float]:
        # Objective stigmator (OLS) in raw JEOL DAC units.
        if self.def_ and hasattr(self.def_, "GetOLs"):
            return self._coerce_xy(self.def_.GetOLs())
        return (0.0, 0.0)

    def get_image_shift(self) -> Tuple[float, float]:
        # Image shift (IS1) in raw JEOL DAC units.
        if self.def_:
            if hasattr(self.def_, "GetIS1"):
                return self._coerce_xy(self.def_.GetIS1())
            if hasattr(self.def_, "GetIS"):
                return self._coerce_xy(self.def_.GetIS())
        return (0.0, 0.0)

    def get_diffraction_shift(self) -> Tuple[float, float]:
        # Diffraction/Projector alignment (PLA) in raw JEOL DAC units.
        if self.def_ and hasattr(self.def_, "GetPLA"):
            return self._coerce_xy(self.def_.GetPLA())
        return (0.0, 0.0)

    # --- Setters ---

    def set_projection_mode(self, mode: str) -> None:
        """Set projection mode.

        Supported inputs:
            * 'IMAGING' / 'DIFFRACTION'
            * JEOL EOS keys like 'TEM:DIFF', 'TEM:MAG', 'STEM:UUDIFF', ...

        This drives EOS3.SelectTemStem() + EOS3.SelectFunctionMode().
        """
        if not mode:
            return

        m = mode.strip().upper()

        # If they pass a full EOS key, obey it.
        if ":" in m:
            self._select_eos_mode_key(m)
            return

        # Otherwise interpret generic intent.
        obs = (self.get_mode() or "TEM").strip().upper()

        if "DIFF" in m:
            if obs == "STEM":
                self._select_eos_mode_key("STEM:UUDIFF")
            else:
                self._select_eos_mode_key("TEM:DIFF")
            return

        # Default to imaging
        if obs == "STEM":
            self._select_eos_mode_key("STEM:SM-MAG")
        else:
            self._select_eos_mode_key("TEM:MAG")

    def set_magnification_index(self, index: int) -> None:
        """Set magnification using the current EOS mode's MagList.

        The input `index` is a *magnification value* (e.g. 200000), not a selector id.
        We pick the closest available entry <= target from the current mode's MagList (unit 'X')
        and then call EOS3.SetSelector().

        If the current mode does not expose magnification (e.g. TEM:DIFF where MagList is in cm/mm),
        this is a no-op (with a warning).
        """
        if not self.eos or not hasattr(self.eos, "SetSelector"):
            return

        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key:
            return

        try:
            mag_list = get_list(key, "MagList") or []
        except Exception:
            return

        if not mag_list:
            return

        unit = str(mag_list[0][1]).strip().upper()
        if unit != "X":
            logger.warning(
                f"set_magnification_index({index}) ignored: current mode {key} MagList unit is '{unit}', "
                f"so EOS is not in a magnification-selectable mode."
            )
            return

        target = float(index)
        values: List[float] = []
        for v, _u, _s in mag_list:
            try:
                values.append(float(v))
            except Exception:
                values.append(float("nan"))

        # Choose the closest entry <= target (or the first entry if target is smaller).
        best_i = 0
        for i, v in enumerate(values):
            if not np.isfinite(v):
                continue
            if v <= target:
                best_i = i

        # Robustness: some installs use 0-based, some 1-based selector indices.
        # We try 1-based first (matches how camera length selection is implemented here),
        # then fall back to 0-based if needed.
        try:
            self.eos.SetSelector(int(best_i + 1))
        except Exception:
            try:
                self.eos.SetSelector(int(best_i))
            except Exception:
                return

    def set_camera_length(self, length: Quantity) -> None:
        """Set camera length (diffraction).

        Implementation:
            * TEM:DIFF uses EOS3.SetSelector(selector_index)
              (values are in EOS_MODE_TABLES[mode]['MagList']).
            * STEM:* uses EOS3.SetStemCamSelector(selector_index)
              (values are in EOS_MODE_TABLES[mode]['StemCamList']).

        Selector indices are 1-based in JEOL lists (table index + 1).
        """
        if not self.eos or length is None:
            return

        key, list_name = self._resolve_eos_table_info()
        if not key or not list_name:
            raise RuntimeError("EOS mode is unavailable; cannot set camera length.")

        # Camera length is only meaningful in diffraction-like modes.
        if key.startswith("TEM:") and "DIFF" not in key:
            logger.warning(f"Ignoring set_camera_length while not in TEM:DIFF (current: {key}).")
            return

        targets = get_list(key, list_name) or []
        if not targets:
            raise RuntimeError(f"No EOS table values for {key}:{list_name}")

        target_mm = length.to(Units.MM).magnitude

        # Pick closest entry in mm.
        best_i = 0
        best_err = float("inf")
        for i, (val, unit, _label) in enumerate(targets):
            try:
                mm = Q_(float(val), str(unit)).to(Units.MM).magnitude
            except Exception:
                continue
            err = abs(mm - target_mm)
            if err < best_err:
                best_err = err
                best_i = i

        selector = int(best_i + 1)

        if key.startswith("STEM:"):
            if hasattr(self.eos, "SetStemCamSelector"):
                self.eos.SetStemCamSelector(selector)
            else:
                raise RuntimeError("EOS3.SetStemCamSelector not available")
        else:
            if hasattr(self.eos, "SetSelector"):
                self.eos.SetSelector(selector)
            else:
                raise RuntimeError("EOS3.SetSelector not available")


    def set_defocus(self, defocus: Quantity) -> None:
        # NOTE: Without a calibration layer, we treat input Quantity magnitude as raw OLc DAC.
        if self.lens and hasattr(self.lens, "SetOLc"):
            val = float(defocus.to(Units.NM).magnitude)
            scaled = int(val * (self.defocus_scale or 1.0))
            self.lens.SetOLc(scaled)

    def set_screen_position(self, position: str) -> None:
        """Raise/lower the fluorescent screen.

        We map:
            * 'DOWN' -> SetScreen(2) (90deg)
            * 'UP'   -> SetScreen(0) (0deg)

        If the hardware doesn't support screen control, this is a no-op.
        """
        if not self.det3 or not hasattr(self.det3, "SetScreen"):
            return
        p = (position or "").strip().upper()
        try:
            if p == "DOWN":
                self.det3.SetScreen(2)
            elif p == "UP":
                self.det3.SetScreen(0)
        except Exception:
            return


    def set_objective_stigmation(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetOLs"):
            self.def_.SetOLs(int(x), int(y))

    def set_image_shift(self, x: float, y: float) -> None:
        if self.def_:
            if hasattr(self.def_, "SetIS1"):
                self.def_.SetIS1(int(x), int(y))
                return
            if hasattr(self.def_, "SetIS"):
                self.def_.SetIS(int(x), int(y))

    def set_diffraction_shift(self, x: float, y: float) -> None:
        if self.def_ and hasattr(self.def_, "SetPLA"):
            self.def_.SetPLA(int(x), int(y))

    # =========================================================================
    # 6. Scan Control (Atomic)
    # =========================================================================

    def get_scan_mode(self) -> str:
        # todo: needs fix
        if self.scan and hasattr(self.scan, "GetScanMode"):
            return str(self.scan.GetScanMode())
        return "0"

    def get_scan_width(self) -> int:
        return int(self._scan_cfg.get('width_px', 512))


    def get_scan_height(self) -> int:
        return int(self._scan_cfg.get('height_px', 512))


    def get_scan_pixel_dwell(self) -> Quantity:
        return Q_(float(self._scan_cfg.get('pixel_dwell_us', 10.0)), Units.US)


    def get_scan_flyback(self) -> Quantity:
        return Q_(float(self._scan_cfg.get('flyback_us', 0.0)), Units.US)


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

    def set_scan_width(self, width: int) -> None:
        self._scan_cfg['width_px'] = int(width)


    def set_scan_height(self, height: int) -> None:
        self._scan_cfg['height_px'] = int(height)


    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        try:
            us = float(time.to(Units.US).magnitude)
        except Exception:
            us = float(time.magnitude)
        self._scan_cfg['pixel_dwell_us'] = us


    def set_scan_flyback(self, time: Quantity) -> None:
        try:
            us = float(time.to(Units.US).magnitude)
        except Exception:
            us = float(time.magnitude)
        self._scan_cfg['flyback_us'] = us


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
        if not self._active_detectors:
            self._refresh_detectors()
        return list(self._active_detectors.keys())

    def get_active_detector_ids(self) -> List[str]:
        # For now: treat all attached detectors as "active".
        return self.list_detectors()

    def get_primary_detector_id(self) -> Optional[str]:
        if self._primary_detector_id is None:
            self._refresh_detectors()
        return self._primary_detector_id

    def get_detector_exposure(self, detector_id: str) -> Quantity:
        if detector is None:
            return Q_(0.0, Units.SEC)
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.exposure or Q_(0.0, Units.SEC)
        except Exception:
            return Q_(0.0, Units.SEC)

    def get_detector_binning(self, detector_id: str) -> int:
        if detector is None:
            return 1
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return int(res.binning_index or 1)
        except Exception:
            return 1

    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        if detector is None:
            return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.roi
        except Exception:
            return None

    def get_detector_integration(self, detector_id: str) -> int:
        if detector is None:
            return 1
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return int(res.frame_integration or 1)
        except Exception:
            return 1

    def get_detector_inserted(self, detector_id: str) -> bool:
        if detector is None:
            return True
        try:
            d = self._get_detector(detector_id)
            if hasattr(d, "get_insert_state"):
                st = d.get_insert_state()
                if isinstance(st, dict):
                    # Common patterns: {"InsertState":0/1} or {"state":"IN"/"OUT"} (varies by install)
                    for k in ("InsertState", "insert_state", "state", "Status", "status"):
                        if k in st:
                            v = st[k]
                            if isinstance(v, str):
                                return v.strip().upper() in ("IN", "INSERT", "INSERTED", "ON", "OPEN")
                            return bool(v)
                # If we can't parse, assume True (safe for software flow; hardware interlocks live elsewhere)
                return True
        except Exception:
            return True
        return True

    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        return None

    # --- Setters ---
    # todo: needs fix
    def set_detector_exposure(self, detector_id: str, exposure: Quantity) -> None:
        if detector is None:
            return
        d = self._get_detector(detector_id)
        # PyJEM uses microseconds for ExposureTimeValue. (Range depends on detector/installation.)
        us = int(exposure.to(Units.US).magnitude)
        if hasattr(d, "set_exposuretime_value"):
            d.set_exposuretime_value(us)
        elif hasattr(d, "set_exposuretime_index"):
            d.set_exposuretime_index(us)

    def set_detector_binning(self, detector_id: str, index: int) -> None:
        if detector is None:
            return
        d = self._get_detector(detector_id)
        if hasattr(d, "set_binningindex"):
            d.set_binningindex(int(index))

    def set_detector_roi(self, detector_id: str, roi: Optional[ROI]) -> None:
        if detector is None:
            return
        d = self._get_detector(detector_id)
        if roi is None:
            return
        # AreaMode imaging window
        if hasattr(d, "set_areamode_imagingarea"):
            d.set_areamode_imagingarea(int(roi.width), int(roi.height), int(roi.x), int(roi.y))

    def set_detector_integration(self, detector_id: str, count: int) -> None:
        if detector is None:
            return
        d = self._get_detector(detector_id)
        if hasattr(d, "set_frameintegration"):
            d.set_frameintegration(int(count))

    def set_detector_insertion(self, detector_id: str, inserted: bool) -> None:
        if detector is None:
            return
        d = self._get_detector(detector_id)
        try:
            if inserted and hasattr(d, "insert"):
                d.insert()
            elif (not inserted) and hasattr(d, "retract"):
                d.retract()
        except Exception:
            pass

    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        if detector is None:
            raise RuntimeError("Detector module missing")

        det_id = request.detector_id or (self.get_primary_detector_id() or "")
        d = self._get_detector(det_id)

        # Apply per-request detector settings (atomic-only).
        if request.settings is not None:
            s = request.settings
            if getattr(s, "exposure", None) is not None:
                self.set_detector_exposure(det_id, s.exposure)
            if getattr(s, "binning_index", None) is not None:
                self.set_detector_binning(det_id, int(s.binning_index))
            if getattr(s, "frame_integration", None) is not None:
                self.set_detector_integration(det_id, int(s.frame_integration))
            if getattr(s, "roi", None) is not None:
                self.set_detector_roi(det_id, s.roi)

        # Capture raw data
        raw = None
        if hasattr(d, "snapshot_rawdata"):
            raw = d.snapshot_rawdata()
        elif hasattr(d, "get_image_cache"):
            raw = d.get_image_cache()
        elif hasattr(d, "livesnapshot"):
            raw = d.livesnapshot("tif")  # returns bytes stream

        # Convert to numpy
        arr = None
        if raw is None:
            arr = np.zeros((1, 1), dtype=np.uint16)
        elif isinstance(raw, (bytes, bytearray)):
            arr = np.frombuffer(raw, dtype=np.uint16)
        elif isinstance(raw, list):
            arr = np.array(raw)
        elif isinstance(raw, dict) and "data" in raw:
            arr = np.array(raw["data"])
        else:
            try:
                arr = np.array(raw)
            except Exception:
                arr = np.zeros((1, 1), dtype=np.uint16)

        # Deterministic reshape using ROI if possible
        roi = None
        if request.settings is not None and getattr(request.settings, "roi", None) is not None:
            roi = request.settings.roi
        if roi is None:
            roi = self.get_detector_roi(det_id) or ROI(_mode="lenient")

        if arr.ndim == 1:
            cols = int(getattr(roi, "width", 0) or 0)
            rows = int(getattr(roi, "height", 0) or 0)
            if cols > 0 and rows > 0 and arr.size == cols * rows:
                arr = arr.reshape((rows, cols))

        metadata = MicroscopeImageMetadata(
            detector_id=det_id,
            exposure=self.get_detector_exposure(det_id),
            binning_index=self.get_detector_binning(det_id),
            roi=self.get_detector_roi(det_id),
            frame_integration=self.get_detector_integration(det_id),
            _mode="lenient"
        )
        return MicroscopeImage(data=arr, metadata=metadata, _mode="lenient")

    # =========================================================================
    # 8. Vacuum Control (Atomic)
    # =========================================================================

    def get_valve_state(self, valve_name: str) -> str:
        """Get a coarse valve state.

        * gun: Gun3.GetBeamValve() -> 0=closed, 1=open
        * column/turbo: VACUUM3.GetValveStatus() returns bitfields; we use configurable bit indices.
        """
        vn = (valve_name or "").strip().lower()

        # Gun valve (beam valve)
        if vn == "gun" and self.gun and hasattr(self.gun, "GetBeamValve"):
            try:
                return "OPEN" if int(self.gun.GetBeamValve()) == 1 else "CLOSED"
            except Exception:
                return "UNKNOWN"

        # Column/turbo valves: model-specific bit assignments
        if self.vac and hasattr(self.vac, "GetValveStatus"):
            try:
                status = self.vac.GetValveStatus()  # [count, v1_bitfield, v2_bitfield]
                if isinstance(status, (list, tuple)) and len(status) >= 2:
                    count = int(status[0]) if status[0] is not None else 0
                    bitfield = int(status[1])
                    bit_map = {"column": 0, "turbo": 1}
                    bit = bit_map.get(vn)
                    if bit is not None and bit < max(count, bit + 1):
                        is_open = ((bitfield >> bit) & 0x1) == 1
                        return "OPEN" if is_open else "CLOSED"
            except Exception:
                return "UNKNOWN"

        return "UNKNOWN"

    def set_valve_state(self, valve_name: str, state: str) -> None:
        """Set valve state (only gun valve is supported here)."""
        vn = (valve_name or "").strip().lower()
        st = (state or "").strip().upper()

        if vn == "gun" and self.gun and hasattr(self.gun, "SetBeamValve"):
            try:
                self.gun.SetBeamValve(1 if st == "OPEN" else 0)
            except Exception:
                return

        # VACUUM3 typically does not expose direct setters for column/turbo via PyJEM.
        return

    def get_pressure(self, gauge_name: str) -> Quantity:
        """Get pressure-like reading.

        PyJEM VACUUM3 exposes gauge monitor values via GetPegInfo/GetPigInfo as raw monitor values.
        Many systems require an additional calibration curve to convert these to Pa.

        We return the raw value *typed* as Pa for now to satisfy the interface, but treat it as
        an uncalibrated monitor reading.
        """
        if not self.vac:
            return Q_(0.0, Units.PA)

        name = (gauge_name or "").strip().lower()
        try:
            # Prefer Ion gauge info if available
            if hasattr(self.vac, "GetPigInfo"):
                val = self.vac.GetPigInfo()
                if isinstance(val, (list, tuple)) and val:
                    return Q_(float(val[0]), Units.PA)

            if hasattr(self.vac, "GetPegInfo"):
                val = self.vac.GetPegInfo()
                if isinstance(val, (list, tuple)) and val:
                    return Q_(float(val[0]), Units.PA)

        except Exception:
            pass

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
        """Resolve EOS table key + the relevant list name for the current mode.

        Notes:
            * TEM magnification and camera-length values are both stored in 'MagList'
              (camera length is meaningful only in TEM:DIFF).
            * STEM camera-length values are stored in 'StemCamList'.
        """
        key = self._get_eos_mode_key()
        key = self._normalize_eos_key(key or "") if key else None
        if not key:
            return None, None
        list_name = "StemCamList" if key.startswith("STEM:") else "MagList"
        return key, list_name

    def _wait_for_stage(self):
        time.sleep(0.1)