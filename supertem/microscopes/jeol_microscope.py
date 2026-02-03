"""
supertem.jeol_microscope

JEOL TEM driver implementation for the SuperTEM hardware abstraction layer.

This module provides :class:`JeolMicroscope`, a concrete implementation of
:class:`~supertem.microscope.TemMicroscope` wrapping the `PyJEM` (TEM3) interface.

===============================================================================
I. Implementation Specifics
===============================================================================

This driver adheres to the strict safety contract defined in `supertem.microscope`.
It maps the standard layers to JEOL hardware as follows:

  1) Atomic Layer: Wraps `PyJEM` calls (TEM3, EOS3, Stage3).
     - **Error Handling:** Raises `PyJEM` exceptions directly in Setters (Fail Loudly).
     - **Data Handling:** Returns `None` in Getters if `PyJEM` fails or returns
       invalid data, strictly following "Null means Unknown".

  2) Helper Layer:
     - **Vendor Validation:** `apply_beam_settings` validates that `alpha_index`
       is within the hardware limit (0-8) before execution.
     - **Mapping:** Translates canonical `defocus` (nm) to `OLc` (DAC) *only if*
       a calibration scale is provided.

===============================================================================
II. Supported Vendor Extras
===============================================================================

This driver utilizes the `Extras.vendor['JEOL']` dictionary to expose hardware
capabilities that do not map to canonical physics.

  - `alpha_index` (int):
    The convergence angle selector (0-8). Used because JEOL does not report
    physical convergence angles (mrad) without external calibration.

  - `defocus_olc_dac` (int):
    The raw Objective Lens Coarse DAC value. Populated in `ProjectionSettings`
    when `defocus_scale` is not configured.

  - `mag_selector` (int):
    The raw magnification index. Used when `Magnification` (float) is ambiguous.

===============================================================================
III. Hardware Quirks & Workarounds
===============================================================================

  - **Stage Hysteresis:** JEOL stages may report "Idle" (0) momentarily during
    direction changes. This driver's `move_stage_absolute` implements a custom
    retry loop that waits for *stable* idle status.

  - **Detector Sync:** If the active detector is offline, `set_scan_active` will
    fallback to the internal scan generator to prevent beam damage (static beam).

  - **Lazy Loading:** `PyJEM` is imported only upon instantiation. This allows
    the class to be imported in simulation/offline environments without crashing.

===============================================================================
IV. Developer Guide (Atomic Method Boilerplate)
===============================================================================

When adding new hardware controls, strictly follow these patterns to maintain
architectural compliance.

**Pattern A: Atomic Getter (Null means Unknown)**
    def get_hardware_value(self) -> Optional[Type]:
        if not self.hardware:
            # Log at DEBUG (not ERROR) to prevent spam during polling
            logger.debug("[TAG] GetValue failed: Hardware disconnected.")
            return None

        try:
            val = self.hardware.GetValue()
            return _clean_or_convert(val)
        except Exception as e:
            logger.debug(f"[TAG] GetValue failed: {e}")
            return None

**Pattern B: Atomic Setter (Fail Loudly)**
    def set_hardware_value(self, value: Type) -> None:
        if not self.hardware:
            # Setters MUST fail loudly if hardware is missing
            logger.error("[TAG] SetValue failed: Hardware disconnected.")
            raise RuntimeError("Hardware disconnected.")

        logger.debug(f"[TAG] SetValue({value})")  # Log intent BEFORE action
        try:
            self.hardware.SetValue(value)
        except Exception as e:
            logger.error(f"[TAG] SetValue failed: {e}")  # ERROR log
            raise  # Always re-raise

===============================================================================
V. Configuration Example
===============================================================================

The JEOL driver relies on specific `extra.vendor["JEOL"]` keys for features that
do not map to standard physics (e.g. Alpha Selector, OLc DAC).

    settings = MicroscopeSettings(
        system=SystemSettings(
            # ... standard limits ...
        ),
        # GLOBAL VENDOR EXTRAS
        extra=Extras(vendor={"JEOL": {}})
    )

    # 1. BEAM SETTINGS (Alpha Selector)
    # The driver reads 'alpha_index' from here to set the convergence angle.
    beam_req = BeamSettings(
        voltage=Q_(200, "kV"),
        extra=Extras(vendor={"JEOL": {
            "alpha_index": 3  # Sets CLA/Alpha selector to index 3
        }})
    )

    # 2. PROJECTION SETTINGS (Raw DACs)
    # If 'defocus_scale' is missing, the driver reads/writes 'defocus_olc_dac'.
    proj_req = ProjectionSettings(
        magnification_index=15,
        extra=Extras(vendor={"JEOL": {
            "defocus_olc_dac": 32768  # Direct hardware value
        }})
    )

    scope = JeolMicroscope(settings)
    scope.connect("localhost")
"""
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from datetime import datetime, timezone

# Import Abstract Base and Strict Structures
from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    MicroscopeSettings,
    SystemInfo,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    DetectorSettings,
    DetectorSystemSettings,
    ScanSettings,
    VacuumSettings,
    ApertureSettings,
    MicroscopeImage,
    MicroscopeImageMetadata,
    AcquisitionRequest,
    Units,
    Q_,
    Quantity,
    Extras,
    Point,
    ROI, DetectorCapabilities
)

# Import Vendor Adapters
from supertem.vendor.JEOL import jeol_adapter
from supertem.vendor.JEOL.jeol_eos_tables import EOS_MODE_TABLES, get_list

logger = logging.getLogger(__name__)


class JeolMicroscope(TemMicroscope):
    """
    JEOL ARM/F2 implementation of the SuperTEM Interface.
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
        """
        Initialize the driver.
        PyJEM modules are loaded into instance variables (self.tem3_mod, self.det_mod)
        to allow independent instantiation and better testing support.
        """
        super().__init__(config)

        # Instance variables for PyJEM modules
        self.tem3_mod = None
        self.det_mod = None

        # Load modules immediately
        self._load_pyjem_modules()

        # Hardware Interface Placeholders
        self.stage = None
        self.eos = None
        self.ht = None
        self.lens = None
        self.def_ = None
        self.apt = None
        self.scan = None
        self.vac = None
        self.gun = None
        self.feg = None
        self.det3 = None
        self._connected = False

        # Detector caching
        self._active_detectors: Dict[str, Any] = {}
        self._primary_detector_id: Optional[str] = None

        # Local state for scan parameters not readable from hardware
        self._scan_cfg: Dict[str, Any] = {}

        # Defocus calibration configuration
        self._has_defocus_calibration: bool = False
        self.defocus_scale: float = 1.0

        if config:
            # Check for attribute first (Pydantic/Dataclass)
            val = getattr(config, 'defocus_scale', None)
            # Fallback to dict if it somehow is one
            if val is None and isinstance(config, dict):
                val = config.get('defocus_scale')

            if val is not None:
                self._has_defocus_calibration = True
                self.defocus_scale = float(val)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _load_pyjem_modules(self):
        """Internal helper to import PyJEM modules into instance variables."""
        model_name = (self.system_settings.info.model or "").upper()
        force_offline = "OFFLINE" in model_name

        # ---------------------------------------------------------
        # A. Load TEM3 Interface
        # ---------------------------------------------------------
        if self.tem3_mod is None:
            if force_offline:
                logger.info("[INIT] Offline mode requested. Forcing PyJEM.offline.TEM3...")
                try:
                    from PyJEM.offline import TEM3 as _T3
                    self.tem3_mod = _T3
                except ImportError as e:
                    logger.error(f"[INIT] Failed to import PyJEM.offline.TEM3: {e}")
                    raise RuntimeError("Offline mode requested but PyJEM.offline is missing.") from e
            else:
                try:
                    from PyJEM import TEM3 as _T3
                    self.tem3_mod = _T3
                    logger.info("[INIT] PyJEM.TEM3 imported (Online Mode).")
                except ImportError:
                    logger.warning("[INIT] PyJEM.TEM3 missing. Falling back to PyJEM.offline...")
                    try:
                        from PyJEM.offline import TEM3 as _T3
                        self.tem3_mod = _T3
                        logger.info("[INIT] PyJEM.offline.TEM3 imported (Fallback).")
                    except ImportError:
                        self.tem3_mod = None
                        logger.warning("[INIT] PyJEM.TEM3 module absent.")

        # ---------------------------------------------------------
        # B. Load Detector Interface
        # ---------------------------------------------------------
        if self.det_mod is None:
            if force_offline:
                logger.info("[INIT] Offline mode requested. Forcing PyJEM.offline.detector...")
                try:
                    from PyJEM.offline import detector as _d
                    self.det_mod = _d
                except ImportError as e:
                    logger.error(f"[INIT] Failed to import PyJEM.offline.detector: {e}")
            else:
                try:
                    from PyJEM import detector as _d
                    self.det_mod = _d
                    logger.info("[INIT] PyJEM.detector imported (Online Mode).")
                except ImportError:
                    logger.warning("[INIT] PyJEM.detector missing. Falling back to PyJEM.offline...")
                    try:
                        from PyJEM.offline import detector as _d
                        self.det_mod = _d
                        logger.info("[INIT] PyJEM.offline.detector imported (Fallback).")
                    except ImportError:
                        self.det_mod = None
                        logger.warning("[INIT] PyJEM.detector module absent.")

    def _require_connected(self) -> None:
        """Raise an error if called before connect()."""
        if not self._connected:
            raise RuntimeError("Microscope is not connected.")

    def _get_detector_function_module(self):
        """
        Retrieve the PyJEM detector function module.
        Newer PyJEM versions nest functions under `detector.function`.
        """
        if self.det_mod is None:
            return None
        fn_mod = getattr(self.det_mod, "function", None)
        return fn_mod if fn_mod is not None else self.det_mod

    def _refresh_detectors(self) -> None:
        """
        Scan for attached detectors, cache their instances, and update
        SystemSettings.detector_system with their hardware capabilities.
        """
        # 1. Reset Internal State
        self._active_detectors = {}
        self._primary_detector_id = None
        self._scan_cfg = {
            'pixel_dwell_us': 10.0, 'flyback_us': 0.0,
            'width_px': 512, 'height_px': 512,
        }

        if self.det_mod is None:
            return

        # 2. PyJEM Discovery Logic
        fn_mod = getattr(self.det_mod, "function", None)
        if fn_mod is None:
            fn_mod = self.det_mod

        getter = getattr(fn_mod, "get_attached_detector", None)
        if not callable(getter):
            return

        try:
            ids = list(getter())
            logger.info(f"[DET] Discovered detectors: {ids}")
        except Exception:
            return

        # 3. Load Capabilities
        caps_map: Dict[str, DetectorCapabilities] = {}

        for det_id in ids:
            try:
                # Create and Cache
                d_obj = self.det_mod.Detector(det_id)
                self._active_detectors[det_id] = d_obj

                # Extract Capabilities (Min/Max settings)
                try:
                    raw = d_obj.get_detectorsetting()
                    _, caps = jeol_adapter.from_jeol_detector_response(raw, det_id)
                    if caps:
                        caps_map[det_id] = caps
                except Exception:
                    pass
            except Exception:
                continue

        if ids:
            self._primary_detector_id = ids[0]

        # 4. Update System Configuration (CORRECTED)
        # We must write to: self.system_settings.detector_system.capabilities_by_id

        sys_config = self.system_settings
        if sys_config is not None:
            if sys_config.detector_system is None:
                sys_config.detector_system = DetectorSystemSettings()

            det_sys = sys_config.detector_system
            det_sys.available_detector_ids = ids
            if det_sys.capabilities_by_id is None:
                det_sys.capabilities_by_id = {}
            det_sys.capabilities_by_id.update(caps_map)

            logger.debug(f"[DET] System capabilities updated for: {list(caps_map.keys())}")

    def _get_detector(self, detector_id: str):
        """Retrieve a cached detector instance by ID, creating it if necessary."""
        if self.det_mod is None:
            raise RuntimeError("PyJEM detector module missing")

        d = self._active_detectors.get(detector_id)
        if d is None:
            d = self.det_mod.Detector(detector_id)
            self._active_detectors[detector_id] = d
            # If no primary is set, make this the primary
            if self._primary_detector_id is None:
                self._primary_detector_id = detector_id
        return d

    def _coerce_xy(self, xy: Any) -> Optional[Tuple[float, float]]:
        """
        Safely convert PyJEM return values (often list [x, y]) into a float tuple.
        Returns None if data is missing or malformed.
        """
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                return float(xy[0]), float(xy[1])
            except Exception:
                return None
        return None

    def _wait_for_stage(self, timeout: float = 30.0) -> None:
        """
        Polls the stage status until all axes report 'Idle' (0).
        This is critical for handling JEOL stage hysteresis.
        """
        if not hasattr(self.stage, "GetStatus"):
            time.sleep(0.5)
            return

        logger.debug("[STAGE] Waiting for IDLE status...")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                status = self.stage.GetStatus()
                if isinstance(status, (list, tuple)):
                    if all(s != 1 for s in status):
                        return
            except Exception:
                pass
            time.sleep(0.2)

        logger.warning(f"Stage move timed out after {timeout}s (GetStatus never settled).")

    def _get_scan_controller_detector(self):
        """Helper to find the Detector instance that controls scanning (STEM)."""
        if self.det_mod is None:
            return None
        try:
            det_id = self.get_primary_detector_id()
            if det_id is None:
                ids = self.list_detectors()
                det_id = ids[0] if ids else None
            if det_id is None:
                return None
            return self._get_detector(det_id)
        except Exception:
            return None

    def _set_imaging_area(self, *, width=None, height=None, x=None, y=None) -> None:
        """Set scanning sub-region (Imaging Area)."""
        d = self._get_scan_controller_detector()
        if d is None:
            logger.error("[SCAN] SetImagingArea failed: No scan controller detector found.")
            raise RuntimeError("Scan detector hardware not connected.")

        w = int(width if width is not None else self._scan_cfg.get("width_px", 512))
        h = int(height if height is not None else self._scan_cfg.get("height_px", 512))
        xx = int(x if x is not None else self._scan_cfg.get("x_px", 0))
        yy = int(y if y is not None else self._scan_cfg.get("y_px", 0))

        if hasattr(d, "set_imaging_area"):
            logger.debug(f"[SCAN] SetImagingArea({w}x{h} @ {xx},{yy})")
            try:
                d.set_imaging_area(w, h, xx, yy)
                self._scan_cfg.update({"width_px": w, "height_px": h, "x_px": xx, "y_px": yy})
            except Exception as e:
                logger.error(f"[SCAN] SetImagingArea failed: {e}")
                raise

    @staticmethod
    def _first_int(d: dict, keys: Tuple[str, ...]) -> Optional[int]:
        for k in keys:
            if k in d:
                try:
                    return int(d[k])
                except Exception:
                    continue
        return None

    @staticmethod
    def _first_float(d: dict, keys: Tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    def _atomic_detector_update(self, detector_id: str, patch: dict) -> None:
        """Helper to simulate atomic updates via read-modify-write if necessary."""
        d = self._get_detector(detector_id)
        if hasattr(d, "set_detectorsetting"):
            # JEOL often allows partial dicts updates
            d.set_detectorsetting(patch)
        else:
            raise RuntimeError(f"Detector {detector_id} does not support settings updates.")

    def _read_hw(self, hardware: Any, method_name: str, tag: str,
                 converter: Optional[Callable[[Any], Any]] = None,
                 default: Any = None) -> Any:
        """
        Atomic Getter (Null means Unknown).
        Checks hardware existence -> Try/Catch -> Log -> Convert.
        """
        if not hardware:
            logger.debug(f"[{tag}] {method_name} failed: Hardware disconnected.")
            return default

        try:
            func = getattr(hardware, method_name)
            val = func() if callable(func) else func
            if converter and val is not None:
                return converter(val)
            return val
        except Exception as e:
            logger.debug(f"[{tag}] {method_name} failed: {e}")
            return default

    def _write_hw(self, hardware: Any, method_name: str, tag: str, *args) -> None:
        """
        Atomic Setter (Fail Loudly).
        Checks hardware existence -> Log Intent -> Try/Catch -> Raise on error.
        """
        if not hardware:
            logger.error(f"[{tag}] {method_name} failed: Hardware disconnected.")
            raise RuntimeError(f"{tag} hardware disconnected.")

        logger.debug(f"[{tag}] {method_name}{args}")
        try:
            getattr(hardware, method_name)(*args)
        except Exception as e:
            logger.error(f"[{tag}] {method_name} failed: {e}")
            raise

    # --- Common Converters ---

    def _to_nm(self, v):
        return Q_(float(v), Units.NM)

    def _to_kv(self, v):
        return Q_(float(v), "V").to(Units.KV)

    def _to_ua(self, v):
        return Q_(float(v), Units.UA)

    def _to_na(self, v):
        return Q_(float(v), Units.UA).to(Units.NA)

    def _to_int(self, v):
        return int(v)

    def _to_bool(self, v):
        return bool(v)

    def _to_deg(self, v):
        return Q_(float(v), Units.DEG)

    # --- EOS Helpers ---

    def _get_eos_mode_key(self) -> Optional[str]:
        """
        Determine the current EOS mode string (e.g., 'TEM:MAG', 'STEM:AMAG').
        Used to look up magnification tables.
        """
        if not self.eos or not hasattr(self.eos, "GetFunctionMode"):
            return None
        try:
            function_mode = self.eos.GetFunctionMode()[0]
            main_mode = self.eos.GetTemStemMode()
        except Exception:
            return None

        key = self._EOS_MODE_MAP.get((int(main_mode), int(function_mode)))
        if key:
            return key

        # Fallback if map is incomplete
        obs = "TEM" if int(main_mode) == 0 else "STEM"
        return f"{obs}:{int(function_mode)}"

    def _normalize_eos_key(self, key: str) -> Optional[str]:
        """
        Match a user-provided mode string against the known EOS_MODE_TABLES keys.
        Case-insensitive.
        """
        if not key:
            return None
        if key in EOS_MODE_TABLES:
            return key
        up = key.upper()
        if up in EOS_MODE_TABLES:
            return up
        for k in EOS_MODE_TABLES.keys():
            if k.upper() == up:
                return k
        return None

    def _select_eos_mode_key(self, key: str) -> None:
        """
        Switch the microscope to the specified EOS mode key.
        Handles the complexity of selecting TEM/STEM mode first, then Function mode.
        """
        if not self.eos:
            logger.error(f"[LENS] SelectFunctionMode({key}) failed: EOS hardware not connected.")
            raise RuntimeError("EOS hardware not connected.")

        logger.debug(f"[LENS] SelectFunctionMode({key})")

        norm = self._normalize_eos_key(key) or key
        if ":" not in norm:
            raise ValueError(f"Invalid EOS mode key: {key!r}")

        obs, func = norm.split(":", 1)
        obs = obs.strip().upper()
        func = func.strip().upper()

        # JEOL internal function mode indices
        tem_funcs = {"MAG": 0, "MAG2": 1, "LOWMAG": 2, "SAMAG": 3, "DIFF": 4}
        stem_funcs = {"ALIGN": 0, "SM-LMAG": 1, "SM-MAG": 2, "AMAG": 3, "UUDIFF": 4, "ROCKING": 5}

        try:
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
        except Exception as e:
            logger.error(f"[LENS] Failed to switch mode to {key}: {e}")
            raise

    def _resolve_eos_table_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine which EOS table list to use based on the current mode.
        Returns: (mode_key, list_name) e.g., ('TEM:DIFF', 'MagList').
        """
        key = self._get_eos_mode_key()
        key = self._normalize_eos_key(key or "") if key else None
        if not key:
            return None, None
        list_name = "StemCamList" if key.startswith("STEM:") else "MagList"
        return key, list_name

    # =========================================================================
    # 1. Connection & Lifecycle
    # =========================================================================

    def connect(self, host: str, port: Optional[int] = None, **kwargs) -> None:
        """
        Connect to the JEOL TEM3 interface and initialize sub-modules.
        """
        if not self.tem3_mod:
            logger.error("[CONN] Cannot connect: PyJEM library not found.")
            raise RuntimeError("PyJEM library not found.")

        try:
            logger.info(f"[CONN] Connecting to TEM3 interface (Host: {host})...")
            self.tem3_mod.connect()

            # Initialize individual hardware controllers
            self.stage = self.tem3_mod.Stage3()
            self.eos = self.tem3_mod.EOS3()
            self.ht = self.tem3_mod.HT3()
            self.lens = self.tem3_mod.Lens3()
            self.def_ = self.tem3_mod.Def3()
            self.apt = self.tem3_mod.Apt3()
            self.scan = self.tem3_mod.Scan3()
            self.vac = self.tem3_mod.VACUUM3()
            self.feg = self.tem3_mod.FEG3()
            self.gun = self.tem3_mod.GUN3()
            self.det3 = self.tem3_mod.Detector3()

            self._connected = True
            self._refresh_detectors()
            logger.info(f"[CONN] Connected to JEOL PyJEM interface (Host: {host}).")

        except Exception as e:
            logger.error(f"[CONN] Failed to connect to PyJEM modules: {e}")
            raise

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[CONN] Disconnected from JEOL PyJEM.")

    def is_connected(self) -> bool:
        return self._connected

    def get_instrument_info(self) -> SystemInfo:
        return self.system_settings.info

    # =========================================================================
    # 2. Global State & Mode
    # =========================================================================

    def get_mode(self) -> str:
        """Get the main observation mode ('TEM' or 'STEM')."""
        if not self.eos or not hasattr(self.eos, "GetTemStemMode"):
            return "UNKNOWN"
        try:
            mode = int(self.eos.GetTemStemMode())
            return "TEM" if mode == 0 else "STEM"
        except Exception as e:
            logger.debug(f"[EOS] GetTemStemMode failed: {e}")
            return "UNKNOWN"

    def set_mode(self, mode: str) -> None:
        """Set the main observation mode ('TEM' or 'STEM')."""
        if not self.eos:
            logger.error("[EOS] SelectTemStem failed: Hardware not connected.")
            raise RuntimeError("EOS hardware not connected.")

        if not hasattr(self.eos, "SelectTemStem"):
            return

        m = (mode or "").strip().upper()
        if m not in {"TEM", "STEM"}:
            return

        logger.debug(f"[EOS] SelectTemStem({m})")
        try:
            self.eos.SelectTemStem(0 if m == "TEM" else 1)
        except Exception as e:
            logger.error(f"[EOS] Failed to set mode {m}: {e}")
            raise

    # =========================================================================
    # 3. Stage Control (Motion)
    # =========================================================================

    # --- Atomic Getters ---

    def get_stage_x(self) -> Optional[Quantity]:
        if not self.stage: return None
        try:
            val = self.stage.GetX() if hasattr(self.stage, "GetX") else self.stage.GetPos()[0]
            return Q_(float(val), Units.NM)
        except Exception:
            return None

    def get_stage_y(self) -> Optional[Quantity]:
        if not self.stage: return None
        try:
            val = self.stage.GetY() if hasattr(self.stage, "GetY") else self.stage.GetPos()[1]
            return Q_(float(val), Units.NM)
        except Exception:
            return None

    def get_stage_z(self) -> Optional[Quantity]:
        if not self.stage: return None
        try:
            val = self.stage.GetZ() if hasattr(self.stage, "GetZ") else self.stage.GetPos()[2]
            return Q_(float(val), Units.NM)
        except Exception:
            return None

    def get_stage_tilt_x(self) -> Optional[Quantity]:
        if not self.stage: return None
        try:
            val = self.stage.GetTiltXAngle() if hasattr(self.stage, "GetTiltXAngle") else self.stage.GetPos()[3]
            return Q_(float(val), Units.DEG)
        except Exception:
            return None

    def get_stage_tilt_y(self) -> Optional[Quantity]:
        if not self.stage: return None
        try:
            val = self.stage.GetTiltYAngle() if hasattr(self.stage, "GetTiltYAngle") else self.stage.GetPos()[4]
            return Q_(float(val), Units.DEG)
        except Exception:
            return None

    def get_stage_r(self) -> Optional[Quantity]:
        # Rotation not typically supported in standard PyJEM Stage3
        return None

    def get_stage_coordinate_system(self) -> Optional[str]:
        """Get the current reference frame name."""
        # JEOL usually operates in a single mechanical coordinate system
        return "Mechanical"

    # --- Atomic Setters ---

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default",
                            wait: bool = True,
                            tolerance_nm: float = 200.0,
                            tolerance_deg: float = 0.1,
                            max_retries: int = 3, **kwargs) -> None:
        """
        Move the stage to coordinates.

        Hardware Quirk:
            JEOL stages sometimes report "Idle" (Status 0) momentarily while changing direction.
            This method implements a retry loop that waits for stability and re-issues
            the command if the final position is not within tolerance.
        """
        if not self.stage:
            logger.error("[STAGE] Move failed: Hardware not connected.")
            raise RuntimeError("Stage hardware not connected.")

        if target.r is not None:
             # If hardware supports rotation (Gon_Rot), implement here.
             # Otherwise, fail if rotation is requested.
             logger.warning("[STAGE] Rotation (r) requested but not supported by this driver version.")
             # Uncomment to enforce strictness:
             # raise ValueError("Stage rotation is not supported by JEOL driver.")

        dt = (drive_type or "motor").strip().lower()
        is_piezo = (dt == "piezo")
        t_args = jeol_adapter.to_jeol_stage_args(target)

        logger.debug(f"[STAGE] IO Write ({dt}): {t_args}")

        def _dispatch():
            if is_piezo:
                # --- PIEZO PATH (SelDrvMode = 1) ---
                if not hasattr(self.stage, "SelDrvMode"):
                    logger.error("[STAGE] Hardware mismatch: 'SelDrvMode' not found.")
                    raise RuntimeError("Piezo control not supported by this stage driver.")

                # Warn if unsupported axes are requested
                if any(k in t_args for k in ['z', 'tx', 'ty']):
                    logger.warning("[STAGE] Piezo mode supports X/Y only. Z/Tilt ignored.")

                try:
                    # 1. Switch to Piezo Mode
                    self.stage.SelDrvMode(1)

                    # 2. Issue Moves (X/Y Only)
                    if 'x' in t_args and hasattr(self.stage, "SetX"):
                        self.stage.SetX(t_args['x'])
                    if 'y' in t_args and hasattr(self.stage, "SetY"):
                        self.stage.SetY(t_args['y'])

                finally:
                    # 3. Restore to Motor Mode (Safety)
                    # We restore immediately so subsequent calls default to standard behavior
                    self.stage.SelDrvMode(0)

            else:
                # --- MOTOR PATH (SelDrvMode = 0) ---
                if hasattr(self.stage, "SelDrvMode"):
                    self.stage.SelDrvMode(0)

                if 'x' in t_args and hasattr(self.stage, "SetX"):
                    self.stage.SetX(t_args['x'])
                if 'y' in t_args and hasattr(self.stage, "SetY"):
                    self.stage.SetY(t_args['y'])
                if 'z' in t_args and hasattr(self.stage, "SetZ"):
                    self.stage.SetZ(t_args['z'])
                if 'tx' in t_args and hasattr(self.stage, "SetTiltXAngle"):
                    self.stage.SetTiltXAngle(t_args['tx'])
                if 'ty' in t_args and hasattr(self.stage, "SetTiltYAngle"):
                    self.stage.SetTiltYAngle(t_args['ty'])

            # Execute

        try:
            _dispatch()
        except Exception as e:
            logger.error(f"[STAGE] Move IO Error: {e}")
            raise

        if not wait:
            return

        # Piezo is open-loop/instant; no retry needed.
        if is_piezo:
            time.sleep(0.1)
            return

        # Motor requires hysteresis retry loop
        for attempt in range(max_retries + 1):
            # Wait for status to settle (Idle)
            self._wait_for_stage(timeout=30.0)

            # Read back current position
            current = self.get_stage_position()
            if current is None:
                logger.warning("[STAGE] Position unreadable during verification.")
                return

            # Check if we are close enough
            if target.is_close(current, tol_nm=tolerance_nm, tol_deg=tolerance_deg):
                logger.debug(f"[STAGE] Move verified within tolerance (Attempt {attempt + 1}).")
                return

            # If not, retry
            if attempt < max_retries:
                logger.info(f"[STAGE] Hysteresis Correction {attempt + 1}/{max_retries}: Adjusting position.")
                try:
                    _dispatch()
                except Exception as e:
                    logger.error(f"[STAGE] Correction IO Error: {e}")
                time.sleep(0.5)
            else:
                logger.warning(f"[STAGE] Move finished but outside tolerance ({tolerance_nm}nm).")

    def stop_stage(self, **kwargs) -> None:
        """Immediately halt stage movement."""
        try:
            if self.stage and hasattr(self.stage, "Stop"):
                logger.debug("[STAGE] Stop()")
                self.stage.Stop()
            elif not self.stage:
                logger.error("[STAGE] Stop failed: Hardware not connected.")
                raise RuntimeError("Stage hardware not connected.")
        except Exception as e:
            logger.error(f"[STAGE] Stop failed: {e}")
            raise

    def home_stage(self, **kwargs) -> None:
        """
        Move the stage to the mechanical origin (0, 0, 0, 0, 0).
        Wraps `TEM3.Stage3.SetOrg`.
        """
        try:
            if self.stage and hasattr(self.stage, "SetOrg"):
                logger.debug("[STAGE] SetOrg() (Homing)")
                self.stage.SetOrg()
            elif not self.stage:
                logger.error("[STAGE] Homing failed: Hardware not connected.")
                raise RuntimeError("Stage hardware not connected.")
            else:
                logger.warning("[STAGE] 'SetOrg' method not found on hardware interface.")
        except Exception as e:
            logger.error(f"[STAGE] SetOrg failed: {e}")
            raise

    # --- Helper Layer Overrides ---

    def get_stage_position(self) -> StagePosition:
        """Override to perform efficient bulk read via GetPos."""
        if not self.stage:
            return StagePosition()
        try:
            # Helper: Aggregates atomic primitives efficiently
            return jeol_adapter.from_jeol_stage_position(self.stage.GetPos())
        except Exception:
            return StagePosition()

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Getters ---

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        return self._read_hw(self.ht, "GetHtValue", "BEAM", self._to_kv)

    def get_probe_mode(self) -> Optional[str]:
        # Logic retention: Map 0/1 to Strings
        val = self._read_hw(self.eos, "GetProbeMode", "BEAM", self._to_int)
        return "Nanoprobe" if val == 1 else "Microprobe" if val == 0 else None

    def get_beam_current(self) -> Optional[Quantity]:
        return self._read_hw(self.gun, "GetEmissionCurrent", "BEAM", self._to_na)

    def get_emission_current(self) -> Optional[Quantity]:
        return self._read_hw(self.gun, "GetEmissionCurrent", "BEAM", self._to_ua)

    def get_spot_size(self) -> Optional[int]:
        return self._read_hw(self.eos, "GetSpotSize", "BEAM", self._to_int)

    def get_convergence_angle(self) -> Optional[Quantity]:
        """
        Physical convergence angle is not available from the hardware directly.
        Use `get_alpha_index()` instead.
        """
        return None

    def get_beam_blank(self) -> bool:
        return self._read_hw(self.def_, "GetBeamBlank", "BEAM", self._to_bool, default=False)

    def get_beam_shift(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetCLA1", "BEAM", self._coerce_xy, default=(None, None))

    def get_beam_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetCLA2", "BEAM", self._coerce_xy, default=(None, None))

    def get_condenser_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetCLs", "BEAM", self._coerce_xy, default=(None, None))

    def get_gun_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetAngBal", "BEAM", self._coerce_xy, default=(None, None))

    # --- Atomic Getters (Vendor Specific) ---

    def get_alpha_index(self) -> Optional[int]:
        return self._read_hw(self.eos, "GetAlpha", "BEAM", self._to_int)

    # --- Atomic Setters ---

    def set_acceleration_voltage(self, voltage: Quantity, **kwargs) -> None:
        self._write_hw(self.ht, "SetHtValue", "BEAM", float(voltage.to("V").magnitude))

    def set_probe_mode(self, mode: str, **kwargs) -> None:
        if not self.eos: raise RuntimeError("EOS hardware not connected.")
        idx = 1 if "nano" in mode.strip().lower() else 0
        self._write_hw(self.eos, "SetProbeMode", "BEAM", idx)

    def set_emission_current(self, current: Quantity, **kwargs) -> None:
        self._write_hw(self.gun, "SetEmissionCurrent", "BEAM", float(current.to(Units.UA).magnitude))

    def set_spot_size(self, index: int, **kwargs) -> None:
        self._write_hw(self.eos, "SelectSpotSize", "BEAM", int(index))

    def set_convergence_angle(self, angle: Quantity, **kwargs) -> None:
        """Physical angle setting not supported. Use `set_alpha_index`."""
        raise NotImplementedError("JEOL driver cannot set physical convergence angle. Use set_alpha_index(idx).")

    def set_beam_blank(self, blank: bool, **kwargs) -> None:
        self._write_hw(self.def_, "SetBeamBlank", "BEAM", 1 if blank else 0)

    def set_beam_shift(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetCLA1", "BEAM", int(x), int(y))

    def set_beam_tilt(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetCLA2", "BEAM", int(x), int(y))

    def set_condenser_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetCLs", "BEAM", int(x), int(y))

    def set_gun_tilt(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetAngBal", "BEAM", int(x), int(y))

    # --- Atomic Setters (Vendor Specific) ---

    def set_alpha_index(self, idx: int, **kwargs) -> None:
        self._write_hw(self.eos, "SetAlphaSelector", "BEAM", int(idx))

    # --- Helper Layer Overrides ---

    def get_beam_settings(self) -> BeamSettings:
        """
        Aggregates beam state.
        Override Reason: Collects vendor-specific 'alpha_index' alongside canonical physics.
        """
        raw_flags: Dict[str, Any] = {}

        # Acceleration voltage
        vq = self.get_acceleration_voltage()
        voltage_val = 0.0
        if vq is None:
            raw_flags["ht_unavailable"] = True
        else:
            try:
                voltage_val = float(vq.to(Units.KV).magnitude)
            except Exception:
                raw_flags["ht_unavailable"] = True

        # Beam current
        cq = self.get_beam_current()
        current_ua = 0.0
        if cq is None:
            raw_flags["beam_current_unavailable"] = True
        else:
            try:
                current_ua = float(cq.to(Units.UA).magnitude)
            except Exception:
                raw_flags["beam_current_unavailable"] = True

        # Spot size
        spot_idx = self.get_spot_size()
        if spot_idx is None:
            raw_flags["spot_size_unavailable"] = True

        # Alpha selector index
        alpha_idx = self.get_alpha_index()
        if alpha_idx is None:
            raw_flags["alpha_unavailable"] = True

        # Beam shift
        beam_shift_dac: Optional[Tuple[int, int]] = None
        bs = self.get_beam_shift()
        if bs[0] is not None and bs[1] is not None:
            beam_shift_dac = (int(round(bs[0])), int(round(bs[1])))
        else:
            raw_flags["beam_shift_unavailable"] = True

        return jeol_adapter.from_jeol_beam_stats(
            voltage_val=voltage_val,
            current_ua=current_ua,
            spot_size_idx=spot_idx,  # type: ignore
            alpha_idx=alpha_idx,    # type: ignore
            beam_shift_dac=beam_shift_dac,
            raw_flags=raw_flags if raw_flags else None
        )

    def apply_beam_settings(self, settings: BeamSettings, **kwargs) -> None:
        """Override to handle alpha_index extra."""
        super().apply_beam_settings(settings, **kwargs)

        vend = getattr(settings.extra, 'vendor', None)
        jeol_v = vend.get('JEOL') if isinstance(vend, dict) else None
        if isinstance(jeol_v, dict) and 'alpha_index' in jeol_v:
            try:
                idx = int(jeol_v['alpha_index'])
                if 0 <= idx <= 8: self.set_alpha_index(idx)
            except Exception:
                pass

    # =========================================================================
    # 5. Projection Control (Imaging/Optics)
    # =========================================================================

    # --- Atomic Getters ---

    def get_optical_mode(self) -> str:
        """Get the logical optical mode (e.g. 'TEM:MAG')."""
        key, _ = self._resolve_eos_table_info()
        return key if key else "UNKNOWN"

    def get_magnification(self) -> Optional[int]:
        """
        Get the magnification value (e.g., 100000).

        Logic:
            1. Try `GetMagValue()` directly.
            2. If unavailable, use `GetSelector()` to look up the value in static tables (`EOS_MODE_TABLES`).
        """
        if not self.eos:
            logger.debug("[LENS] GetMagValue failed: Hardware not connected.")
            return None

        # Method 1: Direct Hardware Query
        if hasattr(self.eos, "GetMagValue"):
            try:
                val = self.eos.GetMagValue()
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    if str(val[1]).strip().upper() == "X":
                        return int(round(float(val[0])))
                else:
                    return int(round(float(val)))
            except Exception as e:
                logger.debug(f"[LENS] GetMagValue failed: {e}")

        # Method 2: Table Lookup via Selector
        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key:
            return None
        try:
            lst = get_list(key, "MagList") or []
        except Exception:
            return None
        if not lst or str(lst[0][1]).strip().upper() != "X":
            return None

        sel = None
        if hasattr(self.eos, "GetCurrentMagSelectorID"):
            try:
                sel = int(self.eos.GetCurrentMagSelectorID())
            except Exception:
                pass
        if sel is None and hasattr(self.eos, "GetSelector"):
            try:
                sel = int(self.eos.GetSelector())
            except Exception:
                pass
        if sel is None:
            return None

        # Check surrounding indices for robustness
        for idx in (sel - 1, sel, sel + 1):
            if 0 <= idx < len(lst):
                try:
                    return int(round(float(lst[idx][0])))
                except Exception:
                    continue
        return None

    def get_camera_length(self) -> Optional[Quantity]:
        """
        Get diffraction camera length.
        Logic: Only valid if in 'DIFF' or 'STEM' mode. Uses `GetMagValue` (TEM) or `GetStemCamValue` (STEM).
        """
        if not self.eos:
            logger.debug("[LENS] GetCameraLength failed: Hardware not connected.")
            return None

        key = self._get_eos_mode_key() or ""
        key_u = key.upper()

        try:
            if key_u.startswith("STEM:"):
                if hasattr(self.eos, "GetStemCamValue"):
                    val, unit, _ = self.eos.GetStemCamValue()
                    return Q_(float(val), str(unit)).to(Units.MM)
                return None

            if "DIFF" in key_u and hasattr(self.eos, "GetMagValue"):
                val, unit, _ = self.eos.GetMagValue()
                return Q_(float(val), str(unit)).to(Units.MM)

        except Exception as e:
            logger.debug(f"[LENS] Camera Length Read failed: {e}")

        return None

    def get_defocus(self) -> Optional[Quantity]:
        if not self._has_defocus_calibration: return None
        return self._read_hw(
            self.lens, "GetOLc", "LENS",
            lambda v: Q_(float(v) / (self.defocus_scale or 1.0), Units.NM)
        )

    def get_screen_position(self) -> str:
        # Custom logic mapping int -> String preserved via lambda or explicit read
        idx = self._read_hw(self.det3, "GetScreen", "LENS", self._to_int)
        if idx is None: return "UNKNOWN"
        return "DOWN" if idx == 2 else "UP"

    def get_objective_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetOLs", "LENS", self._coerce_xy, default=(None, None))

    def get_diffraction_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetILs", "LENS", self._coerce_xy, default=(None, None))

    def get_image_shift(self) -> Tuple[Optional[float], Optional[float]]:
        # Logic retention: Fallback IS1 vs IS
        res = self._read_hw(self.def_, "GetIS1", "LENS", self._coerce_xy)
        return res if res else self._read_hw(self.def_, "GetIS", "LENS", self._coerce_xy, default=(None, None))

    def get_diffraction_shift(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetPLA", "LENS", self._coerce_xy, default=(None, None))

    # --- Atomic Getters (Vendor Specific) ---

    def get_defocus_dac(self) -> Optional[int]:
        return self._read_hw(self.lens, "GetOLc", "LENS", self._to_int)

    # --- Atomic Setters ---

    def set_optical_mode(self, mode: str, **kwargs) -> None:
        """
        Set EOS mode.
        Logic: Maps 'IMAGING'/'DIFFRACTION' to JEOL-specific keys (e.g. 'TEM:MAG', 'TEM:DIFF').
        """
        if not mode:
            logger.error("[LENS] SetOpticalMode failed: Empty mode provided.")
            raise ValueError("Mode cannot be empty.")

        m = mode.strip().upper()
        logger.debug(f"[LENS] SwitchFunctionMode({m})")

        if ":" in m:
            self._select_eos_mode_key(m)
            return

        obs = (self.get_mode() or "TEM").strip().upper()
        if "DIFF" in m:
            self._select_eos_mode_key("STEM:UUDIFF" if obs == "STEM" else "TEM:DIFF")
            return

        self._select_eos_mode_key("STEM:SM-MAG" if obs == "STEM" else "TEM:MAG")

    def set_magnification(self, index: int, **kwargs) -> None:
        """
        Set magnification.
        Logic: Finds the closest selector index in `EOS_MODE_TABLES` for the requested value.
        """
        if not self.eos:
            logger.error("[LENS] SetMagnification failed: Hardware not connected.")
            raise RuntimeError("EOS hardware not connected.")

        logger.debug(f"[LENS] SetSelector({index})")
        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key:
            logger.error("[LENS] SetMagnification failed: Could not determine EOS mode key.")
            raise RuntimeError("Cannot resolve EOS mode for magnification lookup.")

        try:
            mag_list = get_list(key, "MagList") or []
        except Exception:
            mag_list = []

        if not mag_list or str(mag_list[0][1]).strip().upper() != "X":
            logger.error(f"[LENS] SetMagnification failed: MagList unavailable for mode {key}.")
            raise RuntimeError(f"Magnification table not found for mode {key}.")

        target = float(index)
        best_i = 0
        for i, (v, _, _) in enumerate(mag_list):
            try:
                if float(v) <= target:
                    best_i = i
            except Exception:
                continue
        try:
            self.eos.SetSelector(int(best_i + 1))
        except Exception:
            try:
                self.eos.SetSelector(int(best_i))
            except Exception as e:
                logger.error(f"[LENS] SetSelector failed: {e}")
                raise

    def set_camera_length(self, length: Quantity, **kwargs) -> None:
        """
        Set diffraction camera length.
        Logic: Finds the closest selector index in `MagList` (TEM) or `StemCamList` (STEM).
        """
        logger.debug(f"[LENS] SetCameraLength({length})")

        if not self.eos:
            # FIX: Fail Loudly
            logger.error("[LENS] SetCameraLength failed: Hardware not connected.")
            raise RuntimeError("EOS hardware not connected.")

        if length is None:
            raise ValueError("Camera length cannot be None.")

        key, list_name = self._resolve_eos_table_info()
        if not key or not list_name:
            logger.error("[LENS] SetCameraLength failed: Could not resolve table info (Not in DIFF/STEM mode?).")
            raise RuntimeError(f"Camera length control unavailable in mode: {self.get_mode()}")

        targets = get_list(key, list_name) or []
        if not targets:
            logger.error(f"[LENS] SetCameraLength failed: Lookup list '{list_name}' is empty/missing for {key}.")
            raise RuntimeError(f"Camera length table empty for mode {key}")

        first_unit = str(targets[0][1]).strip().lower()
        if first_unit not in ['cm', 'mm', 'm']:
            logger.error(f"[LENS] SetCameraLength failed: Current mode '{key}' uses non-length units '{first_unit}'.")
            raise RuntimeError(f"Cannot set Camera Length in mode {key} (Table unit: {first_unit})")

        target_mm = length.to(Units.MM).magnitude
        best_i, best_err = 0, float("inf")

        found_match = False  # Track if we actually calculated a valid error
        for i, (val, unit, _) in enumerate(targets):
            try:
                # This conversion might still fail if there's garbage data,
                # so we keep the try/except but track success.
                mm = Q_(float(val), str(unit)).to(Units.MM).magnitude
                err = abs(mm - target_mm)
                if err < best_err:
                    best_err = err
                    best_i = i
                    found_match = True
            except Exception:
                continue

        if not found_match:
            raise RuntimeError(f"No valid camera length entries found in table for {key}")

        selector = int(best_i + 1)
        try:
            if key.startswith("STEM:") and hasattr(self.eos, "SetStemCamSelector"):
                self.eos.SetStemCamSelector(selector)
            elif hasattr(self.eos, "SetSelector"):
                self.eos.SetSelector(selector)
            else:
                raise AttributeError("No suitable selector method found on EOS3.")
        except Exception as e:
            logger.error(f"[LENS] SetCameraLength failed (Selector={selector}): {e}")
            raise

    def set_defocus(self, defocus: Quantity, **kwargs) -> None:
        if not self._has_defocus_calibration:
            raise ValueError("JEOL driver cannot set physical defocus without calibration.")
        val = float(defocus.to(Units.NM).magnitude)
        dac = int(val * (self.defocus_scale or 1.0))
        self._write_hw(self.lens, "SetOLc", "LENS", dac)

    def set_screen_position(self, position: str, **kwargs) -> None:
        p = (position or "").strip().upper()
        val = 2 if p == "DOWN" else 0
        self._write_hw(self.det3, "SetScreen", "LENS", val)

    def set_objective_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetOLs", "LENS", int(x), int(y))

    def set_diffraction_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetILs", "LENS", int(x), int(y))

    def set_image_shift(self, x: float, y: float, **kwargs) -> None:
        # Logic retention: try IS1, fallback to IS inside try-catch block of helper?
        # Helper is atomic. We check if method exists.
        if hasattr(self.def_, "SetIS1"):
            self._write_hw(self.def_, "SetIS1", "LENS", int(x), int(y))
        else:
            self._write_hw(self.def_, "SetIS", "LENS", int(x), int(y))

    def set_diffraction_shift(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetPLA", "LENS", int(x), int(y))

    # --- Atomic Setters (Vendor Specific) ---

    def set_defocus_dac(self, dac: int) -> None:
        self._write_hw(self.lens, "SetOLc", "LENS", int(dac))

    # --- Helper Layer Overrides ---

    def get_projection_settings(self) -> ProjectionSettings:
        """
        Override Reason: Populate vendor-specific 'defocus_olc_dac' if uncalibrated.
        """
        ps = super().get_projection_settings()
        if not self._has_defocus_calibration:
            dac = self.get_defocus_dac()
            if dac is not None:
                ps.extra.vendor.setdefault("JEOL", {})["defocus_olc_dac"] = dac
                ps.extra.notes["ProjectionSettings.defocus_uncalibrated"] = (
                    "JEOL OLc reported in DAC units; physical nm defocus requires defocus_scale calibration."
                )
        return ps

    def apply_projection_settings(self, settings: ProjectionSettings, **kwargs) -> None:
        """
        Override Reason: Support setting defocus via 'defocus_olc_dac' when physical calibration is missing.
        """
        if (settings.defocus is not None) and (not self._has_defocus_calibration):
            raise ValueError(
                "JEOL driver cannot apply ProjectionSettings.defocus (nm) without defocus_scale calibration. "
                "Use ProjectionSettings.extra.vendor['JEOL']['defocus_olc_dac'] instead."
            )
        super().apply_projection_settings(settings)
        try:
            vend = getattr(settings.extra, "vendor", None) or {}
            jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None
            if isinstance(jeol_v, dict) and 'defocus_olc_dac' in jeol_v:
                v = jeol_v.get('defocus_olc_dac')
                if v is not None:
                    self.set_defocus_dac(int(v))
        except Exception:
            raise

    # =========================================================================
    # 6. Detector Control & Acquisition
    # =========================================================================

    # --- Atomic Getters ---

    def list_detectors(self) -> List[str]:
        """Return list of discovered detector IDs."""
        if not self._active_detectors:
            self._refresh_detectors()
        return list(self._active_detectors.keys())

    def get_active_detector_ids(self) -> List[str]:
        return self.list_detectors()

    def get_primary_detector_id(self) -> Optional[str]:
        if self._primary_detector_id is None:
            self._refresh_detectors()
        return self._primary_detector_id

    def get_detector_inserted(self, detector_id: str) -> bool:
        """Check if detector is mechanically inserted."""
        if self.det_mod is None:
            logger.debug("[DET] GetInserted failed: Hardware not connected.")
            return True
        try:
            d = self._get_detector(detector_id)
            if hasattr(d, "get_insert_state"):
                st = d.get_insert_state()
                if isinstance(st, dict):
                    for k in ("InsertState", "state", "Status"):
                        if k in st:
                            v = st[k]
                            if isinstance(v, str):
                                return v.strip().upper() in ("IN", "INSERT", "ON")
                            return bool(v)
        except Exception as e:
            logger.debug(f"[DET] GetInserted({detector_id}) failed: {e}")
        return True

    def get_detector_exposure(self, detector_id: str) -> Optional[Quantity]:
        """Get detector exposure time."""
        if self.det_mod is None:
            logger.debug("[DET] GetExposure failed: Hardware not connected.")
            return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.exposure
        except Exception as e:
            logger.debug(f"[DET] GetExposure({detector_id}) failed: {e}")
            return None

    def get_detector_binning_index(self, detector_id: str) -> Optional[int]:
        """Get binning index."""
        if self.det_mod is None:
            logger.debug("[DET] GetBinning failed: Hardware not connected.")
            return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.binning_index
        except Exception as e:
            logger.debug(f"[DET] GetBinning({detector_id}) failed: {e}")
            return None

    def get_detector_binning_xy(self, detector_id: str) -> Optional[Tuple[int, int]]:
        b = self.get_detector_binning_index(detector_id)
        return (b, b) if b is not None else None

    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        """Get Region of Interest."""
        if self.det_mod is None:
            logger.debug("[DET] GetROI failed: Hardware not connected.")
            return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.roi
        except Exception as e:
            logger.debug(f"[DET] GetROI({detector_id}) failed: {e}")
            return None

    def get_detector_gain_index(self, detector_id: str) -> Optional[int]:
        d = self._get_detector(detector_id)
        if not hasattr(d, "get_detectorsetting"): return None
        try:
            st = d.get_detectorsetting()
            return int(st.get("GainIndex")) if "GainIndex" in st else None
        except Exception:
            return None

    def get_detector_offset_index(self, detector_id: str) -> Optional[int]:
        d = self._get_detector(detector_id)
        try:
            st = d.get_detectorsetting() if hasattr(d, "get_detectorsetting") else {}
            return int(st.get("OffsetIndex")) if "OffsetIndex" in st else None
        except Exception:
            return None

    def get_detector_digital_rotation(self, detector_id: str) -> Optional[Quantity]:
        # See scan rotation, usually shared or part of setting
        return None

    def get_detector_frame_integration(self, detector_id: str) -> Optional[int]:
        """Get frame integration count."""
        if self.det_mod is None:
            logger.debug("[DET] GetIntegration failed: Hardware not connected.")
            return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.frame_integration
        except Exception as e:
            logger.debug(f"[DET] GetIntegration({detector_id}) failed: {e}")
            return None

    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        # Typically not exposed directly by PyJEM unless calculated
        return None

    def get_detector_total_frames(self, detector_id: str) -> Optional[int]:
        return None  # PyJEM specific implementation needed

    def get_detector_readout_mode(self, detector_id: str) -> Optional[str]:
        return None

    def get_detector_shutter_mode(self, detector_id: str) -> Optional[str]:
        return None

    def get_detector_save_frames(self, detector_id: str) -> Optional[bool]:
        return None

    # --- Atomic Setters ---

    def set_detector_insertion(self, detector_id: str, inserted: bool, **kwargs) -> None:
        """Insert or retract detector."""
        if self.det_mod is None:
            logger.error("[DET] SetInsertion failed: Detector hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)
            logger.debug(f"[DET] SetInsertion({detector_id}, {inserted})")
            if inserted and hasattr(d, "insert"):
                d.insert()
            elif (not inserted) and hasattr(d, "retract"):
                d.retract()
        except Exception as e:
            logger.error(f"[DET] SetInsertion failed: {e}")
            raise

    def set_detector_exposure(self, detector_id: str, exposure: Quantity, **kwargs) -> None:
        """Set detector exposure time."""
        if self.det_mod is None:
            logger.error("[DET] SetExposure failed: Detector hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)
            us = int(exposure.to(Units.US).magnitude)
            logger.debug(f"[DET] SetExposure({detector_id}, {us}us)")

            if hasattr(d, "set_exposuretime_value"):
                d.set_exposuretime_value(us)
            elif hasattr(d, "set_exposuretime_index"):
                d.set_exposuretime_index(us)
        except Exception as e:
            logger.error(f"[DET] SetExposure failed: {e}")
            raise

    def set_detector_binning_index(self, detector_id: str, index: int, **kwargs) -> None:
        """Set detector binning."""
        if self.det_mod is None:
            logger.error("[DET] SetBinning failed: Detector hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)
            logger.debug(f"[DET] SetBinning({detector_id}, {index})")
            if hasattr(d, "set_binningindex"):
                d.set_binningindex(int(index))
        except Exception as e:
            logger.error(f"[DET] SetBinning failed: {e}")
            raise

    def set_detector_binning_xy(self, detector_id: str, binning: Tuple[int, int], **kwargs) -> None:
        if binning[0] != binning[1]:
            raise ValueError("JEOL PyJEM only supports symmetric binning.")
        self.set_detector_binning_index(detector_id, binning[0])

    def set_detector_roi(self, detector_id: str, roi: Optional[ROI], **kwargs) -> None:
        """Set detector ROI."""
        if self.det_mod is None:
            logger.error("[DET] SetROI failed: Detector hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)
            if roi:
                logger.debug(f"[DET] SetROI({detector_id}, {roi})")
                if hasattr(d, "set_areamode_imagingarea"):
                    d.set_areamode_imagingarea(int(roi.width), int(roi.height), int(roi.x), int(roi.y))
        except Exception as e:
            logger.error(f"[DET] SetROI failed: {e}")
            raise

    def set_detector_gain_index(self, detector_id: str, index: int, **kwargs) -> None:
        # Atomic simulation via bulk update
        self._atomic_detector_update(detector_id, {"GainIndex": int(index)})

    def set_detector_offset_index(self, detector_id: str, index: int, **kwargs) -> None:
        self._atomic_detector_update(detector_id, {"OffsetIndex": int(index)})

    def set_detector_digital_rotation(self, detector_id: str, angle: Quantity, **kwargs) -> None:
        # Check set_scanrotation
        deg = float(angle.to(Units.DEG).magnitude)
        d = self._get_detector(detector_id)
        if hasattr(d, "set_scanrotation"):
            d.set_scanrotation(deg)
        else:
            raise NotImplementedError("Digital rotation not supported on this detector.")

    def set_detector_frame_integration(self, detector_id: str, count: int, **kwargs) -> None:
        """Set frame integration count."""
        if self.det_mod is None:
            logger.error("[DET] SetIntegration failed: Detector hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)
            logger.debug(f"[DET] SetIntegration({detector_id}, {count})")
            if hasattr(d, "set_frameintegration"):
                d.set_frameintegration(int(count))
        except Exception as e:
            logger.error(f"[DET] SetIntegration failed: {e}")
            raise

    def set_detector_frame_rate(self, detector_id: str, rate: Quantity, **kwargs) -> None:
        raise NotImplementedError("Setting frame rate explicitly not supported.")

    def set_detector_total_frames(self, detector_id: str, count: int, **kwargs) -> None:
        raise NotImplementedError("Movie mode frame count control not implemented.")

    def set_detector_readout_mode(self, detector_id: str, mode: str, **kwargs) -> None:
        # Could map to 'ReadoutMode' key in settings
        raise NotImplementedError("Readout mode control not implemented.")

    def set_detector_shutter_mode(self, detector_id: str, mode: str, **kwargs) -> None:
        raise NotImplementedError("Shutter mode control not implemented.")

    def set_detector_save_frames(self, detector_id: str, save: bool, **kwargs) -> None:
        raise NotImplementedError("Save frames flag control not implemented.")

    def acquire_image(self, request: AcquisitionRequest, **kwargs) -> MicroscopeImage:
        # 1. Hardware Availability Check
        if self.det_mod is None:
            raise RuntimeError("Detector module missing.")

        # 2. Resolve Detector ID
        det_id = (request.detector_id
                  or getattr(request.detector, "detector_id", None)
                  or (self.get_primary_detector_id() or ""))

        if not det_id:
            raise RuntimeError("No detector_id provided and no primary detector available.")

        d = self._get_detector(det_id)

        # 3. Apply Settings
        # REMOVED: Handled by Base Class (TemMicroscope.perform_capture)
        # to prevent double-programming the hardware.

        # 4. Trigger Capture (Snapshot)
        raw = None
        try:
            # PyJEM allows multiple ways to grab data. We try them in order of preference.
            if hasattr(d, "snapshot"):
                # Standard snapshot (handles exposure wait internally)
                raw = d.snapshot()
            elif hasattr(d, "snapshot_rawdata"):
                # Preferred by some drivers: Raw data matches sensor bit-depth
                raw = d.snapshot_rawdata()
            elif hasattr(d, "get_image_cache"):
                # Fallback: Cached image (for view mode)
                raw = d.get_image_cache()
            elif hasattr(d, "livesnapshot"):
                # Fallback: Live view snapshot
                raw = d.livesnapshot("tif")
            else:
                raise RuntimeError(f"Detector {det_id} has no compatible snapshot methods.")
        except Exception as e:
            logger.error(f"[DET] Hardware Acquisition Failure: {e}")
            raise

        # 5. Process Raw Data -> Numpy Array (RESTORED ROBUST LOGIC)
        arr: np.ndarray
        if raw is None:
            arr = np.zeros((1, 1), dtype=np.uint16)
        elif isinstance(raw, (bytes, bytearray)):
            # Binary buffer
            try:
                arr = np.frombuffer(raw, dtype=np.uint16)
            except Exception:
                arr = np.frombuffer(raw, dtype=np.uint8)
        elif isinstance(raw, list):
            # List of integers
            arr = np.array(raw)
        elif isinstance(raw, dict) and "data" in raw:
            # Json wrapper
            arr = np.array(raw["data"])
        elif hasattr(raw, "data"):
            # Simple object wrapper
            arr = np.array(raw.data)
        else:
            # Fallback
            try:
                arr = np.array(raw)
            except Exception:
                arr = np.zeros((1, 1), dtype=np.uint16)

        # 6. Normalization (RESTORED ROBUST LOGIC)
        # Ensure uint16 for standard microscopy data
        if arr.dtype not in (np.uint8, np.uint16):
            try:
                arr = arr.astype(np.uint16, copy=False)
            except Exception:
                arr = np.array(arr, dtype=np.uint16)

        # Handle 1D Flattened Arrays (Reshape logic)
        # We need the ROI or scan size to know how to fold the array
        roi = None
        if request.detector and getattr(request.detector, "roi", None) is not None:
            roi = request.detector.roi
        else:
            try:
                roi = self.get_detector_roi(det_id)
            except Exception:
                roi = None

        if arr.ndim == 1:
            # Use ROI if available, otherwise fallback to scan config
            cols = int(getattr(roi, "width", 0) or self._scan_cfg.get("width_px", 0))
            rows = int(getattr(roi, "height", 0) or self._scan_cfg.get("height_px", 0))

            if cols > 0 and rows > 0 and arr.size == cols * rows:
                arr = arr.reshape((rows, cols))
            else:
                # Fallback: Try to guess square
                side = int(np.sqrt(arr.size))
                if side * side == arr.size:
                    arr = arr.reshape((side, side))

        # Handle 3D Arrays (e.g. RGB or Single Frame Stack)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]

        # Ensure at least 2D
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)

        # 7. Collect Metadata (MINIMAL / ATOMIC)
        # We only capture what the Base Class cannot know (Vendor Specifics).
        # Standard physics (Voltage, Mag, etc.) are backfilled by the Base Class if missing.
        jeol_vendor: Dict[str, Any] = {"detector_id": det_id}

        # Use new Atomic Getters
        try:
            jeol_vendor["binning_index"] = self.get_detector_binning_index(det_id)
        except Exception:
            pass

        try:
            jeol_vendor["frame_integration"] = self.get_detector_frame_integration(det_id)
        except Exception:
            pass

        if roi:
            try:
                jeol_vendor["roi"] = roi.to_dict() if hasattr(roi, "to_dict") else roi
            except Exception:
                pass

        metadata = MicroscopeImageMetadata(
            created_at=datetime.now(timezone.utc).isoformat(),
            image_size_px=(arr.shape[1], arr.shape[0]),
            extra=Extras(vendor={"JEOL": jeol_vendor}),
            _mode="lenient"
        )

        return MicroscopeImage(data=arr, metadata=metadata)

    # --- Helper Layer Overrides ---

    def get_detector_settings(self, detector_id: str) -> DetectorSettings:
        """
        Override: Get full detector state including vendor extras (Gain, Offset, etc).
        Fetches the complete settings payload from hardware via `get_detectorsetting`
        and uses the adapter to populate standard fields and extras.
        """
        if self.det_mod is None:
            logger.debug("[DET] GetSettings failed: Hardware not connected.")
            return DetectorSettings(detector_id=detector_id)

        try:
            d = self._get_detector(detector_id)
            # Fetch raw dict from PyJEM (contains GainIndex, ScanMode, etc.)
            raw = d.get_detectorsetting()

            # Use adapter to parse standard fields AND pack unknown keys into extra.vendor['JEOL']
            # Note: Expects adapter to return (DetectorSettings, DetectorCapabilities)
            settings, _ = jeol_adapter.from_jeol_detector_response(raw, detector_id)
            return settings
        except Exception as e:
            logger.debug(f"[DET] GetSettings({detector_id}) failed: {e}")
            return DetectorSettings(detector_id=detector_id)

    def apply_detector_settings(self, detector_id: str, settings: DetectorSettings, **kwargs) -> None:
        """
        Override: Apply settings using bulk setter to handle extras (Gain, Offset).
        Standard atomic setters (e.g. set_detector_exposure) do not cover all vendor
        capabilities. This method constructs a full configuration dictionary
        (merging standard fields + extra.vendor['JEOL']) and sends it via `set_detectorsetting`.
        """
        if self.det_mod is None:
            logger.error("[DET] ApplySettings failed: Hardware not connected.")
            raise RuntimeError("Detector hardware not connected.")

        try:
            d = self._get_detector(detector_id)

            # Use adapter to convert standard fields + vendor extras back into a single JEOL dict
            payload = jeol_adapter.to_jeol_detector_config(settings)

            if not payload:
                logger.debug(f"[DET] ApplySettings({detector_id}): No changes in payload.")
                return

            logger.debug(f"[DET] set_detectorsetting({list(payload.keys())})")
            d.set_detectorsetting(payload)

        except Exception as e:
            logger.error(f"[DET] ApplySettings({detector_id}) failed: {e}")
            raise

    # =========================================================================
    # 7. Scan Control (STEM)
    # =========================================================================

    # --- Atomic Getters ---

    def get_scan_mode(self) -> str:
        """Get scan mode (e.g. 'Spot', 'Area'). Querying detector first."""
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    raw = self._first_int(st, ("ScanMode", "ScanModeValue", "ScanModeIndex"))
                    if raw is not None:
                        return {0: "Scan", 1: "Spot", 3: "Area"}.get(raw, str(raw))
                    raw_s = st.get("ScanModeStr") or st.get("ScanModeString")
                    if isinstance(raw_s, str) and raw_s.strip():
                        return raw_s.strip()
            except Exception:
                pass
        return str(self._scan_cfg.get("mode", "Scan"))

    def get_scan_active(self) -> bool:
        """Check if external scan control is active."""
        if self.scan and hasattr(self.scan, "GetExtScanMode"):
            try:
                return bool(int(self.scan.GetExtScanMode()) == 1)
            except Exception:
                pass
        return bool(self._scan_cfg.get("active", False))

    def get_scan_width(self) -> Optional[int]:
        """Get active scan width in pixels."""
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    w = self._first_int(st, ("Width", "ImagingAreaWidth"))
                    if w is not None:
                        self._scan_cfg["width_px"] = w
                        return w
            except Exception:
                pass
        return None

    def get_scan_height(self) -> Optional[int]:
        """Get active scan height in pixels."""
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    h = self._first_int(st, ("Height", "ImagingAreaHeight"))
                    if h is not None:
                        self._scan_cfg["height_px"] = h
                        return h
            except Exception:
                pass
        return None

    def get_scan_pixel_dwell(self) -> Optional[Quantity]:
        """Get pixel dwell time (cached from config, as HW read is unreliable)."""
        return None

    def get_scan_flyback(self) -> Optional[Quantity]:
        """Get flyback time (cached from config)."""
        return None

    def get_scan_rotation(self) -> Optional[Quantity]:
        """Get scan rotation."""
        if self.scan and hasattr(self.scan, "GetRotationAngleEx"):
            try:
                return Q_(float(self.scan.GetRotationAngleEx()), Units.DEG)
            except Exception:
                pass

        if self.scan and hasattr(self.scan, "GetRotationAngle"):
            try:
                return Q_(float(self.scan.GetRotationAngle()), Units.DEG)
            except Exception:
                pass

        # Fallback to detector settings
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    ang = self._first_float(st, ("ScanRotation", "ScanRotationValue"))
                    if ang is not None:
                        return Q_(ang, Units.DEG)
            except Exception:
                pass
        return None

    # --- Atomic Setters ---

    def set_scan_mode(self, mode: str, **kwargs) -> None:
        """Set scan mode (e.g. 'Spot', 'Area')."""
        m = (mode or "").strip().lower()
        mapping = {"scan": 0, "full": 0, "full frame": 0, "spot": 1, "area": 3}
        if m not in mapping and m.isdigit():
            mapping[m] = int(m)
        val = mapping.get(m)
        if val is None:
            raise ValueError(f"Unsupported scan mode: {mode}")

        logger.debug(f"[SCAN] Setting Mode: {val}")

        d = self._get_scan_controller_detector()
        if d is None:
            logger.error("[SCAN] SetMode failed: No scan controller detector found.")
            raise RuntimeError("Scan detector hardware not connected.")

        if hasattr(d, "set_scanmode"):
            try:
                d.set_scanmode(int(val))
                self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))
                return
            except Exception as e:
                logger.error(f"[SCAN] SetMode failed: {e}")
                raise

        self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))

    def set_scan_active(self, active: bool, **kwargs) -> None:
        """Start or stop the scan engine."""
        detector_handled = False
        d = self._get_scan_controller_detector()
        logger.debug(f"[SCAN] SetActive({active})")

        # Detector-specific logic (Preferred)
        if d is not None:
            try:
                if active and hasattr(d, "livestart"):
                    d.livestart()
                    detector_handled = True
                elif (not active) and hasattr(d, "livestop"):
                    d.livestop()
                    detector_handled = True
            except Exception as e:
                logger.warning(f"[SCAN] Detector Live Control failed, attempting fallback: {e}")

        # Fallback to internal scan generator
        if not detector_handled:
            if self.scan and hasattr(self.scan, "SetExtScanMode"):
                try:
                    self.scan.SetExtScanMode(1 if active else 0)
                except Exception as e:
                    logger.error(f"[SCAN] SetExtScanMode failed: {e}")
                    raise
            else:
                # If detector failed and no fallback exists
                logger.error("[SCAN] SetActive failed: Detector failed and no internal scan control.")
                raise RuntimeError("Scan control failed.")

    def set_scan_width(self, width: int, **kwargs) -> None:
        self._set_imaging_area(width=int(width))

    def set_scan_height(self, height: int, **kwargs) -> None:
        self._set_imaging_area(height=int(height))

    def set_scan_pixel_dwell(self, time: Quantity, **kwargs) -> None:
        logger.error("[SCAN] set_scan_pixel_dwell not supported by JEOL driver IO.")
        raise NotImplementedError("Hardware dwell time control not supported.")

    def set_scan_flyback(self, time: Quantity, **kwargs) -> None:
        logger.error("[SCAN] set_scan_flyback not supported by JEOL driver IO.")
        raise NotImplementedError("Hardware flyback time control not supported.")

    def set_scan_rotation(self, angle: Quantity, **kwargs) -> None:
        deg = float(angle.to(Units.DEG).magnitude)
        logger.debug(f"[SCAN] SetRotation({deg})")

        d = self._get_scan_controller_detector()
        detector_success = False

        # Try Detector First
        if d is not None and hasattr(d, "set_scanrotation"):
            try:
                d.set_scanrotation(float(deg))
                detector_success = True
                return
            except Exception as e:
                logger.warning(f"[SCAN] Detector SetRotation failed, attempting fallback: {e}")

        # Try Scan Coils Fallback
        if self.scan:
            try:
                if hasattr(self.scan, "SetRotationAngleEx"):
                    self.scan.SetRotationAngleEx(float(deg))
                    return
                if hasattr(self.scan, "SetRotationAngle"):
                    self.scan.SetRotationAngle(int(round(deg)) % 360)
                    return
            except Exception as e:
                logger.error(f"[SCAN] Hardware SetRotation failed: {e}")
                raise

        # If we reached here, neither worked
        if not detector_success:
            logger.error("[SCAN] SetRotation failed: No capable hardware found.")
            raise RuntimeError("SetRotation failed on both detector and scan coils.")

    # =========================================================================
    # 8. Vacuum Control
    # =========================================================================

    # --- Atomic Getters ---

    def get_column_valve_state(self) -> str:
        # Logic retention: Bitfield 0 check
        res = self._read_hw(self.vac, "GetValveStatus", "VAC")
        if res and isinstance(res, (list, tuple)) and len(res) > 1:
            return "OPEN" if (res[1] & 1) else "CLOSED"
        return "UNKNOWN"

    def get_gun_valve_state(self) -> str:
        val = self._read_hw(self.gun, "GetBeamValve", "VAC", self._to_int)
        return "OPEN" if val == 1 else "CLOSED" if val is not None else "UNKNOWN"

    def get_turbo_pump_state(self) -> str:
        res = self._read_hw(self.vac, "GetValveStatus", "VAC")
        if res and isinstance(res, (list, tuple)) and len(res) > 1:
            return "ON" if (res[1] & 2) else "OFF"
        return "UNKNOWN"

    def get_column_pressure(self) -> Optional[Quantity]:
        # Logic retention: GetPegInfo()[0]
        res = self._read_hw(self.vac, "GetPegInfo", "VAC")
        if res and len(res) > 0:
            return Q_(float(res[0]), Units.PA)
        return None

    def get_gun_pressure(self) -> Optional[Quantity]:
        # Usually not exposed in basic PyJEM
        return None

    def get_buffer_tank_pressure(self) -> Optional[Quantity]:
        res = self._read_hw(self.vac, "GetPigInfo", "VAC")
        if res and len(res) > 0:
            return Q_(float(res[0]), Units.PA)
        return None

    # --- Atomic Setters ---

    def set_column_valve_state(self, state: str, **kwargs) -> None:
        raise NotImplementedError("Column Valve control not supported.")

    def set_gun_valve_state(self, state: str, **kwargs) -> None:
        self._write_hw(self.gun, "SetBeamValve", "VAC", 1 if state == "OPEN" else 0)

    def set_turbo_pump_state(self, state: str, **kwargs) -> None:
        raise NotImplementedError("Turbo Pump control not supported.")

    # =========================================================================
    # 9. Aperture Control
    # =========================================================================

    # --- Atomic Getters ---

    def list_apertures(self) -> List[str]:
        return list(self._APERTURE_MAP.keys())

    def get_aperture_inserted(self, aperture_id: str) -> bool:
        # Inferred from size index > 0
        idx = self.get_aperture_size_index(aperture_id)
        return idx is not None and idx > 0

    def get_aperture_size_index(self, aperture_id: str) -> Optional[int]:
        kind = self._APERTURE_MAP.get(aperture_id)
        if kind is None: return None

        # Use helper for the state change and the read
        self._write_hw(self.apt, "SelectExpKind", "APT", kind)
        return self._read_hw(self.apt, "GetExpSize", "APT", self._to_int, args=(kind,))

    def get_aperture_size_label(self, aperture_id: str) -> Optional[str]:
        return None

    def get_aperture_position(self, aperture_id: str) -> Optional[Point]:
        kind = self._APERTURE_MAP.get(aperture_id)
        if kind is None: return None

        self._write_hw(self.apt, "SelectExpKind", "APT", kind)
        res = self._read_hw(self.apt, "GetPosition", "APT")
        return Point(x=float(res[0]), y=float(res[1])) if res else None

    # --- Atomic Setters ---

    def set_aperture_inserted(self, aperture_id: str, inserted: bool, **kwargs) -> None:
        if not inserted:
            self.set_aperture_size_index(aperture_id, 0)
        else:
             raise ValueError("Cannot set inserted=True without specifying size_index.")

    def set_aperture_size_index(self, aperture_id: str, index: int, **kwargs) -> None:
        kind = self._APERTURE_MAP.get(aperture_id)
        if kind is None: raise ValueError(f"Unknown ID {aperture_id}")

        self._write_hw(self.apt, "SelectExpKind", "APT", kind)
        self._write_hw(self.apt, "SetExpSize", "APT", kind, int(index))
        time.sleep(2)  # Mechanical delay remains

    def set_aperture_position(self, aperture_id: str, x: float, y: float, **kwargs) -> None:
        kind = self._APERTURE_MAP.get(aperture_id)
        if kind is None: raise ValueError(f"Unknown ID {aperture_id}")

        self._write_hw(self.apt, "SelectExpKind", "APT", kind)
        self._write_hw(self.apt, "SetPosition", "APT", int(x), int(y))
