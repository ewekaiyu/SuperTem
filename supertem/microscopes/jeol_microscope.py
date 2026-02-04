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
        self.mds = None
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

    def _to_int_plus_one(self, v):
        return int(v + 1)

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
            self.mds = self.tem3_mod.MDS3()

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
        # GetPos returns [x, y, z, tx, ty]
        return self._read_hw(
            self.stage, "GetPos", "STAGE",
            lambda v: Q_(float(v[0]), Units.NM) if len(v) > 0 else None
        )

    def get_stage_y(self) -> Optional[Quantity]:
        return self._read_hw(
            self.stage, "GetPos", "STAGE",
            lambda v: Q_(float(v[1]), Units.NM) if len(v) > 1 else None
        )

    def get_stage_z(self) -> Optional[Quantity]:
        return self._read_hw(
            self.stage, "GetPos", "STAGE",
            lambda v: Q_(float(v[2]), Units.NM) if len(v) > 2 else None
        )

    def get_stage_tilt_x(self) -> Optional[Quantity]:
        return self._read_hw(
            self.stage, "GetPos", "STAGE",
            lambda v: Q_(float(v[3]), Units.DEG) if len(v) > 3 else None
        )

    def get_stage_tilt_y(self) -> Optional[Quantity]:
        return self._read_hw(
            self.stage, "GetPos", "STAGE",
            lambda v: Q_(float(v[4]), Units.DEG) if len(v) > 4 else None
        )

    def get_stage_r(self) -> Optional[Quantity]:
        """
        Get Rotation (deg).
        Feature Detection: Checks for 6-axis support (ARM200F+) via GetPosEx.
        GetPosEx returns [x, y, z, tx, ty, rot]
        """

        def _extract_rot(val):
            if isinstance(val, (list, tuple)) and len(val) >= 6:
                return Q_(float(val[5]), Units.DEG)
            return None

        # Try 6-axis method first
        val = self._read_hw(self.stage, "GetPosEx", "STAGE", _extract_rot)
        if val is not None:
            return val

        # Fallback: F200/5-axis machines do not support rotation -> None
        return None

    def get_stage_coordinate_system(self) -> Optional[str]:
        return "Mechanical"

    # ---  Atomic Getters (Vendor Specific) ---

    def get_stage_holder_inserted(self) -> str:
        """
        Check if holder is inserted.
        GetHolderStts: 0=Out, 1=In
        """
        val = self._read_hw(self.stage, "GetHolderStts", "STAGE", self._to_int)
        if val == 1:
            return "INSERTED"
        elif val == 0:
            return "RETRACTED"
        return "UNKNOWN"

    def get_stage_piezo_position(self) -> Optional[Tuple[float, float]]:
        """
        Get raw piezo offset (x, y) in nm.
        GetPiezoPosi returns [x, y]
        """
        return self._read_hw(self.stage, "GetPiezoPosi", "STAGE", self._coerce_xy)

    def get_stage_speed_mode(self, drive_mode: int = 0) -> Dict[str, str]:
        """
        Get speed settings for Motor(0) or Piezo(1).
        GetSpeedMode returns [xy, z, tiltxy] (0=slow, 1=normal, 2=fast)
        """
        raw = self._read_hw(self.stage, "GetSpeedMode", "STAGE", args=(drive_mode,))

        speed_map = {0: "slow", 1: "normal", 2: "fast"}
        if raw and len(raw) >= 3:
            return {
                "xy": speed_map.get(raw[0], "unknown"),
                "z": speed_map.get(raw[1], "unknown"),
                "tilt": speed_map.get(raw[2], "unknown")
            }
        return {}

    def get_stage_axis_status(self) -> Dict[str, str]:
        """
        Get detailed status for each axis (detects Limit Errors).
        Returns: Dict mapping axis ('x','y', etc.) to status.
        """
        status_map = {0: "REST", 1: "MOVING", 2: "LIMIT_ERROR"}
        raw = None
        axes = ["x", "y", "z", "tx", "ty", "r"]

        # Try 6-axis status first
        if hasattr(self.stage, "GetStatusEx"):
            try:
                raw = self.stage.GetStatusEx()
            except Exception:
                pass

        # Fallback to 5-axis
        if not raw:
            raw = self._read_hw(self.stage, "GetStatus", "STAGE", default=[])
            axes = ["x", "y", "z", "tx", "ty"]

        result = {}
        if isinstance(raw, (list, tuple)):
            for i, code in enumerate(raw):
                if i < len(axes):
                    result[axes[i]] = status_map.get(code, f"UNKNOWN_{code}")
        return result

    # --- Atomic Setters ---

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default",
                            wait: bool = True,
                            tolerance_nm: float = 200.0,
                            tolerance_deg: float = 0.1,
                            max_retries: int = 3, **kwargs) -> None:
        """
        Move stage to coordinates with 5-axis vs 6-axis feature detection.
        """
        if not self.stage:
            logger.error("[STAGE] Move failed: Hardware not connected.")
            raise RuntimeError("Stage hardware not connected.")

        dt = (drive_type or "motor").strip().lower()
        is_piezo = (dt == "piezo")

        # Parse canonical target into dictionary of raw values (nm/deg)
        t_args = jeol_adapter.to_jeol_stage_args(target)

        logger.debug(f"[STAGE] IO Write ({dt}): {t_args}")

        def _dispatch():
            # SelDrvMode: 0=Motor, 1=Piezo
            mode_idx = 1 if is_piezo else 0

            # Switch Drive Mode
            if hasattr(self.stage, "SelDrvMode"):
                self.stage.SelDrvMode(mode_idx)

            try:
                # Execute Moves (Methods exist on Stage3 class)
                # Note: SetX/SetY work for both Motor and Piezo based on SelDrvMode
                if 'x' in t_args and hasattr(self.stage, "SetX"):
                    self.stage.SetX(t_args['x'])
                if 'y' in t_args and hasattr(self.stage, "SetY"):
                    self.stage.SetY(t_args['y'])

                # Piezo usually X/Y only; ignore Z/Tilt/Rot unless hardware supports it explicitly
                if not is_piezo:
                    if 'z' in t_args and hasattr(self.stage, "SetZ"):
                        self.stage.SetZ(t_args['z'])
                    if 'tx' in t_args and hasattr(self.stage, "SetTiltXAngle"):
                        self.stage.SetTiltXAngle(t_args['tx'])
                    if 'ty' in t_args and hasattr(self.stage, "SetTiltYAngle"):
                        self.stage.SetTiltYAngle(t_args['ty'])

                    # --- Rotation Support (Feature Detection) ---
                    # SetRotation exists on ARM200F+ [cite: 2058]
                    if 'r' in t_args:
                        if hasattr(self.stage, "SetRotation"):
                            self.stage.SetRotation(float(t_args['r']))
                        else:
                            logger.warning(
                                f"[STAGE] Rotation {t_args['r']} ignored (Hardware not 6-axis compatible).")

            finally:
                # Always restore to Motor mode for safety if we switched to Piezo
                if is_piezo and hasattr(self.stage, "SelDrvMode"):
                    self.stage.SelDrvMode(0)

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
            self._wait_for_stage(timeout=30.0)

            # Verification Read
            current = self.get_stage_position()

            if target.is_close(current, tol_nm=tolerance_nm, tol_deg=tolerance_deg):
                logger.debug(f"[STAGE] Move verified within tolerance (Attempt {attempt + 1}).")
                return

            if attempt < max_retries:
                logger.info(f"[STAGE] Hysteresis Correction {attempt + 1}/{max_retries}: Adjusting position.")
                try:
                    _dispatch()
                except Exception:
                    pass
                time.sleep(0.5)
            else:
                logger.warning(f"[STAGE] Move finished but outside tolerance.")

    def stop_stage(self, **kwargs) -> None:
        """ Stop all drives. """
        self._write_hw(self.stage, "Stop", "STAGE")

    def home_stage(self, **kwargs) -> None:
        """ SetOrg: Move to origin. """
        self._write_hw(self.stage, "SetOrg", "STAGE")

    # --- Atomic Setters (Vendor Specific) ---

    def set_stage_speed_mode(self, speed: str, axis: str = "xy", drive_mode: int = 0) -> None:
        """
        Set speed mode.
        SetSpeedMode(mode, xy, z, tilt)
        """
        speed_map = {"slow": 0, "normal": 1, "fast": 2}
        s_idx = speed_map.get(speed.lower(), 1)

        # Read current state first to preserve other axes
        current_indices = [1, 1, 1]
        try:
            raw = self.stage.GetSpeedMode(drive_mode)
            if raw: current_indices = list(raw)
        except Exception:
            pass

        if axis in ["xy", "all"]: current_indices[0] = s_idx
        if axis in ["z", "all"]: current_indices[1] = s_idx
        if axis in ["tilt", "all"]: current_indices[2] = s_idx

        self._write_hw(self.stage, "SetSpeedMode", "STAGE",
                       drive_mode, current_indices[0], current_indices[1], current_indices[2])

    def set_stage_drive_frequency(self, frequency_hz: int, axis: str = "xy", drive_mode: int = 1) -> None:
        """
        Advanced: Tune the drive frequency (f1) to reduce vibration.
        Typically used for Piezo (drive_mode=1).
        """
        # 0=trackball(manual), 1=switch, 2=command(computer control)
        KIND_COMMAND = 2

        current = [0, 0, 0, 0, 0]  # x, y, z, tx, ty
        try:
            # Getf1OverRate(kind, drive_mode) [cite: 2006]
            raw = self.stage.Getf1OverRate(KIND_COMMAND, drive_mode)
            if raw: current = list(raw)
        except Exception:
            pass

        if axis in ["xy", "all"]:
            current[0] = frequency_hz  # X
            current[1] = frequency_hz  # Y
        if axis in ["z", "all"]:
            current[2] = frequency_hz
        if axis in ["tilt", "all"]:
            current[3] = frequency_hz  # Tx
            current[4] = frequency_hz  # Ty

        logger.info(f"[STAGE] Tuning Drive Frequency (f1) for {axis} to {frequency_hz} (Mode: {drive_mode})")
        # Note: Unpacking *current list into individual args
        self._write_hw(self.stage, "Setf1OverRate", "STAGE",
                       KIND_COMMAND, drive_mode, *current)

    def set_stage_acceleration(self, axis_index: int, accel: int, decel: int) -> None:
        """
        Set acceleration/deceleration rates.
        axis_index: 2=Z, 3=TiltX, 4=TiltY
        Rate: 64 (Slow) - 65535 (Fast)
        """
        if axis_index not in [2, 3, 4]:
            logger.warning("[STAGE] Acceleration control only supported for Z (2), Tx (3), Ty (4).")
            return

        self._write_hw(self.stage, "SetAccelAndDclrRate", "STAGE", axis_index, int(accel), int(decel))

    # --- Helper Layer Overrides ---

    def get_stage_position(self) -> StagePosition:
        """
        Override to perform efficient bulk read and include extras.
        """
        if not self.stage:
            return StagePosition()

        # 1. Base Read (5-axis standard) [cite: 1953]
        raw_5 = self._read_hw(self.stage, "GetPos", "STAGE", default=[])
        pos = jeol_adapter.from_jeol_stage_position(raw_5)

        # 2. Check for 6-axis Rotation (Feature Detection) [cite: 1960]
        if hasattr(self.stage, "GetPosEx"):
            try:
                raw_6 = self.stage.GetPosEx()
                if isinstance(raw_6, (list, tuple)) and len(raw_6) >= 6:
                    pos.r = Q_(float(raw_6[5]), Units.DEG)
            except Exception:
                pass

        # 3. Add Status, Piezo & Holder Info to Extras
        extras = {}

        piezo = self.get_stage_piezo_position()
        if piezo:
            extras["piezo_offset_nm"] = piezo

        holder = self.get_stage_holder_inserted()
        if holder != "UNKNOWN":
            extras["holder_status"] = holder

        # Add detailed limit switch info if any errors exist
        status = self.get_stage_axis_status()
        if any(s == "LIMIT_ERROR" for s in status.values()):
            extras["axis_status"] = status

        if extras:
            pos.extra.vendor["JEOL"] = extras

        return pos

    def perform_stage_action(self, action: str, **kwargs) -> None:
        """
        Override: Routes high-level actions to JEOL-specific atomic methods.
        Arguments come from 'StageControlRequest.extra.options'.
        """
        act = action.upper().strip()

        if act == "STOP":
            self.stop_stage(**kwargs)

        elif act == "HOME":
            self.home_stage(**kwargs)

        elif act == "SET_SPEED":
            # Unpack options: defaults to 'normal', 'xy', Motor(0)
            speed = kwargs.get("speed", "normal")
            axis = kwargs.get("axis", "xy")
            drive_mode = kwargs.get("drive_mode", 0)

            logger.info(f"[STAGE] Setting Speed: {speed} (Axis: {axis}, Mode: {drive_mode})")
            self.set_stage_speed_mode(speed, axis=axis, drive_mode=int(drive_mode))

        elif act == "TUNE_FREQUENCY":
            # options: freq=1000, axis='xy', drive_mode=1
            freq = int(kwargs.get("freq", 1000))
            self.set_stage_drive_frequency(freq, axis=kwargs.get("axis", "xy"))

        elif act == "SET_ACCEL":
            # options: axis_idx=2 (Z), val=10000
            idx = int(kwargs.get("axis_idx", 2))
            val = int(kwargs.get("val", 10000))
            self.set_stage_acceleration(idx, val, val)

        elif act == "ZERO_PIEZO":
            logger.info("[STAGE] Zeroing Piezo position")
            zero = StagePosition(x=Q_(0, "nm"), y=Q_(0, "nm"))
            self.move_stage_absolute(zero, drive_type="piezo", wait=True)

        else:
            super().perform_stage_action(action, **kwargs)

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Getters ---

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        # HT3.GetHtValue -> float (Volts). Convert to kV.
        return self._read_hw(self.ht, "GetHtValue", "BEAM", self._to_kv)

    def get_probe_mode(self) -> Optional[str]:
        # EOS3.GetProbeMode -> 0= TEM, 1= EDS, 2= NBD, 3= CBD
        val = self._read_hw(self.eos, "GetProbeMode", "BEAM", self._to_int)
        if val == 0: return "TEM"
        if val == 1: return "EDS"
        if val == 2: return "NBD"
        if val == 3: return "CBD"

        return None

    def get_beam_current(self) -> Optional[Quantity]:
        # JEOL hardware typically does not report "Probe Current" directly
        return None

    def get_emission_current(self) -> Optional[Quantity]:
        return self._read_hw(self.gun, "GetEmissionCurrentValue", "BEAM", self._to_ua)

    def get_spot_size(self) -> Optional[int]:
        # EOS3.GetSpotSize -> int (0-based index)
        # NOTE: In manufacturer UI, index is often 1-based (1..5)
        return self._read_hw(self.eos, "GetSpotSize", "BEAM", self._to_int_plus_one)

    def get_convergence_angle(self) -> Optional[Quantity]:
        # Physical angle requires calibration. Returns None.
        return None

    def get_beam_blank(self) -> bool:
        # Def3.GetBeamBlank -> 0=OFF(Unblanked), 1=ON(Blanked)
        val = self._read_hw(self.def_, "GetBeamBlank", "BEAM", self._to_int)
        return (val == 1)

    def get_beam_shift(self) -> Tuple[Optional[float], Optional[float]]:
        # Def3.GetShifBal -> [x, y] (User Beam Shift)
        return self._read_hw(self.def_, "GetShifBal", "BEAM", self._coerce_xy, default=(None, None))

    def get_beam_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        # Def3.GetTiltBal -> [x, y] (User Beam Tilt)
        return self._read_hw(self.def_, "GetTiltBal", "BEAM", self._coerce_xy, default=(None, None))

    def get_condenser_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        # Def3.GetCLs -> [x, y]
        return self._read_hw(self.def_, "GetCLs", "BEAM", self._coerce_xy, default=(None, None))

    def get_gun_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        # Def3.GetAngBal -> [x, y] (Angle Balance)
        return self._read_hw(self.def_, "GetAngBal", "BEAM", self._coerce_xy, default=(None, None))

    # --- Atomic Getters (Vendor Specific) ---

    def get_alpha_index(self) -> Optional[int]:
        """Vendor: Get Alpha (Convergence) Selector Index (0-8)."""
        # NOTE: In display the index is one higher
        return self._read_hw(self.eos, "GetAlpha", "BEAM", self._to_int_plus_one)

    def get_brightness_value(self) -> Optional[int]:
        """Vendor: Get CL3 Lens Value (0-65535). Controls Brightness."""
        return self._read_hw(self.lens, "GetCL3", "BEAM", self._to_int)

    def get_mds_mode(self) -> str:
        """
        Vendor: Get Minimum Dose System (MDS) status.
        Returns: 'OFF', 'SEARCH', 'FOCUS', 'PHOTO', or 'UNKNOWN'.
        """
        if self._read_hw(self.mds, "GetSearchMode", "BEAM") == 1:
            return "SEARCH"
        if self._read_hw(self.mds, "GetFocusMode", "BEAM") == 1:
            return "FOCUS"
        if self._read_hw(self.mds, "GetPhotoMode", "BEAM") == 1:
            return "PHOTO"
        if self.mds:
            return "OFF"
        return "UNKNOWN"

    # --- Atomic Setters ---

    def set_acceleration_voltage(self, voltage: Quantity, **kwargs) -> None:
        # HT3.SetHtValue(Volts)
        volts = float(voltage.to(Units.V).magnitude)
        self._write_hw(self.ht, "SetHtValue", "BEAM", volts)

    def set_probe_mode(self, mode: str, **kwargs) -> None:
        # EOS3.SelectProbeMode(0= TEM, 1= EDS, 2= NBD, 3= CBD)
        m = mode.strip().upper()
        if m == "TEM":
            idx = 0
        elif m == "EDS":
            idx = 1
        elif m == "NBD":
            idx = 2
        elif m == "CBD":
            idx = 3
        else:
            logger.error(f"[BEAM] Cannot set {mode} as probe mode. Available: TEM, EDS, NBD, CBD")
            raise ValueError(f"{mode} not in available modes (TEM, EDS, NBD, CBD).")
        self._write_hw(self.eos, "SelectProbeMode", "BEAM", idx)

    def set_beam_current(self, current: Quantity, **kwargs) -> None:
        # WARNING: This typically sets Emission Current on JEOL.
        raise NotImplementedError("Setting beam current is not supported on JEOL.")

    def set_emission_current(self, current: Quantity, **kwargs) -> None:
        uA = float(current.to(Units.UA).magnitude)
        self._write_hw(self.gun, "SetEmissionCurrentValue", "BEAM", uA)

    def set_spot_size(self, index: int, **kwargs) -> None:
        # EOS3.SelectSpotSize(0-N)
        # Input is 1-based (from UI), HW is 0-based
        self._write_hw(self.eos, "SelectSpotSize", "BEAM", int(index - 1))

    def set_convergence_angle(self, angle: Quantity, **kwargs) -> None:
        # Cannot set physical angle without calibration mapping.
        raise NotImplementedError("Use 'alpha_index' extra to set convergence on JEOL.")

    def set_beam_blank(self, blank: bool, **kwargs) -> None:
        # Def3.SetBeamBlank(1=ON/Blanked, 0=OFF/Unblanked)
        val = 1 if blank else 0
        self._write_hw(self.def_, "SetBeamBlank", "BEAM", val)

    def set_beam_shift(self, x: float, y: float, **kwargs) -> None:
        # Def3.SetShifBal - User Beam Shift
        self._write_hw(self.def_, "SetShifBal", "BEAM", int(x), int(y))

    def set_beam_tilt(self, x: float, y: float, **kwargs) -> None:
        # Def3.SetTiltBal - User Beam Tilt
        self._write_hw(self.def_, "SetTiltBal", "BEAM", int(x), int(y))

    def set_condenser_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetCLs", "BEAM", int(x), int(y))

    def set_gun_tilt(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetAngBal", "BEAM", int(x), int(y))

    # --- Atomic Setters (Vendor Specific) ---

    def set_alpha_index(self, idx: int, **kwargs) -> None:
        """Vendor: Set Alpha Selector (0-8)."""
        self._write_hw(self.eos, "SetAlphaSelector", "BEAM", int(idx - 1))

    def set_brightness_value(self, val: int, **kwargs) -> None:
        """Vendor: Set CL3 Lens (Brightness) Value (0-65535)."""
        self._write_hw(self.lens, "SetCL3", "BEAM", int(val))

    def set_mds_mode(self, mode: str) -> None:
        """
        Vendor: Set MDS Mode.
        mode: 'OFF', 'SEARCH', 'FOCUS', 'PHOTO'
        """
        m = mode.strip().upper()

        if m == "OFF":
            # MDS3.EndMdsMode()
            self._write_hw(self.mds, "EndMdsMode", "BEAM")
        elif m == "SEARCH":
            # MDS3.SetSearchMode(1)
            self._write_hw(self.mds, "SetSearchMode", "BEAM", 1)
        elif m == "FOCUS":
            # MDS3.SetFocusMode(1)
            self._write_hw(self.mds, "SetFocusMode", "BEAM", 1)
        elif m == "PHOTO":
            # MDS3.SetPhotoMode(1)
            self._write_hw(self.mds, "SetPhotoMode", "BEAM", 1)
        else:
            logger.error(f"[BEAM] Set MDS failed: Unknown mode {mode}")
            raise ValueError(f"Unknown MDS mode: {mode}")

    def set_ht_wobbler(self, active: bool) -> None:
        """Vendor: Control HT Wobbler (Voltage Center)."""
        state = 1 if active else 0
        self._write_hw(self.gun, "SetHtWobbler", "BEAM", state)

    def set_a2_wobbler(self, active: bool) -> None:
        """Vendor: Control A2 Wobbler (Gun Alignment)."""
        state = 1 if active else 0
        self._write_hw(self.gun, "SetA2Wobbler", "BEAM", state)

    # --- Helper Layer Overrides (Logic & Validation) ---

    def get_beam_settings(self) -> BeamSettings:
        """
        Aggregates beam state.
        Override: Adds 'alpha_index' and 'brightness_value' (CL3) to extras.
        """
        # 1. Get Base Settings (Calls standard atomics)
        bs = super().get_beam_settings()

        # 2. Get Vendor Extras
        alpha = self.get_alpha_index()
        cl3 = self.get_brightness_value()

        vendor_extras = {}
        if alpha is not None:
            vendor_extras["alpha_index"] = alpha
        if cl3 is not None:
            vendor_extras["brightness_value"] = cl3

        # 3. Merge into extras
        if vendor_extras:
            current_extras = bs.extra.vendor if (bs.extra and bs.extra.vendor) else {}
            current_extras.setdefault("JEOL", {}).update(vendor_extras)

            if not bs.extra:
                bs.extra = Extras(vendor=current_extras)
            else:
                bs.extra.vendor = current_extras

        return bs

    def apply_beam_settings(self, settings: BeamSettings, **kwargs) -> None:
        """
        Override: Handles standard settings + Vendor Extras (Alpha, Brightness).
        """
        # 1. Apply Standard Settings (Voltage, Spot, etc)
        super().apply_beam_settings(settings, **kwargs)

        # 2. Handle JEOL Extras
        vend = getattr(settings.extra, 'vendor', None)
        jeol_v = vend.get('JEOL') if isinstance(vend, dict) else None

        if isinstance(jeol_v, dict):
            # A. Alpha Index (Convergence)
            if 'alpha_index' in jeol_v:
                try:
                    idx = int(jeol_v['alpha_index'])
                    # Validation: Hardware usually 0-8
                    if not (0 <= idx <= 8):
                        raise ValueError(f"Alpha index {idx} out of range (0-8).")
                    self.set_alpha_index(idx)
                except Exception as e:
                    logger.error(f"[BEAM] Failed to set alpha_index: {e}")
                    raise

            # B. Brightness (CL3)
            if 'brightness_value' in jeol_v:
                try:
                    val = int(jeol_v['brightness_value'])
                    self.set_brightness_value(val)
                except Exception as e:
                    logger.error(f"[BEAM] Failed to set brightness_value: {e}")
                    raise

    def perform_beam_action(self, action: str, **kwargs) -> None:
        """
        Override: Handles 'FLASH_FEG', 'OPEN_VALVE', etc.
        """
        act = action.upper().strip()

        if act == "FLASH_FEG":
            # PyJEM FEG3: ExecAutoFlashing(1) -> Start
            if hasattr(self.feg, "ExecAutoFlashing"):
                logger.info("[BEAM] Executing FEG Auto-Flash...")
                self._write_hw(self.feg, "ExecAutoFlashing", "BEAM", 1)
            else:
                raise RuntimeError("FEG Flashing not supported (FEG3 module missing or incompatible).")

        elif act in ["OPEN_VALVE", "OPEN_V1"]:
            logger.info("[BEAM] Opening Gun Valve (V1)...")
            self.set_gun_valve_state("OPEN")

        elif act in ["CLOSE_VALVE", "CLOSE_V1"]:
            logger.info("[BEAM] Closing Gun Valve (V1)...")
            self.set_gun_valve_state("CLOSED")

        elif act == "SET_MDS":
            # kwargs: mode (str)
            mode = kwargs.get("mode", "OFF")
            self.set_mds_mode(mode)

        elif act == "WOBBLE_HT":
            active = bool(kwargs.get("active", True))
            logger.info(f"[BEAM] HT Wobbler active={active}")
            self.set_ht_wobbler(active)

        elif act == "WOBBLE_A2":
            active = bool(kwargs.get("active", True))
            logger.info(f"[BEAM] A2 (Gun) Wobbler active={active}")
            self.set_a2_wobbler(active)

        else:
            super().perform_beam_action(action, **kwargs)

    # =========================================================================
    # 5. Projection Control (Imaging/Optics)
    # =========================================================================

    # --- Atomic Getters ---

    def get_optical_mode(self) -> str:
        """Get the logical optical mode (e.g. 'TEM:MAG')."""
        key, _ = self._resolve_eos_table_info()
        return key if key else "UNKNOWN"

    def get_magnification(self) -> Optional[int]:
        """Get the magnification value (e.g., 100000)."""
        if not self.eos:
            logger.debug("[LENS] GetMagValue failed: Hardware not connected.")
            return None

        if hasattr(self.eos, "GetMagValue"):
            try:
                val = self.eos.GetMagValue()
                # PyJEM returns [value, unit, label] e.g. [50000, 'X', 'x50k']
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    if str(val[1]).strip().upper() == "X":
                        return int(round(float(val[0])))
                else:
                    return int(round(float(val)))
            except Exception as e:
                logger.debug(f"[LENS] GetMagValue failed: {e}")
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
        mapping = {0: "UP", 1: "INTERCEPT", 2: "DOWN"}
        return mapping.get(idx, "UNKNOWN")

    def get_objective_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetOLs", "LENS", self._coerce_xy, default=(None, None))

    def get_diffraction_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetILs", "LENS", self._coerce_xy, default=(None, None))

    def get_image_shift(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetIS1", "LENS", self._coerce_xy, default=(None, None))

    def get_diffraction_shift(self) -> Tuple[Optional[float], Optional[float]]:
        return self._read_hw(self.def_, "GetPLA", "LENS", self._coerce_xy, default=(None, None))

    # --- Atomic Getters (Vendor Specific) ---

    def get_defocus_dac(self) -> Optional[int]:
        return self._read_hw(self.lens, "GetOLc", "LENS", self._to_int)

    def get_defocus_fine_dac(self) -> Optional[int]:
        """Ref: Lens3.GetOLf"""
        return self._read_hw(self.lens, "GetOLf", "LENS", self._to_int)

    def get_defocus_superfine_dac(self) -> Optional[int]:
        """Ref: Lens3.GetOLSuperFineValue"""
        return self._read_hw(self.lens, "GetOLSuperFineValue", "LENS", self._to_int)

    def get_image_shift2(self) -> Tuple[Optional[float], Optional[float]]:
        """Ref: Def3.GetIS2 """
        return self._read_hw(self.def_, "GetIS2", "LENS", self._coerce_xy, default=(None, None))

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
        mapping = {"UP": 0, "INTERCEPT": 1, "DOWN": 2}
        val = mapping.get(p)
        if val is None:
            raise ValueError(f"Invalid screen position '{p}'. Use UP, DOWN, or INTERCEPT.")
        self._write_hw(self.det3, "SetScreen", "LENS", val)

    def set_objective_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetOLs", "LENS", int(x), int(y))

    def set_diffraction_stigmation(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetILs", "LENS", int(x), int(y))

    def set_image_shift(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetIS1", "LENS", int(x), int(y))

    def set_diffraction_shift(self, x: float, y: float, **kwargs) -> None:
        self._write_hw(self.def_, "SetPLA", "LENS", int(x), int(y))

    # --- Atomic Setters (Vendor Specific) ---

    def set_defocus_dac(self, dac: int) -> None:
        self._write_hw(self.lens, "SetOLc", "LENS", int(dac))

    def set_defocus_fine_dac(self, dac: int) -> None:
        """Ref: Lens3.SetOLf [cite: 1703]"""
        self._write_hw(self.lens, "SetOLf", "LENS", int(dac))

    def set_defocus_superfine_dac(self, dac: int) -> None:
        """
        Set OLS (SuperFine).
        Ref: Lens3.SetOLSuperFineValue [cite: 1688]
        """
        # Ensure switch is ON [cite: 1682]
        self._write_hw(self.lens, "SetOLSuperFineSw", "LENS", 1)
        self._write_hw(self.lens, "SetOLSuperFineValue", "LENS", int(dac))

    def set_standard_focus(self) -> None:
        """
        Execute Standard Focus (Hysteresis Reset).
        Ref: Lens3.SetStdFocus
        """
        self._write_hw(self.lens, "SetStdFocus", "LENS")

    def set_image_shift2(self, x: float, y: float) -> None:
        """Ref: Def3.SetIS2 [cite: 516]"""
        self._write_hw(self.def_, "SetIS2", "LENS", int(x), int(y))

    def set_diffraction_focus(self, val: int, absolute: bool = True) -> None:
        """
        Control Diffraction Focus.
        Ref: Lens3.SetDiffFocus (Absolute) [cite: 1609]
        Ref: EOS3.SetDiffFocus (Relative Knob) [cite: 929]
        """
        if absolute:
            self._write_hw(self.lens, "SetDiffFocus", "LENS", int(val))
        else:
            self._write_hw(self.eos, "SetDiffFocus", "LENS", int(val))

    def step_objective_focus(self, steps: int) -> None:
        """
        Simulate Objective Focus Knob (Relative).
        Ref: EOS3.SetObjFocus [cite: 945]
        """
        self._write_hw(self.eos, "SetObjFocus", "LENS", int(steps))

    # --- Helper Layer Overrides ---

    def get_projection_settings(self) -> ProjectionSettings:
        """
        Override Reason: Populate vendor-specific 'defocus_olc_dac' if uncalibrated.
        """
        ps = super().get_projection_settings()

        # Populate extras with Fine/SuperFine DACs
        extras = ps.extra.vendor.setdefault("JEOL", {})

        # Coarse (if uncalibrated)
        if not self._has_defocus_calibration:
            dac = self.get_defocus_dac()
            if dac is not None:
                extras["defocus_olc_dac"] = dac
                ps.extra.notes["defocus"] = "Uncalibrated OLc DAC"

        # Fine
        f_dac = self.get_defocus_fine_dac()
        if f_dac is not None:
            extras["defocus_olf_dac"] = f_dac

        # SuperFine
        sf_dac = self.get_defocus_superfine_dac()
        if sf_dac is not None:
            extras["defocus_ols_dac"] = sf_dac

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

            if isinstance(jeol_v, dict):
                # Coarse DAC (fallback)
                if 'defocus_olc_dac' in jeol_v:
                    self.set_defocus_dac(int(jeol_v['defocus_olc_dac']))

                # Fine DAC
                if 'defocus_olf_dac' in jeol_v:
                    self.set_defocus_fine_dac(int(jeol_v['defocus_olf_dac']))

                # SuperFine DAC
                if 'defocus_ols_dac' in jeol_v:
                    self.set_defocus_superfine_dac(int(jeol_v['defocus_ols_dac']))

        except Exception as e:
            logger.error(f"[LENS] Failed to apply projection extras: {e}")
            raise

    def perform_projection_action(self, action: str, **kwargs) -> None:
        """
        Override: Handles STD_FOCUS, STEP_FOCUS, etc.
        """
        act = action.upper().strip()

        if act == "STD_FOCUS":
            logger.info("[LENS] Executing Standard Focus...")
            self.set_standard_focus()

        elif act == "STEP_FOCUS":
            # Simulate knob turn
            steps = int(kwargs.get("steps", 1))
            self.step_objective_focus(steps)

        elif act == "STEP_DIFF_FOCUS":
            steps = int(kwargs.get("steps", 1))
            self.set_diffraction_focus(steps, absolute=False)

        else:
            super().perform_projection_action(action, **kwargs)

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
        """Atomic: Robust check for V1 (FEG or Thermionic)."""
        # 1. Try FEG3 (Modern/FEG)
        if hasattr(self.feg, "GetBeamValve"):
            val = self._read_hw(self.feg, "GetBeamValve", "VAC")
            if val == 1: return "OPEN"
            if val == 0: return "CLOSED"

        # 2. Fallback to GUN3 (Thermionic)
        # Cite: PyJEM_TEM3_reorganized_clean.docx (GUN3.GetBeamValve)
        val = self._read_hw(self.gun, "GetBeamValve", "VAC")
        if val == 1: return "OPEN"
        if val == 0: return "CLOSED"

        return "UNKNOWN"

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
        """Atomic: Set V1 State (Open/Close). Handles FEG vs Thermionic."""
        is_open = 1 if state.upper() == "OPEN" else 0

        # Prefer FEG3 if available
        if hasattr(self.feg, "SetBeamValve"):
            self._write_hw(self.feg, "SetBeamValve", "VAC", is_open)
        else:
            # Cite: PyJEM_TEM3_reorganized_clean.docx (GUN3.SetBeamValve)
            self._write_hw(self.gun, "SetBeamValve", "VAC", is_open)

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
