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
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
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
from supertem.vendor.JEOL.jeol_eos_tables import DEFAULT_TABLES

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

    # TEM Function Indices
    _TEM_FUNC_MAP = {
        0: "MAG",
        1: "MAG2",
        2: "LOWMAG",
        3: "SAMAG",
        4: "DIFF"
    }
    # STEM Function Indices [cite: 819, 888]
    _STEM_FUNC_MAP = {
        0: "ALIGN",
        1: "SM-LMAG",
        2: "SM-MAG",
        3: "AMAG",
        4: "UUDIFF",  # Often maps to "DIFF" logic in STEM
        5: "ROCKING"
    }

    # Reverse maps for Setters
    _TEM_FUNC_NAME_TO_IDX = {v: k for k, v in _TEM_FUNC_MAP.items()}
    _STEM_FUNC_NAME_TO_IDX = {v: k for k, v in _STEM_FUNC_MAP.items()}

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

        # Load Tables: Start with Defaults, then Override with Config
        self.optical_tables = DEFAULT_TABLES.copy()

        if config:
            # Check for attribute first (Pydantic/Dataclass)
            val = getattr(config, 'defocus_scale', None)
            # Fallback to dict if it somehow is one
            if val is None and isinstance(config, dict):
                val = config.get('defocus_scale')

            if val is not None:
                self._has_defocus_calibration = True
                self.defocus_scale = float(val)

            # Table Overrides (e.g., config.extra.vendor["JEOL"]["optical_tables"]["STEM:SM-MAG"])
            if hasattr(config, "extra") and config.extra:
                vendor = getattr(config.extra, "vendor", {})
                if vendor and "JEOL" in vendor:
                    custom = vendor["JEOL"].get("optical_tables")
                    if custom:
                        # Deep merge or update
                        for mode_key, tables in custom.items():
                            if mode_key in self.optical_tables:
                                self.optical_tables[mode_key].update(tables)
                            else:
                                self.optical_tables[mode_key] = tables
                        logger.info(f"Loaded custom JEOL optical tables for: {list(custom.keys())}")

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

    def _get_current_mode_key(self) -> str:
        """
        Robustly determine the current EOS mode key (e.g., 'STEM:SM-MAG').
        Sources of Truth:
          1. EOS3.GetTemStemMode() -> TEM vs STEM
          2. EOS3.GetFunctionMode() -> Function Index

        Returns:
            "TEM:MAG", "STEM:SM-MAG", etc., or "UNKNOWN" on failure.
        """
        if not self.eos:
            return "UNKNOWN"

        try:
            # 1. Determine Base Mode (TEM=0, STEM=1)
            # Direct hardware call protected by try/except
            base_mode_idx = int(self.eos.GetTemStemMode())
            base_mode = "STEM" if base_mode_idx == 1 else "TEM"

            # 2. Determine Function Index
            # Handle PyJEM version differences (list vs scalar return)
            func_data = self.eos.GetFunctionMode()
            func_idx = -1

            if isinstance(func_data, (list, tuple)):
                func_idx = int(func_data[0])
            else:
                func_idx = int(func_data)

            # 3. Map Index to Standardized Name (using class constants)
            if base_mode == "TEM":
                func_name = self._TEM_FUNC_MAP.get(func_idx, f"UNKNOWN-{func_idx}")
            else:
                func_name = self._STEM_FUNC_MAP.get(func_idx, f"UNKNOWN-{func_idx}")

            return f"{base_mode}:{func_name}"

        except Exception as e:
            # Getter Pattern: Log failure at DEBUG, return safe default
            logger.debug(f"[EOS] Mode detection failed: {e}")
            return "UNKNOWN"

    def _get_table_entry(self, mode_key: str, list_type: str) -> List[float]:
        """
        Retrieve the specific MagList or CamList for the active mode.
        Falls back to defaults if specific sub-mode table is missing.
        """
        # 1. Direct Lookup
        if mode_key in self.optical_tables:
            return self.optical_tables[mode_key].get(list_type, [])

        # 2. Fallback Logic
        # If we are in STEM but the specific mode (e.g. unknown new mode) isn't in tables,
        # fallback to 'STEM:SM-MAG' as the safe default for this instrument.
        if "STEM" in mode_key and "STEM:SM-MAG" in self.optical_tables:
            logger.debug(f"[EOS] Table '{mode_key}' missing, using 'STEM:SM-MAG' fallback.")
            return self.optical_tables["STEM:SM-MAG"].get(list_type, [])

        return []

    def _find_closest_index(self, table: List[float], value: float) -> int:
        if not table: return 0
        arr = np.array(table)
        idx = (np.abs(arr - value)).argmin()
        return int(idx)

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
            raise RuntimeError("EOS hardware not connected.")

        target = mode.strip().upper()
        if target not in ["TEM", "STEM"]:
            raise ValueError(f"Invalid mode '{mode}'. Use TEM or STEM.")

        logger.info(f"[EOS] Switching to {target} mode...")
        try:
            self.eos.SelectTemStem(0 if target == "TEM" else 1)

            # Stabilization wait (hardware mode switching is slow)
            time.sleep(1.0)
        except Exception as e:
            logger.error(f"[EOS] Failed to set mode {target}: {e}")
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

    def get_condenser_lens_1(self) -> Optional[int]:
        """Atomic: Get CL1 (Spot Size Lens)."""
        return self._read_hw(self.lens, "GetCL1", "BEAM", self._to_int)

    def get_condenser_lens_2(self) -> Optional[int]:
        """Atomic: Get CL2 (Convergence Lens)."""
        return self._read_hw(self.lens, "GetCL2", "BEAM", self._to_int)

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

    def set_condenser_lens_1(self, dac: int) -> None:
        """Atomic: Set CL1."""
        self._write_hw(self.lens, "SetCL1", "BEAM", int(dac))

    def set_condenser_lens_2(self, dac: int) -> None:
        """Atomic: Set CL2."""
        self._write_hw(self.lens, "SetCL2", "BEAM", int(dac))

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
        cl1 = self.get_condenser_lens_1()
        cl2 = self.get_condenser_lens_2()

        vendor_extras = {}
        if alpha is not None:
            vendor_extras["alpha_index"] = alpha
        if cl3 is not None:
            vendor_extras["brightness_value"] = cl3
        if cl1 is not None:
            vendor_extras["condenser_lens_1"] = cl1
        if cl2 is not None:
            vendor_extras["condenser_lens_2"] = cl2

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

            if 'condenser_lens_1' in jeol_v:
                self.set_condenser_lens_1(int(jeol_v['condenser_lens_1']))

            if 'condenser_lens_2' in jeol_v:
                self.set_condenser_lens_2(int(jeol_v['condenser_lens_2']))

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

        elif act == "WOBBLE_TILT":
            amp_x = int(kwargs.get("amp_x", 200))
            amp_y = int(kwargs.get("amp_y", 0))
            cycles = int(kwargs.get("cycles", 5))
            delay = float(kwargs.get("delay", 0.15))

            cx_raw, cy_raw = self.get_beam_tilt()
            if cx_raw is None or cy_raw is None:
                raise RuntimeError("Cannot wobble: Current Beam Tilt unknown.")

            cx, cy = float(cx_raw), float(cy_raw)
            logger.info(f"[BEAM] Starting Tilt Wobbler (Center: {cx:.0f},{cy:.0f})")

            try:
                for _ in range(cycles):
                    self.set_beam_tilt(cx + amp_x, cy + amp_y)
                    time.sleep(delay)
                    self.set_beam_tilt(cx - amp_x, cy - amp_y)
                    time.sleep(delay)
            finally:
                self.set_beam_tilt(cx, cy)
                logger.debug("[BEAM] Tilt Wobbler finished.")

        else:
            super().perform_beam_action(action, **kwargs)

    # =========================================================================
    # 5. Projection Control (Imaging/Optics)
    # =========================================================================

    # --- Atomic Getters ---

    def get_optical_mode(self) -> str:
        """Get the logical optical mode (e.g. 'TEM:MAG')."""
        return self._get_current_mode_key()

    def get_magnification(self) -> Optional[int]:
        """Get the magnification value (e.g., 100000)."""
        if not self.eos: return None
        mode_key = self._get_current_mode_key()

        # In DIFF mode, 'Magnification' is actually Camera Length
        if "DIFF" in mode_key:
            return None

        # Use _read_hw to preserve "Null means Unknown" logging logic
        # EOS3.GetMagValue returns [val, unit, label]
        def _extract_mag(raw):
            if isinstance(raw, (list, tuple)) and len(raw) > 0:
                return int(float(raw[0]))
            return int(float(raw))

        return self._read_hw(self.eos, "GetMagValue", "LENS", _extract_mag)

    def get_camera_length(self) -> Optional[Quantity]:
        """
        Universal Getter.
        TEM:DIFF -> Uses GetMagValue.
        STEM:* -> Uses GetStemCamValue.
        """
        if not self.eos: return None
        mode_key = self._get_current_mode_key()

        def _extract_mm(raw):
            val = raw[0] if isinstance(raw, (list, tuple)) else raw
            return Q_(float(val), Units.MM)

        if "TEM:DIFF" in mode_key:
            return self._read_hw(self.eos, "GetMagValue", "LENS", _extract_mm)
        elif "STEM" in mode_key:
            return self._read_hw(self.eos, "GetStemCamValue", "LENS", _extract_mm)

        return None

    def get_defocus(self) -> Optional[Quantity]:
        """
        Calculates physical defocus (nm) based on F200-specific DAC offsets.
        """
        # Safety: Only run this math for F200 models
        model = (self.system_settings.info.model or "").upper()
        if "F200" not in model:
            logger.debug(f"[LENS] get_defocus skipped: Model '{model}' is not F200 (Physical scaling unknown).")
            return None

        mode_key = self._get_current_mode_key()
        mag = self.get_magnification() or 0

        # TEM:LOWMAG -> OM (Objective Mini)
        if mode_key == "TEM:LOWMAG":
            raw_dac = self.get_objective_mini_lens_1()
            if raw_dac is None: return None

            # F200 Thresholds
            if mag < 600: std_val = 0xA7C4
            elif mag < 15000: std_val = 0xBD31
            else: std_val = 0xC85A

            # 1 bit = 1500 nm
            return Q_((raw_dac - std_val) * 1500.0, Units.NM)

        # TEM:MAG -> OLf (Objective Fine)
        elif mode_key == "TEM:MAG":
            raw_dac = self.get_objective_lens_fine()
            if raw_dac is None: return None

            # F200 Thresholds
            std_val = 0x8010 if mag < 1500000 else 0x7D10
            # 1 bit = 1.4 nm
            return Q_((raw_dac - std_val) * 1.4, Units.NM)

        return None

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

    def get_objective_lens_coarse(self) -> Optional[int]:
        """Atomic: Get OLc (Objective Lens Coarse)."""
        return self._read_hw(self.lens, "GetOLc", "LENS", self._to_int)

    def get_objective_lens_fine(self) -> Optional[int]:
        """Atomic: Get OLf (Objective Lens Fine)."""
        return self._read_hw(self.lens, "GetOLf", "LENS", self._to_int)

    def get_objective_lens_superfine(self) -> Optional[int]:
        """Atomic: Get OLS (Objective Lens SuperFine)."""
        return self._read_hw(self.lens, "GetOLSuperFineValue", "LENS", self._to_int)

    def get_objective_mini_lens_1(self) -> Optional[int]:
        """Atomic: Get OM (Objective Mini-lens)."""
        return self._read_hw(self.lens, "GetOM", "LENS", self._to_int)

    def get_objective_mini_lens_2(self) -> Optional[int]:
        """Atomic: Get OM2."""
        return self._read_hw(self.lens, "GetOM2", "LENS", self._to_int)

    def get_focus_lens_coarse(self) -> Optional[int]:
        """Atomic: Get FLc (Focus Lens Coarse)."""
        return self._read_hw(self.lens, "GetFLc", "LENS", self._to_int)

    def get_focus_lens_fine(self) -> Optional[int]:
        """Atomic: Get FLf (Focus Lens Fine)."""
        return self._read_hw(self.lens, "GetFLf", "LENS", self._to_int)

    def get_intermediate_lens_1(self) -> Optional[int]:
        """Atomic: Get IL1 (Intermediate Lens 1)."""
        return self._read_hw(self.lens, "GetIL1", "LENS", self._to_int)

    def get_intermediate_lens_2(self) -> Optional[int]:
        """Atomic: Get IL2."""
        return self._read_hw(self.lens, "GetIL2", "LENS", self._to_int)

    def get_intermediate_lens_3(self) -> Optional[int]:
        """Atomic: Get IL3."""
        return self._read_hw(self.lens, "GetIL3", "LENS", self._to_int)

    def get_intermediate_lens_4(self) -> Optional[int]:
        """Atomic: Get IL4."""
        return self._read_hw(self.lens, "GetIL4", "LENS", self._to_int)

    def get_projector_lens_1(self) -> Optional[int]:
        """Atomic: Get PL1."""
        return self._read_hw(self.lens, "GetPL1", "LENS", self._to_int)

    def get_projector_lens_2(self) -> Optional[int]:
        """Atomic: Get PL2."""
        return self._read_hw(self.lens, "GetPL2", "LENS", self._to_int)

    def get_projector_lens_3(self) -> Optional[int]:
        """Atomic: Get PL3."""
        return self._read_hw(self.lens, "GetPL3", "LENS", self._to_int)

    def get_image_shift_2(self) -> Tuple[Optional[float], Optional[float]]:
        """Ref: Def3.GetIS2 """
        return self._read_hw(self.def_, "GetIS2", "LENS", self._coerce_xy, default=(None, None))

    def get_diffraction_focus(self) -> Optional[int]:
        """
        Get the logical 'diffraction focus' property.

        Logic:
          - F200: Returns IL1 (Intermediate Lens 1).
          - Generic: Returns None (or specific register if known).
        """
        model = (self.system_settings.info.model or "").upper()
        if "F200" in model:
            return self.get_intermediate_lens_1()

        logger.debug(f"[LENS] get_diffraction_focus skipped: Model '{model}' is not F200 (IL1 mapping unknown).")
        return None

    # --- Atomic Setters ---

    def set_optical_mode(self, mode: str, **kwargs) -> None:
        """
        Set the optical function mode (e.g., 'TEM:MAG', 'STEM:SM-MAG').
        Automatically switches TEM/STEM base mode if necessary.
        """
        if not self.eos: raise RuntimeError("EOS hardware not connected.")

        target = mode.strip().upper()

        # Parse Input
        if ":" in target:
            base, func = target.split(":", 1)
        else:
            base = self.get_mode()
            func = target

        base = base.strip()
        func = func.strip().replace("_", "-")

        # Switch Base Mode (TEM/STEM) if needed
        current_base = self.get_mode()
        if base != current_base:
            self.set_mode(base)

        # Resolve Function Index
        func_idx = None
        if base == "TEM":
            func_idx = self._TEM_FUNC_NAME_TO_IDX.get(func)
        elif base == "STEM":
            func_idx = self._STEM_FUNC_NAME_TO_IDX.get(func)

        if func_idx is None:
            valid = list(self._TEM_FUNC_MAP.values()) if base == "TEM" else list(self._STEM_FUNC_MAP.values())
            raise ValueError(f"Invalid function '{func}' for mode {base}. Valid: {valid}")

        logger.info(f"[EOS] Setting Optical Mode: {base}:{func} (Index {func_idx})")

        # Safe Execution using _write_hw to catch/log errors
        self._write_hw(self.eos, "SelectFunctionMode", "EOS", func_idx)

    def set_magnification(self, value: int, **kwargs) -> None:
        """
        Universal Setter.
        Always uses 'MagList' and 'SetSelector'.
        """
        if not self.eos: raise RuntimeError("Hardware disconnected")

        mode_key = self._get_current_mode_key()

        # 1. Validation: Can't set Mag in Diff mode
        if "DIFF" in mode_key:
            raise RuntimeError(f"Cannot set Magnification in Diffraction mode ({mode_key}). Use Camera Length.")

        # 2. Get Table
        table = self._get_table_entry(mode_key, "MagList")
        if not table:
            raise RuntimeError(f"Magnification table (MagList) missing for mode {mode_key}.")

        # 3. Find Index
        idx = self._find_closest_index(table, float(value))

        # 4. Log High-Level Intent (Info)
        logger.info(f"[LENS] SetMag ({mode_key}): {table[idx]}x (Index {idx})")

        # 5. Execute via Safe Wrapper (Preserves Debug/Error logging)
        self._write_hw(self.eos, "SetSelector", "LENS", idx)

    def set_camera_length(self, length: Quantity, **kwargs) -> None:
        """
        Universal Setter.
        [cite_start]TEM:DIFF -> Uses 'MagList' table -> SetSelector [cite: 953]
        [cite_start]STEM:* -> Uses 'CamList' table -> SetStemCamSelector [cite: 975]
        """
        if not self.eos: raise RuntimeError("Hardware disconnected")

        mode_key = self._get_current_mode_key()
        target_mm = length.to(Units.MM).magnitude

        if "TEM:DIFF" in mode_key:
            # Special Case: TEM Diff uses the Mag Selector logic
            table = self._get_table_entry(mode_key, "MagList")
            if not table: raise RuntimeError(f"Camera Length table missing for {mode_key}")

            idx = self._find_closest_index(table, target_mm)
            logger.info(f"[LENS] SetCL ({mode_key}): {table[idx]}mm via SetSelector (Index {idx})")

            # Safe execution
            self._write_hw(self.eos, "SetSelector", "LENS", idx)

        elif "STEM" in mode_key:
            # Standard Case: STEM uses the dedicated Cam Selector logic
            table = self._get_table_entry(mode_key, "CamList")
            if not table: raise RuntimeError(f"Camera Length table (CamList) missing for {mode_key}")

            idx = self._find_closest_index(table, target_mm)
            logger.info(f"[LENS] SetCL ({mode_key}): {table[idx]}mm via SetStemCamSelector (Index {idx})")

            # Safe execution
            self._write_hw(self.eos, "SetStemCamSelector", "LENS", idx)

        else:
            raise RuntimeError(f"Cannot set Camera Length in mode {mode_key}.")

    def set_defocus(self, defocus: Quantity, **kwargs) -> None:
        """
        Sets absolute physical defocus (nm).
        Only supported on models with known calibration (e.g., F200).
        """
        # 1. Enforce Contract: Reject Steps/Integers
        if not isinstance(defocus, Quantity):
            raise TypeError(
                "set_defocus requires a Quantity (nm). "
                "For relative steps, use the 'STEP_FOCUS' action via perform_projection_action."
            )

        # 2. Model-Specific Absolute Setting (F200 Logic)
        model = (self.system_settings.info.model or "").upper()
        if "F200" not in model:
            logger.warning("[LENS] Absolute physical defocus setting only supported for F200.")
            return

        target_nm = float(defocus.to(Units.NM).magnitude)
        mode_key = self._get_current_mode_key()
        mag = self.get_magnification() or 0

        if mode_key == "TEM:LOWMAG":
            if mag < 600:
                std = 0xA7C4
            elif mag < 15000:
                std = 0xBD31
            else:
                std = 0xC85A
            target_dac = int(std + (target_nm / 1500.0))
            self.set_objective_mini_lens_1(target_dac)

        elif mode_key == "TEM:MAG":
            std = 0x8010 if mag < 1500000 else 0x7D10
            target_dac = int(std + (target_nm / 1.4))
            self.set_objective_lens_fine(target_dac)

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

    def set_objective_lens_coarse(self, dac: int) -> None:
        """Atomic: Set OLc."""
        self._write_hw(self.lens, "SetOLc", "LENS", int(dac))

    def set_objective_lens_fine(self, dac: int) -> None:
        """Atomic: Set OLf."""
        self._write_hw(self.lens, "SetOLf", "LENS", int(dac))

    def set_objective_lens_superfine(self, dac: int) -> None:
        """Atomic: Set OLS."""
        self._write_hw(self.lens, "SetOLSuperFineSw", "LENS", 1)
        self._write_hw(self.lens, "SetOLSuperFineValue", "LENS", int(dac))

    def set_objective_mini_lens_1(self, dac: int) -> None:
        """Atomic: Set OM."""
        self._write_hw(self.lens, "SetOM", "LENS", int(dac))

    def set_objective_mini_lens_2(self, dac: int) -> None:
        """Atomic: Set OM2."""
        self._write_hw(self.lens, "SetOM2", "LENS", int(dac))

    def set_focus_lens_coarse(self, dac: int) -> None:
        """Atomic: Set FLc."""
        self._write_hw(self.lens, "SetFLc", "LENS", int(dac))

    def set_focus_lens_fine(self, dac: int) -> None:
        """Atomic: Set FLf."""
        self._write_hw(self.lens, "SetFLf", "LENS", int(dac))

    def set_intermediate_lens_1(self, dac: int) -> None:
        """Atomic: Set IL1."""
        self._write_hw(self.lens, "SetIL1", "LENS", int(dac))

    def set_intermediate_lens_2(self, dac: int) -> None:
        """Atomic: Set IL2."""
        self._write_hw(self.lens, "SetIL2", "LENS", int(dac))

    def set_intermediate_lens_3(self, dac: int) -> None:
        """Atomic: Set IL3."""
        self._write_hw(self.lens, "SetIL3", "LENS", int(dac))

    def set_intermediate_lens_4(self, dac: int) -> None:
        """Atomic: Set IL4."""
        self._write_hw(self.lens, "SetIL4", "LENS", int(dac))

    def set_projector_lens_1(self, dac: int) -> None:
        """Atomic: Set PL1."""
        self._write_hw(self.lens, "SetPL1", "LENS", int(dac))

    def set_projector_lens_2(self, dac: int) -> None:
        """Atomic: Set PL2."""
        self._write_hw(self.lens, "SetPL2", "LENS", int(dac))

    def set_projector_lens_3(self, dac: int) -> None:
        """Atomic: Set PL3."""
        self._write_hw(self.lens, "SetPL3", "LENS", int(dac))

    def set_standard_focus(self) -> None:
        """Atomic: Execute Standard Focus."""
        self._write_hw(self.lens, "SetStdFocus", "LENS")

    def set_image_shift_2(self, x: float, y: float) -> None:
        self._write_hw(self.def_, "SetIS2", "LENS", int(x), int(y))

    def set_diffraction_focus(self, index: int, relative: bool = False) -> None:
        """
        Set 'diffraction focus'.
        - Relative: Uses universal 'SetDiffFocus' (Knob turn).
        - Absolute: Uses F200 'SetIL1' (Physical register).
        """
        val = int(index)

        # 1. Universal Relative Step
        if relative:
            # Universal command for all JEOL models
            self._write_hw(self.eos, "SetDiffFocus", "LENS", val)
            return

        # 2. Model-Specific Absolute Setting
        model = (self.system_settings.info.model or "").upper()
        if "F200" in model:
            self.set_intermediate_lens_1(val)
        else:
            logger.warning("[LENS] Absolute diffraction focus not supported for this model (Requires F200 IL1 logic).")

    def set_relative_focus_steps(self, steps: int) -> None:
        """Atomic: Adjust focus by relative hardware steps (Knob turn)."""
        self._write_hw(self.eos, "SetObjFocus", "LENS", int(steps))

    # --- Helper Layer Overrides ---

    def get_projection_settings(self) -> ProjectionSettings:
        """
        Aggregates optical state.
        Ensures raw DACs are ALWAYS stored in extras by calling atomic getters.
        """
        # 1. Get Standard Physics
        ps = super().get_projection_settings()

        # 2. Raw Hardware Registers (Source of Truth)
        extras = ps.extra.vendor.setdefault("JEOL", {})
        extras["objective_lens_coarse"] = self.get_objective_lens_coarse()
        extras["objective_lens_fine"] = self.get_objective_lens_fine()
        extras["objective_lens_superfine"] = self.get_objective_lens_superfine()
        extras["objective_mini_lens_1"] = self.get_objective_mini_lens_1()
        extras["objective_mini_lens_2"] = self.get_objective_mini_lens_2()
        extras["focus_lens_coarse"] = self.get_focus_lens_coarse()
        extras["focus_lens_fine"] = self.get_focus_lens_fine()
        extras["intermediate_lens_1"] = self.get_intermediate_lens_1()
        extras["intermediate_lens_2"] = self.get_intermediate_lens_2()
        extras["intermediate_lens_3"] = self.get_intermediate_lens_3()
        extras["intermediate_lens_4"] = self.get_intermediate_lens_4()
        extras["projector_lens_1"] = self.get_projector_lens_1()
        extras["projector_lens_2"] = self.get_projector_lens_2()
        extras["projector_lens_3"] = self.get_projector_lens_3()

        # 3. Logical Properties
        extras["diffraction_focus_index"] = self.get_diffraction_focus()
        extras["image_shift_2"] = self.get_image_shift_2()

        return ps

    def apply_projection_settings(self, settings: ProjectionSettings, **kwargs) -> None:
        """
        Applies settings.
        Priority:
        1. Standard 'defocus' (nm) -> Calls set_defocus() -> Calculates OM/OLf.
        2. Vendor Extras -> Explicitly sets OLc/OLf/OM via Atomic Setters.
        """
        # 1. Apply Standard Physics (Safe to call super)
        super().apply_projection_settings(settings, **kwargs)

        # 2. Apply Register Overrides (Manual DAC Control)
        vend = getattr(settings.extra, "vendor", {})
        jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None

        if isinstance(jeol_v, dict):
            if "objective_lens_coarse" in jeol_v:
                self.set_objective_lens_coarse(jeol_v["objective_lens_coarse"])
            if "objective_lens_fine" in jeol_v:
                self.set_objective_lens_fine(jeol_v["objective_lens_fine"])
            if "objective_lens_superfine" in jeol_v:
                self.set_objective_lens_superfine(jeol_v["objective_lens_superfine"])
            if "objective_mini_lens_1" in jeol_v:
                self.set_objective_mini_lens_1(jeol_v["objective_mini_lens_1"])
            if "objective_mini_lens_2" in jeol_v:
                self.set_objective_mini_lens_2(jeol_v["objective_mini_lens_2"])
            if "focus_lens_coarse" in jeol_v:
                self.set_focus_lens_coarse(jeol_v["focus_lens_coarse"])
            if "focus_lens_fine" in jeol_v:
                self.set_focus_lens_fine(jeol_v["focus_lens_fine"])
            if "intermediate_lens_1" in jeol_v:
                self.set_intermediate_lens_1(jeol_v["intermediate_lens_1"])
            if "intermediate_lens_2" in jeol_v:
                self.set_intermediate_lens_2(jeol_v["intermediate_lens_2"])
            if "intermediate_lens_3" in jeol_v:
                self.set_intermediate_lens_3(jeol_v["intermediate_lens_3"])
            if "intermediate_lens_4" in jeol_v:
                self.set_intermediate_lens_4(jeol_v["intermediate_lens_4"])
            if "projector_lens_1" in jeol_v:
                self.set_projector_lens_1(jeol_v["projector_lens_1"])
            if "projector_lens_2" in jeol_v:
                self.set_projector_lens_2(jeol_v["projector_lens_2"])
            if "projector_lens_3" in jeol_v:
                self.set_projector_lens_2(jeol_v["projector_lens_3"])
            if "image_shift_2" in jeol_v:
                val = jeol_v["image_shift_2"]
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    self.set_image_shift_2(val[0], val[1])
            if "diffraction_focus_index" in jeol_v:
                self.set_diffraction_focus(jeol_v["diffraction_focus_index"])

    def perform_projection_action(self, action: str, **kwargs) -> None:
        act = action.upper().strip()

        if act == "STD_FOCUS":
            logger.info("[LENS] Executing Standard Focus...")
            self.set_standard_focus()

        # CASE 1: Relative Steps (Knob clicks) - Hardware Command
        elif act == "STEP_FOCUS":
            steps = int(kwargs.get("steps", 1))
            logger.debug(f"[LENS] Stepping focus by {steps} clicks")
            self.set_relative_focus_steps(steps)  # The new Atomic setter we discussed

        # CASE 2: Relative Nanometers (Physical Shift) - Software Calculation
        elif act == "SHIFT_FOCUS_NM":
            # 1. Validate Input
            amount = kwargs.get("amount")
            if not isinstance(amount, Quantity):
                # Fallback: Try to construct Quantity if raw float provided (assuming nm)
                if isinstance(amount, (int, float)):
                    amount = Q_(amount, Units.NM)
                else:
                    raise ValueError("SHIFT_FOCUS_NM requires 'amount' as a Quantity (nm).")

            # 2. Read Current State (The "Read" phase)
            current = self.get_defocus()
            if current is None:
                raise RuntimeError(
                    "Cannot shift focus physically: Current absolute defocus is unknown (Calibration missing?).")

            # 3. Calculate New Target (The "Modify" phase)
            target = current + amount
            logger.info(f"[LENS] Shifting focus: {current} + {amount} -> {target}")

            # 4. Execute Absolute Move (The "Write" phase)
            self.set_defocus(target)

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
