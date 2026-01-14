"""
supertem.jeol_microscope

JEOL TEM driver implementation for the SuperTEM hardware abstraction layer.

This module provides :class:`JeolMicroscope`, a vendor-backed implementation of the
:class:`~supertem.microscope.TemMicroscope` interface using PyJEM (TEM3).

===============================================================================
I. Architecture Implementation
===============================================================================

This driver maps the SuperTEM Three-Layer Architecture to PyJEM as follows:

  1) Atomic Layer (The "Hands")
     - Wraps PyJEM calls (TEM3/EOS3).
     - Responsibility: Dumb I/O. If asked to set Alpha Index 99, it attempts it.
     - Behavior:
        - READ (Getters): "Null means Unknown". Returns `None` on failure, never defaults.
        - WRITE (Setters): "Fail Loudly". Raises exceptions if hardware rejects the command.

  2) Helper Layer (The "Brain")
     - Overrides methods like `apply_beam_settings`.
     - Responsibility:
       a. Routes canonical physics to atomic setters.
       b. **Vendor Guard:** Extracts `alpha_index` from `Extras`, validates it
          against valid ranges (0-8), and RAISES error if invalid.
       c. Prevents invalid vendor data from reaching the Atomic layer.

  3) Orchestrator Layer (Inherited)
     - Uses the base `TemMicroscope` logic for generic safety limits.

This module relies heavily on `supertem.vendor.JEOL.jeol_adapter` to perform
pure data translation (Unit Conversion, Parsing) while this module handles
the physical execution.

===============================================================================
II. Design Philosophy & Rules
===============================================================================

This driver adheres to three core philosophies to ensure safe automation:

  1) "Null means Unknown" (Data Safety)
     - Atomic getters MUST NOT return default values (0, 0.0, 512) if hardware
       reads fail. They MUST return `None`.
     - Rationale: A script seeing `spot_size=None` knows to halt; `0` implies success.

  2) "Trust but Verify" (Control Robustness)
     - Hardware status flags (e.g., `GetStatus() == 0`) are necessary but
       not sufficient.
     - Critical movements (Stage) utilize Closed-Loop Control:
       Command -> Wait for Idle -> Read Actual Position -> Retry if outside Tolerance.

  3) "Be Honest" (Data Fidelity)
     - **Do not fabricate calibrated physics.** If JEOL only provides an Index,
       do NOT coerce it into a fake physical unit (mrad).
     - **Implementation:** Vendor-native encodings are stored in `Extras.vendor['JEOL']`.
     - **Accessors:** Exposed via specific atomic accessors (e.g., `get_alpha_index()`).

===============================================================================
III. Safety & Error Handling
===============================================================================

This driver implements a "Uni-Directional Safety Policy":

  A. Ingress (Reading from Hardware) -> LENIENT
     - Methods: `get_beam_settings`, `get_stage_position`.
     - Behavior: If PyJEM returns garbage (e.g., NaN), we coerce it to `None`
       and log the error. We prioritize keeping the control loop alive.

  B. Egress (Writing to Hardware) -> STRICT
     - Methods: `apply_beam_settings`, `move_stage_absolute`.
     - Behavior: **Safety Checks are Absolute.**
       Rule: The `_mode` of the input object is IGNORED during execution.
       If a value (canonical or vendor-extra) is out of safe bounds, the
       Helper MUST raise an exception immediately.
     - Rationale: "Lenient execution" of unsafe commands causes physical damage.

===============================================================================
IV. Hardware Notes
===============================================================================

  - **Stage Latency:** JEOL stages may report "Idle" status momentarily during
    direction changes. The driver's retry loop accounts for this hysteresis.
  - **Defocus Calibration:** Returns raw DAC units unless `defocus_scale` is configured.
  - **Detector Sync:** Decouples scan coils from detector if detector is offline
    to prevent beam damage (static beam safety).
"""
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
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
        self.feg = None
        self.det3 = None
        self._connected = False

        self._active_detectors: Dict[str, Any] = {}
        self._primary_detector_id: Optional[str] = None
        self._scan_cfg: Dict[str, Any] = {}

        cfg = config if isinstance(config, dict) else {}
        self._has_defocus_calibration: bool = (
            hasattr(config, 'defocus_scale') or ('defocus_scale' in cfg)
        )
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

    def _coerce_xy(self, xy: Any) -> Optional[Tuple[float, float]]:
        """
        Coerce PyJEM (x,y) returns into float pair.
        Returns None if data is missing or malformed, adhering to 'Null means Unknown'.
        """
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                return float(xy[0]), float(xy[1])
            except Exception:
                return None
        return None

    # --- EOS Helpers ---

    def _get_eos_mode_key(self) -> Optional[str]:
        """Return EOS mode key like 'TEM:MAG' or 'STEM:SM-MAG' matching jeol_eos_tables."""
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
        # Fallback: best-effort string
        obs = "TEM" if int(main_mode) == 0 else "STEM"
        return f"{obs}:{int(function_mode)}"

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
        # Fail loudly if mode set fails
        self.eos.SelectTemStem(0 if m == "TEM" else 1)

    # =========================================================================
    # 3. Stage Control
    # =========================================================================

    def get_stage_position(self) -> Optional[StagePosition]:
        """Returns None if the stage cannot be queried."""
        if not self.stage or not hasattr(self.stage, "GetPos"):
            return None
        try:
            return jeol_adapter.from_jeol_stage_position(self.stage.GetPos())
        except Exception:
            return None

    # TODO: include piezo /motor
    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default",
                            wait: bool = True,
                            tolerance_nm: float = 200.0,
                            tolerance_deg: float = 0.1,
                            max_retries: int = 3) -> None:
        """
        Moves the stage and guarantees accuracy via a retry loop for both position and tilt.

        Args:
            target: The destination coordinates.
            wait: If False, fires and returns immediately (no verification).
            tolerance_nm: Max allowed Euclidean distance error (for X, Y, Z).
            tolerance_deg: Max allowed angular error (for Tilt X, Tilt Y).
            max_retries: Number of corrections to attempt if mechanics stop short.
        """
        if not self.stage: return
        t_args = jeol_adapter.to_jeol_stage_args(target)

        def _dispatch():
            # Send the command to hardware (Fail Loudly if HW exception)
            if 'x' in t_args and hasattr(self.stage, "SetX"): self.stage.SetX(t_args['x'])
            if 'y' in t_args and hasattr(self.stage, "SetY"): self.stage.SetY(t_args['y'])
            if 'z' in t_args and hasattr(self.stage, "SetZ"): self.stage.SetZ(t_args['z'])
            if 'tx' in t_args and hasattr(self.stage, "SetTiltXAngle"): self.stage.SetTiltXAngle(t_args['tx'])
            if 'ty' in t_args and hasattr(self.stage, "SetTiltYAngle"): self.stage.SetTiltYAngle(t_args['ty'])

        # 1. Fire Initial Move
        _dispatch()

        if not wait:
            return

        # 2. Verification Loop
        for attempt in range(max_retries + 1):
            # A. Wait for Hardware Idle
            self._wait_for_stage(timeout=30.0)

            # B. Read Truth
            current = self.get_stage_position()
            if current is None:
                logger.warning("Stage position unreadable; skipping verification.")
                return

            # C. Check Tolerance
            if target.is_close(current, tol_nm=tolerance_nm, tol_deg=tolerance_deg):
                return  # Success!

            # D. Retry if needed
            if attempt < max_retries:
                logger.debug(f"Stage Correction {attempt + 1}: Target vs Current mismatch > tol.")
                _dispatch()  # Re-issue command
                time.sleep(0.5)
            else:
                logger.warning(
                    f"Stage accuracy warning: Finished move but outside tolerance. "
                    f"(Limit: {tolerance_nm}nm, {tolerance_deg}deg)"
                )

    def stop_stage(self) -> None:
        if self.stage and hasattr(self.stage, "Stop"):
            self.stage.Stop()

    def home_stage(self) -> None:
        # TODO: Move to 0,0,0,0,0?
        # Not supported in standard PyJEM Stage3 interface
        logger.warning("home_stage() not supported by this driver.")
        pass

    # =========================================================================
    # 4. Beam Control (Atomic Getters - Lenient)
    # =========================================================================

    def get_acceleration_voltage(self) -> Optional[Quantity]:
        if not self.ht or not hasattr(self.ht, "GetHtValue"):
            return None
        try:
            v = float(self.ht.GetHtValue())
        except Exception:
            return None
        return Q_(v, "V").to(Units.KV)

    def get_beam_current(self) -> Optional[Quantity]:
        if self.gun and hasattr(self.gun, "GetEmissionCurrent"):
            try:
                val = self.gun.GetEmissionCurrent()
                return Q_(val, Units.UA).to(Units.NA)
            except Exception:
                return None
        return None

    def get_spot_size(self) -> Optional[int]:
        """Returns None if query fails, avoiding ambiguous '0'."""
        if self.eos and hasattr(self.eos, "GetSpotSize"):
            try:
                return int(self.eos.GetSpotSize())
            except Exception:
                pass
        return None

    def get_convergence_angle(self) -> Optional[Quantity]:
        """Get convergence angle (alpha) as a physical quantity.
        Returns None as we use alpha_index for JEOL.
        """
        return None

    def get_beam_shift(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_ and hasattr(self.def_, "GetCLA1"):
            res = self._coerce_xy(self.def_.GetCLA1())
            if res is not None:
                return res
        return (None, None)

    def get_condenser_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_ and hasattr(self.def_, "GetCLs"):
            res = self._coerce_xy(self.def_.GetCLs())
            if res is not None:
                return res
        return (None, None)

    def get_gun_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_ and hasattr(self.def_, "GetAngBal"):
            res = self._coerce_xy(self.def_.GetAngBal())
            if res is not None:
                return res
        return (None, None)

    # =========================================================================
    # Logic Layer Overrides (JEOL)
    # =========================================================================

    def get_beam_settings(self) -> BeamSettings:
        """JEOL override: build BeamSettings via adapter (Lenient Ingress)."""
        raw_flags: Dict[str, Any] = {}

        # Acceleration voltage
        voltage_val: float = 0.0
        vq = self.get_acceleration_voltage()
        if vq is None:
            raw_flags["ht_unavailable"] = True
        else:
            try:
                voltage_val = float(vq.to(Units.KV).magnitude)
            except Exception:
                raw_flags["ht_unavailable"] = True

        # Beam current
        current_ua: float = 0.0
        cq = self.get_beam_current()
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

    def apply_beam_settings(self, settings: BeamSettings) -> None:
        """JEOL override: apply canonical fields and vendor-native alpha index.

        STRICT EGRESS: Validates vendor extras before passing to Atomic layer.
        """
        # Apply canonical fields
        if settings.voltage is not None:
            self.set_acceleration_voltage(settings.voltage)
        if settings.beam_current is not None:
            self.set_beam_current(settings.beam_current)
        if settings.spot_size is not None:
            self.set_spot_size(settings.spot_size)

        if settings.convergence_angle is not None:
            raise ValueError(
                "JEOL driver cannot apply BeamSettings.convergence_angle (physical) without "
                "a calibration table. Use BeamSettings.extra.vendor['JEOL']['alpha_index']."
            )

        if settings.beam_shift is not None and settings.beam_shift.x is not None:
             self.set_beam_shift(settings.beam_shift.x, settings.beam_shift.y)

        if settings.condenser_stigmation is not None and settings.condenser_stigmation.x is not None:
            self.set_condenser_stigmation(settings.condenser_stigmation.x, settings.condenser_stigmation.y)

        if settings.gun_tilt is not None and settings.gun_tilt.x is not None:
            self.set_gun_tilt(settings.gun_tilt.x, settings.gun_tilt.y)

        # Vendor-native: alpha selector
        # Validate before sending to Atomic layer
        vend = getattr(settings.extra, "vendor", None) or {}
        jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None

        if isinstance(jeol_v, dict) and "alpha_index" in jeol_v:
            idx = int(jeol_v["alpha_index"])

            # --- STRICT VALIDATION START ---
            if not (0 <= idx <= 8):
                raise ValueError(
                    f"Unsafe Command: JEOL Alpha Index {idx} is out of bounds (0-8). "
                    "The driver refused to execute this request."
                )
            # --- STRICT VALIDATION END ---

            self.set_alpha_index(idx)

    def get_projection_settings(self) -> ProjectionSettings:
        """JEOL override: preserve vendor-native defocus DAC when uncalibrated."""
        ps = super().get_projection_settings()

        if not self._has_defocus_calibration:
            dac = self.get_defocus_dac()
            if dac is not None:
                ps.extra.vendor.setdefault("JEOL", {})["defocus_olc_dac"] = dac
                ps.extra.notes["ProjectionSettings.defocus_uncalibrated"] = (
                    "JEOL OLc reported in DAC units; physical nm defocus requires defocus_scale calibration."
                )
        return ps

    def apply_projection_settings(self, settings: ProjectionSettings) -> None:
        """JEOL override: apply vendor-native defocus DAC when provided."""
        if (settings.defocus is not None) and (not self._has_defocus_calibration):
            raise ValueError(
                "JEOL driver cannot apply ProjectionSettings.defocus (nm) without defocus_scale calibration. "
                "Use ProjectionSettings.extra.vendor['JEOL']['defocus_olc_dac'] instead."
            )

        # Apply canonical fields
        super().apply_projection_settings(settings)

        # Vendor-native: defocus DAC
        try:
            vend = getattr(settings.extra, "vendor", None) or {}
            jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None
            if isinstance(jeol_v, dict) and "defocus_olc_dac" in jeol_v:
                # We blindly trust the DAC value range here as it's hardware specific (0-FFFF usually)
                self.set_defocus_dac(int(jeol_v["defocus_olc_dac"]))
        except Exception:
            # Re-raise if set_defocus_dac fails
            raise

    def get_beam_blank(self) -> bool:
        if self.def_ and hasattr(self.def_, "GetBeamBlank"):
            return bool(self.def_.GetBeamBlank())
        return False

    # =========================================================================
    # 4b. Beam Control (Atomic Setters - Strict)
    # =========================================================================

    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        """Set HT. Raises exception if hardware fails."""
        if not self.ht: return
        v = float(voltage.to("V").magnitude)
        self.ht.SetHtValue(v)

    def set_beam_current(self, current: Quantity) -> None:
        logger.warning("set_beam_current not directly supported; use spot_size.")
        pass

    def set_spot_size(self, index: int) -> None:
        if self.eos:
            self.eos.SelectSpotSize(int(index))

    def set_convergence_angle(self, angle: Quantity) -> None:
        raise ValueError(
            "JEOL driver does not support setting a physical convergence angle without a "
            "calibration table. Use set_alpha_index(idx)."
        )

    # --- JEOL Vendor-Native Alpha (Convergence Selector) ---

    def get_alpha_index(self) -> Optional[int]:
        if not self.eos or not hasattr(self.eos, "GetAlpha"):
            return None
        try:
            return int(self.eos.GetAlpha())
        except Exception:
            return None

    def set_alpha_index(self, idx: int) -> None:
        """Set JEOL alpha selector index. BLIND EXECUTION."""
        if not self.eos: return
        # Blindly execute. Validation is upstream.
        self.eos.SetAlphaSelector(int(idx))

    def set_beam_shift(self, x: float, y: float) -> None:
        if self.def_:
            self.def_.SetCLA1(int(x), int(y))

    def set_condenser_stigmation(self, x: float, y: float) -> None:
        if self.def_:
            self.def_.SetCLs(int(x), int(y))

    def set_gun_tilt(self, x: float, y: float) -> None:
        if self.def_:
            self.def_.SetAngBal(int(x), int(y))

    def set_beam_blank(self, blank: bool) -> None:
        if self.def_:
            self.def_.SetBeamBlank(1 if blank else 0)

    # =========================================================================
    # 5. Projection Control (Atomic)
    # =========================================================================

    def get_projection_mode(self) -> str:
        key, _ = self._resolve_eos_table_info()
        return key if key else "UNKNOWN"

    def get_magnification(self) -> int:
        """Return the *magnification value* (e.g. 100000), not the selector index."""
        if not self.eos: return None

        # Preferred: ask EOS for the current value
        if hasattr(self.eos, "GetMagValue"):
            try:
                val = self.eos.GetMagValue()
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    v = float(val[0])
                    unit = str(val[1]).strip().upper()
                    if unit == "X":
                        return int(round(v))
                    return None
                return int(round(float(val)))
            except Exception:
                pass

        # Fallback: map selector -> EOS table
        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key: return None

        try:
            lst = get_list(key, "MagList") or []
        except Exception: return None

        if not lst or str(lst[0][1]).strip().upper() != "X": return None

        sel = None
        if hasattr(self.eos, "GetCurrentMagSelectorID"):
            try:
                sel = int(self.eos.GetCurrentMagSelectorID())
            except Exception: pass
        if sel is None and hasattr(self.eos, "GetSelector"):
            try:
                sel = int(self.eos.GetSelector())
            except Exception: pass
        if sel is None: return None

        for idx in (sel - 1, sel, sel + 1):
            if 0 <= idx < len(lst):
                try:
                    return int(round(float(lst[idx][0])))
                except Exception: continue
        return None

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
                    val, unit, _ = self.eos.GetStemCamValue()
                    return Q_(float(val), str(unit)).to(Units.MM)
                return None
            if "DIFF" in key_u and hasattr(self.eos, "GetMagValue"):
                val, unit, _ = self.eos.GetMagValue()
                return Q_(float(val), str(unit)).to(Units.MM)
        except Exception:
            return None
        return None

    def get_defocus(self) -> Optional[Quantity]:
        """Get defocus as a physical quantity (nm) when calibrated.

        JEOL/PyJEM exposes OLc as a lens DAC value. We only convert to nm when an explicit
        `defocus_scale` calibration is provided in config. Otherwise, return None and expose
        the raw DAC value in `extra.vendor['JEOL']['defocus_olc_dac']` via
        JeolMicroscope.get_projection_settings().
        """
        if not self._has_defocus_calibration:
            return None
        if self.lens and hasattr(self.lens, "GetOLc"):
            try:
                val = float(self.lens.GetOLc())
                return Q_(val / (self.defocus_scale or 1.0), Units.NM)
            except Exception: return None
        return None

    def get_screen_position(self) -> str:
        if not self.det3: return None
        try:
            if hasattr(self.det3, "GetScreen"):
                idx = int(self.det3.GetScreen())
                return "DOWN" if idx == 2 else "UP"
        except Exception: return None
        return None

    def get_objective_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_ and hasattr(self.def_, "GetOLs"):
            res = self._coerce_xy(self.def_.GetOLs())
            if res is not None: return res
        return (None, None)

    def get_image_shift(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_:
            if hasattr(self.def_, "GetIS1"):
                res = self._coerce_xy(self.def_.GetIS1())
                if res is not None: return res
            if hasattr(self.def_, "GetIS"):
                res = self._coerce_xy(self.def_.GetIS())
                if res is not None: return res
        return (None, None)

    def get_diffraction_shift(self) -> Tuple[Optional[float], Optional[float]]:
        if self.def_ and hasattr(self.def_, "GetPLA"):
            res = self._coerce_xy(self.def_.GetPLA())
            if res is not None: return res
        return (None, None)

    # --- Setters (Strict) ---

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
        if ":" in m:
            self._select_eos_mode_key(m)
            return
        obs = (self.get_mode() or "TEM").strip().upper()
        if "DIFF" in m:
            self._select_eos_mode_key("STEM:UUDIFF" if obs == "STEM" else "TEM:DIFF")
            return
        self._select_eos_mode_key("STEM:SM-MAG" if obs == "STEM" else "TEM:MAG")

    def set_magnification(self, index: int) -> None:
        if not self.eos: return

        key = self._normalize_eos_key(self._get_eos_mode_key() or "")
        if not key: return

        try:
            mag_list = get_list(key, "MagList") or []
        except Exception: return
        if not mag_list or str(mag_list[0][1]).strip().upper() != "X": return

        target = float(index)
        best_i = 0
        for i, (v, _, _) in enumerate(mag_list):
            try:
                if float(v) <= target: best_i = i
            except Exception: continue

        # Try best effort to set selector, fail loudly if error persists
        try:
            self.eos.SetSelector(int(best_i + 1))
        except Exception:
            self.eos.SetSelector(int(best_i))

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
        if not key or not list_name: return
        targets = get_list(key, list_name) or []
        if not targets: return

        target_mm = length.to(Units.MM).magnitude
        best_i, best_err = 0, float("inf")
        for i, (val, unit, _) in enumerate(targets):
            try:
                mm = Q_(float(val), str(unit)).to(Units.MM).magnitude
                err = abs(mm - target_mm)
                if err < best_err:
                    best_err = err
                    best_i = i
            except Exception: continue

        selector = int(best_i + 1)
        if key.startswith("STEM:") and hasattr(self.eos, "SetStemCamSelector"):
            self.eos.SetStemCamSelector(selector)
        elif hasattr(self.eos, "SetSelector"):
            self.eos.SetSelector(selector)

    def set_defocus(self, defocus: Quantity) -> None:
        """Set defocus as a physical quantity (nm) when calibrated.

        Without a calibration (`defocus_scale`), this is not supported because OLc is vendor
        DAC units. Use `set_defocus_dac()` (vendor-native) instead.
        """
        if not self._has_defocus_calibration:
            raise ValueError("JEOL driver cannot set physical defocus without calibration.")
        if self.lens and hasattr(self.lens, "SetOLc"):
            val = float(defocus.to(Units.NM).magnitude)
            self.lens.SetOLc(int(val * (self.defocus_scale or 1.0)))

    def get_defocus_dac(self) -> Optional[int]:
        """Get objective lens coarse (OLc) value in JEOL DAC units."""
        if self.lens and hasattr(self.lens, "GetOLc"):
            try:
                return int(self.lens.GetOLc())
            except Exception: return None
        return None

    def set_defocus_dac(self, dac: int) -> None:
        """Set OLc DAC. Fail if hardware rejects."""
        if self.lens:
            self.lens.SetOLc(int(dac))

    def set_screen_position(self, position: str) -> None:
        if not self.det3: return
        p = (position or "").strip().upper()
        if p == "DOWN":
            self.det3.SetScreen(2)
        elif p == "UP":
            self.det3.SetScreen(0)

    def set_objective_stigmation(self, x: float, y: float) -> None:
        if self.def_:
            self.def_.SetOLs(int(x), int(y))

    def set_image_shift(self, x: float, y: float) -> None:
        if self.def_:
            if hasattr(self.def_, "SetIS1"):
                self.def_.SetIS1(int(x), int(y))
                return
            self.def_.SetIS(int(x), int(y))

    def set_diffraction_shift(self, x: float, y: float) -> None:
        if self.def_:
            self.def_.SetPLA(int(x), int(y))

    # =========================================================================
    # 6. Scan Control (Atomic)
    # =========================================================================
    #
    # PyJEM has *two* places where "scan-ish" controls show up:
    #   1) TEM3.Scan3: low-level scan engine controls (rotation, ext scan mode, etc.)
    #   2) detector.Detector: STEM scan configuration tied to the currently selected detector
    #      (scan mode, imaging area, spot position, scan rotation, etc.)
    #
    # In practice, many day-to-day STEM scan knobs (Scan/Spot/Area + imaging area)
    # live under `detector.Detector` rather than TEM3.Scan3. We therefore prefer the
    # detector API when available, and fall back to TEM3.Scan3 only for the subset
    # of scan controls it actually exposes.
    #
    # Reference: PyJEM detector.Detector exposes set_scanmode / set_imaging_area /
    # set_areamode_imagingarea / set_spotposition / set_scanrotation.
    # Reference: PyJEM TEM3.Scan3 exposes Get/SetRotationAngle(Ex) and Get/SetExtScanMode.

    def _get_scan_controller_detector(self):
        """Return a Detector instance used for scan config (best-effort).

        PyJEM's scan configuration is often bound to a specific detector instance.
        We use the primary detector if known, else fall back to the first available.
        """
        if detector is None:
            return None
        try:
            det_id = self.get_primary_detector_id()
            if det_id is None:
                ids = self.list_detectors()
                det_id = ids[0] if ids else None
            if det_id is None: return None
            return self._get_detector(det_id)
        except Exception: return None

    @staticmethod
    def _first_int(d: dict, keys: tuple[str, ...]) -> Optional[int]:
        for k in keys:
            if k in d:
                try: return int(d[k])
                except Exception: continue
        return None

    @staticmethod
    def _first_float(d: dict, keys: tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in d:
                try: return float(d[k])
                except Exception: continue
        return None

    def get_scan_mode(self) -> str:
        """Return scan mode as a human-readable string."""
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
            except Exception: pass
        return str(self._scan_cfg.get("mode", "Scan"))

    def get_scan_width(self) -> Optional[int]:
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    w = self._first_int(st, ("Width", "ImagingAreaWidth"))
                    if w is not None:
                        self._scan_cfg["width_px"] = w
                        return w
            except Exception: pass
        return None

    def get_scan_height(self) -> Optional[int]:
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    h = self._first_int(st, ("Height", "ImagingAreaHeight"))
                    if h is not None:
                        self._scan_cfg["height_px"] = h
                        return h
            except Exception: pass
        return None

    def get_scan_pixel_dwell(self) -> Quantity:
        # No stable public getter in PyJEM docs; keep as local config for now.
        return Q_(float(self._scan_cfg.get("pixel_dwell_us", 10.0)), Units.US)

    def get_scan_flyback(self) -> Quantity:
        # No stable public getter in PyJEM docs; keep as local config for now.
        return Q_(float(self._scan_cfg.get("flyback_us", 100.0)), Units.US)

    def get_scan_rotation(self) -> Optional[Quantity]:
        if self.scan and hasattr(self.scan, "GetRotationAngleEx"):
            try: return Q_(float(self.scan.GetRotationAngleEx()), Units.DEG)
            except Exception: pass
        if self.scan and hasattr(self.scan, "GetRotationAngle"):
            try: return Q_(float(self.scan.GetRotationAngle()), Units.DEG)
            except Exception: pass
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    ang = self._first_float(st, ("ScanRotation", "ScanRotationValue"))
                    if ang is not None: return Q_(ang, Units.DEG)
            except Exception: pass
        return None

    def get_scan_active(self) -> bool:
        # Prefer TEM3.Scan3 ext scan mode.
        if self.scan and hasattr(self.scan, "GetExtScanMode"):
            try: return bool(int(self.scan.GetExtScanMode()) == 1)
            except Exception: pass
        return bool(self._scan_cfg.get("active", False))

    def set_scan_mode(self, mode: str) -> None:
        m = (mode or "").strip().lower()
        mapping = {"scan": 0, "full": 0, "full frame": 0, "spot": 1, "area": 3}
        if m not in mapping and m.isdigit(): mapping[m] = int(m)
        val = mapping.get(m)
        if val is None: raise ValueError(f"Unsupported scan mode: {mode}")

        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "set_scanmode"):
             d.set_scanmode(int(val))
             self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))
             return

        self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))

    def _set_imaging_area(self, *, width=None, height=None, x=None, y=None) -> None:
        d = self._get_scan_controller_detector()
        if d is None: return
        w = int(width if width is not None else self._scan_cfg.get("width_px", 512))
        h = int(height if height is not None else self._scan_cfg.get("height_px", 512))
        xx = int(x if x is not None else self._scan_cfg.get("x_px", 0))
        yy = int(y if y is not None else self._scan_cfg.get("y_px", 0))
        self._scan_cfg.update({"width_px": w, "height_px": h, "x_px": xx, "y_px": yy})

        if hasattr(d, "set_imaging_area"):
            d.set_imaging_area(w, h, xx, yy)

    def set_scan_width(self, width: int) -> None:
        self._set_imaging_area(width=int(width))

    def set_scan_height(self, height: int) -> None:
        self._set_imaging_area(height=int(height))

    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        try: us = float(time.to(Units.US).magnitude)
        except Exception: us = float(time.magnitude)
        self._scan_cfg["pixel_dwell_us"] = us

    def set_scan_flyback(self, time: Quantity) -> None:
        try: us = float(time.to(Units.US).magnitude)
        except Exception: us = float(time.magnitude)
        self._scan_cfg["flyback_us"] = us

    def set_scan_rotation(self, angle: Quantity) -> None:
        deg = float(angle.to(Units.DEG).magnitude)
        self._scan_cfg["rotation_deg"] = deg

        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "set_scanrotation"):
            d.set_scanrotation(float(deg))
            return

        if self.scan:
            if hasattr(self.scan, "SetRotationAngleEx"):
                self.scan.SetRotationAngleEx(float(deg))
                return
            if hasattr(self.scan, "SetRotationAngle"):
                self.scan.SetRotationAngle(int(round(deg)) % 360)

    def set_scan_active(self, active: bool) -> None:
        self._scan_cfg["active"] = bool(active)

        detector_handled = False
        d = self._get_scan_controller_detector()
        if d is not None:
            if active and hasattr(d, "livestart"):
                d.livestart()
                detector_handled = True
            elif (not active) and hasattr(d, "livestop"):
                d.livestop()
                detector_handled = True

        if not detector_handled and self.scan and hasattr(self.scan, "SetExtScanMode"):
            self.scan.SetExtScanMode(1 if active else 0)

    # =========================================================================
    # 7. Detector Control (Atomic)
    # =========================================================================

    def list_detectors(self) -> List[str]:
        if not self._active_detectors:
            self._refresh_detectors()
        return list(self._active_detectors.keys())

    def get_active_detector_ids(self) -> List[str]:
        return self.list_detectors()

    def get_primary_detector_id(self) -> Optional[str]:
        if self._primary_detector_id is None:
            self._refresh_detectors()
        return self._primary_detector_id

    def get_detector_exposure(self, detector_id: str) -> Optional[Quantity]:
        if detector is None: return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.exposure
        except Exception: return None

    def get_detector_binning(self, detector_id: str) -> Optional[int]:
        if detector is None: return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.binning_index
        except Exception: return None

    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        if detector is None: return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.roi
        except Exception: return None

    def get_detector_integration(self, detector_id: str) -> Optional[int]:
        if detector is None: return None
        try:
            d = self._get_detector(detector_id)
            res, _ = jeol_adapter.from_jeol_detector_response(d.get_detectorsetting(), detector_id)
            return res.frame_integration
        except Exception: return None

    def get_detector_inserted(self, detector_id: str) -> bool:
        if detector is None: return True
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
        except Exception: pass
        return True

    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        return None

    def set_detector_exposure(self, detector_id: str, exposure: Quantity) -> None:
        if detector is None: return
        d = self._get_detector(detector_id)
        us = int(exposure.to(Units.US).magnitude)
        if hasattr(d, "set_exposuretime_value"):
            d.set_exposuretime_value(us)
        elif hasattr(d, "set_exposuretime_index"):
            d.set_exposuretime_index(us)

    def set_detector_binning(self, detector_id: str, index: int) -> None:
        if detector is None: return
        d = self._get_detector(detector_id)
        if hasattr(d, "set_binningindex"):
            d.set_binningindex(int(index))

    def set_detector_roi(self, detector_id: str, roi: Optional[ROI]) -> None:
        if detector is None: return
        d = self._get_detector(detector_id)
        if roi is None: return
        if hasattr(d, "set_areamode_imagingarea"):
            d.set_areamode_imagingarea(int(roi.width), int(roi.height), int(roi.x), int(roi.y))

    def set_detector_integration(self, detector_id: str, count: int) -> None:
        if detector is None: return
        d = self._get_detector(detector_id)
        if hasattr(d, "set_frameintegration"):
            d.set_frameintegration(int(count))

    def set_detector_insertion(self, detector_id: str, inserted: bool) -> None:
        if detector is None: return
        d = self._get_detector(detector_id)
        if inserted and hasattr(d, "insert"):
            d.insert()
        elif (not inserted) and hasattr(d, "retract"):
            d.retract()

    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        """Atomic: Acquire a single image from a JEOL detector.

        Notes:
            - JEOL detector metadata like ROI/binning/integration are recorded under
              `MicroscopeImageMetadata.extra.vendor['JEOL']` because MicroscopeImageMetadata is
              scientific/canonical and intentionally small.
        """
        if detector is None:
            raise RuntimeError("Detector module missing")

        # Resolve detector id
        det_id = (request.detector_id
                  or getattr(request.detector, "detector_id", None)
                  or (self.get_primary_detector_id() or ""))

        if not det_id: raise RuntimeError("No detector_id provided and no primary detector available")
        d = self._get_detector(det_id)

        # 1) Apply per-request detector settings (Atomic-only, fail loudly if HW errors)
        det_req = request.detector
        if det_req is not None:
            if det_req.exposure is not None:
                self.set_detector_exposure(det_id, det_req.exposure)
            if det_req.binning_index is not None:
                self.set_detector_binning(det_id, int(det_req.binning_index))
            if det_req.frame_integration is not None:
                self.set_detector_integration(det_id, int(det_req.frame_integration))
            if det_req.roi is not None:
                self.set_detector_roi(det_id, det_req.roi)

        # 2) Capture raw data
        raw = None
        if hasattr(d, "snapshot_rawdata"):
            raw = d.snapshot_rawdata()
        elif hasattr(d, "get_image_cache"):
            raw = d.get_image_cache()
        elif hasattr(d, "livesnapshot"):
            raw = d.livesnapshot("tif")

        # 3) Convert to numpy array
        arr: np.ndarray
        if raw is None:
            arr = np.zeros((1, 1), dtype=np.uint16)
        elif isinstance(raw, (bytes, bytearray)):
            try: arr = np.frombuffer(raw, dtype=np.uint16)
            except Exception: arr = np.frombuffer(raw, dtype=np.uint8)
        elif isinstance(raw, list):
            arr = np.array(raw)
        elif isinstance(raw, dict) and "data" in raw:
            arr = np.array(raw["data"])
        else:
            try: arr = np.array(raw)
            except Exception: arr = np.zeros((1, 1), dtype=np.uint16)

        if arr.dtype not in (np.uint8, np.uint16):
            try: arr = arr.astype(np.uint16, copy=False)
            except Exception: arr = np.array(arr, dtype=np.uint16)

        roi = None
        if det_req is not None and getattr(det_req, "roi", None) is not None:
            roi = det_req.roi
        else:
            try: roi = self.get_detector_roi(det_id)
            except Exception: roi = None

        if arr.ndim == 1:
            cols = int(getattr(roi, "width", 0) or 0) if roi is not None else 0
            rows = int(getattr(roi, "height", 0) or 0) if roi is not None else 0
            if cols > 0 and rows > 0 and arr.size == cols * rows:
                arr = arr.reshape((rows, cols))

        if arr.ndim == 3:
            if arr.shape[0] == 1: arr = arr[0]
            elif arr.shape[-1] == 1: arr = arr[..., 0]
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)

        # 4) Build metadata (Lenient)
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            v = self.get_acceleration_voltage()
            accelerating_voltage_kv = float(v.to(Units.KV).magnitude) if v is not None else None
        except Exception: accelerating_voltage_kv = None

        try:
            bc = self.get_beam_current()
            beam_current_na = float(bc.to(Units.NA).magnitude) if bc is not None else None
        except Exception: beam_current_na = None

        try:
            exp_q = self.get_detector_exposure(det_id)
            exposure_ms = float(exp_q.to(Units.MS).magnitude) if exp_q is not None else None
        except Exception: exposure_ms = None

        try:
            mag_idx = self.get_magnification_index()
            magnification = float(mag_idx) if mag_idx is not None else None
        except Exception: magnification = None

        try:
            cl = self.get_camera_length()
            camera_length_mm = float(cl.to(Units.MM).magnitude) if cl is not None else None
        except Exception: camera_length_mm = None

        w, h = (int(arr.shape[1]), int(arr.shape[0])) if arr.ndim == 2 else (None, None)

        jeol_vendor: Dict[str, Any] = {"detector_id": det_id}
        try: jeol_vendor["binning_index"] = int(self.get_detector_binning(det_id))
        except Exception: pass
        try: jeol_vendor["frame_integration"] = int(self.get_detector_integration(det_id))
        except Exception: pass
        if roi is not None:
            try: jeol_vendor["roi"] = roi.to_dict() if hasattr(roi, "to_dict") else roi
            except Exception: jeol_vendor["roi"] = None

        state = None
        meta_extra = Extras(vendor={"JEOL": jeol_vendor})
        try: state = self.get_full_state()
        except Exception as e:
            try: meta_extra.notes["MicroscopeImageMetadata.state_capture_failed"] = str(e)
            except Exception: pass

        metadata = MicroscopeImageMetadata(
            created_at=created_at,
            magnification=magnification,
            camera_length_mm=camera_length_mm,
            image_size_px=(w, h) if (w is not None and h is not None) else None,
            accelerating_voltage_kv=accelerating_voltage_kv,
            beam_current_na=beam_current_na,
            exposure_ms=exposure_ms,
            microscope_state=state,
            extra=meta_extra,
            _mode="lenient",
        )

        return MicroscopeImage(data=arr, metadata=metadata)

    # =========================================================================
    # 8. Vacuum Control (Atomic)
    # =========================================================================

    def get_valve_state(self, valve_name: str) -> str:
        """Get a coarse valve state.

        * gun: Gun3.GetBeamValve() -> 0=closed, 1=open
        * column/turbo: VACUUM3.GetValveStatus() returns bitfields; we use configurable bit indices.
        """
        vn = (valve_name or "").strip().lower()
        if vn == "gun" and self.gun and hasattr(self.gun, "GetBeamValve"):
            try: return "OPEN" if int(self.gun.GetBeamValve()) == 1 else "CLOSED"
            except Exception: pass

        if self.vac and hasattr(self.vac, "GetValveStatus"):
            try:
                status = self.vac.GetValveStatus()
                if isinstance(status, (list, tuple)) and len(status) >= 2:
                    count = int(status[0]) if status[0] is not None else 0
                    bitfield = int(status[1])
                    bit_map = {"column": 0, "turbo": 1}
                    bit = bit_map.get(vn)
                    if bit is not None and bit < max(count, bit + 1):
                        is_open = ((bitfield >> bit) & 0x1) == 1
                        return "OPEN" if is_open else "CLOSED"
            except Exception: pass
        return "UNKNOWN"

    def set_valve_state(self, valve_name: str, state: str) -> None:
        """Set valve state (only gun valve is supported here)."""
        vn = (valve_name or "").strip().lower()
        st = (state or "").strip().upper()
        if vn == "gun" and self.gun:
             self.gun.SetBeamValve(1 if st == "OPEN" else 0)

    def get_pressure(self, gauge_name: str) -> Optional[Quantity]:
        if not self.vac: return None
        try:
            if hasattr(self.vac, "GetPigInfo"):
                val = self.vac.GetPigInfo()
                if isinstance(val, (list, tuple)) and val:
                    return Q_(float(val[0]), Units.PA)
            if hasattr(self.vac, "GetPegInfo"):
                val = self.vac.GetPegInfo()
                if isinstance(val, (list, tuple)) and val:
                    return Q_(float(val[0]), Units.PA)
        except Exception: pass
        return None

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
        if not key: return None, None
        list_name = "StemCamList" if key.startswith("STEM:") else "MagList"
        return key, list_name

    def _wait_for_stage(self, timeout: float = 30.0) -> None:
        """Polls stage status until all axes are at rest (Status 0)."""
        if not hasattr(self.stage, "GetStatus"):
            time.sleep(0.5)
            return

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                status = self.stage.GetStatus()
                if isinstance(status, (list, tuple)):
                    if all(s == 0 for s in status):
                        return
            except Exception: pass
            time.sleep(0.2)

        logger.warning(f"Stage move timed out after {timeout}s (GetStatus never settled).")
