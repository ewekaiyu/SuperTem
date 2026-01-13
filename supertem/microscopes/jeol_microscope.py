"""
supertem.jeol_microscope

JEOL TEM driver implementation for the SuperTEM hardware abstraction layer.

This module provides :class:`JeolMicroscope`, a vendor-backed implementation of the
:class:`~supertem.microscope.TemMicroscope` interface using PyJEM (TEM3).

Core idea
---------
SuperTEM separates "what the microscope *means*" from "what a vendor API *returns*":

1) Canonical (portable) fields live in the typed structures from ``supertem.structures.base``
   (e.g. :class:`~supertem.structures.base.BeamSettings`,
   :class:`~supertem.structures.base.ProjectionSettings`,
   :class:`~supertem.structures.base.DetectorSettings`).

2) Vendor-native encodings (indices, DAC units, mode codes, table keys) must be preserved
   losslessly under ``Extras.vendor["JEOL"]`` rather than being forced into canonical physics.

This matters because PyJEM exposes some controls only as selectors/indices or raw DAC values.
If those values are pushed directly into canonical fields (e.g. treating an alpha selector index
as a convergence angle), you either:
- crash in STRICT parsing, or
- silently store plausible-looking but incorrect physics.

Design philosophy
-----------------
- **Be honest at the atomic layer.**
  If JEOL can only provide an index/DAC, do not fabricate a calibrated physical quantity.
  Expose a vendor-native accessor (e.g. ``get_alpha_index()``, ``get_defocus_dac()``).

- **Translate at the snapshot layer.**
  ``JeolMicroscope`` overrides helper aggregators (e.g. ``get_beam_settings()``,
  ``get_projection_settings()``) to route through ``supertem.jeol_adapter``.
  The adapter produces canonical structures where possible, and stores vendor-only data in
  ``Extras.vendor["JEOL"]`` when mapping is not available.

- **Apply settings with provenance.**
  ``apply_beam_settings()`` / ``apply_projection_settings()`` apply canonical fields, and also
  recognize JEOL-only values supplied via ``settings.extra.vendor["JEOL"]`` (e.g. ``alpha_index``).

JEOL-only keys stored in Extras.vendor["JEOL"]
----------------------------------------------
The following vendor keys are used by this driver (non-exhaustive; extend as needed):

- ``alpha_index``:
  Convergence selector index (JEOL alpha). Canonical ``BeamSettings.convergence_angle`` is
  ``None`` unless a calibration/mapping is added.

- ``defocus_olc_dac``:
  Raw OLc/DAC value for defocus. Canonical ``ProjectionSettings.defocus`` is only populated
  when a calibration scale is provided (see "Calibration" below).

- Capture-specific details (recorded in image metadata extras where applicable):
  ``detector_id``, ``binning_index``, ``frame_integration``, ROI payload, etc.

Calibration
-----------
Some JEOL controls require calibration to convert vendor-native values into physical units.
This driver supports an opt-in defocus calibration via ``defocus_scale`` (nm per DAC unit).
If not provided, defocus is treated as uncalibrated:
- ``get_defocus()`` returns ``None`` and stashes ``defocus_olc_dac`` under vendor extras.
- ``set_defocus()`` refuses uncalibrated nm inputs (use ``set_defocus_dac()`` instead).

Acquisition / acquire_image()
-----------------------------
``acquire_image(request)`` expects the current SuperTEM request model:
- Detector controls come from ``request.detector`` (:class:`~supertem.structures.base.DetectorSettings`).
- The returned :class:`~supertem.structures.base.MicroscopeImageMetadata` stays canonical, while
  JEOL-only capture details are stored under ``metadata.extra.vendor["JEOL"]``.

Extending this driver
---------------------
When adding a new JEOL feature:
1) Prefer adding a vendor-native atomic getter/setter if PyJEM uses indices/DAC/mode codes.
2) Add/extend mapping logic in ``supertem.jeol_adapter`` to translate to canonical structures.
3) Ensure any unmapped vendor data is preserved under ``Extras.vendor["JEOL"]`` with stable keys.
4) Keep the base interface semantics intact: canonical fields represent physical quantities
   (or ``None`` when not representable without calibration).

Dependencies
------------
- PyJEM (TEM3). This module imports PyJEM lazily and will raise a clear error if unavailable.

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

        # Mapping / calibration parameters
        # NOTE: Some vendor APIs expose lens controls only in raw DAC counts. We only expose
        # physical quantities (e.g. nm) when an explicit calibration parameter is provided.
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
        """Get convergence angle (alpha) as a physical quantity.

        JEOL/PyJEM exposes alpha primarily as a discrete selector index (0-8) via EOS3.GetAlpha().
        Without a calibration table mapping index -> angle, we cannot report a physical convergence
        angle in degrees/mrad. In that case, we return None; the selector index is surfaced via
        JeolMicroscope.get_beam_settings() as `extra.vendor['JEOL']['alpha_index']`.
        """
        return None

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

    # =========================================================================
    # Logic Layer Overrides (JEOL)
    # =========================================================================

    def get_beam_settings(self) -> BeamSettings:
        """JEOL override: build BeamSettings via adapter, preserving vendor-native fields.

        JEOL reports alpha as an index. We store it under `extra.vendor['JEOL']['alpha_index']`
        and leave `convergence_angle` as None unless a calibration exists.
        """
        # Raw values for adapter (prefer vendor-native API reads)
        raw_flags: Dict[str, Any] = {}

        # Acceleration voltage
        voltage_val: float = 0.0
        if self.ht and hasattr(self.ht, "GetHtValue"):
            try:
                voltage_val = float(self.ht.GetHtValue())
            except Exception:
                raw_flags["ht_unavailable"] = True
        else:
            vq = self.get_acceleration_voltage()
            if vq is None:
                raw_flags["ht_unavailable"] = True
            else:
                # Adapter heuristic accepts kV if <= 5000
                try:
                    voltage_val = float(vq.to(Units.KV).magnitude)
                except Exception:
                    raw_flags["ht_unavailable"] = True

        # Beam current (PyJEM returns uA typically)
        current_ua: float = 0.0
        if self.gun and hasattr(self.gun, "GetEmissionCurrent"):
            try:
                current_ua = float(self.gun.GetEmissionCurrent())
            except Exception:
                raw_flags["beam_current_unavailable"] = True
        else:
            cq = self.get_beam_current()
            if cq is None:
                raw_flags["beam_current_unavailable"] = True
            else:
                try:
                    current_ua = float(cq.to(Units.UA).magnitude)
                except Exception:
                    raw_flags["beam_current_unavailable"] = True

        # Spot size (already defined as an index in the abstract interface)
        try:
            spot_idx = int(self.get_spot_size())
        except Exception:
            spot_idx = 0
            raw_flags["spot_size_unavailable"] = True

        # Alpha selector index (vendor-native)
        alpha_idx = self.get_alpha_index()
        if alpha_idx is None:
            alpha_idx = -1
            raw_flags["alpha_unavailable"] = True

        # Beam shift (CLA1) DAC
        beam_shift_dac: Optional[Tuple[int, int]] = None
        try:
            bs = self.get_beam_shift()
            bx, by = float(bs[0]), float(bs[1])
            beam_shift_dac = (int(round(bx)), int(round(by)))
            if (beam_shift_dac[0] != bx) or (beam_shift_dac[1] != by):
                raw_flags["beam_shift_float"] = (bx, by)
        except Exception:
            raw_flags["beam_shift_unavailable"] = True

        return jeol_adapter.from_jeol_beam_stats(
            voltage_val=voltage_val,
            current_ua=current_ua,
            spot_size_idx=spot_idx,
            alpha_idx=int(alpha_idx),
            beam_shift_dac=beam_shift_dac,
            raw_flags=raw_flags if raw_flags else None
        )

    def apply_beam_settings(self, settings: BeamSettings) -> None:
        """JEOL override: apply canonical fields and vendor-native alpha index if provided."""
        # Apply canonical fields except convergence_angle (not supported without calibration)
        if settings.voltage is not None:
            self.set_acceleration_voltage(settings.voltage)
        if settings.beam_current is not None:
            self.set_beam_current(settings.beam_current)
        if settings.spot_size is not None:
            self.set_spot_size(settings.spot_size)

        if settings.convergence_angle is not None:
            # Control-plane: do not silently reinterpret a physical angle as an index.
            raise ValueError(
                "JEOL driver cannot apply BeamSettings.convergence_angle (physical) without "
                "a calibration table. Use BeamSettings.extra.vendor['JEOL']['alpha_index']."
            )

        if settings.beam_shift is not None:
            self.set_beam_shift(settings.beam_shift.x, settings.beam_shift.y)
        if settings.condenser_stigmation is not None:
            self.set_condenser_stigmation(settings.condenser_stigmation.x, settings.condenser_stigmation.y)
        if settings.gun_tilt is not None:
            self.set_gun_tilt(settings.gun_tilt.x, settings.gun_tilt.y)

        # Vendor-native: alpha selector
        try:
            vend = getattr(settings.extra, "vendor", None) or {}
            jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None
            if isinstance(jeol_v, dict) and "alpha_index" in jeol_v:
                self.set_alpha_index(int(jeol_v["alpha_index"]))
        except Exception:
            # Best-effort only; do not crash after applying other fields.
            return

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

        # Apply canonical fields (may include defocus if calibrated)
        super().apply_projection_settings(settings)

        # Vendor-native: defocus DAC
        try:
            vend = getattr(settings.extra, "vendor", None) or {}
            jeol_v = vend.get("JEOL") if isinstance(vend, dict) else None
            if isinstance(jeol_v, dict) and "defocus_olc_dac" in jeol_v:
                self.set_defocus_dac(int(jeol_v["defocus_olc_dac"]))
        except Exception:
            return

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
        """Set convergence angle (alpha) as a physical quantity.

        JEOL/PyJEM supports alpha selection as a discrete index (0-8). Without a calibration
        table, we cannot map a physical angle (e.g. mrad) to the correct selector index.
        Use `set_alpha_index()` (vendor-native) instead.
        """
        raise ValueError(
            "JEOL driver does not support setting a physical convergence angle without a "
            "calibration table. Use set_alpha_index(idx) or provide idx via "
            "BeamSettings.extra.vendor['JEOL']['alpha_index']."
        )

    # --- JEOL Vendor-Native Alpha (Convergence Selector) ---

    def get_alpha_index(self) -> Optional[int]:
        """Get JEOL alpha selector index (0-8).

        This is vendor-native (unitless). Prefer this over `get_convergence_angle()` for JEOL.
        """
        if not self.eos or not hasattr(self.eos, "GetAlpha"):
            return None
        try:
            return int(self.eos.GetAlpha())
        except Exception:
            return None

    def set_alpha_index(self, idx: int) -> None:
        """Set JEOL alpha selector index (0-8)."""
        if not self.eos or not hasattr(self.eos, "SetAlphaSelector"):
            return
        try:
            i = int(idx)
        except Exception:
            return
        i = max(0, min(8, i))
        try:
            self.eos.SetAlphaSelector(i)
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
        """Set defocus as a physical quantity (nm) when calibrated.

        Without a calibration (`defocus_scale`), this is not supported because OLc is vendor
        DAC units. Use `set_defocus_dac()` (vendor-native) instead.
        """
        if not self._has_defocus_calibration:
            raise ValueError(
                "JEOL driver cannot set physical defocus (nm) without defocus_scale calibration. "
                "Use set_defocus_dac(dac) or provide dac via ProjectionSettings.extra.vendor['JEOL']['defocus_olc_dac']."
            )
        if self.lens and hasattr(self.lens, "SetOLc"):
            val = float(defocus.to(Units.NM).magnitude)
            scaled = int(val * (self.defocus_scale or 1.0))
            self.lens.SetOLc(scaled)

    # --- JEOL Vendor-Native Defocus (OLc DAC) ---

    def get_defocus_dac(self) -> Optional[int]:
        """Get objective lens coarse (OLc) value in JEOL DAC units."""
        if self.lens and hasattr(self.lens, "GetOLc"):
            try:
                return int(self.lens.GetOLc())
            except Exception:
                return None
        return None

    def set_defocus_dac(self, dac: int) -> None:
        """Set objective lens coarse (OLc) value in JEOL DAC units."""
        if self.lens and hasattr(self.lens, "SetOLc"):
            try:
                self.lens.SetOLc(int(dac))
            except Exception:
                return

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
            if det_id is None:
                return None
            return self._get_detector(det_id)
        except Exception:
            return None

    @staticmethod
    def _first_int(d: dict, keys: tuple[str, ...]) -> Optional[int]:
        for k in keys:
            if k in d:
                try:
                    return int(d[k])
                except Exception:
                    continue
        return None

    @staticmethod
    def _first_float(d: dict, keys: tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    continue
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
            except Exception:
                pass
        return str(self._scan_cfg.get("mode", "Scan"))

    def get_scan_width(self) -> int:
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    w = self._first_int(st, ("Width", "ImagingAreaWidth", "ImagingArea_Width", "ImagingAreaW"))
                    if w is not None:
                        self._scan_cfg["width_px"] = w
                        return w
            except Exception:
                pass
        return int(self._scan_cfg.get("width_px", 512))

    def get_scan_height(self) -> int:
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    h = self._first_int(st, ("Height", "ImagingAreaHeight", "ImagingArea_Height", "ImagingAreaH"))
                    if h is not None:
                        self._scan_cfg["height_px"] = h
                        return h
            except Exception:
                pass
        return int(self._scan_cfg.get("height_px", 512))

    def get_scan_pixel_dwell(self) -> Quantity:
        # No stable public getter in PyJEM docs; keep as local config for now.
        return Q_(float(self._scan_cfg.get("pixel_dwell_us", 10.0)), Units.US)

    def get_scan_flyback(self) -> Quantity:
        # No stable public getter in PyJEM docs; keep as local config for now.
        return Q_(float(self._scan_cfg.get("flyback_us", 100.0)), Units.US)

    def get_scan_rotation(self) -> Quantity:
        # Prefer TEM3.Scan3 for rotation readback.
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

        # Fallback: try detector settings (if present).
        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "get_detectorsetting"):
            try:
                st = d.get_detectorsetting()
                if isinstance(st, dict):
                    ang = self._first_float(st, ("ScanRotation", "ScanRotationValue", "ScanRotationDeg"))
                    if ang is not None:
                        return Q_(ang, Units.DEG)
            except Exception:
                pass

        return Q_(float(self._scan_cfg.get("rotation_deg", 0.0)), Units.DEG)

    def get_scan_active(self) -> bool:
        # Prefer TEM3.Scan3 ext scan mode.
        if self.scan and hasattr(self.scan, "GetExtScanMode"):
            try:
                return bool(int(self.scan.GetExtScanMode()) == 1)
            except Exception:
                pass
        return bool(self._scan_cfg.get("active", False))

    def set_scan_mode(self, mode: str) -> None:
        m = (mode or "").strip().lower()
        mapping = {"scan": 0, "full": 0, "full frame": 0, "spot": 1, "area": 3, "subarea": 3}
        if m not in mapping and m.isdigit():
            mapping[m] = int(m)
        val = mapping.get(m)
        if val is None:
            raise ValueError(f"Unsupported scan mode: {mode!r} (expected Scan/Spot/Area)")

        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "set_scanmode"):
            try:
                d.set_scanmode(int(val))
                self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))
                return
            except Exception:
                pass

        self._scan_cfg["mode"] = {0: "Scan", 1: "Spot", 3: "Area"}.get(int(val), str(val))

    def _set_imaging_area(
        self,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> None:
        """Best-effort wrapper around detector.Detector.set_imaging_area()."""
        d = self._get_scan_controller_detector()
        if d is None:
            return

        w = int(width if width is not None else self._scan_cfg.get("width_px", 512))
        h = int(height if height is not None else self._scan_cfg.get("height_px", 512))
        xx = int(x if x is not None else self._scan_cfg.get("x_px", 0))
        yy = int(y if y is not None else self._scan_cfg.get("y_px", 0))

        self._scan_cfg.update({"width_px": w, "height_px": h, "x_px": xx, "y_px": yy})

        if hasattr(d, "set_imaging_area"):
            try:
                d.set_imaging_area(w, h, xx, yy)
                return
            except Exception:
                pass

    def set_scan_width(self, width: int) -> None:
        self._set_imaging_area(width=int(width))

    def set_scan_height(self, height: int) -> None:
        self._set_imaging_area(height=int(height))

    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        try:
            us = float(time.to(Units.US).magnitude)
        except Exception:
            us = float(time.magnitude)
        self._scan_cfg["pixel_dwell_us"] = us

    def set_scan_flyback(self, time: Quantity) -> None:
        try:
            us = float(time.to(Units.US).magnitude)
        except Exception:
            us = float(time.magnitude)
        self._scan_cfg["flyback_us"] = us

    def set_scan_rotation(self, angle: Quantity) -> None:
        deg = float(angle.to(Units.DEG).magnitude)
        self._scan_cfg["rotation_deg"] = deg

        d = self._get_scan_controller_detector()
        if d is not None and hasattr(d, "set_scanrotation"):
            try:
                d.set_scanrotation(float(deg))
                return
            except Exception:
                pass

        if self.scan and hasattr(self.scan, "SetRotationAngleEx"):
            try:
                self.scan.SetRotationAngleEx(float(deg))
                return
            except Exception:
                pass
        if self.scan and hasattr(self.scan, "SetRotationAngle"):
            try:
                self.scan.SetRotationAngle(int(round(deg)) % 360)
            except Exception:
                pass

    def set_scan_active(self, active: bool) -> None:
        self._scan_cfg["active"] = bool(active)

        d = self._get_scan_controller_detector()
        if d is not None:
            if active and hasattr(d, "livestart"):
                try:
                    d.livestart()
                    return
                except Exception:
                    pass
            if (not active) and hasattr(d, "livestop"):
                try:
                    d.livestop()
                    return
                except Exception:
                    pass

        val = 1 if active else 0
        if self.scan and hasattr(self.scan, "SetExtScanMode"):
            try:
                self.scan.SetExtScanMode(val)
            except Exception:
                pass

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
        """Atomic: Acquire a single image from a JEOL detector.

        Notes:
            - AcquisitionRequest uses `detector` (DetectorSettings) and `image` (ImageOutputSettings).
              There is no `request.settings`.
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

        if not det_id:
            raise RuntimeError("No detector_id provided and no primary detector available")

        d = self._get_detector(det_id)

        # ---------------------------------------------------------------------
        # 1) Apply per-request detector settings (atomic-only)
        # ---------------------------------------------------------------------
        det_req = request.detector
        if det_req is not None:
            try:
                if det_req.exposure is not None:
                    self.set_detector_exposure(det_id, det_req.exposure)
                if det_req.binning_index is not None:
                    self.set_detector_binning(det_id, int(det_req.binning_index))
                if det_req.frame_integration is not None:
                    self.set_detector_integration(det_id, int(det_req.frame_integration))
                if det_req.roi is not None:
                    self.set_detector_roi(det_id, det_req.roi)
                if det_req.gain_index is not None:
                    self.set_detector_gain(det_id, int(det_req.gain_index))
                if det_req.offset_index is not None:
                    self.set_detector_offset(det_id, int(det_req.offset_index))
            except Exception:
                # This is an atomic method; we don't interpret failures here.
                # Higher-level orchestration can decide whether to fail-hard.
                pass

        # ---------------------------------------------------------------------
        # 2) Capture raw data
        # ---------------------------------------------------------------------
        raw = None
        if hasattr(d, "snapshot_rawdata"):
            raw = d.snapshot_rawdata()
        elif hasattr(d, "get_image_cache"):
            raw = d.get_image_cache()
        elif hasattr(d, "livesnapshot"):
            raw = d.livesnapshot("tif")  # may return bytes stream

        # ---------------------------------------------------------------------
        # 3) Convert to numpy array
        # ---------------------------------------------------------------------
        arr: np.ndarray
        if raw is None:
            arr = np.zeros((1, 1), dtype=np.uint16)
        elif isinstance(raw, (bytes, bytearray)):
            # Try uint16 first (common for detectors); fallback to uint8.
            try:
                arr = np.frombuffer(raw, dtype=np.uint16)
            except Exception:
                arr = np.frombuffer(raw, dtype=np.uint8)
        elif isinstance(raw, list):
            arr = np.array(raw)
        elif isinstance(raw, dict) and "data" in raw:
            arr = np.array(raw["data"])
        else:
            try:
                arr = np.array(raw)
            except Exception:
                arr = np.zeros((1, 1), dtype=np.uint16)

        # Ensure a sane dtype for MicroscopeImage (expects uint8/uint16)
        if arr.dtype not in (np.uint8, np.uint16):
            try:
                arr = arr.astype(np.uint16, copy=False)
            except Exception:
                arr = np.array(arr, dtype=np.uint16)

        # ---------------------------------------------------------------------
        # 4) Deterministic reshape using ROI if possible
        # ---------------------------------------------------------------------
        roi = None
        if det_req is not None and getattr(det_req, "roi", None) is not None:
            roi = det_req.roi
        else:
            try:
                roi = self.get_detector_roi(det_id)
            except Exception:
                roi = None

        if arr.ndim == 1:
            cols = int(getattr(roi, "width", 0) or 0) if roi is not None else 0
            rows = int(getattr(roi, "height", 0) or 0) if roi is not None else 0
            if cols > 0 and rows > 0 and arr.size == cols * rows:
                arr = arr.reshape((rows, cols))

        # If still not 2D, attempt a safe squeeze / fallback
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)

        # ---------------------------------------------------------------------
        # 5) Build canonical scientific metadata + stash JEOL-only details
        # ---------------------------------------------------------------------
        created_at = datetime.now(timezone.utc).isoformat()

        # Snapshot some canonical values
        try:
            v = self.get_acceleration_voltage()
            accelerating_voltage_kv = float(v.to(Units.KV).magnitude) if v is not None else None
        except Exception:
            accelerating_voltage_kv = None

        try:
            bc = self.get_beam_current()
            beam_current_na = float(bc.to(Units.NA).magnitude) if bc is not None else None
        except Exception:
            beam_current_na = None

        # Exposure in ms (use current detector setting after any applied request)
        try:
            exp_q = self.get_detector_exposure(det_id)
            exposure_ms = float(exp_q.to(Units.MS).magnitude) if exp_q is not None else None
        except Exception:
            exposure_ms = None

        # Indicated magnification & camera length (best-effort)
        try:
            mag_idx = self.get_magnification_index()
            magnification = float(mag_idx) if mag_idx is not None else None
        except Exception:
            magnification = None

        try:
            cl = self.get_camera_length()
            camera_length_mm = float(cl.to(Units.MM).magnitude) if cl is not None else None
        except Exception:
            camera_length_mm = None

        w = int(arr.shape[1]) if arr.ndim == 2 else None
        h = int(arr.shape[0]) if arr.ndim == 2 else None

        # Vendor-only detector snapshot
        jeol_vendor: Dict[str, Any] = {"detector_id": det_id}
        try:
            jeol_vendor["binning_index"] = int(self.get_detector_binning(det_id))
        except Exception:
            pass
        try:
            jeol_vendor["frame_integration"] = int(self.get_detector_integration(det_id))
        except Exception:
            pass
        if roi is not None:
            try:
                jeol_vendor["roi"] = roi.to_dict() if hasattr(roi, "to_dict") else roi
            except Exception:
                jeol_vendor["roi"] = None

        # Full telemetry snapshot (best-effort). Keep failures as notes.
        state = None
        meta_extra = Extras(vendor={"JEOL": jeol_vendor})
        try:
            state = self.get_full_state()
        except Exception as e:
            try:
                meta_extra.notes["MicroscopeImageMetadata.state_capture_failed"] = str(e)
            except Exception:
                pass

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
