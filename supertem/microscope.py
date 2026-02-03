"""
supertem.microscope

The Hardware Abstraction Layer (HAL) and Control Plane Orchestrator.

This module defines the abstract interface (`TemMicroscope`) that all vendor
drivers (e.g., JEOL, Thermo, Simulated) must implement. It acts as the
operational "Verb" layer corresponding to the "Noun" structures defined in
`supertem.structures.base`.

===============================================================================
I. The Three-Layer Architecture
===============================================================================

To ensure safety and consistency across different hardware vendors, this class
enforces a strict separation of concerns via three distinct execution layers:

  1) The Atomic Layer (The "Hands" - Abstract & Vendor Implemented)
     - Role: Direct, unbuffered hardware I/O.
     - Responsibility: Dumb I/O. If asked to set an unsafe value (e.g. index 99),
       it attempts it without second-guessing.
     - Behavior:
        - READ (Getters): "Null means Unknown". Returns `None` on failure, never defaults.
        - WRITE (Setters): "Fail Loudly". Raises exceptions if hardware rejects the command.
          *Rule:* Do NOT swallow hardware errors (IOError, Timeout) in this layer.

  2) The Helper Layer (The "Brain" - Vendor Overridden)
     - Role: Bulk application, Unpacking, and **Vendor Validation**.
     - Responsibility:
       a. Routes canonical physics (e.g. `voltage`) to atomic setters.
       b. **Vendor Guard:** Extracts vendor-specific keys from `Extras` (e.g. registers,
          indices), validates them against hardware limits, and RAISES error if invalid.
       c. Prevents invalid vendor data from reaching the Atomic layer.
     - Behavior:
       - **Validation:** Enforces vendor-specific safety logic (raises ValueError).
       - **Pass-Through:** Does NOT catch hardware errors. If the Atomic layer explodes
         (e.g., IOError), the Helper layer MUST let the exception bubble up.

  3) The Orchestrator Layer (The "Gatekeeper" - Framework Provided)
     - Role: The Control Plane Interface.
     - Responsibility:
       a. Validate the Intent (`request.validate()`).
       b. Check Canonical Hardware Capabilities (`system.is_safe_...`).
       c. Delegate to Helpers/Atomic methods for execution.
     - Behavior:
       - **Strict Safety:** Raises `RuntimeError` or `ValueError` to prevent unsafe moves.
       - **Bubble Up:** Does NOT catch hardware errors. If the Atomic/Helper layers explode,
         the Orchestrator lets the exception pass through to the user script.

===============================================================================
II. The Safety & Validation Contract
===============================================================================

Safety is handled via a "Dual-Gatekeeper" model:

  A. Canonical Safety (Handled by Orchestrator)
     The base class Orchestrator validates standard physical properties against
     `SystemSettings` limits (e.g., Voltage, Stage Limits).
     *Result:* Safe canonical values reach the Helper layer.

  B. Vendor Safety (Handled by Helper Overrides)
     The Orchestrator CANNOT validate vendor-specific `Extras`. The Vendor Driver
     MUST override Helper methods (e.g. `apply_beam_settings`) to validate these.
     *Result:* The driver refuses to pass invalid indices to the Atomic layer.

===============================================================================
III. Data Integrity & Parse Modes
===============================================================================

Drivers must implement the "Ingress/Egress" policy using `base.py` ParseModes:

  A. Egress (Control Plane / Writing to Hardware) -> ParseMode.STRICT
     - Context: `apply_...` methods and `move_stage...`.
     - Rule: **Fail Fast.** If the input (canonical or vendor extra) is invalid
       or unsafe, raise an Exception immediately. Do not coerce. Do not guess.

  B. Ingress (Data Plane / Reading from Hardware) -> ParseMode.LENIENT
     - Context: `get_...` methods and `acquire_image`.
     - Rule: **Survive.** If hardware returns malformed data (e.g., NaN vacuum),
       coerce it to `None` or a safe default. Do not crash the logging loop.
     - Implementation: Wrap Atomic Getters in try/except blocks that return `None`.

  *Exception:* Critical navigation data (e.g., Stage Position) may use STRICT
  mode on Ingress if corrupted data poses a physical collision risk.

===============================================================================
IV. Type Safety & Return Policy
===============================================================================

To balance Safety (Control Logic) with Accuracy (Physics), this interface enforces
a strict return type policy for Atomic Getters:

  A. Measurements (Optional Objects) -> Return `None` on Failure
     - Types: `Quantity`, `int` (indices), `StagePosition`, `ROI`.
     - Logic: `None` implies "Unknown". Zero is a valid physical value.
     - Example: `get_pressure() -> None` (Sensor offline).
     - Signature: `def get_x(self) -> Optional[Type]`

  B. Discrete States (Strict Primitives) -> Return Sentinel on Failure
     - Types: `str`, `bool`.
     - Logic: Return `"UNKNOWN"` or `False` to ensure control flow safety.
       Allows logic like `if get_mode() == "TEM"` to fail gracefully rather than crashing.
     - Example: `get_mode() -> "UNKNOWN"`, `get_beam_blank() -> False`.
     - Signature: `def get_x(self) -> str` (No Optional)

===============================================================================
V. Logging Strategy (Intent vs. IO)
===============================================================================

To maintain readability and traceability, drivers must strictly follow these
logging rules:

1. Layered Logging Levels
   - **Orchestrator (INFO):** Logs high-level intent.
     *Example:* `[STAGE] Executing Move: Target=(x=10um)...`
   - **Helper (WARNING):** Logs safety interventions or clamps.
     *Example:* `[BEAM] Spot Size 12 clamped to 5.`
   - **Atomic (DEBUG):** Logs raw hardware I/O.
     *Example:* `[PyJEM] Write: HT3.SetHtValue(200000)`

2. Implementation Rules (Atomic Layer)
   Drivers must implement Atomic methods using this specific pattern:

   A. **Consistent Logging (Setters):**
      Always log the value *before* the hardware call.
      *Pattern:* `logger.debug(f"[{TAG}] Setting {Name}: {Value}")`

   B. **Consistent Error Handling:**
      - **Getters (Read):** Catch Exception -> Log DEBUG -> Return None.
        *Reason:* "Null means Unknown". Logging as ERROR causes log spam during
        high-frequency polling.
        *Code:*
          ```python
          try:
              return hardware.get_value()
          except Exception as e:
              logger.debug(f"[{TAG}] Read failed: {e}")
              return None
          ```

      - **Setters (Write):** Catch Exception -> Log ERROR -> Raise.
        *Reason:* "Fail Loudly". Writes change state; silent failure is dangerous.
        *Code:*
          ```python
          try:
              hardware.set_value(val)
          except Exception as e:
              logger.error(f"[{TAG}] Write failed: {e}")
              raise
          ```

===============================================================================
Usage
===============================================================================

  # 1. Instantiate (usually via utils.setup_session)
  scope = JeolMicroscope(settings)

  # 2. Control (Use Orchestrators)
  req = StageMoveRequest(target=StagePosition(x=Q_(10, 'um')))
  scope.execute_stage_move(req)  # -> Checks limits -> Calls move_stage_absolute

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import replace
import datetime
import os
from pathlib import Path

# Import strictly typed structures from base.py
from supertem.structures.base import (
    # Configuration & Safety
    MicroscopeSettings,
    SystemSettings,
    SystemInfo,
    SafetyCheck,

    # State Objects (Snapshots)
    MicroscopeState,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    DetectorSettings,
    ScanSettings,
    VacuumSettings,
    ApertureSettings,  # CORRECTED: Was Aperture
    MicroscopeImage,
    MicroscopeImageMetadata,
    Point,
    ROI,
    ImageOutputSettings,

    # Request Objects (Intents)
    StageMoveRequest,
    StageControlRequest,
    BeamControlRequest,
    ProjectionControlRequest,
    DetectorControlRequest,
    AcquisitionRequest,
    VacuumControlRequest,
    ApertureControlRequest,
    ScanControlRequest,

    # Enums & Constants
    Units,
    Q_,         # For Instantiation (Values)
    Quantity,   # For Type Hinting (Annotations)
    StageDriveType,
    ParseMode
)

logger = logging.getLogger(__name__)


class TemMicroscope(ABC):
    """
    The generic template for all TEM implementations.
    Acts as the bridge between the Control Plane (Requests) and the Hardware Plane (Drivers).
    """

    def __init__(self, settings: Optional[MicroscopeSettings] = None):
        """
        Initialize the microscope interface.

        Args:
            settings: Configuration containing system limits, hardware registry,
                      and safety policies. If None, safe defaults are used.
        """
        if settings is None:
            self._settings = MicroscopeSettings(
                system=SystemSettings(),
                _mode=ParseMode.LENIENT
            )
        else:
            self._settings = settings

    @property
    def system_settings(self) -> SystemSettings:
        """Access the system limits and capabilities configuration."""
        return self._settings.system

    # ---------------------------------------------------------------------
    # Internal helpers (Intent summaries & Extras)
    # ---------------------------------------------------------------------

    @staticmethod
    def _summarize_extras(extra) -> str:
        """Return a compact summary of actionable extras keys.

        We treat vendor/unknown dict keys with non-None values as actionable.
        raw/notes are intentionally ignored.
        """
        if extra is None:
            return ""
        parts = []
        vend = getattr(extra, "vendor", None)
        if isinstance(vend, dict):
            for vname, payload in vend.items():
                if isinstance(payload, dict):
                    keys = [k for k, v in payload.items() if v is not None]
                    if keys:
                        parts.append(f"vendor.{vname}({', '.join(keys)})")
                elif payload is not None:
                    parts.append(f"vendor.{vname}")
        unk = getattr(extra, "unknown", None)
        if isinstance(unk, dict):
            keys = [k for k, v in unk.items() if v is not None]
            if keys:
                parts.append(f"unknown({', '.join(keys)})")
        return "; ".join(parts)

    @classmethod
    def _summarize_patch(cls, target) -> str:
        """Summarize non-None canonical fields + actionable extras."""
        if target is None:
            return "<none>"
        fields = []
        for name, val in getattr(target, "__dict__", {}).items():
            if name.startswith("_") or name == "extra":
                continue
            if val is not None:
                fields.append(name)
        extra_s = cls._summarize_extras(getattr(target, "extra", None))
        if extra_s:
            fields.append(extra_s)
        return ", ".join(fields) if fields else "<empty>"

    @staticmethod
    def _require_point_complete(p: Optional[Point], name: str) -> None:
        """Reject partially-specified Point values.

        For 2D coil fields (Point), we require both x and y if either is provided.
        """
        if p is None:
            return
        x = getattr(p, "x", None)
        y = getattr(p, "y", None)
        if (x is None) ^ (y is None):
            raise ValueError(f"{name} requires both x and y when provided (got x={x}, y={y}).")

    # =========================================================================
    # 1. Connection & Lifecycle
    # =========================================================================

    @abstractmethod
    def connect(self, host: str, port: Optional[int] = None, **kwargs) -> None:
        """
        Establish connection to the microscope.

        Args:
            host: Hostname or IP address.
            port: Port number (optional).
            **kwargs: Vendor-specific arguments (e.g., api_key, instrument_id).
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources and close the connection cleanly."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connection is active and responsive."""
        pass

    @abstractmethod
    def get_instrument_info(self) -> SystemInfo:
        """
        Return static instrument identity.

        Returns:
            SystemInfo containing Model, Serial, Software Version, etc.
        """
        pass

    # =========================================================================
    # 2. Global State & Mode
    # =========================================================================

    @abstractmethod
    def get_mode(self) -> str:
        """
        Get the global instrument mode. (Strict Primitive)

        Returns:
            String: 'TEM', 'STEM', or 'UNKNOWN' on failure.
        """
        pass

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """
        Set the global instrument mode.

        Args:
            mode: The target mode string (must be supported by vendor driver).
        """
        pass

    def get_full_state(self) -> MicroscopeState:
        """
        Capture a comprehensive snapshot of the entire microscope state.

        Aggregates data from all subsystems (Stage, Beam, Optics, etc.) into
        a single timestamped structure matching base.py definition.
        """
        return MicroscopeState(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            mode=self.get_mode(),
            stage_position=self.get_stage_position(),
            beam=self.get_beam_settings(),
            projection=self.get_projection_settings(),
            scan=self.get_scan_settings(),
            vacuum=self.get_vacuum_settings(),
            apertures={a_id: self.get_aperture_settings(a_id)
                       for a_id in self.list_apertures()},
            detectors={d_id: self.get_detector_settings(d_id)
                       for d_id in self.list_detectors()},
            active_detector_ids=self.get_active_detector_ids(),
            primary_detector_id=self.get_primary_detector_id()
        )

    # =========================================================================
    # 3. Stage Control (Motion)
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def get_stage_x(self) -> Optional[Quantity]:
        """Atomic: Get X coordinate (nm)."""
        pass

    @abstractmethod
    def get_stage_y(self) -> Optional[Quantity]:
        """Atomic: Get Y coordinate (nm)."""
        pass

    @abstractmethod
    def get_stage_z(self) -> Optional[Quantity]:
        """Atomic: Get Z coordinate (nm)."""
        pass

    @abstractmethod
    def get_stage_r(self) -> Optional[Quantity]:
        """Atomic: Get Rotation (deg)."""
        pass

    @abstractmethod
    def get_stage_tilt_x(self) -> Optional[Quantity]:
        """Atomic: Get Alpha Tilt (deg)."""
        pass

    @abstractmethod
    def get_stage_tilt_y(self) -> Optional[Quantity]:
        """Atomic: Get Beta Tilt (deg)."""
        pass

    @abstractmethod
    def get_stage_coordinate_system(self) -> Optional[str]:
        """Atomic: Get the current reference frame name."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default", wait: bool = True,
                            **kwargs) -> None:
        """
        Atomic: Move multiple axes simultaneously (Vector Move).
        Target fields that are None should be ignored (no motion).

        Args:
            target: Destination coordinates.
            drive_type: Mechanism hint (e.g. 'piezo', 'motor').
            wait: If True, block until move completes.
            **kwargs: Vendor-specific execution options (e.g. tolerance_nm, retries).
        """
        pass

    @abstractmethod
    def stop_stage(self, **kwargs) -> None:
        """Atomic: Immediately halt all stage motion axes."""
        pass

    @abstractmethod
    def home_stage(self, **kwargs) -> None:
        """Atomic: Return stage to its mechanical origin/zero position."""
        pass

    # --- Helper Layer ---

    def get_stage_position(self) -> StagePosition:
        """
        Helper: Aggregates atomic primitives into a consistent StagePosition object.
        """
        return StagePosition(
            x=self.get_stage_x(),
            y=self.get_stage_y(),
            z=self.get_stage_z(),
            r=self.get_stage_r(),
            tilt_x=self.get_stage_tilt_x(),
            tilt_y=self.get_stage_tilt_y(),
            coordinate_system=self.get_stage_coordinate_system()
        )

    def apply_stage_position(self, target: StagePosition, drive_type: str = "default", wait: bool = True,
                             **kwargs) -> None:
        """
        Helper: Prepares and executes a stage move.
        Drivers can override this to handle vendor-specific 'extras' before moving.
        """
        self.move_stage_absolute(target, drive_type=drive_type, wait=wait, **kwargs)

    def perform_stage_action(self, action: str, **kwargs) -> None:
        """
        Helper: Routes high-level control actions to atomic commands.
        Vendor Override: Useful if 'STOP' requires complex deceleration logic.
        """
        if action == "STOP":
            self.stop_stage(**kwargs)
        elif action == "HOME":
            self.home_stage(**kwargs)
        elif action == "ZERO_ENCODERS":
            # Example of an action that might not be standard, handled gracefully
            logger.warning("[STAGE] ZERO_ENCODERS requested but not implemented in base.")

    # --- Orchestrator Layer ---

    def execute_stage_move(self, request: StageMoveRequest) -> None:
        """Orchestrator: Validates intent and checks safety limits."""
        if not request.validate():
            raise ValueError(f"Invalid StageMoveRequest: {request}")

        tgt_str = f"Target={request.target}" if not request.relative else f"Delta={request.target}"
        logger.info(f"[STAGE] Executing Move: {tgt_str} (Mode: {request.drive_type})")

        # 1. Resolve Absolute Target
        current = self.get_stage_position()
        target_abs = request.target

        if request.relative:
            if current is None:
                raise RuntimeError("Relative move failed: Current stage position is unknown.")
            target_abs = current + request.target

        # 2. Safety Check
        sys = self.system_settings.stage_system
        if sys:
            check = sys.is_safe_move(
                target=request.target if request.relative else target_abs,
                current=current,
                relative=request.relative,
                ignore_step_limit=True
            )
            if not check:
                logger.error(f"[STAGE] Unsafe move rejected. Reasons: {check.reasons}")
                raise RuntimeError(f"Unsafe move rejected: {check.reasons}")

        # 3. Extract Options
        exec_opts = request.extra.options if request.extra else {}

        # 4. Execution
        self.safe_move_stage(
            target_abs,
            drive_type=request.drive_type,
            wait=request.wait_for_settle,
            **exec_opts
        )

    def safe_move_stage(self, target: StagePosition, drive_type: str = "default", wait: bool = True, **kwargs) -> None:
        """
        Safety Helper: Breaks large moves into smaller linear steps if required.
        """
        sys = self.system_settings.stage_system
        if not sys or not sys.max_step_distance:
            self.apply_stage_position(target, drive_type=drive_type, wait=wait, **kwargs)
            return

        current = self.get_stage_position()
        if current is None:
            raise RuntimeError("Safe Move Failed: Cannot read current stage position.")

        max_step_nm = sys.max_step_distance.to(Units.NM).magnitude

        def dist(c: Optional[Quantity], t: Optional[Quantity]) -> float:
            if c is None or t is None: return 0.0
            return abs(t.to(Units.NM).magnitude - c.to(Units.NM).magnitude)

        d_x = dist(current.x, target.x)
        d_y = dist(current.y, target.y)
        d_z = dist(current.z, target.z)
        max_dist = max(d_x, d_y, d_z)

        if max_dist <= max_step_nm:
            self.apply_stage_position(target, drive_type=drive_type, wait=wait, **kwargs)
            return

        # Linear Interpolation
        steps = int(max_dist // max_step_nm) + 1
        logger.info(f"[STAGE] Step Limit: {max_dist:.1f}nm > {max_step_nm:.1f}nm. Breaking into {steps} segments.")

        for i in range(1, steps + 1):
            frac = i / steps
            interim = replace(current)

            def interp(c, t):
                if c is None or t is None: return c
                return c + (t - c) * frac

            if target.x is not None: interim.x = interp(current.x, target.x)
            if target.y is not None: interim.y = interp(current.y, target.y)
            if target.z is not None: interim.z = interp(current.z, target.z)

            if i == steps:
                interim.r = target.r
                interim.tilt_x = target.tilt_x
                interim.tilt_y = target.tilt_y

            self.apply_stage_position(interim, drive_type=drive_type, wait=True, **kwargs)

    def execute_stage_control(self, request: StageControlRequest) -> None:
        """Orchestrator: Handle StageControlRequest (STOP, HOME)."""
        if not request.validate():
            raise ValueError(f"Invalid StageControlRequest: {request}")

        logger.info(f"[STAGE] Executing Control: {request.action}")
        exec_opts = request.extra.options if request.extra else {}

        self.perform_stage_action(request.action, **exec_opts)

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def get_acceleration_voltage(self) -> Optional[Quantity]:
        """Get High Tension (kV). Returns None if unknown."""
        pass

    @abstractmethod
    def get_probe_mode(self) -> Optional[str]:
        """Get probe mode (e.g. 'Microprobe', 'Nanoprobe'). Returns None if unknown."""
        pass

    @abstractmethod
    def get_beam_current(self) -> Optional[Quantity]:
        """Get Beam Current (nA/pA). Returns None if unknown."""
        pass

    @abstractmethod
    def get_emission_current(self) -> Optional[Quantity]:
        """Get Gun Emission Current (uA). Returns None if unknown."""
        pass

    @abstractmethod
    def get_spot_size(self) -> Optional[int]:
        """Get Spot Size Index. Returns None if unknown."""
        pass

    @abstractmethod
    def get_convergence_angle(self) -> Optional[Quantity]:
        """Get Convergence (Alpha) Angle (mrad). Returns None if unknown."""
        pass

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Get Beam Blank Status. True=Blanked."""
        pass

    @abstractmethod
    def get_beam_shift(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Beam Shift Coils (x, y). Returns (None, None) if unknown."""
        pass

    @abstractmethod
    def get_beam_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Beam Tilt Coils (x, y). Returns (None, None) if unknown."""
        pass

    @abstractmethod
    def get_condenser_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Condenser Stigmator Coils (x, y). Returns (None, None) if unknown."""
        pass

    @abstractmethod
    def get_gun_tilt(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Gun Tilt Alignment (x, y). Returns (None, None) if unknown."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_acceleration_voltage(self, voltage: Quantity, **kwargs) -> None:
        """Set High Tension (kV)."""
        pass

    @abstractmethod
    def set_probe_mode(self, mode: str, **kwargs) -> None:
        """Set probe mode."""
        pass

    @abstractmethod
    def set_beam_current(self, current: Quantity, **kwargs) -> None:
        """Set Beam Current (nA)."""
        pass

    @abstractmethod
    def set_emission_current(self, current: Quantity, **kwargs) -> None:
        """Set Gun Emission Current (uA)."""
        pass

    @abstractmethod
    def set_spot_size(self, index: int, **kwargs) -> None:
        """Set Spot Size Index."""
        pass

    @abstractmethod
    def set_convergence_angle(self, angle: Quantity, **kwargs) -> None:
        """Set Convergence Angle (mrad)."""
        pass

    @abstractmethod
    def set_beam_blank(self, blank: bool, **kwargs) -> None:
        """Set Beam Blanker. True = Blank (Block)."""
        pass

    @abstractmethod
    def set_beam_shift(self, x: float, y: float, **kwargs) -> None:
        """Set Beam Shift Coils."""
        pass

    @abstractmethod
    def set_beam_tilt(self, x: float, y: float, **kwargs) -> None:
        """Set Beam Tilt Coils."""
        pass

    @abstractmethod
    def set_condenser_stigmation(self, x: float, y: float, **kwargs) -> None:
        """Set Condenser Stigmator Coils."""
        pass

    @abstractmethod
    def set_gun_tilt(self, x: float, y: float, **kwargs) -> None:
        """Set Gun Tilt Alignment."""
        pass

    # --- Helper Layer ---

    def get_beam_settings(self) -> BeamSettings:
        """Helper: Aggregates atomic beam state into a BeamSettings object."""
        # Note: Point() construction handles the (None, None) case gracefully if needed,
        # but we check explicit returns from atomics.

        bs, bt = self.get_beam_shift(), self.get_beam_tilt()
        cs = self.get_condenser_stigmation()
        gt = self.get_gun_tilt()

        return BeamSettings(
            mode=self.get_mode(),  # Global mode usually lives here
            voltage=self.get_acceleration_voltage(),
            probe_mode=self.get_probe_mode(),
            beam_current=self.get_beam_current(),
            emission_current=self.get_emission_current(),
            spot_size=self.get_spot_size(),
            convergence_angle=self.get_convergence_angle(),
            is_blanked=self.get_beam_blank(),

            # Reconstruction of Points
            beam_shift=Point(x=bs[0], y=bs[1]) if bs[0] is not None else None,
            beam_tilt=Point(x=bt[0], y=bt[1]) if bt[0] is not None else None,
            condenser_stigmation=Point(x=cs[0], y=cs[1]) if cs[0] is not None else None,
            gun_tilt=Point(x=gt[0], y=gt[1]) if gt[0] is not None else None
        )

    def apply_beam_settings(self, settings: BeamSettings, **kwargs) -> None:
        """
        Helper: Applies a partial beam configuration.
        """
        if settings.mode is not None:
            self.set_mode(settings.mode)  # Global set_mode usually handles its own args
        if settings.voltage is not None:
            self.set_acceleration_voltage(settings.voltage, **kwargs)
        if settings.beam_current is not None:
            self.set_beam_current(settings.beam_current, **kwargs)
        if settings.emission_current is not None:
            self.set_emission_current(settings.emission_current, **kwargs)
        if settings.spot_size is not None:
            self.set_spot_size(settings.spot_size, **kwargs)
        if settings.convergence_angle is not None:
            self.set_convergence_angle(settings.convergence_angle, **kwargs)
        if settings.probe_mode is not None:
            self.set_probe_mode(settings.probe_mode, **kwargs)
        if settings.is_blanked is not None:
            self.set_beam_blank(settings.is_blanked, **kwargs)

        if settings.beam_shift:
            self._require_point_complete(settings.beam_shift, 'beam_shift')
            if settings.beam_shift.x is not None:
                self.set_beam_shift(float(settings.beam_shift.x), float(settings.beam_shift.y), **kwargs)

        if settings.beam_tilt:
            self._require_point_complete(settings.beam_tilt, 'beam_tilt')
            if settings.beam_tilt.x is not None:
                self.set_beam_tilt(float(settings.beam_tilt.x), float(settings.beam_tilt.y), **kwargs)

        if settings.condenser_stigmation:
            self._require_point_complete(settings.condenser_stigmation, 'condenser_stigmation')
            if settings.condenser_stigmation.x is not None:
                self.set_condenser_stigmation(float(settings.condenser_stigmation.x),
                                              float(settings.condenser_stigmation.y), **kwargs)

        if settings.gun_tilt:
            self._require_point_complete(settings.gun_tilt, 'gun_tilt')
            if settings.gun_tilt.x is not None:
                self.set_gun_tilt(float(settings.gun_tilt.x), float(settings.gun_tilt.y), **kwargs)

    def perform_beam_action(self, action: str, **kwargs) -> None:
        """Helper: Handles procedural beam commands."""
        # Vendors override this to implement logic
        if action == "DEGAUSS":
            logger.warning("[BEAM] Degauss requested but not implemented.")
        elif action == "NORMALIZE":
            logger.warning("[BEAM] Normalize requested but not implemented.")
        elif action == "ALIGN_GUN":
            logger.warning("[BEAM] Gun Align requested but not implemented.")
        else:
            logger.warning(f"[BEAM] Unknown action '{action}'")

    # --- Orchestrator Layer ---

    def execute_beam_control(self, request: BeamControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid BeamControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")

        intent_str = " ".join(intent) if intent else "No Operation"
        logger.info(f"[BEAM] Control: {intent_str}")

        # Safety Check
        sys = self.system_settings.beam_system
        if sys and request.target:
            check = sys.is_safe_beam(request.target)
            if not check:
                raise RuntimeError(f"Unsafe beam settings: {check.reasons}")

        exec_opts = request.extra.options if request.extra else {}

        # 1. Action (Verb)
        if request.action:
            self.perform_beam_action(request.action, **exec_opts)

        # 2. Settings (Noun)
        if request.target:
            self.apply_beam_settings(request.target, **exec_opts)

    # =========================================================================
    # 5. Projection Control (Imaging/Optics)
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def get_optical_mode(self) -> str:
        """Get optical mode (e.g., 'IMAGING', 'DIFFRACTION')."""
        pass

    @abstractmethod
    def get_magnification(self) -> Optional[int]:
        """Get Magnification index."""
        pass

    @abstractmethod
    def get_camera_length(self) -> Optional[Quantity]:
        """Get Camera Length (mm)."""
        pass

    @abstractmethod
    def get_defocus(self) -> Optional[Quantity]:
        """Get Defocus (nm)."""
        pass

    @abstractmethod
    def get_screen_position(self) -> str:
        """Get Screen Position ('UP', 'DOWN')."""
        pass

    @abstractmethod
    def get_objective_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Objective Stigmator Coils (x, y)."""
        pass

    @abstractmethod
    def get_diffraction_stigmation(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Diffraction Stigmator Coils (x, y)."""
        pass

    @abstractmethod
    def get_image_shift(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Image Shift Coils (x, y)."""
        pass

    @abstractmethod
    def get_diffraction_shift(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Diffraction Shift Coils (x, y)."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_optical_mode(self, mode: str, **kwargs) -> None:
        """Set optical mode."""
        pass

    @abstractmethod
    def set_magnification(self, index: int, **kwargs) -> None:
        """Set Magnification."""
        pass

    @abstractmethod
    def set_camera_length(self, length: Quantity, **kwargs) -> None:
        """Set Camera Length (mm)."""
        pass

    @abstractmethod
    def set_defocus(self, defocus: Quantity, **kwargs) -> None:
        """Set Defocus (nm)."""
        pass

    @abstractmethod
    def set_screen_position(self, position: str, **kwargs) -> None:
        """Set Screen Position ('UP', 'DOWN')."""
        pass

    @abstractmethod
    def set_objective_stigmation(self, x: float, y: float, **kwargs) -> None:
        """Set Objective Stigmator Coils."""
        pass

    @abstractmethod
    def set_diffraction_stigmation(self, x: float, y: float, **kwargs) -> None:
        """Set Diffraction Stigmator Coils."""
        pass

    @abstractmethod
    def set_image_shift(self, x: float, y: float, **kwargs) -> None:
        """Set Image Shift Coils."""
        pass

    @abstractmethod
    def set_diffraction_shift(self, x: float, y: float, **kwargs) -> None:
        """Set Diffraction Shift Coils."""
        pass

    # --- Helper Layer ---

    def get_projection_settings(self) -> ProjectionSettings:
        """Helper: Aggregates atomic projection state."""
        obj_stig = self.get_objective_stigmation()
        diff_stig = self.get_diffraction_stigmation()
        img_shift = self.get_image_shift()
        diff_shift = self.get_diffraction_shift()

        return ProjectionSettings(
            optical_mode=self.get_optical_mode(),
            magnification=self.get_magnification(),
            camera_length=self.get_camera_length(),
            defocus=self.get_defocus(),
            screen_position=self.get_screen_position(),

            objective_stigmation=Point(x=obj_stig[0], y=obj_stig[1]) if obj_stig[0] is not None else None,
            diffraction_stigmation=Point(x=diff_stig[0], y=diff_stig[1]) if diff_stig[0] is not None else None,
            image_shift=Point(x=img_shift[0], y=img_shift[1]) if img_shift[0] is not None else None,
            diffraction_shift=Point(x=diff_shift[0], y=diff_shift[1]) if diff_shift[0] is not None else None
        )

    def apply_projection_settings(self, settings: ProjectionSettings, **kwargs) -> None:
        """Helper: Applies partial projection settings."""
        if settings.optical_mode is not None:
            self.set_optical_mode(settings.optical_mode, **kwargs)
        if settings.magnification is not None:
            self.set_magnification(settings.magnification, **kwargs)
        if settings.camera_length is not None:
            self.set_camera_length(settings.camera_length, **kwargs)
        if settings.defocus is not None:
            self.set_defocus(settings.defocus, **kwargs)
        if settings.screen_position is not None:
            self.set_screen_position(settings.screen_position, **kwargs)

        if settings.objective_stigmation:
            self._require_point_complete(settings.objective_stigmation, 'objective_stigmation')
            if settings.objective_stigmation.x is not None:
                self.set_objective_stigmation(float(settings.objective_stigmation.x),
                                              float(settings.objective_stigmation.y), **kwargs)

        if settings.diffraction_stigmation:
            self._require_point_complete(settings.diffraction_stigmation, 'diffraction_stigmation')
            if settings.diffraction_stigmation.x is not None:
                self.set_diffraction_stigmation(float(settings.diffraction_stigmation.x),
                                                float(settings.diffraction_stigmation.y), **kwargs)

        if settings.image_shift:
            self._require_point_complete(settings.image_shift, 'image_shift')
            if settings.image_shift.x is not None:
                self.set_image_shift(float(settings.image_shift.x), float(settings.image_shift.y), **kwargs)

        if settings.diffraction_shift:
            self._require_point_complete(settings.diffraction_shift, 'diffraction_shift')
            if settings.diffraction_shift.x is not None:
                self.set_diffraction_shift(float(settings.diffraction_shift.x), float(settings.diffraction_shift.y),
                                           **kwargs)

    def perform_projection_action(self, action: str, **kwargs) -> None:
        if action == "NORMALIZE":
            logger.warning("[PROJ] Normalize requested but not implemented.")
        else:
            logger.warning(f"[PROJ] Unknown action '{action}'")

    # --- Orchestrator Layer ---

    def execute_projection_control(self, request: ProjectionControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid ProjectionControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")

        intent_str = " ".join(intent) if intent else "No Operation"
        logger.info(f"[PROJ] Control: {intent_str}")

        sys = self.system_settings.projection_system
        if sys and request.target:
            check = sys.is_safe_projection(request.target)
            if not check:
                raise RuntimeError(f"Unsafe projection settings: {check.reasons}")

        exec_opts = request.extra.options if request.extra else {}

        if request.action:
            self.perform_projection_action(request.action, **exec_opts)

        if request.target:
            self.apply_projection_settings(request.target, **exec_opts)

    # =========================================================================
    # 6. Detector Control & Acquisition
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def list_detectors(self) -> List[str]:
        """List available detector IDs."""
        pass

    @abstractmethod
    def get_active_detector_ids(self) -> List[str]:
        """List currently active detectors."""
        pass

    @abstractmethod
    def get_primary_detector_id(self) -> Optional[str]:
        """Get the ID of the primary detector."""
        pass

    @abstractmethod
    def get_detector_inserted(self, detector_id: str) -> bool:
        """Return True if detector is mechanically inserted."""
        pass

    @abstractmethod
    def get_detector_exposure(self, detector_id: str) -> Optional[Quantity]:
        """Get exposure time (ms)."""
        pass

    @abstractmethod
    def get_detector_binning_index(self, detector_id: str) -> Optional[int]:
        """Get binning index (scalar)."""
        pass

    @abstractmethod
    def get_detector_binning_xy(self, detector_id: str) -> Optional[Tuple[int, int]]:
        """Get binning tuple (x, y)."""
        pass

    @abstractmethod
    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        """Get Region of Interest."""
        pass

    @abstractmethod
    def get_detector_gain_index(self, detector_id: str) -> Optional[int]:
        """Get gain index."""
        pass

    @abstractmethod
    def get_detector_offset_index(self, detector_id: str) -> Optional[int]:
        """Get offset index."""
        pass

    @abstractmethod
    def get_detector_digital_rotation(self, detector_id: str) -> Optional[Quantity]:
        """Get digital rotation (deg)."""
        pass

    @abstractmethod
    def get_detector_frame_integration(self, detector_id: str) -> Optional[int]:
        """Get frame integration count."""
        pass

    @abstractmethod
    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        """Get estimated frame rate (Hz)."""
        pass

    @abstractmethod
    def get_detector_total_frames(self, detector_id: str) -> Optional[int]:
        """Get total frames (movie mode)."""
        pass

    @abstractmethod
    def get_detector_readout_mode(self, detector_id: str) -> Optional[str]:
        """Get readout mode (e.g. 'LINEAR')."""
        pass

    @abstractmethod
    def get_detector_shutter_mode(self, detector_id: str) -> Optional[str]:
        """Get shutter mode (e.g. 'PRE_SPECIMEN')."""
        pass

    @abstractmethod
    def get_detector_save_frames(self, detector_id: str) -> Optional[bool]:
        """Get save frames flag."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_detector_insertion(self, detector_id: str, inserted: bool, **kwargs) -> None:
        """
        Atomic: Insert (True) or Retract (False) the detector.
        """
        pass

    @abstractmethod
    def set_detector_exposure(self, detector_id: str, exposure: Quantity, **kwargs) -> None:
        """
        Atomic: Set exposure time.
        Unit: ms
        """
        pass

    @abstractmethod
    def set_detector_binning_index(self, detector_id: str, index: int, **kwargs) -> None:
        """
        Atomic: Set binning by index (e.g., 0=1x1, 1=2x2).
        """
        pass

    @abstractmethod
    def set_detector_binning_xy(self, detector_id: str, binning: Tuple[int, int], **kwargs) -> None:
        """
        Atomic: Set explicit binning (x, y).
        """
        pass

    @abstractmethod
    def set_detector_roi(self, detector_id: str, roi: Optional[ROI], **kwargs) -> None:
        """
        Atomic: Set Region of Interest (sub-area readout).
        """
        pass

    @abstractmethod
    def set_detector_gain_index(self, detector_id: str, index: int, **kwargs) -> None:
        """
        Atomic: Set gain index.
        """
        pass

    @abstractmethod
    def set_detector_offset_index(self, detector_id: str, index: int, **kwargs) -> None:
        """
        Atomic: Set offset index.
        """
        pass

    @abstractmethod
    def set_detector_digital_rotation(self, detector_id: str, angle: Quantity, **kwargs) -> None:
        """
        Atomic: Set digital rotation.
        Unit: deg
        """
        pass

    @abstractmethod
    def set_detector_frame_integration(self, detector_id: str, count: int, **kwargs) -> None:
        """
        Atomic: Set frame integration count (hardware averaging).
        """
        pass

    @abstractmethod
    def set_detector_frame_rate(self, detector_id: str, rate: Quantity, **kwargs) -> None:
        """
        Atomic: Set target frame rate.
        Unit: Hz
        """
        pass

    @abstractmethod
    def set_detector_total_frames(self, detector_id: str, count: int, **kwargs) -> None:
        """
        Atomic: Set total frames to capture (Movie Mode).
        """
        pass

    @abstractmethod
    def set_detector_readout_mode(self, detector_id: str, mode: str, **kwargs) -> None:
        """
        Atomic: Set readout mode (e.g., 'LINEAR', 'COUNTING').
        """
        pass

    @abstractmethod
    def set_detector_shutter_mode(self, detector_id: str, mode: str, **kwargs) -> None:
        """
        Atomic: Set shutter mode (e.g., 'PRE_SPECIMEN').
        """
        pass

    @abstractmethod
    def set_detector_save_frames(self, detector_id: str, save: bool, **kwargs) -> None:
        """
        Atomic: Set flag to save individual frames in movie mode.
        """
        pass

    @abstractmethod
    def acquire_image(self, request: AcquisitionRequest, **kwargs) -> MicroscopeImage:
        """
        Atomic: Execute raw Hardware Acquisition Cycle.
        Responsibility:
          1. Expose sensor (using previously applied settings).
          2. Block/Wait for readout.
          3. Return raw MicroscopeImage with data.
        """
        pass

    # --- Helper Layer ---

    def get_detector_settings(self, detector_id: str) -> DetectorSettings:
        """Helper: Aggregates atomic detector state."""
        return DetectorSettings(
            detector_id=detector_id,
            inserted=self.get_detector_inserted(detector_id),
            exposure=self.get_detector_exposure(detector_id),
            binning_index=self.get_detector_binning_index(detector_id),
            binning_xy=self.get_detector_binning_xy(detector_id),
            roi=self.get_detector_roi(detector_id),
            gain_index=self.get_detector_gain_index(detector_id),
            offset_index=self.get_detector_offset_index(detector_id),
            digital_rotation=self.get_detector_digital_rotation(detector_id),
            frame_integration=self.get_detector_frame_integration(detector_id),
            frame_rate=self.get_detector_frame_rate(detector_id),
            total_frames=self.get_detector_total_frames(detector_id),
            readout_mode=self.get_detector_readout_mode(detector_id),
            shutter_mode=self.get_detector_shutter_mode(detector_id),
            save_frames=self.get_detector_save_frames(detector_id)
        )

    def apply_detector_settings(self, detector_id: str, settings: DetectorSettings, **kwargs) -> None:
        """
        Helper: Applies a partial detector configuration.
        """
        if settings.inserted is not None:
            self.set_detector_insertion(detector_id, settings.inserted, **kwargs)
        if settings.exposure is not None:
            self.set_detector_exposure(detector_id, settings.exposure, **kwargs)
        if settings.binning_index is not None:
            self.set_detector_binning_index(detector_id, settings.binning_index, **kwargs)
        if settings.binning_xy is not None:
            self.set_detector_binning_xy(detector_id, settings.binning_xy, **kwargs)
        if settings.roi is not None:
            self.set_detector_roi(detector_id, settings.roi, **kwargs)
        if settings.gain_index is not None:
            self.set_detector_gain_index(detector_id, settings.gain_index, **kwargs)
        if settings.offset_index is not None:
            self.set_detector_offset_index(detector_id, settings.offset_index, **kwargs)
        if settings.digital_rotation is not None:
            self.set_detector_digital_rotation(detector_id, settings.digital_rotation, **kwargs)
        if settings.frame_integration is not None:
            self.set_detector_frame_integration(detector_id, settings.frame_integration, **kwargs)
        if settings.frame_rate is not None:
            self.set_detector_frame_rate(detector_id, settings.frame_rate, **kwargs)
        if settings.total_frames is not None:
            self.set_detector_total_frames(detector_id, settings.total_frames, **kwargs)
        if settings.readout_mode is not None:
            self.set_detector_readout_mode(detector_id, settings.readout_mode, **kwargs)
        if settings.shutter_mode is not None:
            self.set_detector_shutter_mode(detector_id, settings.shutter_mode, **kwargs)
        if settings.save_frames is not None:
            self.set_detector_save_frames(detector_id, settings.save_frames, **kwargs)

    def perform_detector_action(self, detector_id: str, action: str, **kwargs) -> None:
        """Helper: Handles detector maintenance (Cooldown, etc)."""
        if action == "INSERT":
            self.set_detector_insertion(detector_id, True, **kwargs)
        elif action == "RETRACT":
            self.set_detector_insertion(detector_id, False, **kwargs)
        elif action == "COOLDOWN":
            logger.warning(f"[{detector_id}] Cooldown requested but not implemented.")
        elif action == "WARMUP":
            logger.warning(f"[{detector_id}] Warmup requested but not implemented.")
        else:
            logger.warning(f"[{detector_id}] Unknown action '{action}'")

    def perform_capture(self, request: AcquisitionRequest, **kwargs) -> MicroscopeImage:
        """
        Helper: The 'Brain' of the acquisition process.

        Responsibilities:
        1. Routes canonical settings to the hardware (via apply_detector_settings).
        2. Triggers the Atomic capture.
        3. Orchestrates metadata enhancement and state snapshotting.

        This method is the primary override point for vendors needing custom
        synchronization or pre-flight checks before the shutter opens.
        """
        # 1. Apply Settings
        if request.detector:
            self.apply_detector_settings(request.detector_id, request.detector, **kwargs)

        # 2. Trigger Atomic Capture
        image = self.acquire_image(request, **kwargs)

        # 3. Enhance Metadata (State Snapshot)
        if image.metadata is None:
            image.metadata = MicroscopeImageMetadata()

        if image.metadata.microscope_state is None:
            try:
                # Capture the full context of the microscope at the moment of image creation
                image.metadata.microscope_state = self.get_full_state()
            except Exception as e:
                # LENIENT: Do not fail the acquisition if metadata/telemetry fails
                logger.warning(f"[{request.detector_id}] Failed to capture state for metadata: {e}")

        return image

    # --- Orchestrator Layer ---

    def execute_detector_control(self, request: DetectorControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid DetectorControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")

        intent_str = " ".join(intent) if intent else "No Operation"
        logger.info(f"[DET] Executing Control on {request.detector_id}: {intent_str}")

        sys = self.system_settings.detector_system
        if sys and request.target:
            if not sys.is_supported(request.target):
                raise RuntimeError("Detector settings not supported")

        exec_opts = request.extra.options if request.extra else {}

        # 1. Action (Delegated to Helper)
        if request.action:
            self.perform_detector_action(request.detector_id, request.action, **exec_opts)

        # 2. Target (Delegated to Helper)
        if request.target:
            self.apply_detector_settings(request.detector_id, request.target, **exec_opts)

    def execute_acquisition(self, request: AcquisitionRequest) -> MicroscopeImage:
        """
        Orchestrator: Handle AcquisitionRequest (Capture Image).

        Responsibilities:
        1. Validate Intent (System Limits).
        2. Delegate Execution to Helper Layer (perform_capture).
        3. Persist Data (Save to Disk).
        """
        # 1. Validate Intent
        if not request.validate():
            raise ValueError(f"Invalid AcquisitionRequest: {request}")

        det_id = request.detector_id
        logger.info(f"[ACQ] Starting acquisition on '{det_id}'")

        # 2. Check Hardware Capabilities
        sys = self.system_settings.detector_system
        if sys and request.detector:
            check = sys.is_supported(request.detector)
            if not check:
                raise RuntimeError(f"Acquisition settings not supported: {check.reasons}")

        exec_opts = request.extra.options if request.extra else {}

        # 3. Delegate to Helper Layer (The Brain)
        image = self.perform_capture(request, **exec_opts)

        # 4. Save Logic (Framework Persistence)
        output_cfg = request.image or self._settings.image
        if output_cfg and output_cfg.path:
            try:
                save_path = Path(output_cfg.path)
                # Auto-generate filename if directory or empty
                if save_path.is_dir() or (not save_path.suffix):
                    fname = f"Image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    save_path = save_path / fname

                final_path = image.save(save_path, file_format=output_cfg.file_format)
                logger.info(f"[ACQ] Image saved to: {final_path}")
            except Exception as e:
                logger.error(f"[ACQ] Failed to save image: {e}")

        return image

    # =========================================================================
    # 7. Scan Control (STEM)
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def get_scan_mode(self) -> str:
        """Get scan engine mode."""
        pass

    @abstractmethod
    def get_scan_active(self) -> bool:
        """Return True if scanning is currently active."""
        pass

    @abstractmethod
    def get_scan_width(self) -> Optional[int]:
        """Get scan width in pixels. Returns None if unknown."""
        pass

    @abstractmethod
    def get_scan_height(self) -> Optional[int]:
        """Get scan height in pixels. Returns None if unknown."""
        pass

    @abstractmethod
    def get_scan_pixel_dwell(self) -> Optional[Quantity]:
        """Get pixel dwell time (us)."""
        pass

    @abstractmethod
    def get_scan_flyback(self) -> Optional[Quantity]:
        """Get flyback time (us)."""
        pass

    @abstractmethod
    def get_scan_rotation(self) -> Optional[Quantity]:
        """Get scan rotation (deg)."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_scan_mode(self, mode: str, **kwargs) -> None:
        """Atomic: Set scan engine mode."""
        pass

    @abstractmethod
    def set_scan_active(self, active: bool, **kwargs) -> None:
        """Atomic: Start (True) or Stop (False) the scan engine."""
        pass

    @abstractmethod
    def set_scan_width(self, px: int, **kwargs) -> None:
        """Atomic: Set scan width (px)."""
        pass

    @abstractmethod
    def set_scan_height(self, px: int, **kwargs) -> None:
        """Atomic: Set scan height (px)."""
        pass

    @abstractmethod
    def set_scan_pixel_dwell(self, time: Quantity, **kwargs) -> None:
        """
        Atomic: Set pixel dwell time.
        Unit: us
        """
        pass

    @abstractmethod
    def set_scan_flyback(self, time: Quantity, **kwargs) -> None:
        """
        Atomic: Set flyback time.
        Unit: us
        """
        pass

    @abstractmethod
    def set_scan_rotation(self, angle: Quantity, **kwargs) -> None:
        """
        Atomic: Set scan rotation.
        Unit: deg
        """
        pass

    # --- Helper Layer ---

    def get_scan_settings(self) -> ScanSettings:
        """Helper: Aggregates atomic scan engine state."""
        return ScanSettings(
            scan_mode=self.get_scan_mode(),
            active=self.get_scan_active(),
            width_px=self.get_scan_width(),
            height_px=self.get_scan_height(),
            pixel_dwell_time=self.get_scan_pixel_dwell(),
            flyback_time=self.get_scan_flyback(),
            scan_rotation=self.get_scan_rotation()
        )

    def apply_scan_settings(self, settings: ScanSettings, **kwargs) -> None:
        """Helper: Applies a partial scan configuration."""
        if settings.scan_mode is not None:
            self.set_scan_mode(settings.scan_mode, **kwargs)
        if settings.active is not None:
            self.set_scan_active(settings.active, **kwargs)
        if settings.width_px is not None:
            self.set_scan_width(settings.width_px, **kwargs)
        if settings.height_px is not None:
            self.set_scan_height(settings.height_px, **kwargs)
        if settings.pixel_dwell_time is not None:
            self.set_scan_pixel_dwell(settings.pixel_dwell_time, **kwargs)
        if settings.flyback_time is not None:
            self.set_scan_flyback(settings.flyback_time, **kwargs)
        if settings.scan_rotation is not None:
            self.set_scan_rotation(settings.scan_rotation, **kwargs)

    def perform_scan_action(self, action: str, **kwargs) -> None:
        """Helper: Handles Start/Stop logic."""
        if action == "START":
            self.set_scan_active(True, **kwargs)
        elif action == "STOP":
            self.set_scan_active(False, **kwargs)
        elif action == "SINGLE_FRAME":
            # Vendor override point for single-shot logic
            logger.warning("[SCAN] SINGLE_FRAME generic fallback: Starting continuous scan.")
            self.set_scan_active(True, **kwargs)

    # --- Orchestrator Layer ---

    def execute_scan_control(self, request: ScanControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid ScanControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")

        logger.info(f"[SCAN] Control: {' '.join(intent)}")

        exec_opts = request.extra.options if request.extra else {}

        # 1. Settings
        if request.target:
            # (Safety checks...)
            self.apply_scan_settings(request.target, **exec_opts)

        # 2. Action (Delegated to Helper)
        if request.action:
            self.perform_scan_action(request.action, **exec_opts)

    # =========================================================================
    # 8. Vacuum Control
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def get_column_valve_state(self) -> str:
        """Atomic: Get Column Valve (V7/V4) state ('OPEN', 'CLOSED', 'UNKNOWN')."""
        pass

    @abstractmethod
    def get_gun_valve_state(self) -> str:
        """Atomic: Get Gun Valve (V1) state ('OPEN', 'CLOSED', 'UNKNOWN')."""
        pass

    @abstractmethod
    def get_turbo_pump_state(self) -> str:
        """Atomic: Get Turbo Pump state ('ON', 'OFF', 'UNKNOWN')."""
        pass

    @abstractmethod
    def get_column_pressure(self) -> Optional[Quantity]:
        """Atomic: Get Column Pressure (Pa)."""
        pass

    @abstractmethod
    def get_gun_pressure(self) -> Optional[Quantity]:
        """Atomic: Get Gun Pressure (Pa)."""
        pass

    @abstractmethod
    def get_buffer_tank_pressure(self) -> Optional[Quantity]:
        """Atomic: Get Buffer Tank Pressure (Pa)."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_column_valve_state(self, state: str, **kwargs) -> None:
        """
        Atomic: Set Column Valve state.
        Values: 'OPEN', 'CLOSED'
        """
        pass

    @abstractmethod
    def set_gun_valve_state(self, state: str, **kwargs) -> None:
        """
        Atomic: Set Gun Valve state.
        Values: 'OPEN', 'CLOSED'
        """
        pass

    @abstractmethod
    def set_turbo_pump_state(self, state: str, **kwargs) -> None:
        """
        Atomic: Set Turbo Pump state.
        Values: 'ON', 'OFF'
        """
        pass

    # --- Helper Layer ---

    def get_vacuum_settings(self) -> VacuumSettings:
        """Helper: Aggregates atomic vacuum state."""
        return VacuumSettings(
            column_valve_state=self.get_column_valve_state(),
            gun_valve_state=self.get_gun_valve_state(),
            turbo_pump_state=self.get_turbo_pump_state(),
            column_pressure=self.get_column_pressure(),
            gun_pressure=self.get_gun_pressure(),
            buffer_tank_pressure=self.get_buffer_tank_pressure()
        )

    def apply_vacuum_settings(self, settings: VacuumSettings, **kwargs) -> None:
        """Helper: Applies partial vacuum configuration."""
        if settings.column_valve_state is not None:
            self.set_column_valve_state(settings.column_valve_state, **kwargs)
        if settings.gun_valve_state is not None:
            self.set_gun_valve_state(settings.gun_valve_state, **kwargs)
        if settings.turbo_pump_state is not None:
            self.set_turbo_pump_state(settings.turbo_pump_state, **kwargs)

    def perform_vacuum_action(self, action: str, **kwargs) -> None:
        if action == "VENT":
            logger.warning("[VAC] Vent requested but not implemented.")
        elif action == "CYCLE":
            logger.warning("[VAC] Cycle requested but not implemented.")

    # --- Orchestrator Layer ---

    def execute_vacuum_control(self, request: VacuumControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid VacuumControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")
        if request.force:
            intent.append("(FORCE)")

        logger.info(f"[VAC] Control: {' '.join(intent)}")

        exec_opts = request.extra.options if request.extra else {}

        if request.force is not None:
            exec_opts['force'] = request.force

        if request.action:
            self.perform_vacuum_action(request.action, **exec_opts)

        if request.target:
            self.apply_vacuum_settings(request.target, **exec_opts)

    # =========================================================================
    # 9. Aperture Control
    # =========================================================================

    # --- Atomic Getters ---

    @abstractmethod
    def list_apertures(self) -> List[str]:
        """Atomic: List supported aperture mechanism IDs."""
        pass

    @abstractmethod
    def get_aperture_inserted(self, aperture_id: str) -> bool:
        """Atomic: Return True if aperture is in the beam path."""
        pass

    @abstractmethod
    def get_aperture_size_index(self, aperture_id: str) -> Optional[int]:
        """Atomic: Get the current size index."""
        pass

    @abstractmethod
    def get_aperture_size_label(self, aperture_id: str) -> Optional[str]:
        """Atomic: Get human-readable size label (Read-Only metadata)."""
        pass

    @abstractmethod
    def get_aperture_position(self, aperture_id: str) -> Optional[Point]:
        """Atomic: Get the mechanical XY position."""
        pass

    # --- Atomic Setters ---

    @abstractmethod
    def set_aperture_inserted(self, aperture_id: str, inserted: bool, **kwargs) -> None:
        """
        Atomic: Insert or Retract the mechanism.
        """
        pass

    @abstractmethod
    def set_aperture_size_index(self, aperture_id: str, index: int, **kwargs) -> None:
        """
        Atomic: Select a specific hole size by index.
        """
        pass

    @abstractmethod
    def set_aperture_position(self, aperture_id: str, x: float, y: float, **kwargs) -> None:
        """
        Atomic: Align the aperture mechanism mechanically.
        """
        pass

    # --- Helper Layer ---

    def get_aperture_settings(self, aperture_id: str) -> ApertureSettings:
        """Helper: Aggregates atomic aperture state."""
        return ApertureSettings(
            aperture_id=aperture_id,
            inserted=self.get_aperture_inserted(aperture_id),
            size_index=self.get_aperture_size_index(aperture_id),
            size_label=self.get_aperture_size_label(aperture_id),
            position=self.get_aperture_position(aperture_id)
        )

    def apply_aperture_settings(self, aperture_id: str, settings: ApertureSettings, **kwargs) -> None:
        """
        Helper: Applies a fully resolved aperture configuration.
        """
        if settings.inserted is not None:
            self.set_aperture_inserted(aperture_id, settings.inserted, **kwargs)

        if settings.size_index is not None:
            self.set_aperture_size_index(aperture_id, settings.size_index, **kwargs)

        if settings.position is not None:
            # We enforce that if position is provided, it must be complete (X and Y)
            # or the driver handles partials. Here we pass what we have.
            x_val = settings.position.x if settings.position.x is not None else 0.0
            y_val = settings.position.y if settings.position.y is not None else 0.0
            self.set_aperture_position(aperture_id, x_val, y_val, **kwargs)

    def perform_aperture_action(self, aperture_id: str, action: str, **kwargs) -> None:
        if action == "RESET":
            logger.warning(f"[{aperture_id}] Reset requested but not implemented.")
        elif action == "CALIBRATE":
            logger.warning(f"[{aperture_id}] Calibrate requested but not implemented.")

    # --- Orchestrator Layer ---

    def execute_aperture_control(self, request: ApertureControlRequest) -> None:
        """
        Orchestrator: Handle ApertureControlRequest.
        Handles ID matching, safety checks, and relative position logic.
        """
        if not request.validate():
            raise ValueError(f"Invalid ApertureControlRequest: {request}")

        intent = []
        if request.action:
            intent.append(f"Action={request.action}")
        if request.target:
            intent.append(f"Target={self._summarize_patch(request.target)}")
        if request.relative:
            intent.append("(Relative)")

        logger.info(f"[APT] Control on '{request.aperture_id}': {' '.join(intent)}")

        sys = self.system_settings.aperture_system
        if sys:
            check = sys.is_supported(request.target)
            if not check:
                raise RuntimeError(f"Aperture request rejected: {check.reasons}")

        target = request.target
        a_id = request.aperture_id

        # Handle Relative Position Logic
        if request.relative and target.position:
            # We need the current position to calculate the delta
            current_pos = self.get_aperture_position(a_id)
            if current_pos is None:
                raise RuntimeError(f"Relative move failed: Current position of '{a_id}' is unknown")

            # Calculate new absolute position (manual vector addition)
            cur_x = current_pos.x if current_pos.x is not None else 0.0
            cur_y = current_pos.y if current_pos.y is not None else 0.0

            new_x = cur_x + (target.position.x if target.position.x is not None else 0.0)
            new_y = cur_y + (target.position.y if target.position.y is not None else 0.0)

            # Update target with absolute position
            # We use replace() to avoid mutating the original request object
            target = replace(target, position=replace(target.position, x=new_x, y=new_y))

        exec_opts = request.extra.options if request.extra else {}

        if request.action:
            self.perform_aperture_action(request.aperture_id, request.action, **exec_opts)

        if request.target:
            self.apply_aperture_settings(request.aperture_id, target, **exec_opts)
