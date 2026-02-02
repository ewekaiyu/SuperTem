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
            apertures=self.get_all_apertures(),
            detectors={d_id: self.get_detector_settings(d_id)
                       for d_id in self.list_detectors()},
            active_detector_ids=self.get_active_detector_ids(),
            primary_detector_id=self.get_primary_detector_id()
        )

    # =========================================================================
    # 3. Stage Control (Motion)
    # =========================================================================

    # --- Atomic Layer ---
    @abstractmethod
    def get_stage_position(self) -> Optional[StagePosition]:
        """Atomic: Read current physical stage coordinates. Returns None if unknown."""
        pass

    @abstractmethod
    def get_stage_coordinate_system(self) -> Optional[str]:
        """Get the current reference frame name."""
        pass

    @abstractmethod
    def move_stage_absolute(
            self,
            target: StagePosition,
            drive_type: str = "default",
            wait: bool = True,
            tolerance_nm: float = 200.0,
            tolerance_deg: float = 0.1,
            max_retries: int = 3
    ) -> None:
        """Atomic: Move stage to a specific absolute coordinate."""
        pass

    @abstractmethod
    def set_stage_coordinate_system(self, system_id: str) -> None:
        """Atomic: Set the reference frame."""
        pass

    @abstractmethod
    def stop_stage(self) -> None:
        """Atomic: Immediately halt all stage motion axes."""
        pass

    @abstractmethod
    def home_stage(self) -> None:
        """Atomic: Return stage to its mechanical origin/zero position."""
        pass

    # --- Logic Layer ---
    def execute_stage_move(self, request: StageMoveRequest) -> None:
        """
        Orchestrator: Handle StageMoveRequest.
        Validates safety, handles relative logic, and interpolates large moves.
        """
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

            # 3. Safe Execution (interpolates if needed)
            self.safe_move_stage(target_abs, drive_type=request.drive_type, wait=request.wait_for_settle)
        else:
            self.move_stage_absolute(target_abs, drive_type=request.drive_type, wait=request.wait_for_settle)

    def execute_stage_control(self, request: StageControlRequest) -> None:
        """Orchestrator: Handle StageControlRequest (STOP, HOME)."""
        if not request.validate():
            raise ValueError(f"Invalid StageControlRequest: {request}")

        logger.info(f"[STAGE] Executing Control: {request.action}")

        if request.action == "STOP":
            self.stop_stage()
        elif request.action == "HOME":
            self.home_stage()
        # Drivers can extend for RESET_ERROR etc.

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Getters ---
    @abstractmethod
    def get_acceleration_voltage(self) -> Optional[Quantity]:
        """Get High Tension (kV). Returns None if unknown."""
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

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Get Beam Blank Status. True=Blanked."""
        pass

    @abstractmethod
    def get_probe_mode(self) -> Optional[str]:
        """Get probe mode (e.g. 'Microprobe', 'Nanoprobe'). Returns None if unknown."""
        pass

    # --- Atomic Setters ---
    @abstractmethod
    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        """Set High Tension (kV)."""
        pass

    @abstractmethod
    def set_beam_current(self, current: Quantity) -> None:
        """Set Beam Current (nA)."""
        pass

    @abstractmethod
    def set_emission_current(self, current: Quantity) -> None:
        """Set Gun Emission Current (uA)."""
        pass

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """Set Spot Size Index."""
        pass

    @abstractmethod
    def set_convergence_angle(self, angle: Quantity) -> None:
        """Set Convergence Angle (mrad)."""
        pass

    @abstractmethod
    def set_beam_shift(self, x: float, y: float) -> None:
        """Set Beam Shift Coils."""
        pass

    @abstractmethod
    def set_beam_tilt(self, x: float, y: float) -> None:
        """Set Beam Tilt Coils."""
        pass

    @abstractmethod
    def set_condenser_stigmation(self, x: float, y: float) -> None:
        """Set Condenser Stigmator Coils."""
        pass

    @abstractmethod
    def set_gun_tilt(self, x: float, y: float) -> None:
        """Set Gun Tilt Alignment."""
        pass

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """Set Beam Blanker. True = Blank (Block)."""
        pass

    @abstractmethod
    def set_probe_mode(self, mode: str) -> None:
        """Set probe mode."""
        pass

    # --- Logic Layer ---
    def get_beam_settings(self) -> BeamSettings:
        """Aggregator: returns full BeamSettings snapshot."""
        bs = self.get_beam_shift()
        bt = self.get_beam_tilt()
        cs = self.get_condenser_stigmation()
        gt = self.get_gun_tilt()
        return BeamSettings(
            mode=self.get_mode(),
            voltage=self.get_acceleration_voltage(),
            beam_current=self.get_beam_current(),
            emission_current=self.get_emission_current(),
            spot_size=self.get_spot_size(),
            convergence_angle=self.get_convergence_angle(),
            probe_mode=self.get_probe_mode(),
            is_blanked=self.get_beam_blank(),
            beam_shift=Point(x=bs[0], y=bs[1]),
            beam_tilt=Point(x=bt[0], y=bt[1]),
            condenser_stigmation=Point(x=cs[0], y=cs[1]),
            gun_tilt=Point(x=gt[0], y=gt[1])
        )

    def apply_beam_settings(self, settings: BeamSettings) -> None:
        """
        Helper: Applies a partial beam configuration.

        Notes:
        - Canonical fields are applied when not None.
        - For 2D coil fields (Point), partial specification is rejected (strict).
        - Vendor-specific extras are validated/applied by vendor overrides.
        """
        if settings.mode is not None:
            self.set_mode(settings.mode)
        if settings.voltage is not None:
            self.set_acceleration_voltage(settings.voltage)
        if settings.beam_current is not None:
            self.set_beam_current(settings.beam_current)
        if settings.emission_current is not None:
            self.set_emission_current(settings.emission_current)
        if settings.spot_size is not None:
            self.set_spot_size(settings.spot_size)
        if settings.convergence_angle is not None:
            self.set_convergence_angle(settings.convergence_angle)
        if settings.probe_mode is not None:
            self.set_probe_mode(settings.probe_mode)
        if settings.is_blanked is not None:
            self.set_beam_blank(settings.is_blanked)

        if settings.beam_shift:
            self._require_point_complete(settings.beam_shift, 'beam_shift')
            if settings.beam_shift.x is not None:
                self.set_beam_shift(float(settings.beam_shift.x), float(settings.beam_shift.y))

        if settings.beam_tilt:
            self._require_point_complete(settings.beam_tilt, 'beam_tilt')
            if settings.beam_tilt.x is not None:
                self.set_beam_tilt(float(settings.beam_tilt.x), float(settings.beam_tilt.y))

        if settings.condenser_stigmation:
            self._require_point_complete(settings.condenser_stigmation, 'condenser_stigmation')
            if settings.condenser_stigmation.x is not None:
                self.set_condenser_stigmation(float(settings.condenser_stigmation.x),
                                              float(settings.condenser_stigmation.y))

        if settings.gun_tilt:
            self._require_point_complete(settings.gun_tilt, 'gun_tilt')
            if settings.gun_tilt.x is not None:
                self.set_gun_tilt(float(settings.gun_tilt.x), float(settings.gun_tilt.y))

    def execute_beam_control(self, request: BeamControlRequest) -> None:
        """Orchestrator: Handle BeamControlRequest."""
        if not request.validate():
            raise ValueError(f"Invalid BeamControlRequest: {request}")

        logger.info(f"[BEAM] Executing Control: {self._summarize_patch(request.target)}")

        sys = self.system_settings.beam_system
        if sys:
            check = sys.is_safe_beam(request.target)
            if not check:
                logger.error(f"[BEAM] Unsafe settings rejected: {check.reasons}")
                raise RuntimeError(f"Unsafe beam settings rejected: {check.reasons}")

        if request.target:
            self.apply_beam_settings(request.target)

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
    def set_optical_mode(self, mode: str) -> None:
        """Set optical mode."""
        pass

    @abstractmethod
    def set_magnification(self, index: int) -> None:
        """Set Magnification."""
        pass

    @abstractmethod
    def set_camera_length(self, length: Quantity) -> None:
        """Set Camera Length (mm)."""
        pass

    @abstractmethod
    def set_defocus(self, defocus: Quantity) -> None:
        """Set Defocus (nm)."""
        pass

    @abstractmethod
    def set_screen_position(self, position: str) -> None:
        """Set Screen Position ('UP', 'DOWN')."""
        pass

    @abstractmethod
    def set_objective_stigmation(self, x: float, y: float) -> None:
        """Set Objective Stigmator Coils."""
        pass

    @abstractmethod
    def set_diffraction_stigmation(self, x: float, y: float) -> None:
        """Set Diffraction Stigmator Coils."""
        pass

    @abstractmethod
    def set_image_shift(self, x: float, y: float) -> None:
        """Set Image Shift Coils."""
        pass

    @abstractmethod
    def set_diffraction_shift(self, x: float, y: float) -> None:
        """Set Diffraction Shift Coils."""
        pass

    # --- Logic Layer ---

    def get_projection_settings(self) -> ProjectionSettings:
        """Aggregator: returns full ProjectionSettings snapshot."""
        obj_st = self.get_objective_stigmation()
        dif_st = self.get_diffraction_stigmation()
        img_sh = self.get_image_shift()
        dif_sh = self.get_diffraction_shift()
        return ProjectionSettings(
            optical_mode=self.get_optical_mode(),
            magnification=self.get_magnification(),
            defocus=self.get_defocus(),
            camera_length=self.get_camera_length(),
            screen_position=self.get_screen_position(),
            objective_stigmation=Point(x=obj_st[0], y=obj_st[1]),
            diffraction_stigmation=Point(x=dif_st[0], y=dif_st[1]),
            image_shift=Point(x=img_sh[0], y=img_sh[1]),
            diffraction_shift=Point(x=dif_sh[0], y=dif_sh[1])
        )

    def apply_projection_settings(self, settings: ProjectionSettings) -> None:
        """Helper: Applies partial projection settings.

        Notes:
        - Canonical fields are applied when not None.
        - For 2D coil fields (Point), partial specification is rejected (strict).
        - Vendor-specific extras are validated/applied by vendor overrides.
        """
        if settings.optical_mode is not None:
            self.set_optical_mode(settings.optical_mode)
        if settings.magnification is not None:
            self.set_magnification(settings.magnification)
        if settings.camera_length is not None:
            self.set_camera_length(settings.camera_length)
        if settings.defocus is not None:
            self.set_defocus(settings.defocus)
        if settings.screen_position is not None:
            self.set_screen_position(settings.screen_position)

        if settings.objective_stigmation:
            self._require_point_complete(settings.objective_stigmation, 'objective_stigmation')
            if settings.objective_stigmation.x is not None:
                self.set_objective_stigmation(float(settings.objective_stigmation.x),
                                              float(settings.objective_stigmation.y))

        if settings.diffraction_stigmation:
            self._require_point_complete(settings.diffraction_stigmation, 'diffraction_stigmation')
            if settings.diffraction_stigmation.x is not None:
                self.set_diffraction_stigmation(float(settings.diffraction_stigmation.x),
                                                float(settings.diffraction_stigmation.y))

        if settings.image_shift:
            self._require_point_complete(settings.image_shift, 'image_shift')
            if settings.image_shift.x is not None:
                self.set_image_shift(float(settings.image_shift.x), float(settings.image_shift.y))

        if settings.diffraction_shift:
            self._require_point_complete(settings.diffraction_shift, 'diffraction_shift')
            if settings.diffraction_shift.x is not None:
                self.set_diffraction_shift(float(settings.diffraction_shift.x), float(settings.diffraction_shift.y))

    def execute_projection_control(self, request: ProjectionControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid ProjectionControlRequest: {request}")

        logger.info(f"[PROJ] Executing Control: {self._summarize_patch(request.target)}")

        sys = self.system_settings.projection_system
        if sys:
            check = sys.is_safe_projection(request.target)
            if not check:
                logger.error(f"[PROJ] Unsafe settings rejected: {check.reasons}")
                raise RuntimeError(f"Unsafe projection settings rejected: {check.reasons}")

        if request.target:
            self.apply_projection_settings(request.target)

    # =========================================================================
    # 6. Scan Control (STEM)
    # =========================================================================

    # --- Atomic Getters ---
    @abstractmethod
    def get_scan_mode(self) -> str:
        """Get scan engine mode."""
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

    @abstractmethod
    def get_scan_active(self) -> bool:
        """Return True if scanning is currently active."""
        pass

    # --- Atomic Setters ---
    @abstractmethod
    def set_scan_mode(self, mode: str) -> None:
        """Set scan engine mode."""
        pass

    @abstractmethod
    def set_scan_width(self, px: int) -> None:
        """Set scan width (px)."""
        pass

    @abstractmethod
    def set_scan_height(self, px: int) -> None:
        """Set scan height (px)."""
        pass

    @abstractmethod
    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        """Set dwell time (us)."""
        pass

    @abstractmethod
    def set_scan_flyback(self, time: Quantity) -> None:
        """Set flyback time (us)."""
        pass

    @abstractmethod
    def set_scan_rotation(self, angle: Quantity) -> None:
        """Set scan rotation (deg)."""
        pass

    @abstractmethod
    def set_scan_active(self, active: bool) -> None:
        """Start (True) or Stop (False) the scan."""
        pass

    # --- Logic Layer ---

    def get_scan_settings(self) -> ScanSettings:
        """Aggregator: returns full ScanSettings snapshot."""
        return ScanSettings(
            scan_mode=self.get_scan_mode(),
            width_px=self.get_scan_width(),
            height_px=self.get_scan_height(),
            pixel_dwell_time=self.get_scan_pixel_dwell(),
            flyback_time=self.get_scan_flyback(),
            scan_rotation=self.get_scan_rotation()
        )

    def apply_scan_settings(self, settings: ScanSettings) -> None:
        """Helper: Applies partial scan settings."""
        if settings.scan_mode is not None: self.set_scan_mode(settings.scan_mode)
        if settings.width_px is not None: self.set_scan_width(settings.width_px)
        if settings.height_px is not None: self.set_scan_height(settings.height_px)
        if settings.pixel_dwell_time is not None: self.set_scan_pixel_dwell(settings.pixel_dwell_time)
        if settings.flyback_time is not None: self.set_scan_flyback(settings.flyback_time)
        if settings.scan_rotation is not None: self.set_scan_rotation(settings.scan_rotation)

    def execute_scan_control(self, request: ScanControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid ScanControlRequest: {request}")

        logger.info(f"[SCAN] Executing Control: Action={request.action}")

        if request.target and request.action in ("START", "SINGLE_FRAME"):
            sys = self.system_settings.scan_system
            if sys:
                check = sys.is_safe_scan(request.target)
                if not check:
                    logger.error(f"[SCAN] Unsafe settings rejected: {check.reasons}")
                    raise RuntimeError(f"Unsafe scan settings rejected: {check.reasons}")
            self.apply_scan_settings(request.target)

        if request.action == "START":
            self.set_scan_active(True)
        elif request.action == "STOP":
            self.set_scan_active(False)
        elif request.action == "SINGLE_FRAME":
            # Logic for single frame could involve START -> Wait -> STOP, or driver specific logic
            self.set_scan_active(True)

    # =========================================================================
    # 7. Detector Control & Acquisition
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
    def get_detector_exposure(self, detector_id: str) -> Optional[Quantity]:
        """Get exposure time (ms)."""
        pass

    @abstractmethod
    def get_detector_binning(self, detector_id: str) -> Optional[int]:
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
    def get_detector_integration(self, detector_id: str) -> Optional[int]:
        """Get frame integration count."""
        pass

    @abstractmethod
    def get_detector_inserted(self, detector_id: str) -> bool:
        """Return True if detector is mechanically inserted."""
        pass

    @abstractmethod
    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        """Get estimated frame rate (Hz)."""
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
    def set_detector_exposure(self, detector_id: str, exposure: Quantity) -> None:
        """Set exposure time (ms)."""
        pass

    @abstractmethod
    def set_detector_binning(self, detector_id: str, index: int) -> None:
        """Set binning index."""
        pass

    @abstractmethod
    def set_detector_binning_xy(self, detector_id: str, binning: Tuple[int, int]) -> None:
        """Set binning tuple (x, y)."""
        pass

    @abstractmethod
    def set_detector_roi(self, detector_id: str, roi: Optional[ROI]) -> None:
        """Set Region of Interest."""
        pass

    @abstractmethod
    def set_detector_integration(self, detector_id: str, count: int) -> None:
        """Set frame integration count."""
        pass

    @abstractmethod
    def set_detector_insertion(self, detector_id: str, inserted: bool) -> None:
        """Insert (True) or Retract (False) detector."""
        pass

    @abstractmethod
    def set_detector_frame_rate(self, detector_id: str, rate: Quantity) -> None:
        """Set target frame rate (Hz)."""
        pass

    @abstractmethod
    def set_detector_gain_index(self, detector_id: str, index: int) -> None:
        """Set gain index."""
        pass

    @abstractmethod
    def set_detector_offset_index(self, detector_id: str, index: int) -> None:
        """Set offset index."""
        pass

    @abstractmethod
    def set_detector_digital_rotation(self, detector_id: str, angle: Quantity) -> None:
        """Set digital rotation (deg)."""
        pass

    @abstractmethod
    def set_detector_total_frames(self, detector_id: str, count: int) -> None:
        """Set total frames (movie mode)."""
        pass

    @abstractmethod
    def set_detector_readout_mode(self, detector_id: str, mode: str) -> None:
        """Set readout mode."""
        pass

    @abstractmethod
    def set_detector_shutter_mode(self, detector_id: str, mode: str) -> None:
        """Set shutter mode."""
        pass

    @abstractmethod
    def set_detector_save_frames(self, detector_id: str, save: bool) -> None:
        """Set save frames flag."""
        pass

    @abstractmethod
    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        """
        Atomic: Execute Acquisition Cycle.
        1. Configure hardware (if request.detector / request.image provided).
        2. Expose sensor.
        3. Readout and return data.
        """
        pass

    # --- Logic Layer ---

    def get_detector_settings(self, detector_id: str) -> DetectorSettings:
        """Aggregator: returns settings for a specific detector."""
        return DetectorSettings(
            detector_id=detector_id,
            exposure=self.get_detector_exposure(detector_id),
            binning_index=self.get_detector_binning(detector_id),
            binning_xy=self.get_detector_binning_xy(detector_id),
            roi=self.get_detector_roi(detector_id),
            frame_integration=self.get_detector_integration(detector_id),
            frame_rate=self.get_detector_frame_rate(detector_id),
            inserted=self.get_detector_inserted(detector_id),
            gain_index=self.get_detector_gain_index(detector_id),
            offset_index=self.get_detector_offset_index(detector_id),
            digital_rotation=self.get_detector_digital_rotation(detector_id),
            total_frames=self.get_detector_total_frames(detector_id),
            readout_mode=self.get_detector_readout_mode(detector_id),
            shutter_mode=self.get_detector_shutter_mode(detector_id),
            save_frames=self.get_detector_save_frames(detector_id)
        )

    def apply_detector_settings(self, detector_id: str, settings: DetectorSettings) -> None:
        """Helper: Apply partial detector settings."""
        if settings.exposure is not None:
            self.set_detector_exposure(detector_id, settings.exposure)
        if settings.binning_index is not None:
            self.set_detector_binning(detector_id, settings.binning_index)
        if settings.binning_xy is not None:
            self.set_detector_binning_xy(detector_id, settings.binning_xy)
        if settings.frame_integration is not None:
            self.set_detector_integration(detector_id, settings.frame_integration)
        if settings.roi is not None:
            self.set_detector_roi(detector_id, settings.roi)
        if settings.frame_rate is not None:
            self.set_detector_frame_rate(detector_id, settings.frame_rate)
        if settings.inserted is not None:
            self.set_detector_insertion(detector_id, settings.inserted)
        if settings.gain_index is not None:
            self.set_detector_gain_index(detector_id, settings.gain_index)
        if settings.offset_index is not None:
            self.set_detector_offset_index(detector_id, settings.offset_index)
        if settings.digital_rotation is not None:
            self.set_detector_digital_rotation(detector_id, settings.digital_rotation)
        if settings.total_frames is not None:
            self.set_detector_total_frames(detector_id, settings.total_frames)
        if settings.readout_mode is not None:
            self.set_detector_readout_mode(detector_id, settings.readout_mode)
        if settings.shutter_mode is not None:
            self.set_detector_shutter_mode(detector_id, settings.shutter_mode)
        if settings.save_frames is not None:
            self.set_detector_save_frames(detector_id, settings.save_frames)

    def execute_detector_control(self, request: DetectorControlRequest) -> None:
        """
        Orchestrator: Handle DetectorControlRequest.
        Handles INSERT/RETRACT actions and applies settings.
        """
        if not request.validate():
            raise ValueError(f"Invalid DetectorControlRequest: {request}")

        logger.info(f"[DET] Executing Control: {request.action or 'Configure'} on {request.detector_id}")

        sys = self.system_settings.detector_system
        if sys and request.target:
            check = sys.is_supported(request.target)
            if not check:
                logger.error(f"[DET] Unsupported settings: {check.reasons}")
                raise RuntimeError(f"Detector settings not supported: {check.reasons}")

        if request.action == "INSERT":
            self.set_detector_insertion(request.detector_id, True)
        elif request.action == "RETRACT":
            self.set_detector_insertion(request.detector_id, False)

        if request.target:
            self.apply_detector_settings(request.detector_id, request.target)

    def execute_acquisition(self, request: AcquisitionRequest) -> MicroscopeImage:
        """
        Orchestrator: Handle AcquisitionRequest.
        1. Validate
        2. Apply Settings
        3. Capture
        4. Save (Optional)
        """
        if not request.validate():
            raise ValueError(f"Invalid AcquisitionRequest: {request}")

        det_id = request.detector_id
        logger.info(f"[ACQ] Starting acquisition on '{det_id}'")

        # 1. Check Capabilities
        sys = self.system_settings.detector_system
        if sys and request.detector:
            # We merge defaults here conceptually, but validation checks pure intent
            check = sys.is_supported(request.detector)
            if not check:
                raise RuntimeError(f"Acquisition settings not supported: {check.reasons}")

        # 2. Apply Settings
        if request.detector:
            self.apply_detector_settings(det_id, request.detector)

        # 3. Capture (Atomic)
        image = self.acquire_image(request)

        # 4. Enhance Metadata (Inject System State if driver didn't)
        if image.metadata is None:
            image.metadata = MicroscopeImageMetadata()

        # Inject full state snapshot if missing
        if image.metadata.microscope_state is None:
            try:
                image.metadata.microscope_state = self.get_full_state()
            except Exception as e:
                logger.warning(f"Failed to capture full state for metadata: {e}")

        # 5. Save Logic
        output_cfg = request.image or self._settings.image
        if output_cfg and output_cfg.path:
            try:
                # Basic filename generation if path is a directory
                save_path = Path(output_cfg.path)
                if save_path.is_dir() or (not save_path.suffix):
                    fname = f"Image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    save_path = save_path / fname

                final_path = image.save(save_path, file_format=output_cfg.file_format)
                logger.info(f"[ACQ] Image saved to: {final_path}")
            except Exception as e:
                logger.error(f"[ACQ] Failed to save image: {e}")

        return image

    # =========================================================================
    # 8. Vacuum Control
    # =========================================================================

    # --- Atomic Methods ---
    @abstractmethod
    def get_valve_state(self, valve_name: str) -> str:
        """Get Valve State ('OPEN', 'CLOSED' or 'UNKNOWN')."""
        pass

    @abstractmethod
    def set_valve_state(self, valve_name: str, state: str) -> None:
        """Set Valve State ('OPEN', 'CLOSED')."""
        pass

    @abstractmethod
    def get_pressure(self, gauge_name: str) -> Optional[Quantity]:
        """Get Pressure (Pa). Returns None if unknown."""
        pass

    # --- Logic Layer ---

    def get_vacuum_settings(self) -> VacuumSettings:
        """Aggregator: returns full vacuum status."""
        return VacuumSettings(
            column_valve_state=self.get_valve_state('column'),
            gun_valve_state=self.get_valve_state('gun'),
            turbo_pump_state=self.get_valve_state('turbo'),
            column_pressure=self.get_pressure('column'),
            gun_pressure=self.get_pressure('gun'),
            buffer_tank_pressure=self.get_pressure('buffer')
        )

    def apply_vacuum_settings(self, settings: VacuumSettings) -> None:
        """Helper: Apply vacuum state changes.

        Notes:
        - Canonical fields are applied when not None.
        - Vendor-specific extras are validated/applied by vendor overrides.
        """
        if settings.column_valve_state is not None:
            self.set_valve_state('column', settings.column_valve_state)
        if settings.gun_valve_state is not None:
            self.set_valve_state('gun', settings.gun_valve_state)
        if settings.turbo_pump_state is not None:
            self.set_valve_state('turbo', settings.turbo_pump_state)

    def execute_vacuum_control(self, request: VacuumControlRequest) -> None:
        """Orchestrator: Handle VacuumControlRequest."""
        if not request.validate():
            raise ValueError(f"Invalid VacuumControlRequest: {request}")

        logger.info(f"[VAC] Executing Control: {self._summarize_patch(request.target)}")
        if request.target:
            self.apply_vacuum_settings(request.target)

    # =========================================================================
    # 9. Aperture Control
    # =========================================================================

    # --- Atomic Methods ---
    @abstractmethod
    def list_apertures(self) -> List[str]:
        """List supported aperture mechanism IDs (e.g. 'CLA', 'OLA')."""
        pass

    @abstractmethod
    def get_aperture(self, aperture_id: str) -> Optional[ApertureSettings]:
        """Get state of an aperture. Returns None if unknown."""
        pass

    @abstractmethod
    def set_aperture(self, aperture_id: str, target: ApertureSettings) -> None:
        """Set aperture state."""
        pass

    # --- Logic Layer ---
    def get_all_apertures(self) -> Dict[str, ApertureSettings]:
        return {a_id: self.get_aperture(a_id) for a_id in self.list_apertures()
                if self.get_aperture(a_id) is not None}

    def execute_aperture_control(self, request: ApertureControlRequest) -> None:
        """
        Orchestrator: Handle ApertureControlRequest.
        Handles ID matching, safety checks, and relative position logic.
        """
        if not request.validate():
            raise ValueError(f"Invalid ApertureControlRequest: {request}")

        logger.info(f"[APT] Executing Control on '{request.aperture_id}'")

        # 1. Safety Check
        sys = self.system_settings.aperture_system
        if sys:
            check = sys.is_supported(request.target)
            if not check:
                raise RuntimeError(f"Aperture request rejected: {check.reasons}")

        final_target = request.target

        # 2. Handle Relative Movement
        if request.relative and request.target.position:
            current = self.get_aperture(request.aperture_id)
            if current is None:
                raise RuntimeError(f"Relative move failed: State of '{request.aperture_id}' unknown")

            if current.position:
                new_pos = replace(request.target.position)  # Start with delta structure

                # Perform vector addition (Manual because Point doesn't have __add__)
                cur_x = current.position.x if current.position.x is not None else 0.0
                cur_y = current.position.y if current.position.y is not None else 0.0

                if request.target.position.x is not None:
                    new_pos.x = cur_x + request.target.position.x
                else:
                    new_pos.x = current.position.x

                if request.target.position.y is not None:
                    new_pos.y = cur_y + request.target.position.y
                else:
                    new_pos.y = current.position.y

                final_target = replace(final_target, position=new_pos)

        # 3. Execute
        self.set_aperture(request.aperture_id, final_target)

    # =========================================================================
    # 10. Safety Helpers
    # =========================================================================

    def safe_move_stage(self, target: StagePosition,
                        drive_type: str = "default",
                        wait: bool = True) -> None:
        """
        Safety Helper: Executes a stage move in smaller steps if required.

        Checks `SystemSettings.stage_system.max_step_distance`. If the move
        exceeds this limit, it breaks the trajectory into linear segments
        and moves sequentially.

        Args:
            target: Absolute destination.
            drive_type: 'mechanical', 'piezo', or 'default'.
            wait: Block until complete.
        """
        sys = self.system_settings.stage_system
        if not sys or not sys.max_step_distance:
            self.move_stage_absolute(target, drive_type, wait)
            return

        current = self.get_stage_position()
        if current is None:
            raise RuntimeError("Safe Move Failed: Cannot read current stage position to calculate steps.")

        max_step_nm = sys.max_step_distance.to(Units.NM).magnitude

        # Calculate max delta in NM
        def dist(c: Optional[Quantity], t: Optional[Quantity]) -> float:
            if c is None or t is None: return 0.0
            return abs(t.to(Units.NM).magnitude - c.to(Units.NM).magnitude)

        d_x = dist(current.x, target.x)
        d_y = dist(current.y, target.y)
        d_z = dist(current.z, target.z)
        max_dist = max(d_x, d_y, d_z)

        if max_dist <= max_step_nm:
            self.move_stage_absolute(target, drive_type, wait)
            return

        # Linear Interpolation
        steps = int(max_dist // max_step_nm) + 1
        logger.info(f"[STAGE] Step Limit: {max_dist:.1f}nm > {max_step_nm:.1f}nm. Breaking into {steps} segments.")

        for i in range(1, steps + 1):
            frac = i / steps
            interim = replace(current)  # Copy structure

            # Interpolate known axes
            def interp(c, t):
                if c is None or t is None: return c
                return c + (t - c) * frac

            if target.x is not None: interim.x = interp(current.x, target.x)
            if target.y is not None: interim.y = interp(current.y, target.y)
            if target.z is not None: interim.z = interp(current.z, target.z)

            # Rotation/Tilt usually not interpolated by distance logic, passed through on final step
            if i == steps:
                interim.r = target.r
                interim.tilt_x = target.tilt_x
                interim.tilt_y = target.tilt_y

            self.move_stage_absolute(interim, drive_type, wait=True)