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
IV. Type Safety & Units
===============================================================================

All Atomic interfaces use strict typing:
  - `Quantity` (from pint) is used for all physical values.
  - `int` / `str` / `bool` are used for discrete states.
  - Vendor drivers must handle unit conversion (e.g., converting the input
    `10 nm` to the `1e-8 meters` expected by a specific API).

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
import time

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
    Aperture,
    MicroscopeImage,
    Point,
    ROI,

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
    StageDriveType
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
            # Fallback for bare initialization (not recommended for production)
            self._settings = MicroscopeSettings(
                system=SystemSettings(),
                _mode="lenient"
            )
        else:
            self._settings = settings

    @property
    def system_settings(self) -> SystemSettings:
        """Access the system limits and capabilities configuration."""
        return self._settings.system

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
        Get the global instrument mode.

        Returns:
            String: e.g., 'TEM', 'STEM', 'SEM', 'DIFF', 'EDX'.
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

    # --- Atomic Layer (Abstract) ---

    @abstractmethod
    def get_stage_position(self) -> StagePosition:
        """
        Atomic: Read current physical stage coordinates.

        Returns:
            StagePosition: Objects with x, y, z, r, tilt_x, tilt_y.
        """
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
        """
        Atomic: Move stage to a specific absolute coordinate.

        Args:
            target: Destination. Axes set to None (e.g., target.x=None) MUST be ignored.
            drive_type: Hint mechanism ('piezo', 'mechanical').
            wait: If True, block until motion completes.
        """
        pass

    @abstractmethod
    def stop_stage(self) -> None:
        """Atomic: Immediately halt all stage motion axes."""
        pass

    @abstractmethod
    def home_stage(self) -> None:
        """Atomic: Return stage to its mechanical origin/zero position."""
        pass

    # --- Logic Layer (Concrete) ---

    def move_stage_relative(self, delta: StagePosition,
                            drive_type: str = "default",
                            wait: bool = True) -> None:
        """
        Helper: Calculate absolute target from delta and execute move.
        """
        current = self.get_stage_position()
        target = current + delta  # Vector addition handled by StagePosition
        # We delegate to the safe mover to ensure step sizes are respected even for relative moves
        self.safe_move_stage(target, drive_type=drive_type, wait=wait)

    def execute_stage_move(self, request: StageMoveRequest) -> None:
        """
        Orchestrator: Handle StageMoveRequest.

        Features:
        - Validates request (structure and types).
        - Checks `SystemSettings` for safety limits (logic).
        - Handles Relative vs Absolute logic.
        - Uses `safe_move_stage` to interpolate large moves if needed.
        """
        if not request.validate():
            raise ValueError(f"Invalid StageMoveRequest: {request}")

        # LOG INTENT
        tgt_str = f"Target={request.target}" if not request.relative else f"Delta={request.target}"
        logger.info(f"[STAGE] Executing Move: {tgt_str} (Mode: {request.drive_type})")

        # 1. Resolve Target (Absolute)
        current = self.get_stage_position()
        target_abs = request.target

        if request.relative:
            target_abs = current + request.target
            # If absolute addition resulted in None for some axes, fill them from current
            # to allow for a complete safety check of the final destination.
            # (Note: move_stage_absolute typically ignores Nones, but safety check needs context)

        # 2. Safety Check (Destination & Capability)
        sys = self.system_settings.stage_system
        if sys:
            # We check the move using the resolved absolute target.
            # We pass 'current' to allow step size calculation in the check.
            check = sys.is_safe_move(
                target=request.target if request.relative else target_abs,
                current=current,
                relative=request.relative
            )
            if not check:
                logger.error(f"[STAGE] Unsafe move rejected. Reasons: {check.reasons}")
                raise RuntimeError(f"Unsafe move rejected: {check.reasons}")

            # 3. Execution (via Safe Mover)
            # The safe mover handles "max_step_distance" interpolation.
            self.safe_move_stage(target_abs, drive_type=request.drive_type, wait=request.wait_for_settle)
        else:
            # Fallback (No safety system defined)
            self.move_stage_absolute(target_abs, drive_type=request.drive_type, wait=request.wait_for_settle)

    def execute_stage_control(self, request: StageControlRequest) -> None:
        """
        Orchestrator: Handle StageControlRequest (STOP, HOME).
        """
        if not request.validate():
            raise ValueError(f"Invalid StageControlRequest: {request}")

        logger.info(f"[STAGE] Executing Control: {request.action}")

        if request.action == "STOP":
            self.stop_stage()
        elif request.action == "HOME":
            self.home_stage()
        # "ABORT", "RESET_ERROR" could be implemented if driver supports them

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Getters (Abstract) ---
    @abstractmethod
    def get_acceleration_voltage(self) -> Optional[Quantity]:
        """Get High Tension. Units: Electric Potential (kV)."""
        pass

    @abstractmethod
    def get_beam_current(self) -> Optional[Quantity]:
        """Get Beam Current. Units: Electric Current (nA/pA)."""
        pass

    @abstractmethod
    def get_spot_size(self) -> int:
        """Get Spot Size Index (unitless integer)."""
        pass

    @abstractmethod
    def get_convergence_angle(self) -> Optional[Quantity]:
        """Get Convergence (Alpha) Angle. Units: Angle (mrad)."""
        pass

    @abstractmethod
    def get_beam_shift(self) -> Tuple[float, float]:
        """Get Beam Shift Coils. Units: Logical (-1..1) or Physical (Arb)."""
        pass

    @abstractmethod
    def get_condenser_stigmation(self) -> Tuple[float, float]:
        """Get Condenser Stigmator Coils. Units: Logical or Physical."""
        pass

    @abstractmethod
    def get_gun_tilt(self) -> Tuple[float, float]:
        """Get Gun Tilt Alignment. Units: Logical or Physical."""
        pass

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Get Beam Blank Status. True = Blanked (Beam OFF)."""
        pass

    # --- Atomic Setters (Abstract) ---
    @abstractmethod
    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        """Set High Tension. Expected Units: Volts/kV."""
        pass

    @abstractmethod
    def set_beam_current(self, current: Quantity) -> None:
        """Set Beam Current. Expected Units: Amperes/nA."""
        pass

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """Set Spot Size Index."""
        pass

    @abstractmethod
    def set_convergence_angle(self, angle: Quantity) -> None:
        """Set Convergence Angle. Expected Units: Radians/mrad."""
        pass

    @abstractmethod
    def set_beam_shift(self, x: float, y: float) -> None:
        """Set Beam Shift Coils (x, y)."""
        pass

    @abstractmethod
    def set_condenser_stigmation(self, x: float, y: float) -> None:
        """Set Condenser Stigmator Coils (x, y)."""
        pass

    @abstractmethod
    def set_gun_tilt(self, x: float, y: float) -> None:
        """Set Gun Tilt Alignment (x, y)."""
        pass

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """Set Beam Blanker. True = Blank Beam (Block)."""
        pass

    # --- Logic Layer (Concrete) ---

    def get_beam_settings(self) -> BeamSettings:
        """Aggregator: returns full BeamSettings snapshot."""
        bs = self.get_beam_shift()
        cs = self.get_condenser_stigmation()
        gt = self.get_gun_tilt()

        return BeamSettings(
            voltage=self.get_acceleration_voltage(),
            beam_current=self.get_beam_current(),
            spot_size=self.get_spot_size(),
            convergence_angle=self.get_convergence_angle(),
            beam_shift=Point(x=bs[0], y=bs[1]),
            condenser_stigmation=Point(x=cs[0], y=cs[1]),
            gun_tilt=Point(x=gt[0], y=gt[1])
        )

    def apply_beam_settings(self, settings: BeamSettings) -> None:
        """
        Helper: Applies a partial beam configuration.
        Iterates over the `settings` object. If a field is NOT None, the
        corresponding atomic setter is called.
        """
        if settings.voltage is not None:
            self.set_acceleration_voltage(settings.voltage)
        if settings.beam_current is not None:
            self.set_beam_current(settings.beam_current)
        if settings.spot_size is not None:
            self.set_spot_size(settings.spot_size)
        if settings.convergence_angle is not None:
            self.set_convergence_angle(settings.convergence_angle)

        if settings.beam_shift:
            self.set_beam_shift(settings.beam_shift.x or 0.0, settings.beam_shift.y or 0.0)
        if settings.condenser_stigmation:
            self.set_condenser_stigmation(settings.condenser_stigmation.x or 0.0,
                                          settings.condenser_stigmation.y or 0.0)
        if settings.gun_tilt:
            self.set_gun_tilt(settings.gun_tilt.x or 0.0, settings.gun_tilt.y or 0.0)

    def execute_beam_control(self, request: BeamControlRequest) -> None:
        """Orchestrator: Handle BeamControlRequest."""
        if not request.validate():
            raise ValueError(f"Invalid BeamControlRequest: {request}")

        logger.info(f"[BEAM] Executing Control: Setting {list(request.target.__dict__.keys())}")

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
    def get_projection_mode(self) -> str:
        """Get optical mode (e.g., 'IMAGING', 'DIFFRACTION')."""
        pass

    @abstractmethod
    def get_magnification(self) -> int:
        """Get Magnification (unitless integer)."""
        pass

    @abstractmethod
    def get_camera_length(self) -> Optional[Quantity]:
        """Get Camera Length (Diffraction). Units: Length (mm)."""
        pass

    @abstractmethod
    def get_defocus(self) -> Optional[Quantity]:
        """Get Defocus. Units: Length (nm)."""
        pass

    @abstractmethod
    def get_screen_position(self) -> str:
        """Get Fluorescent Screen Position ('UP' or 'DOWN')."""
        pass

    @abstractmethod
    def get_objective_stigmation(self) -> Tuple[float, float]:
        """Get Objective Stigmator Coils (x, y)."""
        pass

    @abstractmethod
    def get_image_shift(self) -> Tuple[float, float]:
        """Get Image Shift Coils (x, y)."""
        pass

    @abstractmethod
    def get_diffraction_shift(self) -> Tuple[float, float]:
        """Get Diffraction Shift Coils (x, y)."""
        pass

    # --- Atomic Setters ---
    @abstractmethod
    def set_projection_mode(self, mode: str) -> None:
        """Set optical mode."""
        pass

    @abstractmethod
    def set_magnification(self, index: int) -> None:
        """Set Magnification."""
        pass

    @abstractmethod
    def set_camera_length(self, length: Quantity) -> None:
        """Set Camera Length. Expected Units: Length (mm/cm)."""
        pass

    @abstractmethod
    def set_defocus(self, defocus: Quantity) -> None:
        """Set Defocus. Expected Units: Length (nm/um)."""
        pass

    @abstractmethod
    def set_screen_position(self, position: str) -> None:
        """Set Screen Position ('UP'/'DOWN')."""
        pass

    @abstractmethod
    def set_objective_stigmation(self, x: float, y: float) -> None:
        """Set Objective Stigmator Coils."""
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
        img_sh = self.get_image_shift()
        dif_sh = self.get_diffraction_shift()

        return ProjectionSettings(
            optical_mode=self.get_projection_mode(),
            magnification_index=self.get_magnification_index(),
            defocus=self.get_defocus(),
            camera_length=self.get_camera_length(),
            screen_position=self.get_screen_position(),
            objective_stigmation=Point(x=obj_st[0], y=obj_st[1]),
            image_shift=Point(x=img_sh[0], y=img_sh[1]),
            diffraction_shift=Point(x=dif_sh[0], y=dif_sh[1])
        )

    def apply_projection_settings(self, settings: ProjectionSettings) -> None:
        """Helper: Applies partial projection settings."""
        if settings.optical_mode is not None:
            self.set_projection_mode(settings.optical_mode)
        if settings.magnification_index is not None:
            self.set_magnification_index(settings.magnification_index)
        if settings.camera_length is not None:
            self.set_camera_length(settings.camera_length)
        if settings.defocus is not None:
            self.set_defocus(settings.defocus)
        if settings.screen_position is not None:
            self.set_screen_position(settings.screen_position)

        if settings.objective_stigmation:
            self.set_objective_stigmation(settings.objective_stigmation.x or 0.0,
                                          settings.objective_stigmation.y or 0.0)
        if settings.image_shift:
            self.set_image_shift(settings.image_shift.x or 0.0, settings.image_shift.y or 0.0)
        if settings.diffraction_shift:
            self.set_diffraction_shift(settings.diffraction_shift.x or 0.0,
                                       settings.diffraction_shift.y or 0.0)

    def execute_projection_control(self, request: ProjectionControlRequest) -> None:
        if not request.validate():
            raise ValueError(f"Invalid ProjectionControlRequest: {request}")

        logger.info("[PROJ] Executing Control Request")

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
    def get_scan_width(self) -> int:
        """Get scan width in pixels."""
        pass

    @abstractmethod
    def get_scan_height(self) -> int:
        """Get scan height in pixels."""
        pass

    @abstractmethod
    def get_scan_pixel_dwell(self) -> Quantity:
        """Get pixel dwell time. Units: Time (us/ns)."""
        pass

    @abstractmethod
    def get_scan_flyback(self) -> Quantity:
        """Get flyback time. Units: Time (us/ns)."""
        pass

    @abstractmethod
    def get_scan_rotation(self) -> Quantity:
        """Get scan rotation. Units: Angle (deg/rad)."""
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
        """Set width (pixels)."""
        pass

    @abstractmethod
    def set_scan_height(self, px: int) -> None:
        """Set height (pixels)."""
        pass

    @abstractmethod
    def set_scan_pixel_dwell(self, time: Quantity) -> None:
        """Set dwell time. Expected Units: Time."""
        pass

    @abstractmethod
    def set_scan_flyback(self, time: Quantity) -> None:
        """Set flyback time. Expected Units: Time."""
        pass

    @abstractmethod
    def set_scan_rotation(self, angle: Quantity) -> None:
        """Set scan rotation. Expected Units: Angle."""
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
    # 7. Detector Control
    # =========================================================================

    # --- Atomic Getters ---
    @abstractmethod
    def list_detectors(self) -> List[str]:
        """Return list of available detector IDs."""
        pass

    @abstractmethod
    def get_active_detector_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_primary_detector_id(self) -> Optional[str]:
        pass

    @abstractmethod
    def get_detector_exposure(self, detector_id: str) -> Quantity:
        """Get exposure time. Units: Time (s/ms)."""
        pass

    @abstractmethod
    def get_detector_binning(self, detector_id: str) -> int:
        """Get binning index (e.g., 1 for 1x1, 2 for 2x2)."""
        pass

    @abstractmethod
    def get_detector_roi(self, detector_id: str) -> Optional[ROI]:
        """Get Region of Interest."""
        pass

    @abstractmethod
    def get_detector_integration(self, detector_id: str) -> int:
        """Get frame integration count."""
        pass

    @abstractmethod
    def get_detector_inserted(self, detector_id: str) -> bool:
        """Return True if detector is mechanically inserted."""
        pass

    @abstractmethod
    def get_detector_frame_rate(self, detector_id: str) -> Optional[Quantity]:
        pass

    # ---Atomic Setters ---
    @abstractmethod
    def set_detector_exposure(self, detector_id: str, exposure: Quantity) -> None:
        """Set exposure time."""
        pass

    @abstractmethod
    def set_detector_binning(self, detector_id: str, index: int) -> None:
        """Set binning index."""
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
        """Mechanically insert (True) or retract (False) the detector."""
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
            roi=self.get_detector_roi(detector_id),
            frame_integration=self.get_detector_integration(detector_id),
            frame_rate=self.get_detector_frame_rate(detector_id)
        )

    def apply_detector_settings(self, detector_id: str, settings: DetectorSettings) -> None:
        """Helper: Apply partial detector settings."""
        if settings.exposure is not None:
            self.set_detector_exposure(detector_id, settings.exposure)
        if settings.binning_index is not None:
            self.set_detector_binning(detector_id, settings.binning_index)
        if settings.frame_integration is not None:
            self.set_detector_integration(detector_id, settings.frame_integration)
        if settings.roi is not None:
            self.set_detector_roi(detector_id, settings.roi)

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
            # Check if capabilities support the request
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

    # =========================================================================
    # 8. Vacuum Control
    # =========================================================================

    # --- Atomic Methods ---
    @abstractmethod
    def get_valve_state(self, valve_name: str) -> str:
        """Get Valve State ('OPEN', 'CLOSED'). Name examples: 'column', 'gun'."""
        pass

    @abstractmethod
    def set_valve_state(self, valve_name: str, state: str) -> None:
        """Set Valve State ('OPEN', 'CLOSED')."""
        pass

    @abstractmethod
    def get_pressure(self, gauge_name: str) -> Quantity:
        """Get Pressure. Units: Pressure (Pa/Torr). Name: 'column', 'gun', etc."""
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
        """Helper: Apply vacuum state changes."""
        if settings.column_valve_state:
            self.set_valve_state('column', settings.column_valve_state)
        if settings.gun_valve_state:
            self.set_valve_state('gun', settings.gun_valve_state)
        if settings.turbo_pump_state:
            self.set_valve_state('turbo', settings.turbo_pump_state)

    def execute_vacuum_control(self, request: VacuumControlRequest) -> None:
        """Orchestrator: Handle VacuumControlRequest."""
        if not request.validate():
            raise ValueError(f"Invalid VacuumControlRequest: {request}")

        logger.info("[VAC] Executing Control Request")
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
    def get_aperture(self, aperture_id: str) -> Aperture:
        """Get state (inserted, size, position) of an aperture."""
        pass

    @abstractmethod
    def set_aperture(self, aperture_id: str, target: Aperture) -> None:
        """Set aperture state."""
        pass

    # --- Logic Layer ---

    def get_all_apertures(self) -> Dict[str, Aperture]:
        """Aggregator: returns state of all apertures."""
        return {a_id: self.get_aperture(a_id) for a_id in self.list_apertures()}

    def execute_aperture_control(self, request: ApertureControlRequest) -> None:
        """
        Orchestrator: Handle ApertureControlRequest.
        Features:
        - Logic to handle Relative Position moves.
        """
        if not request.validate():
            raise ValueError(f"Invalid ApertureControlRequest: {request}")

        logger.info(f"[APT] Executing Control on '{request.aperture_id}'")

        final_target = request.target

        # Handle Relative Movement logic
        if request.relative and request.target.position:
            current = self.get_aperture(request.aperture_id)
            if current is None:
                raise RuntimeError(
                    f"Cannot execute relative move: Failed to read current state of '{request.aperture_id}'")

            if current.position:
                new_pos = replace(request.target.position)
                # Apply delta to current position (manual vector addition)
                if request.target.position.x is not None and current.position.x is not None:
                    new_pos.x = current.position.x + request.target.position.x
                if request.target.position.y is not None and current.position.y is not None:
                    new_pos.y = current.position.y + request.target.position.y
                final_target = replace(final_target, position=new_pos)

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
        # If no step limit is defined, pass through directly
        if not sys or not sys.max_step_distance:
            self.move_stage_absolute(target, drive_type, wait)
            return

        current = self.get_stage_position()

        if current is None:
            raise RuntimeError("Safe Move Failed: Cannot read current stage position to calculate steps.")

        max_step_nm = sys.max_step_distance.to(Units.NM).magnitude

        # Calculate max delta across active axes
        def dist(c, t):
            if c is None or t is None: return 0.0
            return abs(t.to(Units.NM).magnitude - c.to(Units.NM).magnitude)

        d_x = dist(current.x, target.x)
        d_y = dist(current.y, target.y)
        d_z = dist(current.z, target.z)
        max_dist = max(d_x, d_y, d_z)

        # If move is within limit, execute directly
        if max_dist <= max_step_nm:
            self.move_stage_absolute(target, drive_type, wait)
            return

        # Otherwise, step it out via Linear Interpolation
        steps = int(max_dist // max_step_nm) + 1
        logger.info(f"Move exceeds max step ({max_dist:.1f}nm > {max_step_nm:.1f}nm). "
                    f"Breaking into {steps} segments.")

        for i in range(1, steps + 1):
            frac = i / steps
            interim = replace(current)  # Start with current structure

            # Interpolate only axes that are being moved (not None)
            if target.x is not None and current.x is not None:
                interim.x = current.x + (target.x - current.x) * frac
            if target.y is not None and current.y is not None:
                interim.y = current.y + (target.y - current.y) * frac
            if target.z is not None and current.z is not None:
                interim.z = current.z + (target.z - current.z) * frac

            # Commit the step
            self.move_stage_absolute(interim, drive_type, wait=True)