"""
supertem.microscope

Abstract Base Class (ABC) for Transmission Electron Microscope (TEM) control.

This module defines the "Hardware Abstraction Layer" (HAL) for the SuperTEM ecosystem.
It strictly enforces a separation of concerns between:

1. The Atomic Layer (Abstract)
   - Low-level, hardware-specific primitives.
   - Drivers (JEOL, ThermoFisher, etc.) MUST implement these methods.
   - These methods do one thing only (e.g., `set_spot_size`, `move_stage_absolute`).
   - They accept simple types (int, float, Quantity) or pure data structures (StagePosition).

2. The Logic Layer (Concrete)
   - High-level orchestration and safety.
   - Implemented here in the base class (do not override unless necessary).
   - These methods accept `Request` objects (Intents) from `supertem.structures.base`.
   - They perform validation, safety checks against `SystemSettings`, and calculate
     trajectories (e.g., `safe_move_stage`) before calling the Atomic Layer.

Usage:
    class JeolMicroscope(TemMicroscope):
        def connect(self, ...): ...
        def move_stage_absolute(self, target, ...): ...
        # ... implement all abstract methods ...

    scope = JeolMicroscope(settings=my_config)
    scope.connect()
    scope.execute_stage_move(StageMoveRequest(target=pos, relative=True))
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import replace

from pint import Quantity

# Import strictly typed structures from base.py
from supertem.structures.base import (
    # Configuration
    MicroscopeSettings,
    SystemSettings,
    SystemInfo,

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
    Q_
)

logger = logging.getLogger(__name__)


class TemMicroscope(ABC):
    """
    The generic template for all TEM implementations.

    Acts as the bridge between the Control Plane (Requests/Intents) and the 
    Hardware Plane (Drivers/SDKs).
    """

    def __init__(self, settings: Optional[MicroscopeSettings] = None):
        """
        Initialize the microscope interface.

        Args:
            settings: Initial configuration containing system limits, capabilities,
                      and hardware registry. If None, a lenient default is created.
        """
        if settings is None:
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
        Establish connection to the microscope control interface.

        Args:
            host: IP address or hostname.
            port: Port number (optional).
            **kwargs: Vendor-specific connection parameters.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources and close the connection cleanly."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check connection status.

        Returns:
            True if the connection is active and responsive.
        """
        pass

    @abstractmethod
    def get_instrument_info(self) -> SystemInfo:
        """
        Return static instrument identity.

        Returns:
            SystemInfo object containing Model, Serial Number, Software Version, etc.
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
            String identifier (e.g., 'TEM', 'STEM', 'SEM', 'DIFF').
        """
        pass

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """
        Set the global instrument mode.

        Args:
            mode: Target mode string (vendor-specific constants).
        """
        pass

    def get_full_state(self) -> MicroscopeState:
        """
        Capture a comprehensive snapshot of the entire microscope state.

        Aggregates data from all subsystems (Stage, Beam, Optics, Vacuum, etc.)
        into a single timestamped structure.

        Returns:
            MicroscopeState: The complete telemetry object.
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
            active_detector_ids=self.get_active_detector_ids()
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
            StagePosition object populated with current x, y, z, tilt, etc.
        """
        pass

    @abstractmethod
    def move_stage_absolute(self, target: StagePosition,
                            drive_type: str = "default",
                            wait: bool = True) -> None:
        """
        Atomic: Move stage to a specific absolute coordinate.

        Args:
            target: Destination coordinates. Axes set to None should be ignored.
            drive_type: Mechanism hint ("piezo", "mechanical", "default").
            wait: If True, block until movement completes.
        """
        pass

    @abstractmethod
    def stop_stage(self) -> None:
        """Atomic: Immediately halt all stage motion axes."""
        pass

    @abstractmethod
    def home_stage(self) -> None:
        """Atomic: Return stage to its mechanical origin/zero."""
        pass

    # --- Logic Layer (Concrete) ---

    def move_stage_relative(self, delta: StagePosition,
                            drive_type: str = "default",
                            wait: bool = True) -> None:
        """
        Helper: Calculate absolute target from delta and execute move.

        Args:
            delta: Relative distances to move.
            drive_type: Mechanism hint.
            wait: If True, block until completion.
        """
        current = self.get_stage_position()
        # StagePosition supports vector addition (current + delta)
        target = current + delta
        self.move_stage_absolute(target, drive_type=drive_type, wait=wait)

    def execute_stage_move(self, request: StageMoveRequest) -> None:
        """
        Orchestrator: Handle a movement request with safety checks and routing.

        1. Validates the Request object.
        2. Checks `SystemSettings` for collisions, limits, or unsafe conditions.
        3. Routes to `safe_move_stage` (for step-wise moves) or `move_stage_relative`.

        Args:
            request: A validated StageMoveRequest object containing target and flags.

        Raises:
            ValueError: If request is malformed.
            RuntimeError: If the move is deemed unsafe by SystemSettings.
        """
        if not request.validate():
            raise ValueError(f"Invalid StageMoveRequest: {request}")

        # Safety Gatekeeping
        sys = self.system_settings.stage_system
        if sys:
            current = self.get_stage_position()
            check = sys.is_safe_move(
                target=request.target,
                current=current,
                relative=request.relative
            )
            if not check:
                raise RuntimeError(f"Unsafe move rejected: {check.reasons}")

            # Calculate Absolute Target for the safe mover
            target_abs = request.target
            if request.relative:
                target_abs = current + request.target

            # Execute via Safe Mover (enforces step size limits)
            self.safe_move_stage(target_abs, drive_type=request.drive_type, wait=request.wait_for_settle)

        else:
            # Fallback (No safety system loaded)
            if request.relative:
                self.move_stage_relative(request.target, drive_type=request.drive_type, wait=request.wait_for_settle)
            else:
                self.move_stage_absolute(request.target, drive_type=request.drive_type, wait=request.wait_for_settle)

    def execute_stage_control(self, request: StageControlRequest) -> None:
        """
        Orchestrator: Handle non-motion stage commands (Stop, Home).

        Args:
            request: StageControlRequest with action string.
        """
        if not request.validate():
            raise ValueError(f"Invalid StageControlRequest: {request}")

        if request.action == "STOP":
            self.stop_stage()
        elif request.action == "HOME":
            self.home_stage()
        # "RESET_ERROR" or others can be implemented by subclasses if needed

    # =========================================================================
    # 4. Beam Control (Illumination)
    # =========================================================================

    # --- Atomic Layer (Getters) ---

    @abstractmethod
    def get_acceleration_voltage(self) -> Optional[Quantity]:
        """Get High Tension (kV)."""
        pass

    @abstractmethod
    def get_beam_current(self) -> Optional[Quantity]:
        """Get measured beam current (nA)."""
        pass

    @abstractmethod
    def get_spot_size(self) -> int:
        """Get spot size index."""
        pass

    @abstractmethod
    def get_convergence_angle(self) -> Optional[Quantity]:
        """Get alpha/convergence angle (mrad)."""
        pass

    @abstractmethod
    def get_beam_shift(self) -> Tuple[float, float]:
        """Get beam shift coils (x, y)."""
        pass

    @abstractmethod
    def get_condenser_stigmation(self) -> Tuple[float, float]:
        """Get condenser stigmator coils (x, y)."""
        pass

    @abstractmethod
    def get_gun_tilt(self) -> Tuple[float, float]:
        """Get gun tilt alignment (x, y)."""
        pass

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Get beam blanker status (True=Blanked)."""
        pass

    # --- Atomic Layer (Setters) ---

    @abstractmethod
    def set_acceleration_voltage(self, voltage: Quantity) -> None:
        """Set High Tension (kV)."""
        pass

    @abstractmethod
    def set_beam_current(self, current: Quantity) -> None:
        """Set target beam current (nA)."""
        pass

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """Set spot size index."""
        pass

    @abstractmethod
    def set_convergence_angle(self, angle: Quantity) -> None:
        """Set alpha/convergence angle (mrad)."""
        pass

    @abstractmethod
    def set_beam_shift(self, x: float, y: float) -> None:
        """Set beam shift coils."""
        pass

    @abstractmethod
    def set_condenser_stigmation(self, x: float, y: float) -> None:
        """Set condenser stigmator coils."""
        pass

    @abstractmethod
    def set_gun_tilt(self, x: float, y: float) -> None:
        """Set gun tilt alignment coils."""
        pass

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """Set beam blanker status (True=Blanked)."""
        pass

    # --- Logic Layer ---

    def get_beam_settings(self) -> BeamSettings:
        """
        Aggregator: returns full BeamSettings snapshot.
        """
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

    def apply_beam_settings(self, request: BeamControlRequest) -> None:
        """
        Orchestrator: Applies a partial beam configuration.
        Only fields that are NOT None in the request.target are applied.
        """
        tgt = request.target

        # Check safety (optional but recommended hook)
        if self.system_settings.beam_system:
            # Basic check if target is compliant
            check = self.system_settings.beam_system.is_safe_beam(tgt)
            if not check:
                logger.warning(f"Beam settings out of bounds: {check.reasons}")
                # In lenient mode we might warn, strict mode raises.
                # For now, we proceed as the driver might have its own limits.

        if tgt.voltage is not None:
            self.set_acceleration_voltage(tgt.voltage)
        if tgt.beam_current is not None:
            self.set_beam_current(tgt.beam_current)
        if tgt.spot_size is not None:
            self.set_spot_size(tgt.spot_size)
        if tgt.convergence_angle is not None:
            self.set_convergence_angle(tgt.convergence_angle)

        if tgt.beam_shift:
            self.set_beam_shift(tgt.beam_shift.x or 0.0, tgt.beam_shift.y or 0.0)
        if tgt.condenser_stigmation:
            self.set_condenser_stigmation(tgt.condenser_stigmation.x or 0.0,
                                          tgt.condenser_stigmation.y or 0.0)
        if tgt.gun_tilt:
            self.set_gun_tilt(tgt.gun_tilt.x or 0.0, tgt.gun_tilt.y or 0.0)

    # =========================================================================
    # 5. Projection Control (Optics/Imaging)
    # =========================================================================

    # --- Atomic Layer (Getters) ---

    @abstractmethod
    def get_projection_mode(self) -> str:
        """Get optical mode (e.g. 'IMAGING', 'DIFFRACTION')."""
        pass

    @abstractmethod
    def get_magnification_index(self) -> int:
        """Get magnification index."""
        pass

    @abstractmethod
    def get_camera_length(self) -> Optional[Quantity]:
        """Get diffraction camera length (mm)."""
        pass

    @abstractmethod
    def get_defocus(self) -> Optional[Quantity]:
        """Get defocus (nm)."""
        pass

    @abstractmethod
    def get_objective_stigmation(self) -> Tuple[float, float]:
        """Get objective stigmator coils (x, y)."""
        pass

    @abstractmethod
    def get_image_shift(self) -> Tuple[float, float]:
        """Get image shift coils (x, y)."""
        pass

    @abstractmethod
    def get_diffraction_shift(self) -> Tuple[float, float]:
        """Get diffraction shift coils (x, y)."""
        pass

    @abstractmethod
    def get_screen_position(self) -> str:
        """Get fluorescent screen state (e.g. 'UP', 'DOWN')."""
        pass

    # --- Atomic Layer (Setters) ---

    @abstractmethod
    def set_projection_mode(self, mode: str) -> None:
        """Set optical mode."""
        pass

    @abstractmethod
    def set_magnification_index(self, index: int) -> None:
        """Set magnification index."""
        pass

    @abstractmethod
    def set_camera_length(self, length: Quantity) -> None:
        """Set diffraction camera length (mm)."""
        pass

    @abstractmethod
    def set_defocus(self, defocus: Quantity) -> None:
        """Set defocus (nm)."""
        pass

    @abstractmethod
    def set_objective_stigmation(self, x: float, y: float) -> None:
        """Set objective stigmator coils."""
        pass

    @abstractmethod
    def set_image_shift(self, x: float, y: float) -> None:
        """Set image shift coils."""
        pass

    @abstractmethod
    def set_diffraction_shift(self, x: float, y: float) -> None:
        """Set diffraction shift coils."""
        pass

    @abstractmethod
    def set_screen_position(self, position: str) -> None:
        """Set fluorescent screen ('UP' or 'DOWN')."""
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

    def apply_projection_settings(self, request: ProjectionControlRequest) -> None:
        """Orchestrator: Applies a partial projection configuration."""
        tgt = request.target

        # Check system limits (optional hook)
        if self.system_settings.projection_system:
            check = self.system_settings.projection_system.is_safe_projection(tgt)
            if not check:
                logger.warning(f"Projection settings out of bounds: {check.reasons}")

        if tgt.optical_mode is not None:
            self.set_projection_mode(tgt.optical_mode)
        if tgt.magnification_index is not None:
            self.set_magnification_index(tgt.magnification_index)
        if tgt.camera_length is not None:
            self.set_camera_length(tgt.camera_length)
        if tgt.defocus is not None:
            self.set_defocus(tgt.defocus)
        if tgt.screen_position is not None:
            self.set_screen_position(tgt.screen_position)

        if tgt.objective_stigmation:
            self.set_objective_stigmation(tgt.objective_stigmation.x or 0,
                                          tgt.objective_stigmation.y or 0)
        if tgt.image_shift:
            self.set_image_shift(tgt.image_shift.x or 0, tgt.image_shift.y or 0)
        if tgt.diffraction_shift:
            self.set_diffraction_shift(tgt.diffraction_shift.x or 0,
                                       tgt.diffraction_shift.y or 0)

    # =========================================================================
    # 6. Detector & Acquisition
    # =========================================================================

    # --- Atomic Layer ---

    @abstractmethod
    def list_detectors(self) -> List[str]:
        """List available detector identifiers."""
        pass

    @abstractmethod
    def get_active_detector_ids(self) -> List[str]:
        """List currently active/inserted detectors."""
        pass

    @abstractmethod
    def get_detector_settings(self, detector_id: str) -> DetectorSettings:
        """Get current settings (exposure, binning, ROI) for a specific detector."""
        pass

    @abstractmethod
    def set_detector_settings(self, detector_id: str, settings: DetectorSettings) -> None:
        """Configure detector parameters (Exposure, Binning, etc)."""
        pass

    @abstractmethod
    def set_detector_insertion(self, detector_id: str, inserted: bool) -> None:
        """Mechanically insert (True) or retract (False) the detector."""
        pass

    @abstractmethod
    def acquire_image(self, request: AcquisitionRequest) -> MicroscopeImage:
        """
        Execute acquisition: Apply settings -> Expose -> Return Image.

        Args:
            request: AcquisitionRequest containing detector ID, settings, and output options.
        Returns:
            MicroscopeImage containing the raw array and metadata.
        """
        pass

    # --- Logic Layer ---

    def execute_detector_control(self, request: DetectorControlRequest) -> None:
        """
        Orchestrator: Handle mechanical detector actions or configuration updates.
        """
        if request.action == "INSERT":
            self.set_detector_insertion(request.detector_id, True)
        elif request.action == "RETRACT":
            self.set_detector_insertion(request.detector_id, False)
        elif request.action == "COOLDOWN":
            # Optional: vendor specific, can be no-op
            pass

        if request.target:
            self.set_detector_settings(request.detector_id, request.target)

    # =========================================================================
    # 7. Scan (STEM) Control
    # =========================================================================

    # --- Atomic Layer ---

    @abstractmethod
    def get_scan_active(self) -> bool:
        """Check if scanning (rastering) is currently active."""
        pass

    @abstractmethod
    def set_scan_active(self, active: bool) -> None:
        """Start (True) or Stop (False) the beam raster."""
        pass

    @abstractmethod
    def get_scan_settings(self) -> ScanSettings:
        """Get current STEM parameters."""
        pass

    @abstractmethod
    def set_scan_settings(self, settings: ScanSettings) -> None:
        """Set STEM parameters (dwell time, resolution, etc)."""
        pass

    # --- Logic Layer ---

    def execute_scan_control(self, request: ScanControlRequest) -> None:
        """Orchestrator: Handle Scan Start/Stop commands."""
        if request.action == "START":
            # Validate safety
            if self.system_settings.scan_system and request.target:
                check = self.system_settings.scan_system.is_safe_scan(request.target)
                if not check:
                    logger.warning(f"Scan settings out of bounds: {check.reasons}")

            if request.target:
                self.set_scan_settings(request.target)
            self.set_scan_active(True)

        elif request.action == "STOP":
            self.set_scan_active(False)
        elif request.action == "SINGLE_FRAME":
            # Optional implementation dependent logic
            pass

    # =========================================================================
    # 8. Vacuum Control
    # =========================================================================

    # --- Atomic Layer ---

    @abstractmethod
    def get_column_valve_state(self) -> str:
        """Get Column Valve (V7) state: 'OPEN' or 'CLOSED'."""
        pass

    @abstractmethod
    def set_column_valve_state(self, state: str) -> None:
        """Set Column Valve state."""
        pass

    @abstractmethod
    def get_gun_valve_state(self) -> str:
        """Get Gun Valve (V1) state: 'OPEN' or 'CLOSED'."""
        pass

    @abstractmethod
    def set_gun_valve_state(self, state: str) -> None:
        """Set Gun Valve state."""
        pass

    @abstractmethod
    def get_turbo_pump_state(self) -> str:
        """Get Turbo Pump state: 'ON' or 'OFF'."""
        pass

    @abstractmethod
    def set_turbo_pump_state(self, state: str) -> None:
        """Set Turbo Pump state."""
        pass

    @abstractmethod
    def get_pressure(self, gauge: str) -> Quantity:
        """
        Get pressure reading from a named gauge.
        Args:
            gauge: 'column', 'gun', or 'buffer'.
        Returns:
            Pressure in Pascals (Quantity).
        """
        pass

    # --- Logic Layer ---

    def get_vacuum_settings(self) -> VacuumSettings:
        """Aggregator: returns full VacuumSettings snapshot."""
        return VacuumSettings(
            column_valve_state=self.get_column_valve_state(),
            gun_valve_state=self.get_gun_valve_state(),
            turbo_pump_state=self.get_turbo_pump_state(),
            column_pressure=self.get_pressure('column'),
            gun_pressure=self.get_pressure('gun'),
            buffer_tank_pressure=self.get_pressure('buffer')
        )

    def apply_vacuum_settings(self, request: VacuumControlRequest) -> None:
        """Orchestrator: Applies vacuum state changes."""
        tgt = request.target
        # Note: Implementations should enforce safety logic (e.g. don't open valve if pressure high)
        if tgt.column_valve_state:
            self.set_column_valve_state(tgt.column_valve_state)
        if tgt.gun_valve_state:
            self.set_gun_valve_state(tgt.gun_valve_state)
        if tgt.turbo_pump_state:
            self.set_turbo_pump_state(tgt.turbo_pump_state)

    # =========================================================================
    # 9. Aperture Control
    # =========================================================================

    # --- Atomic Layer ---

    @abstractmethod
    def list_apertures(self) -> List[str]:
        """List supported aperture mechanisms (e.g. 'CLA', 'OLA')."""
        pass

    @abstractmethod
    def get_aperture(self, aperture_id: str) -> Aperture:
        """Get state of a specific aperture."""
        pass

    @abstractmethod
    def set_aperture(self, aperture_id: str, target: Aperture) -> None:
        """Set aperture state (Insert/Retract, Size, or Position)."""
        pass

    # --- Logic Layer ---

    def get_all_apertures(self) -> Dict[str, Aperture]:
        """Aggregator: returns state of all apertures."""
        return {a_id: self.get_aperture(a_id) for a_id in self.list_apertures()}

    def execute_aperture_control(self, request: ApertureControlRequest) -> None:
        """
        Orchestrator: Handle aperture changes.
        """
        if request.relative and request.target.position:
            # Calculate absolute position for relative moves
            current = self.get_aperture(request.aperture_id)
            if current.position:
                new_pos = replace(request.target.position)
                # Helper: point addition logic (conceptual)
                # In real impl, use Point addition if defined or manual field sum
                if request.target.position.x is not None and current.position.x is not None:
                    new_pos.x = current.position.x + request.target.position.x
                if request.target.position.y is not None and current.position.y is not None:
                    new_pos.y = current.position.y + request.target.position.y

                # Update target with absolute position
                request.target.position = new_pos

        self.set_aperture(request.aperture_id, request.target)

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

        # Otherwise, calculate steps for linear interpolation
        steps = int(max_dist // max_step_nm) + 1

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

            # Note: Rotations/Tilts are usually not interpolated here unless specified
            # Commit the step
            self.move_stage_absolute(interim, drive_type, wait=True)