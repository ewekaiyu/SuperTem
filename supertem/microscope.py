from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from pint import Quantity
import math

from supertem.structures.base import SystemSettings, ImageSettings, TemStagePosition, TemImage, Q_, ensure_quantity, magnitude, TemDetectorSettings


class TemMicroscope(ABC):
    """
    Abstract base class: define the smallest useful "atomic" TEM operations.

    Vendor implementations (JEOL / ThermoFisher / Hitachi / ...) should inherit this class and
    implement these methods using their own control libraries (PyJEM, AutoScript, etc.).
    """

    # -----------------------
    # Connection / Status
    # -----------------------

    @abstractmethod
    def connect_to_microscope(self, ip_address: str, port: int, timeout_s: float = 5.0) -> None:
        """Connect to the microscope control interface."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect / release resources."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the microscope is connected."""

    @abstractmethod
    def get_instrument_info(self) -> Dict[str, Any]:
        """Return instrument identification, version, model, etc."""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the microscope status (stage, vacuum, beam, etc.)."""

    # -----------------------
    # Imaging mode / function mode
    # -----------------------

    @abstractmethod
    def get_mode(self) -> str:
        """
        Return current observation mode as a string.

        Recommended convention:
          - "TEM:<FUNCTION>"  e.g. "TEM:MAG", "TEM:DIFF"
          - "STEM:<FUNCTION>" e.g. "STEM:SM-MAG"
        Implementations can return just "TEM" / "STEM" if function mode is not available.
        """

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Set observation mode. Accepts "TEM", "STEM", or "TEM:DIFF", etc (implementation-defined)."""

    # -----------------------
    # Beam / HT / Emission
    # -----------------------

    @abstractmethod
    def set_acceleration_voltage(self, voltage: Optional[Quantity]) -> None:
        """Set accelerating voltage (Quantity)."""

    @abstractmethod
    def get_acceleration_voltage(self) -> Optional[Quantity]:
        """Get accelerating voltage as a Quantity."""

    @abstractmethod
    def get_emission_current(self) -> Optional[Quantity]:
        """Get emission current (units are vendor-defined, often µA or nA)."""

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """Enable/disable beam blanking."""

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Return current beam blank status."""

    # -----------------------
    # Optics / Imaging (Mag / Spot / Focus / Stig)
    # -----------------------

    @abstractmethod
    def set_magnification(self, mag: float) -> None:
        """Set magnification (unitless, in X)."""

    @abstractmethod
    def get_magnification(self) -> float:
        """Get magnification (unitless, in X)."""


    @abstractmethod
    def set_camera_length(self, camera_length: Optional[Quantity]) -> None:
        """Set camera length (a length Quantity, e.g. cm or m)."""

    @abstractmethod
    def get_camera_length(self) -> Optional[Quantity]:
        """Get camera length as a Quantity (or None if not applicable)."""

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """Set spot size index (or equivalent condenser control)."""

    @abstractmethod
    def get_spot_size(self) -> int:
        """Get spot size index."""

    @abstractmethod
    def set_defocus(self, defocus: Optional[Quantity]) -> None:
        """
        Set defocus.

        Preferably in **nm**, but some vendor APIs expose only "knob units".
        In that case, the concrete implementation should interpret defocus_nm as
        device units (and clearly document it).
        """

    @abstractmethod
    def get_defocus(self) -> Optional[Quantity]:
        """Get current defocus in nm (or device units; see implementation)."""

    @abstractmethod
    def set_stigmation(self, x: float, y: float) -> None:
        """
        Set stigmation (x, y).
        Units may be vendor-defined; implementers should document.
        """

    @abstractmethod
    def get_stigmation(self) -> Tuple[float, float]:
        """Get current stigmation (x, y)."""

    @abstractmethod
    def align_beam(self) -> None:
        """Run a basic beam alignment routine if supported (optional / vendor-defined)."""

    # -----------------------
    # Apertures
    # -----------------------
    @abstractmethod
    def list_apertures(self) -> List[str]:
        """Return supported aperture 'kinds' or names for this microscope."""

    @abstractmethod
    def get_aperture_status(self) -> Dict[str, Any]:
        """Select which aperture is the active target (vendor-defined)."""

    @abstractmethod
    def insert_aperture(self, kind: str, size: Optional[int]) -> None:
        """Insert the currently selected aperture (if supported)."""

    @abstractmethod
    def retract_aperture(self, kind: str) -> None:
        """Retract the currently selected aperture (if supported)."""

    # -----------------------
    # Detectors
    # -----------------------

    @abstractmethod
    def list_detectors(self) -> List[str]:
        """List available detector identifiers (names or IDs)."""

    @abstractmethod
    def select_detector(self, name: str) -> None:
        """Select active detector (vendor-defined)."""

    @abstractmethod
    def get_detector_settings(self) -> TemDetectorSettings:
        """Return current detector settings (brightness/contrast/position/etc)."""

    @abstractmethod
    def set_detector_settings(self, settings: Optional[TemDetectorSettings]) -> None:
        """Apply detector settings (brightness/contrast/position/etc)."""

    # -----------------------
    # Image acquisition / Live
    # -----------------------

    @abstractmethod
    def acquire_image(self, settings: Optional[ImageSettings] = None) -> TemImage:
        """Acquire a still image with optional ImageSettings."""

    @abstractmethod
    def start_live(self, settings: Optional[ImageSettings] = None) -> None:
        """Start live imaging / continuous acquisition if supported."""

    @abstractmethod
    def stop_live(self) -> None:
        """Stop live imaging."""

    @abstractmethod
    def get_live_frame(self) -> bytes:
        """Return one live frame (implementation-defined encoding, e.g. TIFF/PNG/raw bytes)."""

    # -----------------------
    # Stage control
    # -----------------------

    @abstractmethod
    def get_stage_position(self) -> TemStagePosition:
        """Return current stage position."""

    @abstractmethod
    def move_stage_absolute(
        self, pos: TemStagePosition, exact: bool = False, tolerance_length: Optional[Quantity] = None,
        tolerance_angle: Optional[Quantity] = None
    ) -> None:
        """Move stage to an absolute position. Set exact to True to ensure exact position within tolerance."""

    @abstractmethod
    def move_stage_relative(
        self, pos: TemStagePosition, exact: bool = False, tolerance_length: Optional[Quantity] = None,
        tolerance_angle: Optional[Quantity] = None
    ) -> None:
        """Move stage relatively. Set exact to True to ensure exact position within tolerance."""

    @abstractmethod
    def set_stage_drive_mode(self, mode: str) -> None:
        """Set stage drive mode, e.g. 'motor' or 'piezo' (vendor-defined)."""

    @abstractmethod
    def stop_stage(self) -> None:
        """Stop stage motion."""

    @abstractmethod
    def get_stage_status(self) -> Dict[str, Any]:
        """Return stage status (axis status, errors, limits, etc)."""

    @abstractmethod
    def insert_holder(self) -> None:
        """Insert sample holder (if supported)."""

    @abstractmethod
    def retract_holder(self) -> None:
        """Retract sample holder (if supported)."""

    # -----------------------
    # Automation hooks / Diagnostics
    # -----------------------
    @abstractmethod
    def run_autofunction(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a vendor/user-defined auto-function (autofocus, autostig, etc)."""

    @abstractmethod
    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover/declare supported capabilities for this implementation."""

    @abstractmethod
    def get_log(self, n: int = 100) -> List[str]:
        """Return last n log lines (if supported)."""

    @abstractmethod
    def send_raw_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Escape hatch: send a raw vendor command (implementation-defined)."""

    # -----------------------
    # Convenience: safe stage movement (default implementation)
    # -----------------------

    def safe_move_stage(
            self,
            pos: TemStagePosition,
            max_step: Optional[Quantity] = None,
            *,
            exact: bool = True,
            tolerance_length: Optional[Quantity] = None,
            tolerance_angle: Optional[Quantity] = None,
    ) -> None:
        """
        Move stage to `pos` in safe incremental steps.

        Each axis (x, y, z) moves one at a time.
        If delta > max_step, move in multiple smaller steps.
        Final move is an exact absolute move to target (handles tilt).
        """

        cur = self.get_stage_position()
        max_step = Q_(1, "micrometer")
        max_step = ensure_quantity(max_step, "nanometer")
        pos.x = ensure_quantity(pos.x, "nanometer")
        pos.y = ensure_quantity(pos.y, "nanometer")
        pos.z = ensure_quantity(pos.z, "nanometer")
        pos.tilt_x = ensure_quantity(pos.tilt_x, "degree")
        pos.tilt_y = ensure_quantity(pos.tilt_y, "degree")

        # --- Helper to move one axis safely ---
        def _move_axis(axis: str, target: Quantity):
            cur_val = getattr(cur, axis)
            delta = target - cur_val
            step_nm = max_step.to("nanometer").magnitude
            delta_nm = delta.to("nanometer").magnitude

            if abs(delta_nm) <= step_nm:
                step_target = TemStagePosition(**{axis: target})
                self.move_stage_absolute(
                    pos=step_target,
                    exact=False,
                    tolerance_length=tolerance_length,
                    tolerance_angle=tolerance_angle,
                )
                setattr(cur, axis, target)
                return

            n_steps = int(abs(delta_nm) // step_nm)
            step = delta / (n_steps + 1)
            for _ in range(n_steps):
                next_pos = TemStagePosition(**{axis: getattr(cur, axis) + step})
                self.move_stage_absolute(
                    pos=next_pos,
                    exact=False,
                    tolerance_length=tolerance_length,
                    tolerance_angle=tolerance_angle,
                )
                setattr(cur, axis, getattr(cur, axis) + step)

            # Final step to target
            self.move_stage_absolute(
                pos=TemStagePosition(**{axis: target}),
                exact=False,
                tolerance_length=tolerance_length,
                tolerance_angle=tolerance_angle,
            )
            setattr(cur, axis, target)

        # --- Step each axis independently ---
        if pos.x is not None:
            _move_axis("x", pos.x)
        if pos.y is not None:
            _move_axis("y", pos.y)
        if pos.z is not None:
            _move_axis("z", pos.z)

        # --- Final absolute move for tilt / precision ---
        self.move_stage_absolute(
            pos=pos,
            exact=exact,
            tolerance_length=tolerance_length,
            tolerance_angle=tolerance_angle,
        )
