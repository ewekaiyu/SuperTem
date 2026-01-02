from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from supertem.structures.base import SystemSettings, ImageSettings, TemStagePosition, TemImage




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
    def set_acceleration_voltage(self, kv: float) -> None:
        """Set accelerating voltage in kV."""

    @abstractmethod
    def get_acceleration_voltage(self) -> float:
        """Get accelerating voltage in kV."""

    @abstractmethod
    def get_emission_current(self) -> float:
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
        """Set magnification (TEM) or camera length / equivalent selector (STEM), depending on vendor."""

    @abstractmethod
    def get_magnification(self) -> float:
        """Get magnification (TEM) or camera length (STEM), depending on vendor."""

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """Set spot size index (or equivalent condenser control)."""

    @abstractmethod
    def get_spot_size(self) -> int:
        """Get spot size index."""

    @abstractmethod
    def set_defocus(self, defocus_nm: float) -> None:
        """
        Set defocus.

        Preferably in **nm**, but some vendor APIs expose only "knob units".
        In that case, the concrete implementation should interpret defocus_nm as
        device units (and clearly document it).
        """

    @abstractmethod
    def get_defocus(self) -> float:
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
    def set_aperture(self, kind: str) -> None:
        """Select which aperture is the active target (vendor-defined)."""

    @abstractmethod
    def insert_aperture(self, size: Optional[int]) -> None:
        """Insert the currently selected aperture (if supported)."""

    @abstractmethod
    def retract_aperture(self) -> None:
        """Retract the currently selected aperture (if supported)."""

    # -----------------------
    # Detectors
    # -----------------------

    @abstractmethod
    def list_detectors(self) -> List[str]:
        """List available detector identifiers (names or IDs)."""

    @abstractmethod
    def select_detector(self, detector: str) -> None:
        """Select active detector (vendor-defined)."""

    @abstractmethod
    def get_detector_settings(self) -> Dict[str, Any]:
        """Return current detector settings (brightness/contrast/position/etc)."""

    @abstractmethod
    def set_detector_settings(self, settings: Dict[str, Any]) -> None:
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
        self, pos: TemStagePosition, wait: bool = True, tolerance_nm: float = 10.0
    ) -> None:
        """Move stage to an absolute position."""

    @abstractmethod
    def move_stage_relative(
        self, dx: float, dy: float, dz: float, wait: bool = True, tolerance_nm: float = 10.0
    ) -> None:
        """Move stage relatively by dx/dy/dz (in the same units used in TemStagePosition)."""

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

    def safe_move_stage(self, pos: TemStagePosition, max_step_nm: float = 50000.0) -> None:
        """
        Move in smaller increments to reduce the chance of hitting limits/collisions.

        This default implementation assumes `TemStagePosition.x/y/z` are in **nm** and
        uses `move_stage_relative` to step in equal increments.
        """
        cur = self.get_stage_position()
        if cur.x is None or cur.y is None or cur.z is None:
            # Can't do anything sensible without a current position.
            self.move_stage_absolute(pos, wait=True)
            return

        dx = (pos.x - cur.x) if pos.x is not None else 0.0
        dy = (pos.y - cur.y) if pos.y is not None else 0.0
        dz = (pos.z - cur.z) if pos.z is not None else 0.0

        max_delta = max(abs(dx), abs(dy), abs(dz), 0.0)
        if max_delta <= max_step_nm:
            self.move_stage_absolute(pos, wait=True)
            return

        steps = int(max_delta / max_step_nm) + 1
        step_dx, step_dy, step_dz = dx / steps, dy / steps, dz / steps

        for _ in range(steps):
            self.move_stage_relative(step_dx, step_dy, step_dz, wait=True)