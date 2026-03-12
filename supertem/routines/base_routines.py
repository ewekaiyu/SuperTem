"""
supertem.routines.base_routines

Abstract blueprints for complex, multi-step hardware workflows (Tier 2).

This module operates entirely within the Cognitive Plane of the SuperTEM architecture.
It defines the universal contracts that all vendor-specific algorithms must follow.

===============================================================================
I. Module Responsibility (Tier 2: The Algorithms)
===============================================================================
Unlike the Hardware Abstraction Layer (Tier 3) which must remain stateless and
instantaneous, Routines represent Goal-Oriented, Cognitive Tasks.

Routines ARE explicitly allowed (and expected) to:
  - Block the main execution thread.
  - Use `time.sleep()` and bounded `while` loops for time-series sequences.
  - Analyze images, calculate FFTs, and perform closed-loop feedback logic.
  - Calculate relative math to achieve a scientific target.

===============================================================================
II. The Golden Rule (No Direct Hardware I/O)
===============================================================================
Routines MUST NOT communicate with the hardware drivers directly (e.g., zero
PyJEM imports or atomic hardware calls).

They must accomplish their goals by constructing mathematically safe,
strictly-parsed Request payloads (e.g., `BeamControlRequest`) and delegating
them back down to the Control Plane Orchestrator (`base_microscope`) for execution.

===============================================================================
III. Error Handling & Aborts
===============================================================================
If a Routine fails to achieve its scientific goal (e.g., "Image too dark for
cross-correlation" or "Sample drifted out of bounds"), it must cleanly abort,
attempt to leave the hardware in a baseline safe state (e.g., blanking the beam),
and raise an exception to inform the Tier 1 Protocol.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from supertem.microscopes.base_microscope import TemMicroscope
from supertem.registry import SuperTEMContext


class BaseRoutine(ABC):
    """The root class for all executable workflows (Tier 2)."""

    def __init__(self, scope: TemMicroscope, context: SuperTEMContext):
        self.scope = scope
        self.context = context
        # Extract settings directly from the scope so routines can access it easily
        self.settings = scope._settings

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the routine. Subclasses must implement this."""
        pass


# =============================================================================
# Phase 1: Preparation Routines
# =============================================================================

class MicroscopeInitializationRoutine(BaseRoutine):
    """Abstract contract for waking up the microscope safely."""
    @abstractmethod
    def execute(self, target_aperture: str, auto_emission: bool = True, **kwargs) -> Dict[str, Any]:
        pass

class SampleSearchRoutine(BaseRoutine):
    """Abstract contract for low-mag grid navigation and sample detection."""
    @abstractmethod
    def execute(self, max_mag: int = 40000, **kwargs) -> Dict[str, Any]:
        pass


# =============================================================================
# Phase 2: Optical Adjustment Routines
# =============================================================================

class BeamCenteringRoutine(BaseRoutine):
    """Abstract contract for iteratively balancing brightness, shift, and aperture."""
    @abstractmethod
    def execute(self, iterations: int = 3, **kwargs) -> Dict[str, Any]:
        pass

class EucentricHeightRoutine(BaseRoutine):
    """Abstract contract for utilizing the image wobbler to set Z-height."""
    @abstractmethod
    def execute(self, wobble_angle_mrad: float = 10.0, slight_defocus: bool = True, **kwargs) -> Dict[str, Any]:
        pass


# =============================================================================
# Phase 3: STEM Transition & Ronchigram Tuning
# =============================================================================

class STEMTransitionRoutine(BaseRoutine):
    """Abstract contract for transitioning optical and detector states to STEM mode."""
    @abstractmethod
    def execute(self, camera_length_mm: float = 800.0, insert_gatan_camera: bool = True, lift_screen: bool = True, **kwargs) -> Dict[str, Any]:
        pass

class RonchiAstigmatismRoutine(BaseRoutine):
    """Abstract contract for tuning condenser stig coils using a Ronchigram FFT."""
    @abstractmethod
    def execute(self, target_feature: str = "carbon_film", stig_tolerance: float = 0.95, **kwargs) -> Dict[str, Any]:
        pass


# =============================================================================
# Phase 4: Coma-Free Axis & Detector Alignment
# =============================================================================

class ComaFreeAlignmentRoutine(BaseRoutine):
    """Abstract contract for minimizing breathing via projector/lens deflection."""
    @abstractmethod
    def execute(self, wobble_element: str = "ANDO", compensate_with: str = "PLA", **kwargs) -> Dict[str, Any]:
        pass

class DetectorAlignmentRoutine(BaseRoutine):
    """Abstract contract for centering the focused beam onto the STEM detector."""
    @abstractmethod
    def execute(self, camera_length_mm: float = 250.0, target_detector: str = "ADF", **kwargs) -> Dict[str, Any]:
        pass


# =============================================================================
# Phase 5: Final Acquisition
# =============================================================================

class STEMAcquisitionRoutine(BaseRoutine):
    """Abstract contract for ensuring pure optical state and triggering capture."""
    @abstractmethod
    def execute(self, target_aperture: str = "CL_40", retract_cameras: bool = True, output_dir: str = "", **kwargs) -> Dict[str, Any]:
        pass


# =============================================================================
# Legacy / Other Routines
# =============================================================================

class AutoFocusRoutine(BaseRoutine):
    """Abstract contract for an AutoFocus procedure."""
    @abstractmethod
    def execute(self, target_defocus_nm: float = 0.0, **kwargs) -> Dict[str, Any]:
        pass

class GunAlignmentRoutine(BaseRoutine):
    """Abstract contract for aligning the electron source."""
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass