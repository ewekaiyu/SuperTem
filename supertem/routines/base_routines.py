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
    """The root class for all executable workflows."""

    def __init__(self, scope: TemMicroscope, context: SuperTEMContext):
        self.scope = scope
        self.context = context
        # Extract settings directly from the scope so routines can access it easily
        self.settings = scope._settings

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the routine. Subclasses must implement this."""
        pass


class AutoFocusRoutine(BaseRoutine):
    """Abstract contract for an AutoFocus procedure."""

    @abstractmethod
    def execute(self, target_defocus_nm: float = 0.0, **kwargs) -> Dict[str, Any]:
        pass

class GunAlignmentRoutine(BaseRoutine):
    """Abstract contract for aligning the electron source."""

    @abstractmethod
    def execute(self, **kwargs) -> bool:
        pass