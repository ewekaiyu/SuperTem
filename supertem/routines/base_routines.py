"""
supertem.routines.base_routines

Abstract blueprints for complex, multi-step hardware workflows.
Universal scripts will type-hint against these classes.

===============================================================================
The Routine Philosophy (The "Smart Algorithms" / The "Puppeteers")
===============================================================================
Unlike the Hardware Abstraction Layer (HAL) which must remain stateless and
instantaneous, Routines represent Goal-Oriented, Cognitive Tasks.

Routines ARE explicitly allowed to:
  - Block the main thread.
  - Use `time.sleep()` and `while` loops for time-series operations (e.g., Wobblers).
  - Analyze images, calculate FFTs, and perform closed-loop feedback logic.
  - Calculate relative math to achieve a scientific goal.

The Golden Rule:
  Routines MUST NOT communicate with the hardware drivers directly (no PyJEM calls).
  They must accomplish their goals by constructing mathematically safe,
  strictly-parsed Request payloads (e.g., BeamControlRequest) and sending them
  to the `TemMicroscope` Orchestrator.
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