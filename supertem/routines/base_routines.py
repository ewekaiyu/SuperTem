"""
supertem.routines.base_routine

Abstract blueprints for complex, multi-step hardware workflows.
Universal scripts will type-hint against these classes.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from supertem.microscopes.base_microscope import TemMicroscope
from supertem.config import SuperTEMContext

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