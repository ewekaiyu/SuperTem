"""
supertem.routines.routine_factory

The dynamic router that supplies vendor-specific routine implementations
to the universal automation protocol.
"""
import logging
from supertem.config import SuperTEMContext
from supertem.microscopes.base_microscope import TemMicroscope
from supertem.routines.base_routines import BaseRoutine, AutoFocusRoutine, GunAlignmentRoutine

# Import vendor implementations
from supertem.routines.jeol_routines import JeolAutoFocus, JeolGunAlignment

logger = logging.getLogger(__name__)

class RoutineFactory:
    """
    Generates the correct routine classes based on the connected microscope hardware.
    """

    def __init__(self, scope: TemMicroscope, context: SuperTEMContext):
        self.scope = scope
        self.context = context

        # Safely determine the manufacturer from the SystemInfo
        info = self.scope.get_instrument_info()
        self.manufacturer = (info.manufacturer or "UNKNOWN").upper()

    def get_routine(self, routine_name: str) -> BaseRoutine:
        """
        Dynamic router used by protocol.py to fetch tools by name.
        """
        name = routine_name.strip().lower()
        if name == "autofocus":
            return self.get_autofocus()
        elif name == "gun_alignment":
            return self.get_gun_alignment()
        else:
            raise ValueError(f"RoutineFactory: Unknown routine requested '{routine_name}'")

    def get_autofocus(self) -> AutoFocusRoutine:
        """Returns a vendor-specific AutoFocusRoutine."""
        if "JEOL" in self.manufacturer:
            return JeolAutoFocus(self.scope, self.context)
        # elif "THERMO" in self.manufacturer:
        #     return ThermoAutoFocus(self.scope, self.context)
        else:
            logger.error(f"AutoFocus not implemented for manufacturer: {self.manufacturer}")
            raise NotImplementedError(f"No AutoFocus routine for {self.manufacturer}")

    def get_gun_alignment(self) -> GunAlignmentRoutine:
        """Returns a vendor-specific Gun Alignment routine."""
        if "JEOL" in self.manufacturer:
            return JeolGunAlignment(self.scope, self.context)
        else:
            raise NotImplementedError(f"No Gun Alignment routine for {self.manufacturer}")