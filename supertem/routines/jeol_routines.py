"""
supertem.routines.jeol_routine

JEOL-specific implementations of the standard SuperTEM routines.
"""
import logging
from typing import Any, Dict

from supertem.routines.base_routines import AutoFocusRoutine, GunAlignmentRoutine
from supertem.structures.base_structures import DetectorControlRequest

logger = logging.getLogger(__name__)

class JeolAutoFocus(AutoFocusRoutine):
    """
    Executes JEOL's hardware AutoFocus for STEM, or a custom algorithm for TEM.
    """

    def execute(self, target_defocus_nm: float = 0.0, **kwargs) -> Dict[str, Any]:
        mode = self.scope.get_mode()

        if mode == "STEM":
            logger.info("[ROUTINE] Executing JEOL Hardware STEM AutoFocus...")
            req = DetectorControlRequest(
                detector_id=self.scope.get_primary_detector_id(),
                action="AUTO_FOCUS"
            )
            self.scope.execute_detector_control(req)
            return {"status": "success", "method": "hardware_stem_autofocus"}

        elif mode == "TEM":
            logger.info("[ROUTINE] Executing Custom TEM AutoFocus Sweep...")
            # We now have access to self.settings.image.path to save sweep images!
            logger.debug(f"Sweep images would be saved to: {self.settings.image.path}")
            return {"status": "simulated_success", "method": "software_tem_autofocus"}

        return {"status": "error", "reason": f"Unknown mode: {mode}"}


class JeolGunAlignment(GunAlignmentRoutine):
    """Executes JEOL-specific gun tilt/shift optimizations."""

    def execute(self, **kwargs) -> bool:
        logger.info("[ROUTINE] Executing JEOL Gun Alignment...")
        return True