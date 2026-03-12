"""
supertem.routines.routine_factory

The dynamic router that supplies vendor-specific routine implementations
to the universal automation protocol.
"""
import logging
from supertem.registry import SuperTEMContext
from supertem.microscopes.base_microscope import TemMicroscope
from supertem.routines.base_routines import (
    BaseRoutine, AutoFocusRoutine, GunAlignmentRoutine,
    MicroscopeInitializationRoutine, SampleSearchRoutine, BeamCenteringRoutine,
    EucentricHeightRoutine, STEMTransitionRoutine, RonchiAstigmatismRoutine,
    ComaFreeAlignmentRoutine, DetectorAlignmentRoutine, STEMAcquisitionRoutine
)

# Import vendor implementations
from supertem.routines.jeol_routines import (
    JeolAutoFocus, JeolGunAlignment,
    JeolMicroscopeInitialization, JeolSampleSearch, JeolBeamCentering,
    JeolEucentricHeight, JeolSTEMTransition, JeolRonchiAstigmatism,
    JeolComaFreeAlignment, JeolDetectorAlignment, JeolSTEMAcquisition
)

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

        # Phase 1: Preparation
        if name == "initialize_microscope": return self.get_initialize_microscope()
        elif name == "search_sample": return self.get_search_sample()

        # Phase 2: Optical Adjustment
        elif name == "center_beam": return self.get_center_beam()
        elif name == "eucentric_height": return self.get_eucentric_height()

        # Phase 3: STEM Transition & Ronchigram
        elif name == "transition_to_stem": return self.get_transition_to_stem()
        elif name == "ronchi_astigmatism": return self.get_ronchi_astigmatism()

        # Phase 4: Alignment
        elif name == "coma_free_alignment": return self.get_coma_free_alignment()
        elif name == "align_detector": return self.get_align_detector()

        # Phase 5: Acquisition
        elif name == "stem_acquisition": return self.get_stem_acquisition()

        # Legacy
        elif name == "autofocus": return self.get_autofocus()
        elif name == "gun_alignment": return self.get_gun_alignment()

        else:
            raise ValueError(f"RoutineFactory: Unknown routine requested '{routine_name}'")

    # =========================================================================
    # Phase 1: Preparation Factory Methods
    # =========================================================================
    def get_initialize_microscope(self) -> MicroscopeInitializationRoutine:
        if "JEOL" in self.manufacturer: return JeolMicroscopeInitialization(self.scope, self.context)
        raise NotImplementedError(f"Initialization not implemented for {self.manufacturer}")

    def get_search_sample(self) -> SampleSearchRoutine:
        if "JEOL" in self.manufacturer: return JeolSampleSearch(self.scope, self.context)
        raise NotImplementedError(f"Sample Search not implemented for {self.manufacturer}")

    # =========================================================================
    # Phase 2: Optical Adjustment Factory Methods
    # =========================================================================
    def get_center_beam(self) -> BeamCenteringRoutine:
        if "JEOL" in self.manufacturer: return JeolBeamCentering(self.scope, self.context)
        raise NotImplementedError(f"Beam Centering not implemented for {self.manufacturer}")

    def get_eucentric_height(self) -> EucentricHeightRoutine:
        if "JEOL" in self.manufacturer: return JeolEucentricHeight(self.scope, self.context)
        raise NotImplementedError(f"Eucentric Height not implemented for {self.manufacturer}")

    # =========================================================================
    # Phase 3: STEM Transition Factory Methods
    # =========================================================================
    def get_transition_to_stem(self) -> STEMTransitionRoutine:
        if "JEOL" in self.manufacturer: return JeolSTEMTransition(self.scope, self.context)
        raise NotImplementedError(f"STEM Transition not implemented for {self.manufacturer}")

    def get_ronchi_astigmatism(self) -> RonchiAstigmatismRoutine:
        if "JEOL" in self.manufacturer: return JeolRonchiAstigmatism(self.scope, self.context)
        raise NotImplementedError(f"Ronchi Astigmatism not implemented for {self.manufacturer}")

    # =========================================================================
    # Phase 4: Alignment Factory Methods
    # =========================================================================
    def get_coma_free_alignment(self) -> ComaFreeAlignmentRoutine:
        if "JEOL" in self.manufacturer: return JeolComaFreeAlignment(self.scope, self.context)
        raise NotImplementedError(f"Coma-Free Alignment not implemented for {self.manufacturer}")

    def get_align_detector(self) -> DetectorAlignmentRoutine:
        if "JEOL" in self.manufacturer: return JeolDetectorAlignment(self.scope, self.context)
        raise NotImplementedError(f"Detector Alignment not implemented for {self.manufacturer}")

    # =========================================================================
    # Phase 5: Acquisition Factory Methods
    # =========================================================================
    def get_stem_acquisition(self) -> STEMAcquisitionRoutine:
        if "JEOL" in self.manufacturer: return JeolSTEMAcquisition(self.scope, self.context)
        raise NotImplementedError(f"STEM Acquisition not implemented for {self.manufacturer}")

    # =========================================================================
    # Legacy Factory Methods
    # =========================================================================
    def get_autofocus(self) -> AutoFocusRoutine:
        if "JEOL" in self.manufacturer: return JeolAutoFocus(self.scope, self.context)
        raise NotImplementedError(f"AutoFocus not implemented for {self.manufacturer}")

    def get_gun_alignment(self) -> GunAlignmentRoutine:
        if "JEOL" in self.manufacturer: return JeolGunAlignment(self.scope, self.context)
        raise NotImplementedError(f"Gun Alignment not implemented for {self.manufacturer}")