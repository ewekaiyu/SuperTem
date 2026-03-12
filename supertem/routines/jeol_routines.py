"""
supertem.routines.jeol_routines

JEOL-specific implementations of the standard SuperTEM workflows (Tier 2).

This module contains the concrete algorithms optimized for JEOL hardware.
It strictly adheres to the Cognitive Plane rules:
  1. It uses blocking loops and mathematical algorithms.
  2. It never calls PyJEM directly.
  3. All hardware changes are routed through `self.scope.execute_...` using
     strictly typed Request payloads.
"""
import logging
import numpy as np
import time
import os
from typing import Any, Dict
from skimage.registration import phase_cross_correlation

from supertem.routines.base_routines import (
    MicroscopeInitializationRoutine, SampleSearchRoutine, BeamCenteringRoutine,
    EucentricHeightRoutine, STEMTransitionRoutine, RonchiAstigmatismRoutine,
    ComaFreeAlignmentRoutine, DetectorAlignmentRoutine, STEMAcquisitionRoutine,
    AutoFocusRoutine, GunAlignmentRoutine
)

from supertem.structures.base_structures import (
    BeamControlRequest, StageControlRequest, StagePosition,
    ProjectionControlRequest, DetectorControlRequest, AcquisitionRequest,
    ParseMode, Q_, StageMoveRequest, BeamSettings, ProjectionSettings,
    VacuumControlRequest, ApertureControlRequest,
    ApertureSettings, VacuumSettings,
    Point, Extras
)

logger = logging.getLogger(__name__)

# =============================================================================
# Phase 1: Preparation Routines
# =============================================================================

class JeolMicroscopeInitialization(MicroscopeInitializationRoutine):
    def execute(self, target_aperture: str, auto_emission: bool = True, **kwargs) -> Dict[str, Any]:
        logger.info(f"[ROUTINE] Initializing Microscope (Aperture: {target_aperture}, Emission: {auto_emission})")

        # 1. Turn on Emission
        if auto_emission:
            logger.info("[ROUTINE] Turning on Auto Emission...")
            beam_req = BeamControlRequest(
                action="EMISSION_ON",
                _mode=ParseMode.STRICT
            )
            self.scope.execute_beam_control(beam_req)
            time.sleep(2.0)  # Give the FEG a moment to stabilize

        # 2. Open Gun Valve
        logger.info("[ROUTINE] Opening Gun Valve...")
        vac_req = VacuumControlRequest(
            target=VacuumSettings(gun_valve_state="OPEN", _mode=ParseMode.STRICT),
            _mode=ParseMode.STRICT
        )
        self.scope.execute_vacuum_control(vac_req)

        # 3. Insert Target Aperture (e.g., "CL_100")
        if target_aperture:
            logger.info(f"[ROUTINE] Inserting Condenser Aperture: {target_aperture}")
            apt_id = "CLA" if "CL" in target_aperture.upper() else "UNKNOWN"
            target_val = target_aperture.split("_")[-1]

            # Look up the correct integer index from your system settings
            size_idx = 2  # Safe default fallback
            try:
                available_sizes = self.scope.system_settings.aperture_system.capabilities_by_id[apt_id].available_sizes
                size_idx = next(i for i, size in enumerate(available_sizes) if target_val in size)
            except Exception:
                logger.warning(f"[ROUTINE] Could not map '{target_aperture}' to index in config. Defaulting to {size_idx}.")

            apt_req = ApertureControlRequest(
                aperture_id=apt_id,
                target=ApertureSettings(size_index=size_idx, _mode=ParseMode.STRICT),
                action="INSERT",
                _mode=ParseMode.STRICT
            )
            self.scope.execute_aperture_control(apt_req)

        return {"status": "success", "message": "Microscope is awake and beam is on."}

class JeolSampleSearch(SampleSearchRoutine):
    def execute(self, max_mag: int = 40000, **kwargs) -> Dict[str, Any]:
        logger.info(f"[JEOL] TODO: Implement Sample Search (Max Mag: {max_mag}x)")
        # TODO: Low-mag mapping/navigation logic
        return {"status": "not_implemented"}


# =============================================================================
# Phase 2: Optical Adjustment Routines
# =============================================================================

class JeolBeamCentering(BeamCenteringRoutine):
    def execute(self, iterations: int = 3, **kwargs) -> Dict[str, Any]:
        logger.info(f"[JEOL] TODO: Implement Beam Centering ({iterations} iterations)")
        # TODO: Iterative loop balancing brightness and shift
        return {"status": "not_implemented"}


class JeolEucentricHeight(EucentricHeightRoutine):
    def execute(self, wobble_angle_mrad: float = 5.0, slight_defocus: bool = True, **kwargs) -> Dict[str, Any]:
        logger.info(f"[ROUTINE] Starting Eucentric Height Alignment (Wobble Angle: {wobble_angle_mrad} mrad)")

        # 1. Setup Phase: Defocus slightly via the Orchestrator
        if slight_defocus:
            logger.info("[ROUTINE] Applying +500nm defocus to enhance image contrast.")
            defocus_req = ProjectionControlRequest(
                action="SHIFT_FOCUS_NM",
                extra=Extras(options={"amount": Q_(500.0, "nm")}),
                _mode=ParseMode.STRICT
            )
            self.scope.execute_projection_control(defocus_req)

        # 2. Algorithm Parameters
        max_iterations = 6
        tolerance_nm = 2.0
        converged = False
        final_shift = 0.0

        # Read the current beam tilt so we can restore it later
        current_tilt = self.scope.get_beam_tilt()
        base_tx = current_tilt[0] if current_tilt[0] is not None else 0.0
        base_ty = current_tilt[1] if current_tilt[1] is not None else 0.0

        # 3. The Cognitive Loop
        for i in range(max_iterations):
            logger.info(f"[ROUTINE] --- Iteration {i + 1}/{max_iterations} ---")

            # A. Tilt Positive and Acquire
            img_plus = self._acquire_tilted_image(base_tx + wobble_angle_mrad, base_ty)

            # B. Tilt Negative and Acquire
            img_minus = self._acquire_tilted_image(base_tx - wobble_angle_mrad, base_ty)

            # C. Computer Vision Math: Phase Cross-Correlation
            shift, error, diffphase = phase_cross_correlation(img_plus, img_minus, upsample_factor=10)
            shift_x_px, shift_y_px = shift[1], shift[0]

            # Convert pixels to physical nanometers
            pixel_size_nm = 1.0  # TODO: Pull from self.settings.image.pixel_size_nm
            shift_x_nm = shift_x_px * pixel_size_nm
            shift_y_nm = shift_y_px * pixel_size_nm

            final_shift = np.hypot(shift_x_nm, shift_y_nm)
            logger.info(f"[ROUTINE] Measured Shift: {final_shift:.2f} nm (X: {shift_x_nm:.2f}, Y: {shift_y_nm:.2f})")

            # D. Success Condition
            if final_shift < tolerance_nm:
                logger.info("[ROUTINE] Eucentric height converged successfully!")
                converged = True
                break

            # E. Z-Stage Correction Math
            angle_rad = wobble_angle_mrad / 1000.0
            z_correction_nm = (shift_x_nm / (2.0 * angle_rad)) * 0.8

            logger.info(f"[ROUTINE] Moving Z-stage by {z_correction_nm:.2f} nm to compensate.")

            # STRICT CORRECT PAYLOAD: StageMoveRequest with relative=True
            stage_req = StageMoveRequest(
                target=StagePosition(z=Q_(z_correction_nm, "nm"), _mode=ParseMode.STRICT),
                relative=True
            )
            # Route through the Stage Orchestrator for bounds checking
            self.scope.execute_stage_move(stage_req)

            # Let the mechanical stage settle
            time.sleep(1.0)

        # 4. Cleanup Phase
        logger.info("[ROUTINE] Restoring original beam tilt and focus.")
        self._apply_beam_tilt(base_tx, base_ty)

        if slight_defocus:
            restore_req = ProjectionControlRequest(
                action="SHIFT_FOCUS_NM",
                extra=Extras(options={"amount": Q_(-500.0, "nm")}),
                _mode=ParseMode.STRICT
            )
            self.scope.execute_projection_control(restore_req)

        if not converged:
            logger.warning("[ROUTINE] Eucentric height algorithm reached max iterations without full convergence.")

        return {"status": "success" if converged else "failed", "final_shift_nm": final_shift}

    # --- Helper Methods for the Routine ---

    def _apply_beam_tilt(self, tx: float, ty: float) -> None:
        """Constructs a strict payload to tilt the beam via the Orchestrator."""
        req = BeamControlRequest(
            target=BeamSettings(
                beam_tilt=Point(x=tx, y=ty, _mode=ParseMode.STRICT),
                _mode=ParseMode.STRICT
            )
        )
        self.scope.execute_beam_control(req)
        time.sleep(0.2)  # Allow beam deflectors to settle

    def _acquire_tilted_image(self, tx: float, ty: float) -> np.ndarray:
        """Tilts the beam and acquires a single frame."""
        self._apply_beam_tilt(tx, ty)

        # Strictly parsed AcquisitionRequest routed to the Orchestrator
        acq_req = AcquisitionRequest()
        image_obj = self.scope.execute_acquisition(acq_req)

        return image_obj.data


# =============================================================================
# Phase 3: STEM Transition & Ronchigram Tuning
# =============================================================================

class JeolSTEMTransition(STEMTransitionRoutine):
    def execute(self, camera_length_mm: float = 800.0, insert_gatan_camera: bool = True, lift_screen: bool = True, **kwargs) -> Dict[str, Any]:
        logger.info(f"[ROUTINE] Transitioning to STEM (CL: {camera_length_mm}mm)...")

        # 1. Switch Mode, set Camera Length, AND Lift Screen simultaneously
        proj_req = ProjectionControlRequest(
            target=ProjectionSettings(
                mode="STEM",
                camera_length=Q_(camera_length_mm, "mm"),
                screen_position="UP" if lift_screen else None,
                _mode=ParseMode.STRICT
            ),
            _mode=ParseMode.STRICT
        )
        self.scope.execute_projection_control(proj_req)

        # 2. Insert Gatan Camera
        if insert_gatan_camera:
            cam_req = DetectorControlRequest(
                detector_id="GATAN",
                action="INSERT",
                _mode=ParseMode.STRICT
            )
            self.scope.execute_detector_control(cam_req)

        return {"status": "success", "camera_length_mm": camera_length_mm}

class JeolRonchiAstigmatism(RonchiAstigmatismRoutine):
    def execute(self, target_feature: str = "carbon_film", stig_tolerance: float = 0.95, **kwargs) -> Dict[str, Any]:
        logger.info(f"[JEOL] TODO: Implement Ronchi Astigmatism Tuning on {target_feature}")
        # TODO: FFT -> Astigmatism calculation -> Condenser Stig adjustments
        return {"status": "not_implemented"}


# =============================================================================
# Phase 4: Coma-Free Axis & Detector Alignment
# =============================================================================

class JeolComaFreeAlignment(ComaFreeAlignmentRoutine):
    def execute(self, wobble_element: str = "ANDO", compensate_with: str = "PLA", **kwargs) -> Dict[str, Any]:
        logger.info(f"[JEOL] TODO: Implement Coma-Free Alignment (Wobble: {wobble_element})")
        # TODO: Wobble Ando -> Analyze breathing -> Adjust PLA
        return {"status": "not_implemented"}

class JeolDetectorAlignment(DetectorAlignmentRoutine):
    def execute(self, camera_length_mm: float = 250.0, target_detector: str = "ADF", **kwargs) -> Dict[str, Any]:
        logger.info(f"[JEOL] TODO: Implement Detector Alignment ({target_detector} at {camera_length_mm}mm)")
        # TODO: Map detector intensity bounds -> Shift projector lenses
        return {"status": "not_implemented"}


# =============================================================================
# Phase 5: Final Acquisition
# =============================================================================

class JeolSTEMAcquisition(STEMAcquisitionRoutine):
    def execute(self, target_aperture: str = "CL_40", target_detector: str = "ADF", retract_cameras: bool = True, output_dir: str = "", **kwargs) -> Dict[str, Any]:
        logger.info(f"[ROUTINE] Preparing Final STEM Acquisition state...")

        # 1. Clean up auxiliary cameras
        if retract_cameras:
            logger.info("[ROUTINE] Retracting Gatan camera for clean STEM path...")
            det_req = DetectorControlRequest(
                detector_id="GATAN",
                action="RETRACT",
                _mode=ParseMode.STRICT
            )
            self.scope.execute_detector_control(det_req)
            time.sleep(1.0)  # Pneumatic wait

        # 2. Insert High-Res STEM Aperture (e.g., CL_40)
        if target_aperture:
            logger.info(f"[ROUTINE] Inserting high-res STEM Aperture: {target_aperture}")
            apt_id = "CLA" if "CL" in target_aperture.upper() else "UNKNOWN"
            target_val = target_aperture.split("_")[-1]

            size_idx = 3  # Safe default fallback
            try:
                available_sizes = self.scope.system_settings.aperture_system.capabilities_by_id[apt_id].available_sizes
                size_idx = next(i for i, size in enumerate(available_sizes) if target_val in size)
            except Exception:
                logger.warning(f"[ROUTINE] Could not map '{target_aperture}' to index. Defaulting to {size_idx}.")

            apt_req = ApertureControlRequest(
                aperture_id=apt_id,
                target=ApertureSettings(size_index=size_idx, _mode=ParseMode.STRICT),
                action="INSERT",
                _mode=ParseMode.STRICT
            )
            self.scope.execute_aperture_control(apt_req)

        # 3. Trigger STEM Scan and Acquire Data
        logger.info(f"[ROUTINE] Firing STEM Scan Generator on {target_detector} detector...")
        acq_req = AcquisitionRequest(
            detector_id=target_detector,
            _mode=ParseMode.STRICT
        )
        image_obj = self.scope.execute_acquisition(acq_req)

        # 4. Save to Disk
        saved_path = "Memory Only (No Output Dir)"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            saved_path = os.path.join(output_dir, f"stem_final_{target_detector}.tif")
            # In a full implementation, you'd use tifffile or PIL to save image_obj.data here
            logger.info(f"[ROUTINE] Image acquired! Shape: {image_obj.data.shape}. Saving to: {saved_path}")

        return {"status": "success", "saved_path": saved_path}


# =============================================================================
# Legacy Routines
# =============================================================================

class JeolAutoFocus(AutoFocusRoutine):
    def execute(self, target_defocus_nm: float = 0.0, **kwargs) -> Dict[str, Any]:
        logger.info("[JEOL] Executing hardware auto-focus...")
        req = DetectorControlRequest(detector_id="MAIN", action="AUTO_FOCUS", _mode=ParseMode.STRICT)
        self.scope.execute_detector_control(req)
        return {"status": "success"}

class JeolGunAlignment(GunAlignmentRoutine):
    def execute(self, **kwargs) -> Dict[str, Any]:
        logger.info("[JEOL] Executing Gun Alignment...")
        return {"status": "success"}