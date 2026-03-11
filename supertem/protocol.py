"""
supertem.protocol

The Automation Executor (Tier 1: The "Scientist").

This module operates at the absolute peak of the SuperTEM Orchestration Plane.
It serves as the main execution entry point for automated scientific workflows.

===============================================================================
I. Module Responsibility
===============================================================================
The ProtocolExecutor is responsible for translating declarative, human-readable
YAML experiments into hardware actions.

The Golden Rule (No Math, No Logic):
  This module contains ZERO computer vision algorithms, mathematical calculations,
  or hardware I/O. It is strictly a coordinator. It reads instructions from a
  file and delegates the work downstream.

===============================================================================
II. The Execution Pipeline
===============================================================================
When `executor.execute()` is called, it triggers the entire SuperTEM lifecycle
in a strict dependency-injected sequence:

  1) Context & Registry: Utilizes `RegistryManager` to load the active protocol
     YAML file from the configured environment.
  2) Session Orchestration: Passes the context to `setup_session()` to initialize
     logging, create run directories, and boot the hardware (HAL).
  3) Factory Injection: Injects the active hardware session into the
     `RoutineFactory` to gain access to vendor-specific scientific algorithms.
  4) Step Delegation: Iterates through the YAML sequence:
       - Complex steps (e.g., "autofocus") are delegated to the Cognitive Plane
         via the `RoutineFactory`.
       - Primitive steps (e.g., "stage") are parsed into STRICT requests and
         sent directly to the Control Plane Orchestrator (`base_microscope`).

===============================================================================
III. Error Handling & Safety
===============================================================================
If any step in the sequence fails (whether from a mathematical bounds check in
the Orchestrator or an algorithmic failure in a Routine), the Protocol catches
the exception, logs the abort sequence, and cleanly terminates the experiment
to prevent runaway hardware states.
"""
import yaml
import logging
from supertem.registry import SuperTEMContext, RegistryManager
from supertem.session import setup_session
from supertem.routines.routine_factory import RoutineFactory
from supertem.structures.base_structures import (
    StageMoveRequest, StageControlRequest, BeamControlRequest,
    ProjectionControlRequest, DetectorControlRequest, AcquisitionRequest,
    ParseMode
)

logger = logging.getLogger(__name__)

class ProtocolExecutor:
    def __init__(self, profile_name: str = None):
        self.context = SuperTEMContext.production()
        self.profile_name = profile_name
        self.scope = None
        self.factory = None

    def load_yaml(self) -> dict:
        registry = RegistryManager(self.context)
        protocol_path = registry.get_active_protocol_path()
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load protocol YAML: {e}")
            return {}

    def execute(self):
        # 1. READ INSTRUCTIONS FIRST
        protocol_data = self.load_yaml()
        protocol_name = protocol_data.get("name", "Unnamed_Protocol")
        steps = protocol_data.get("steps", [])

        if not steps:
            logger.warning("No steps found in protocol. Exiting.")
            return

        logger.info(f"Starting Protocol: {protocol_name}")

        # 2. TURN ON MICROSCOPE (Passing the protocol name for the folder creation)
        self.scope = setup_session(
            context=self.context,
            session_name=protocol_name,
            profile_name=self.profile_name
        )

        # 3. INITIALIZE ROUTINES
        # We pass scope._settings because the routines need access to the dynamic image save path
        self.factory = RoutineFactory(self.scope, self.context, self.scope._settings)

        # 4. EXECUTE STEPS
        for i, step in enumerate(steps):
            routine_name = step.get("routine")
            step_type = step.get("type", "").lower()

            try:
                if routine_name:
                    params = step.get("params", {})
                    logger.info(f"--- Step {i+1}: Executing ROUTINE [{routine_name.upper()}] ---")
                    tool = self.factory.get_routine(routine_name)
                    tool.execute(**params)

                elif step_type:
                    raw_payload = step.get("request", {})
                    logger.info(f"--- Step {i+1}: Executing PRIMITIVE [{step_type.upper()}] ---")

                    if step_type == "stage":
                        req = StageControlRequest.from_dict(raw_payload, mode=ParseMode.STRICT)
                        self.scope.execute_stage_control(req)
                    elif step_type == "beam":
                        req = BeamControlRequest.from_dict(raw_payload, mode=ParseMode.STRICT)
                        self.scope.execute_beam_control(req)
                    elif step_type == "acquisition":
                        req = AcquisitionRequest.from_dict(raw_payload, mode=ParseMode.STRICT)
                        self.scope.execute_acquisition(req)
                    else:
                        logger.warning(f"Unknown primitive type: {step_type}")
                else:
                    logger.warning(f"Step {i+1} is malformed. Skipping.")

            except Exception as e:
                logger.error(f"Protocol aborted at step {i+1} due to error: {e}")
                break

        logger.info("Protocol Execution Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s: %(message)s")
    executor = ProtocolExecutor()
    executor.execute()