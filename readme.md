# SuperTEM

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SuperTEM is a strictly-typed, hardware-agnostic Hardware Abstraction Layer (HAL) and automation framework for Transmission Electron Microscopes (TEMs). 

By enforcing strict boundaries between physical hardware I/O, mathematical state validation, and cognitive scientific algorithms, SuperTEM provides a universal, crash-resistant Python control layer for microscopes from different vendors (e.g., JEOL, Thermo Fisher, Nion).

## 🏛️ The SuperTEM Architecture (The 4 Tiers)

To prevent "spaghetti hardware state" and thread collisions, SuperTEM strictly classifies all operations into four execution tiers:

1. **Protocols (The Scientist):** Declarative YAML configurations that string together complex sequences. They contain zero math, algorithms, or hardware logic.
2. **Routines (The Algorithms):** Goal-oriented, closed-loop software processes (e.g., AutoFocus, Grid Mapping). This is the *only* layer allowed to use `time.sleep()`, run continuous `while` loops, or calculate relative math based on image feedback. Routines never talk to hardware directly.
3. **The HAL / Orchestrator (The Reflexes):** The Control Plane gatekeeper (`base_microscope.py`). It validates intents, performs mathematical bounds checking (`sys.is_safe_...`), and manages state interlocks (e.g., ensuring the gun valve is open before unblanking the beam). 
4. **The Atomic Drivers (The Nerve Endings):** Direct, unbuffered hardware I/O (e.g., `jeol_microscope.py` wrapping `PyJEM`). Pure execution with zero logic. 

## 🛡️ Core Safety Philosophies

SuperTEM is built for production-grade, safety-critical hardware automation. It enforces the following rules:

* **The Dual-Gatekeeper:** Targets (Nouns) undergo mathematical Bounds Checking by the Orchestrator. Actions (Verbs) undergo Contextual State Interlock checking by the Vendor Helper layer.
* **Ingress vs. Egress (Data Plane vs. Control Plane):** * *Ingress (Telemetry):* Uses `ParseMode.LENIENT`. Hardware is messy; if a sensor returns garbage, the system survives and logs the anomaly without crashing.
  * *Egress (Commands):* Uses `ParseMode.STRICT`. If an untrusted payload attempts to move the stage with a string instead of a float, the system fails instantly before the command leaves Python.
* **Null Means Unknown & Fail Loudly:** When querying hardware, a failed read returns `None` (Null means Unknown) to keep polling loops clean. When writing to hardware, a rejected command raises an exception immediately (Fail Loudly).
* **Vendor Payload Mutation:** Proprietary hardware quirks (e.g., JEOL Alpha Selectors) are isolated inside `Extra` dataclasses, allowing the core structures to remain universally compatible without sacrificing vendor-specific power.

## 📂 Project Structure

```text
supertem/
├── structures/         # The Data Plane: Strongly-typed Nouns (Settings, Intents, States)
│   ├── base_structures.py   # Universal canonical models
│   └── jeol_structures.py   # Vendor-specific mutated payloads
├── microscopes/        # The Control Plane: The HAL and Vendor Drivers (Verbs)
│   ├── base_microscope.py   # The Orchestrator (Bounds checking & routing)
│   └── jeol_microscope.py   # The Atomic Driver (PyJEM execution)
├── routines/           # The Cognitive Plane: Algorithms and Math
│   ├── base_routines.py     # Abstract blueprints (e.g., AutoFocusRoutine)
│   ├── jeol_routines.py     # Vendor-optimized routine implementations
│   └── routine_factory.py   # Dynamic router for dependency injection
├── registry.py         # Configuration bootstrap & Dependency Injection context
├── session.py          # Session orchestration and environment setup
└── protocol.py         # The Automation Executor (YAML sequence runner)
```

## 🚀 Installation

```bash
pip install SuperTEM
```
> **Note:** SuperTEM is the abstraction framework. To actually drive hardware, you must install the corresponding vendor SDK in your environment (e.g., `PyJEM` for JEOL).

## ⚡ Quickstart

SuperTEM utilizes explicit Dependency Injection to manage hardware states and environments. 

### 1. Executing a Full Protocol (The Scientist)
Because the `ProtocolExecutor` manages the environment context, logging, and hardware initialization internally, running an entire automated YAML experiment only takes two lines of code.

```python
from supertem.protocol import ProtocolExecutor

# 1. Instantiate the executor (automatically loads the active YAML and sets up logging)
executor = ProtocolExecutor(profile_name="jeol_production")

# 2. Run the entire automated experiment
executor.execute()
```

### 2. Executing a Scientific Workflow (Routine)
If you are writing custom Python scripts and want to utilize a specific algorithm without a YAML file, you can call the Routine directly.

```python
from supertem.registry import SuperTEMContext
from supertem.session import setup_session
from supertem.routines.routine_factory import RoutineFactory

context = SuperTEMContext.production()
scope = setup_session(context=context, session_name="Routine_Test")

# Ask the Factory for the correct Routine based on the connected hardware
factory = RoutineFactory(scope, context, scope._settings)
autofocus = factory.get_routine("autofocus")

# Execute the time-blocking algorithm safely
result = autofocus.execute(target_defocus_nm=0.0)
print(result)
```

### 3. Executing a Direct Command (Action)
If you just want to move the hardware directly via the Orchestrator, bypassing all routines.

```python
from supertem.registry import SuperTEMContext
from supertem.session import setup_session
from supertem.structures.base_structures import StageMoveRequest, StagePosition, Q_, ParseMode

context = SuperTEMContext.production()
scope = setup_session(context=context, session_name="Manual_Run")

# Create a mathematically safe, strictly-parsed request
request = StageMoveRequest(
    target=StagePosition(x=Q_(10, 'um')),
    mode=ParseMode.STRICT
)

# Execute via the Orchestrator (handles bounds-checking and interlocks automatically)
scope.execute_stage_move(request)
```

## 🤝 Contributing

Contributions are welcome! If you are building a new vendor driver (e.g., Thermo Fisher) or adding new complex scientific routines, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/my-new-driver`).
3. **Important:** Ensure your code adheres to the "SuperTEM Constitution" outlined in the module docstrings (especially the Execution Tier rules in `base_microscope.py` and `base_routines.py`).
4. Commit your changes and push to the branch.
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.