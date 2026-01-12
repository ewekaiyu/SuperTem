"""
supertem.config

Configuration Bootstrap, Registry Management, and Default Factory.

This module serves as the entry point for the application's state, responsible for:
  - Bootstrapping the runtime environment (filesystem, logs, databases).
  - Managing the "Registry of Configurations" (Active vs. Available profiles).
  - Providing the "Factory Reset" defaults compliant with Schema v2.
  - Atomic persistence of configuration changes to disk.

It acts as the *Source of Truth* that feeds the `base.py` ingestion lifecycle.

===============================================================================
I. The Bootstrap Lifecycle
===============================================================================

On import, this module executes a self-healing initialization sequence:

  1) Environment Validation
     - Ensures all required directory trees exist (`log/`, `data/`, `db/`).
     - Prevents "FileNotFound" crashes in downstream modules.

  2) Default Generation (The "Safe Mode")
     - Checks for the existence of critical YAML definitions.
     - If missing, atomically writes the `DEFAULT_` dictionaries defined in this
       file to disk.
     - GOAL: The system is always runnable, even on a fresh install or after
       user configuration corruption.

  3) Registry Loading
     - Loads the "Index" files (`microscope-config-index.yaml`).
     - Resolves the "Active" configuration path.
     - Fallback: If the active config is missing, reverts to the internal default.

===============================================================================
II. Schema v2: Alignment with supertem.structure.base
===============================================================================

The `DEFAULT_MICROSCOPE_CONFIGURATION_YAML` defined here is strictly typed to
match the `MicroscopeSettings` dataclass hierarchy in `base.py`.

1. Strict Hierarchy (Root -> System -> Subsystem)
   - OLD: Flat dictionary (`stage: {}`, `beam: {}`).
   - NEW: Nested Object Graph.
     `MicroscopeSettings` -> `system` -> `stage_system` -> `x_limits_nm`
   - Rationale: Separates "Hardware Limits" (System) from "User Preferences" (Image).

2. Unit-Explicit Naming (The "Zero Ambiguity" Rule)
   - Keys MUST include their unit suffix to satisfy `base.py`'s strict alias mapping.
   - Examples:
       `x_limits` -> `x_limits_nm`
       `voltage`  -> `voltage_limits_kv`
       `exposure` -> `exposure_ms`
   - Rationale: Prevents `50` from being interpreted as "50 meters" by Pint.

3. Separation of Concerns: Limits vs. State
   - **Limits (Gatekeeping):** Defined at the `system` level (e.g., `voltage_limits_kv`).
     These are the hard boundaries the `Validator` enforces.
   - **State (Snapshots):** Defined in nested defaults (e.g., `default_beam`).
     These are the values applied during a system reset or `Lens.normalize()`.

4. The Detector Registry
   - Requires explicit `available_detectors` list.
   - Requires `capabilities_by_id` to enable `DetectorSystemSettings.is_supported()`.

===============================================================================
III. Registry & Profile Management
===============================================================================

SuperTEM supports multiple hardware profiles (e.g., "Simulated", "JEOL-2100",
"Thermo-Krios"). This module acts as the Librarian for these profiles.

- **The Index:** A YAML file mapping human-readable names to file paths.
- **The Active Profile:** The `default` key in the index determines which file
  is loaded by the main application logic.
- **Atomic I/O:** All writes to index files use `_atomic_dump_yaml` (write to tmp
  -> OS replace) to prevent corruption if the power fails during a write.

===============================================================================
IV. Usage Guide
===============================================================================

1. Accessing Configuration
   - Do NOT import `DEFAULT_...` dicts directly for runtime logic.
   - Use `get_microscope_config_path(name)` to load specific settings.
   - Use `DEFAULT_CONFIGURATION_PATH` for the currently active profile.

2. Modifying the Registry
   - Use `add_microscope_config(...)` to register a new YAML file.
   - Use `set_default_microscope_config(...)` to switch the active machine.
   - These changes persist immediately to `microscope-config-index.yaml`.

"""
from __future__ import annotations

import os
import yaml
import logging

import supertem

# =============================================================================
# Constants & Versions
# =============================================================================

METADATA_VERSION = "1.0.0"

# Paths
BASE_PATH = os.path.dirname(supertem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "supertem", "config")
LOG_PATH = os.path.join(BASE_PATH, "supertem", "log")
DATA_PATH = os.path.join(BASE_PATH, "supertem", "log", "data")

# Sub-paths
MICROSCOPE_CONFIG_INDEX_PATH = os.path.join(CONFIG_PATH, "microscope-config-index.yaml")
PROTOCOL_INDEX_PATH = os.path.join(CONFIG_PATH, "protocol-index.yaml")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(CONFIG_PATH, "microscope-configuration.yaml")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")

DATA_ML_PATH = os.path.join(DATA_PATH, "ml")
DATA_CC_PATH = os.path.join(DATA_PATH, "crosscorrelation")
DATA_TILE_PATH = os.path.join(DATA_PATH, "tile")
MODELS_PATH = os.path.join(BASE_PATH, "supertem", "segmentation", "models")
DATABASE_PATH = os.path.join(BASE_PATH, "supertem", "db", "supertem.db")


__DEFAULT_MANUFACTURER__ = "JEOL"
__DEFAULT_IP_ADDRESS__ = "192.168.0.1"


# =============================================================================
# Default Configurations
# =============================================================================

# This structure matches supertem.structure.base.MicroscopeSettings
DEFAULT_MICROSCOPE_CONFIGURATION_YAML = {
    "system": {
        "info": {
            "name": "default-configuration",
            "ip_address": __DEFAULT_IP_ADDRESS__,
            "manufacturer": __DEFAULT_MANUFACTURER__,
            "model": "Unknown",
            "serial_number": "Unknown",
            "hardware_version": "Unknown",
            "software_version": "Unknown",
        },
        "stage_system": {
            "enabled": True,
            "can_x": True, "can_y": True, "can_z": True,
            "can_r": True, "can_tilt_x": True, "can_tilt_y": False,

            # Limits are required by base.py if axis is enabled
            "x_limits_nm": [-1000000.0, 1000000.0],
            "y_limits_nm": [-1000000.0, 1000000.0],
            "z_limits_nm": [-100000.0, 100000.0],
            "tilt_x_limits_deg": [-70.0, 70.0],
            "r_limits_deg": [-360.0, 360.0],

            "max_step_nm": 50000.0,
            "max_step_angle": 1.0,
            "settle_time_s": 0.5,

            # Legacy/Custom fields go here (auto-collected into Extras)
            "rotation_reference": 0.0,
            "shuttle_pre_tilt": 35.0,
        },
        "beam_system": {
            "enabled": True,
            "voltage_limits_kv": [60.0, 300.0],
            "beam_current_limits_na": [0.0, 50.0],

            # Default state for resets
            "default_beam": {
                "voltage_kv": 200.0,
                "beam_current_na": 0.1,
                "spot_size": 1,
                "defocus_nm": 0.0,
            }
        },
        "detector_system": {
            "enabled": True,
            "available_detectors": ["SimCam"],
            "default_detector_id": "SimCam",

            # Registry of defaults per camera
            "defaults_by_id": {
                "SimCam": {
                    "exposure_ms": 100.0,
                    "binning_index": 1,
                    "frame_integration": 1,
                    "gain_index": 0,
                    "roi": None  # Full frame
                }
            },
            # Registry of capabilities per camera
            "capabilities_by_id": {
                "SimCam": {
                    "can_binning": True,
                    "exposure_ms_min": 0.1,
                    "exposure_ms_max": 10000.0,
                    "binning_index_min": 1,
                    "binning_index_max": 4,
                }
            }
        }
    },
    "image": {
        "file_format": "tiff",
        "path": os.path.join(DATA_PATH, "{date}", "images"),
    },
    "protocol": {
        "name": "demo",
        "description": "Default empty protocol",
        "steps": [],
    }
}

DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML = {
    "configurations": {"default-configuration": {"path": MICROSCOPE_CONFIGURATION_PATH}},
    "default": "default-configuration",
}

DEFAULT_PROTOCOL_YAML = {
    "name": "demo",
    "description": "Default protocol",
    "steps": [],
}

DEFAULT_PROTOCOL_INDEX_YAML = {
    "protocols": {"default-protocol": {"path": PROTOCOL_PATH}},
    "default": "default-protocol",
}

DEFAULT_POSITIONS_YAML = []


# =============================================================================
# Helper Functions
# =============================================================================

def load_yaml(fname: str, default=None):
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return default if data is None else data
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return default

def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _atomic_dump_yaml(path: str, data) -> None:
    parent = os.path.dirname(path)
    if parent:
        _safe_makedirs(parent)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, path)

def _write_yaml_if_missing(path: str, data) -> None:
    if os.path.exists(path):
        return
    _atomic_dump_yaml(path, data)

def bootstrap_config_files() -> None:
    """Ensure required directories + default YAML files exist."""
    _safe_makedirs(CONFIG_PATH)
    _safe_makedirs(LOG_PATH)
    _safe_makedirs(DATA_PATH)
    _safe_makedirs(DATA_ML_PATH)
    _safe_makedirs(DATA_CC_PATH)
    _safe_makedirs(DATA_TILE_PATH)
    _safe_makedirs(os.path.dirname(DATABASE_PATH))

    # Create “real” config/protocol files if missing
    _write_yaml_if_missing(MICROSCOPE_CONFIGURATION_PATH, DEFAULT_MICROSCOPE_CONFIGURATION_YAML)
    _write_yaml_if_missing(PROTOCOL_PATH, DEFAULT_PROTOCOL_YAML)
    _write_yaml_if_missing(POSITION_PATH, DEFAULT_POSITIONS_YAML)

    # Create index files if missing
    _write_yaml_if_missing(MICROSCOPE_CONFIG_INDEX_PATH, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
    _write_yaml_if_missing(PROTOCOL_INDEX_PATH, DEFAULT_PROTOCOL_INDEX_YAML)

# Initialize on import
bootstrap_config_files()


# =============================================================================
# Registry Logic (Microscope Configs)
# =============================================================================

MICROSCOPE_CONFIG_INDEX_YAML = load_yaml(MICROSCOPE_CONFIG_INDEX_PATH, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
if not isinstance(MICROSCOPE_CONFIG_INDEX_YAML, dict):
    MICROSCOPE_CONFIG_INDEX_YAML = DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML.copy()

MICROSCOPE_CONFIG_INDEX_YAML.setdefault("configurations", {})
MICROSCOPE_CONFIG_INDEX_YAML.setdefault("default", "default-configuration")
MICROSCOPE_CONFIG_INDEX = MICROSCOPE_CONFIG_INDEX_YAML["configurations"]
DEFAULT_CONFIGURATION_NAME = MICROSCOPE_CONFIG_INDEX_YAML["default"]

# Ensure default exists
MICROSCOPE_CONFIG_INDEX.setdefault("default-configuration", {"path": MICROSCOPE_CONFIGURATION_PATH})
if DEFAULT_CONFIGURATION_NAME not in MICROSCOPE_CONFIG_INDEX:
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    MICROSCOPE_CONFIG_INDEX_YAML["default"] = DEFAULT_CONFIGURATION_NAME

DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIG_INDEX[DEFAULT_CONFIGURATION_NAME].get("path") or MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    # Fallback if the file pointed to by default doesn't exist
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    MICROSCOPE_CONFIG_INDEX_YAML["default"] = DEFAULT_CONFIGURATION_NAME
    MICROSCOPE_CONFIG_INDEX[DEFAULT_CONFIGURATION_NAME]["path"] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

_atomic_dump_yaml(MICROSCOPE_CONFIG_INDEX_PATH, MICROSCOPE_CONFIG_INDEX_YAML)
logging.info("Default configuration: %s", DEFAULT_CONFIGURATION_NAME)


# =============================================================================
# Registry Logic (Protocols)
# =============================================================================

PROTOCOL_INDEX_YAML = load_yaml(PROTOCOL_INDEX_PATH, DEFAULT_PROTOCOL_INDEX_YAML)
if not isinstance(PROTOCOL_INDEX_YAML, dict):
    PROTOCOL_INDEX_YAML = DEFAULT_PROTOCOL_INDEX_YAML.copy()

PROTOCOL_INDEX_YAML.setdefault("protocols", {})
PROTOCOL_INDEX_YAML.setdefault("default", "default-protocol")
PROTOCOL_INDEX = PROTOCOL_INDEX_YAML["protocols"]
DEFAULT_PROTOCOL_NAME = PROTOCOL_INDEX_YAML["default"]

PROTOCOL_INDEX.setdefault("default-protocol", {"path": PROTOCOL_PATH})
if DEFAULT_PROTOCOL_NAME not in PROTOCOL_INDEX:
    DEFAULT_PROTOCOL_NAME = "default-protocol"
    PROTOCOL_INDEX_YAML["default"] = DEFAULT_PROTOCOL_NAME

DEFAULT_PROTOCOL_PATH = PROTOCOL_INDEX[DEFAULT_PROTOCOL_NAME].get("path") or PROTOCOL_PATH

if not os.path.exists(DEFAULT_PROTOCOL_PATH):
    DEFAULT_PROTOCOL_NAME = "default-protocol"
    PROTOCOL_INDEX_YAML["default"] = DEFAULT_PROTOCOL_NAME
    PROTOCOL_INDEX[DEFAULT_PROTOCOL_NAME]["path"] = PROTOCOL_PATH
    DEFAULT_PROTOCOL_PATH = PROTOCOL_PATH

_atomic_dump_yaml(PROTOCOL_INDEX_PATH, PROTOCOL_INDEX_YAML)
logging.info("Default protocol: %s", DEFAULT_PROTOCOL_NAME)


# =============================================================================
# Accessors
# =============================================================================

def list_microscope_configs():
    return sorted(MICROSCOPE_CONFIG_INDEX.keys())

def get_microscope_config_path(config_name: str) -> str:
    if config_name not in MICROSCOPE_CONFIG_INDEX:
        raise ValueError(f"Microscope config '{config_name}' does not exist.")
    return MICROSCOPE_CONFIG_INDEX[config_name]["path"]

def add_microscope_config(config_name: str, path: str):
    if config_name in MICROSCOPE_CONFIG_INDEX:
        raise ValueError(f"Microscope config '{config_name}' already exists.")
    MICROSCOPE_CONFIG_INDEX[config_name] = {"path": path}
    MICROSCOPE_CONFIG_INDEX_YAML["configurations"] = MICROSCOPE_CONFIG_INDEX
    _atomic_dump_yaml(MICROSCOPE_CONFIG_INDEX_PATH, MICROSCOPE_CONFIG_INDEX_YAML)

def remove_microscope_config(config_name: str):
    global DEFAULT_CONFIGURATION_NAME, DEFAULT_CONFIGURATION_PATH
    if config_name not in MICROSCOPE_CONFIG_INDEX:
        raise ValueError(f"Microscope config '{config_name}' does not exist.")

    del MICROSCOPE_CONFIG_INDEX[config_name]
    MICROSCOPE_CONFIG_INDEX_YAML["configurations"] = MICROSCOPE_CONFIG_INDEX

    if MICROSCOPE_CONFIG_INDEX_YAML.get("default") == config_name:
        MICROSCOPE_CONFIG_INDEX_YAML["default"] = "default-configuration"
        MICROSCOPE_CONFIG_INDEX.setdefault("default-configuration", {"path": MICROSCOPE_CONFIGURATION_PATH})
        DEFAULT_CONFIGURATION_NAME = "default-configuration"
        DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

    _atomic_dump_yaml(MICROSCOPE_CONFIG_INDEX_PATH, MICROSCOPE_CONFIG_INDEX_YAML)

def set_default_microscope_config(config_name: str):
    global DEFAULT_CONFIGURATION_NAME, DEFAULT_CONFIGURATION_PATH
    if config_name not in MICROSCOPE_CONFIG_INDEX:
        raise ValueError(f"Microscope config '{config_name}' does not exist.")
    MICROSCOPE_CONFIG_INDEX_YAML["default"] = config_name
    DEFAULT_CONFIGURATION_NAME = config_name
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIG_INDEX[config_name]["path"]
    _atomic_dump_yaml(MICROSCOPE_CONFIG_INDEX_PATH, MICROSCOPE_CONFIG_INDEX_YAML)


def list_protocols():
    return sorted(PROTOCOL_INDEX.keys())

def get_protocol_path(protocol_name: str) -> str:
    if protocol_name not in PROTOCOL_INDEX:
        raise ValueError(f"Protocol '{protocol_name}' does not exist.")
    return PROTOCOL_INDEX[protocol_name]["path"]

def add_protocol(protocol_name: str, path: str):
    if protocol_name in PROTOCOL_INDEX:
        raise ValueError(f"Protocol '{protocol_name}' already exists.")
    PROTOCOL_INDEX[protocol_name] = {"path": path}
    PROTOCOL_INDEX_YAML["protocols"] = PROTOCOL_INDEX
    _atomic_dump_yaml(PROTOCOL_INDEX_PATH, PROTOCOL_INDEX_YAML)

def remove_protocol(protocol_name: str):
    global DEFAULT_PROTOCOL_NAME, DEFAULT_PROTOCOL_PATH
    if protocol_name not in PROTOCOL_INDEX:
        raise ValueError(f"Protocol '{protocol_name}' does not exist.")
    del PROTOCOL_INDEX[protocol_name]
    PROTOCOL_INDEX_YAML["protocols"] = PROTOCOL_INDEX

    if PROTOCOL_INDEX_YAML.get("default") == protocol_name:
        PROTOCOL_INDEX_YAML["default"] = "default-protocol"
        PROTOCOL_INDEX.setdefault("default-protocol", {"path": PROTOCOL_PATH})
        DEFAULT_PROTOCOL_NAME = "default-protocol"
        DEFAULT_PROTOCOL_PATH = PROTOCOL_PATH

    _atomic_dump_yaml(PROTOCOL_INDEX_PATH, PROTOCOL_INDEX_YAML)

def set_default_protocol(protocol_name: str):
    global DEFAULT_PROTOCOL_NAME, DEFAULT_PROTOCOL_PATH
    if protocol_name not in PROTOCOL_INDEX:
        raise ValueError(f"Protocol '{protocol_name}' does not exist.")
    PROTOCOL_INDEX_YAML["default"] = protocol_name
    DEFAULT_PROTOCOL_NAME = protocol_name
    DEFAULT_PROTOCOL_PATH = PROTOCOL_INDEX[protocol_name]["path"]
    _atomic_dump_yaml(PROTOCOL_INDEX_PATH, PROTOCOL_INDEX_YAML)
