"""
supertem.config

Configuration Bootstrap, Registry Management, and Default Factory.

This module serves as the central entry point for the application's state,
responsible for:
  - Bootstrapping the runtime environment (filesystem trees for logs, data, and db).
  - Managing the "Registry of Configurations" (Active vs. Available hardware profiles).
  - Managing the "Protocol Library" (Storage and indexing of automation workflows).
  - Providing "Factory Reset" defaults that align with the dataclass structures
    defined in supertem.structure.base.
  - Atomic persistence of configuration changes to disk to prevent data corruption.

===============================================================================
I. The Bootstrap Lifecycle
===============================================================================

On instantiation, the RegistryManager executes a self-healing initialization sequence:

  1) Environment Validation
     - Ensures all required directory trees exist (log/, data/, db/, etc.).
     - Prevents "FileNotFound" crashes in downstream modules.

  2) Default Generation (The "Safe Mode")
     - Checks for existence of critical YAML definitions for both the hardware
       and the protocol library.
     - If missing, atomically writes internal DEFAULT_ dictionaries to disk.
     - GOAL: The system remains runnable even on a fresh install or after
       configuration corruption.

  3) Registry Loading
     - Loads Index files (microscope-config-index.yaml & protocol-index.yaml).
     - Resolves the "Active" paths to feed the base.py ingestion lifecycle.

===============================================================================
II. Registry Management (Hardware vs. Protocols)
===============================================================================

The module manages two parallel registries to support different operational needs:

1. Microscope Configurations (The "Hardware Profile")
   - Tracks multiple machine profiles (e.g., "Simulated", "JEOL-2100").
   - Aligns with the MicroscopeSettings structure: System -> Subsystem -> Limits.
   - Enforces Unit-Explicit Naming (e.g., voltage_limits_kv) to eliminate
     interpretive ambiguity during ingestion.

2. Automation Protocols (The "Workflows")
   - Manages a library of executable routines (e.g., "Demo", "Grid-Screening").
   - Uses the same indexing pattern as hardware configs to allow
     switching active protocols without code changes.

===============================================================================
III. Safety & Atomic I/O
===============================================================================

- Atomic Writes: All updates to index files use a write-to-tmp -> OS-replace
  sequence to prevent file corruption during power failures.
- Separation of Concerns: This module handles *storage and location* (The Registry),
  while base.py handles *typing and validation* (The Structure).
"""

from __future__ import annotations
import os
import yaml
import logging
import threading
from typing import Dict, List, Any, Optional

import supertem

# =============================================================================
# Constants & Paths
# =============================================================================

METADATA_VERSION = "1.0.0"

BASE_PATH = os.path.dirname(supertem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "supertem", "config")
LOG_PATH = os.path.join(BASE_PATH, "supertem", "log")
DATA_PATH = os.path.join(BASE_PATH, "supertem", "log", "data")

MICROSCOPE_CONFIG_INDEX_PATH = os.path.join(CONFIG_PATH, "microscope-config-index.yaml")
PROTOCOL_INDEX_PATH = os.path.join(CONFIG_PATH, "protocol-index.yaml")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(CONFIG_PATH, "microscope-configuration.yaml")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")

DATA_ML_PATH = os.path.join(DATA_PATH, "ml")
DATA_CC_PATH = os.path.join(DATA_PATH, "crosscorrelation")
DATA_TILE_PATH = os.path.join(DATA_PATH, "tile")
DATABASE_PATH = os.path.join(BASE_PATH, "supertem", "db", "supertem.db")

__DEFAULT_MANUFACTURER__ = "JEOL"
__DEFAULT_IP_ADDRESS__ = "192.168.0.1"

# =============================================================================
# Default Dictionaries (Fallback State)
# =============================================================================

DEFAULT_MICROSCOPE_CONFIGURATION_YAML = {
    "system": {
        "info": {
            "name": "default-configuration",
            "ip_address": __DEFAULT_IP_ADDRESS__,
            "manufacturer": __DEFAULT_MANUFACTURER__,
            "model": "Unknown",
            "serial_number": "Unknown",
        },
        "stage_system": {
            "enabled": True,
            "can_x": True, "can_y": True, "can_z": True,
            "x_limits_nm": [-1000000.0, 1000000.0],
            "y_limits_nm": [-1000000.0, 1000000.0],
            "z_limits_nm": [-100000.0, 100000.0],
            "max_step_nm": 50000.0,
            "settle_time_s": 0.5,
        },
        "beam_system": {
            "enabled": True,
            "voltage_limits_kv": [60.0, 300.0],
            "default_beam": {"voltage_kv": 200.0, "beam_current_na": 0.1, "spot_size": 1}
        },
        "detector_system": {
            "enabled": True,
            "available_detectors": ["SimCam"],
            "default_detector_id": "SimCam",
            "capabilities_by_id": {
                "SimCam": {"can_binning": True, "exposure_ms_min": 0.1, "exposure_ms_max": 10000.0}
            }
        }
    },
    "image": {"file_format": "tiff", "path": os.path.join(DATA_PATH, "{date}", "images")},
    "protocol": {"name": "demo", "steps": []}
}

DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML = {
    "configurations": {"default-configuration": {"path": MICROSCOPE_CONFIGURATION_PATH}},
    "default": "default-configuration",
}

DEFAULT_PROTOCOL_YAML = {"name": "demo", "description": "Default protocol", "steps": []}
DEFAULT_PROTOCOL_INDEX_YAML = {
    "protocols": {"default-protocol": {"path": PROTOCOL_PATH}},
    "default": "default-protocol",
}
DEFAULT_POSITIONS_YAML = []


# =============================================================================
# Registry Manager (Singleton)
# =============================================================================

class RegistryManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RegistryManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.microscope_index = {}
        self.protocol_index = {}
        self.active_config_name = ""
        self.active_protocol_name = ""
        self.bootstrap()
        self._initialized = True

    def bootstrap(self) -> None:
        """Self-healing setup for folders and YAMLs."""
        for d in [CONFIG_PATH, LOG_PATH, DATA_PATH, DATA_ML_PATH, DATA_CC_PATH, DATA_TILE_PATH,
                  os.path.dirname(DATABASE_PATH)]:
            os.makedirs(d, exist_ok=True)

        self._write_if_missing(MICROSCOPE_CONFIGURATION_PATH, DEFAULT_MICROSCOPE_CONFIGURATION_YAML)
        self._write_if_missing(PROTOCOL_PATH, DEFAULT_PROTOCOL_YAML)
        self._write_if_missing(POSITION_PATH, DEFAULT_POSITIONS_YAML)
        self._write_if_missing(MICROSCOPE_CONFIG_INDEX_PATH, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
        self._write_if_missing(PROTOCOL_INDEX_PATH, DEFAULT_PROTOCOL_INDEX_YAML)
        self.reload()

    def reload(self) -> None:
        """Loads indices and runs a health check on the active config."""
        m_idx = self._load_yaml(MICROSCOPE_CONFIG_INDEX_PATH, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
        self.active_config_name = m_idx.get("default", "default-configuration")
        self.microscope_index = m_idx.get("configurations", {})

        p_idx = self._load_yaml(PROTOCOL_INDEX_PATH, DEFAULT_PROTOCOL_INDEX_YAML)
        self.active_protocol_name = p_idx.get("default", "default-protocol")
        self.protocol_index = p_idx.get("protocols", {})

        # Verify active profile health using base.py logic
        self.validate_profile(self.active_config_name, mode="lenient")

    def validate_profile(self, name: str, mode: str = "strict") -> bool:
        """Integration: Validates a config against base.py via deferred import."""
        try:
            from supertem.structures.base import MicroscopeSettings
        except ImportError:
            # Fallback to relative import if the absolute path fails in certain environments
            from .structures.base import MicroscopeSettings

        path = self.microscope_index.get(name, {}).get("path")
        if not path or not os.path.exists(path):
            return False

        data = self._load_yaml(path, None)
        if data is None:
            return False

        try:
            # Ingest and validate
            settings = MicroscopeSettings.from_dict(data, mode=mode)
            return settings.validate()
        except Exception as e:
            logging.error(f"Validation failed for '{name}': {e}")
            return False

    # --- Accessors ---

    def get_active_config_path(self) -> str:
        return self.microscope_index.get(self.active_config_name, {}).get("path", MICROSCOPE_CONFIGURATION_PATH)

    def set_default_microscope_config(self, name: str) -> None:
        if name not in self.microscope_index: raise ValueError(f"Unknown config: {name}")
        if self.validate_profile(name, mode="strict"):
            self.active_config_name = name
            self._atomic_dump(MICROSCOPE_CONFIG_INDEX_PATH, {"configurations": self.microscope_index, "default": name})

    def get_active_protocol_path(self) -> str:
        return self.protocol_index.get(self.active_protocol_name, {}).get("path", PROTOCOL_PATH)

    # --- Helpers ---

    def _load_yaml(self, path: str, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or default
        except Exception:
            return default

    def _atomic_dump(self, path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, path)

    def _write_if_missing(self, path: str, data: Any) -> None:
        if not os.path.exists(path): self._atomic_dump(path, data)


# Global access point
registry = RegistryManager()