"""
supertem.config

Configuration Bootstrap, Registry Management, and Default Factory.

This module serves as the central entry point for the application's state.
It has been architected to use **Dependency Injection (DI)**, replacing global
singletons with explicit Context objects.

===============================================================================
I. The Bootstrap Lifecycle
===============================================================================

On instantiation, the RegistryManager executes a self-healing initialization sequence:

  1) Environment Validation
     - Ensures all required directory trees exist (log/, data/, db/, etc.).
     - Prevents "FileNotFound" crashes in downstream modules.

  2) Default Generation (The "Safe Mode")
     - Checks for existence of critical YAML definitions.
     - If missing, atomically writes internal DEFAULT_ dictionaries to disk.
     - GOAL: The system remains runnable even on a fresh install.

  3) Registry Loading
     - Loads Index files (microscope-config-index.yaml & protocol-index.yaml).
     - Resolves the "Active" paths to feed the base.py ingestion lifecycle.

===============================================================================
II. Registry Management (Hardware vs. Protocols)
===============================================================================

The module manages two parallel registries to support different operational needs:
  1) Microscope Configurations (The "Hardware Profile") - Tracks machine profiles.
  2) Automation Protocols (The "Workflows") - Manages executable routines.

===============================================================================
III. Safety & Atomic I/O
===============================================================================

- Atomic Writes: Updates use a write-to-tmp -> OS-replace sequence.
- Separation of Concerns: This module handles *storage* (The Registry),
  while base.py handles *structure* (The Typing).

===============================================================================
IV. Usage Patterns (The Context Architecture)
===============================================================================

This module does not expose a global `registry` object. Instead, you must
create a `SuperTEMContext` and pass it downstream.

**Pattern A: Running in Production (Normal Operation)**
    from supertem.config import SuperTEMContext
    from supertem.utils import setup_session

    # 1. Create the Production Context (Points to /opt/supertem/...)
    ctx = SuperTEMContext.production()

    # 2. Initialize the Session
    scope, settings = setup_session(context=ctx, manufacturer="JEOL")

**Pattern B: Running Unit Tests (Isolated Environment)**
    # 1. Create a Test Context (Points to a temporary folder)
    ctx = SuperTEMContext.testing(tmp_path)

    # 2. Verify behavior without touching real config files
    registry = RegistryManager(ctx)
    assert registry.active_config_name == "default-configuration"
"""

from __future__ import annotations
import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import supertem

# =============================================================================
# The Context (Dependency Injection Container)
# =============================================================================

@dataclass
class SuperTEMContext:
    """
    Holds the runtime environment configuration (paths and mode).
    Pass this object to functions instead of relying on global variables.
    """
    base_path: Path
    config_path: Path
    log_path: Path
    data_path: Path

    # Database and specific data subfolders
    db_path: Path
    data_ml_path: Path
    data_cc_path: Path
    data_tile_path: Path

    @classmethod
    def production(cls) -> SuperTEMContext:
        """Factory for the standard production environment."""
        base = Path(os.path.dirname(supertem.__path__[0]))
        # Standard SuperTEM folder structure
        supertem_root = base / "supertem"

        return cls(
            base_path=base,
            config_path=supertem_root / "config",
            log_path=supertem_root / "log",
            data_path=supertem_root / "log" / "data",
            db_path=supertem_root / "db" / "supertem.db",
            data_ml_path=supertem_root / "log" / "data" / "ml",
            data_cc_path=supertem_root / "log" / "data" / "crosscorrelation",
            data_tile_path=supertem_root / "log" / "data" / "tile"
        )

    @classmethod
    def testing(cls, tmp_path: Path) -> SuperTEMContext:
        """Factory for unit tests (runs in a temporary isolated folder)."""
        return cls(
            base_path=tmp_path,
            config_path=tmp_path / "config",
            log_path=tmp_path / "log",
            data_path=tmp_path / "data",
            db_path=tmp_path / "db" / "supertem.db",
            data_ml_path=tmp_path / "data" / "ml",
            data_cc_path=tmp_path / "data" / "crosscorrelation",
            data_tile_path=tmp_path / "data" / "tile"
        )

# =============================================================================
# Constants & Defaults
# =============================================================================

from supertem.structures.base import SCHEMA_VERSION
__DEFAULT_MANUFACTURER__ = "JEOL"
__DEFAULT_IP_ADDRESS__ = "192.168.0.1"

DEFAULT_MICROSCOPE_CONFIGURATION_YAML = {
    "version": SCHEMA_VERSION,

    "system": {
        "info": {
            "name": "default-configuration",
            "ip_address": __DEFAULT_IP_ADDRESS__,
            "manufacturer": __DEFAULT_MANUFACTURER__,
            "model": "Simulated TEM",
            "serial_number": "000000",
            "software_version": "1.0.0"
        },
        "stage_system": {
            "enabled": True,
            "can_x": True, "can_y": True, "can_z": True,
            "can_r": True, "can_tilt_x": False, "can_tilt_y": False,
            "x_limits_nm": [-2000000.0, 2000000.0],
            "y_limits_nm": [-2000000.0, 2000000.0],
            "z_limits_nm": [-100000.0, 100000.0],
            "r_limits_deg": [-180.0, 180.0],
            "tilt_x_limits_deg": [-70.0, 70.0],
            "tilt_y_limits_deg": [-70.0, 70.0],
            "max_step_nm": 50000.0,
            "max_step_deg": 5.0,
            "settle_time_s": 0.5,
            "timeout_s": 30.0,
            "eucentric_z_nm": 0.0
        },
        "beam_system": {
            "enabled": True,
            "voltage_limits_kv": [60.0, 300.0],
            "beam_current_limits_na": [0.0, 100.0],
            "spot_size_limits": [1, 5],
            "convergence_angle_limits_mrad": [0.0, 50.0],
            "default_beam": {
                "voltage_kv": 200.0,
                "beam_current_na": 0.1,
                "spot_size": 1,
                "convergence_angle_mrad": 10.0
            }
        },
        "projection_system": {
            "enabled": True,
            "camera_length_limits_mm": [50.0, 5000.0],
            "magnification_limits": [50, 2000000],
            "defocus_limits_nm": [-10000.0, 10000.0],
            "default_projection": {
                "optical_mode": "IMAGING",
                "magnification": 5000,
                "defocus_nm": 0.0
            }
        },
        "scan_system": {
            "enabled": True,
            "available_scan_modes": ["Full Frame", "Spot"],
            "pixel_dwell_time_limits_us": [0.1, 1000.0],
            "flyback_time_limits_us": [0.0, 500.0],
            "scan_rotation_limits_deg": [0.0, 360.0]
        },
        "detector_system": {
            "enabled": True,
            "available_detectors": ["SimCam"],
            "default_detector_id": "SimCam",
            "defaults_by_id": {
                "SimCam": {
                    "exposure_ms": 100.0,
                    "binning_index": 1,
                    "frame_integration": 1,
                    "readout_mode": "LINEAR",
                    "save_frames": False
                }
            },
            "capabilities_by_id": {
                "SimCam": {
                    "can_binning": True,
                    "binning_index_min": 1,
                    "binning_index_max": 4,
                    "exposure_ms_min": 0.1,
                    "exposure_ms_max": 10000.0,
                    "roi_min": [64, 64],
                    "roi_max": [4096, 4096],
                    "can_gain": True,
                    "gain_index_min": 0,
                    "gain_index_max": 3
                }
            }
        }
    },
    "image": {
        "file_format": "tiff",
        # Note: Paths here are relative templates, resolved at runtime
        "path": "{session_path}/images"
    }
}

DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML = {
    "configurations": {"default-configuration": {"path": "microscope-configuration.yaml"}},
    "default": "default-configuration",
}

DEFAULT_PROTOCOL_YAML = {"name": "demo", "description": "Default protocol", "steps": []}
DEFAULT_PROTOCOL_INDEX_YAML = {
    "protocols": {"default-protocol": {"path": "protocol.yaml"}},
    "default": "default-protocol",
}
DEFAULT_POSITIONS_YAML = []


# =============================================================================
# Registry Manager (Context-Aware)
# =============================================================================

class RegistryManager:
    """
    Manages loading and saving of configuration state.

    Refactored to support Dependency Injection:
    - No longer a Singleton.
    - Requires a `SuperTEMContext` to determine file locations.
    """

    def __init__(self, context: SuperTEMContext):
        self.context = context
        self.microscope_index = {}
        self.protocol_index = {}
        self.active_config_name = ""
        self.active_protocol_name = ""

        # Resolve filenames relative to context
        self.microscope_index_path = self.context.config_path / "microscope-config-index.yaml"
        self.protocol_index_path = self.context.config_path / "protocol-index.yaml"
        self.default_microscope_config_path = self.context.config_path / "microscope-configuration.yaml"
        self.default_protocol_path = self.context.config_path / "protocol.yaml"
        self.position_path = self.context.config_path / "positions.yaml"

        self.bootstrap()

    def bootstrap(self) -> None:
        """Self-healing setup for folders and YAMLs."""
        # Ensure directories exist based on context
        dirs_to_make = [
            self.context.config_path,
            self.context.log_path,
            self.context.data_path,
            self.context.data_ml_path,
            self.context.data_cc_path,
            self.context.data_tile_path,
            self.context.db_path.parent
        ]
        for d in dirs_to_make:
            os.makedirs(d, exist_ok=True)

        self._write_if_missing(self.default_microscope_config_path, DEFAULT_MICROSCOPE_CONFIGURATION_YAML)
        self._write_if_missing(self.default_protocol_path, DEFAULT_PROTOCOL_YAML)
        self._write_if_missing(self.position_path, DEFAULT_POSITIONS_YAML)
        self._write_if_missing(self.microscope_index_path, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
        self._write_if_missing(self.protocol_index_path, DEFAULT_PROTOCOL_INDEX_YAML)
        self.reload()

    def reload(self) -> None:
        """Loads indices and runs a health check on the active config."""
        m_idx = self._load_yaml(self.microscope_index_path, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
        self.active_config_name = m_idx.get("default", "default-configuration")
        self.microscope_index = m_idx.get("configurations", {})

        p_idx = self._load_yaml(self.protocol_index_path, DEFAULT_PROTOCOL_INDEX_YAML)
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

        path_str = self.microscope_index.get(name, {}).get("path")
        if not path_str:
            return False

        # Resolve path: If absolute, use it. If relative, join with config_path.
        full_path = Path(path_str)
        if not full_path.is_absolute():
            full_path = self.context.config_path / full_path

        if not full_path.exists():
            return False

        data = self._load_yaml(full_path, None)
        if data is None:
            return False

        file_version = data.get("version", "0.0.0")
        if file_version != SCHEMA_VERSION:
            logging.warning(f"Config '{name}' version mismatch! File: v{file_version}, Code: v{SCHEMA_VERSION}")
            # Future: self._migrate_config(data, file_version)

        try:
            # Ingest and validate
            settings = MicroscopeSettings.from_dict(data, mode=mode)
            return settings.validate()
        except Exception as e:
            logging.error(f"Validation failed for '{name}': {e}")
            return False

    # --- Accessors ---

    def get_active_config_path(self) -> Path:
        p = self.microscope_index.get(self.active_config_name, {}).get("path", "microscope-configuration.yaml")
        return self._resolve_path(p)

    def set_default_microscope_config(self, name: str) -> None:
        if name not in self.microscope_index: raise ValueError(f"Unknown config: {name}")
        if self.validate_profile(name, mode="strict"):
            self.active_config_name = name
            self._atomic_dump(self.microscope_index_path, {"configurations": self.microscope_index, "default": name})

    def get_active_protocol_path(self) -> Path:
        p = self.protocol_index.get(self.active_protocol_name, {}).get("path", "protocol.yaml")
        return self._resolve_path(p)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.context.config_path / path

    # --- Helpers ---

    def _load_yaml(self, path: Path, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or default
        except Exception:
            return default

    def _atomic_dump(self, path: Path, data: Any) -> None:
        # Cast to Path to ensure string paths from legacy code don't break
        path_obj = Path(path)
        tmp = path_obj.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, path_obj)

    def _write_if_missing(self, path: Path, data: Any) -> None:
        if not path.exists(): self._atomic_dump(path, data)