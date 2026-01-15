"""
supertem.utils

Operational Utilities, Session Lifecycle Management, and Unit-Aware I/O.

This module acts as the "Glue Layer" between the abstract data structures defined
in base.py and the physical microscope hardware. It provides the high-level
orchestration required to initialize sessions, manage persistent storage, and
handle media generation.

===============================================================================
I. Session Orchestration (The Setup Lifecycle)
===============================================================================

The primary entry point for any automation routine is `setup_session()`.
This function manages the transition from static configuration to live execution:

  1) Configuration Resolution
     - Instantiates a `RegistryManager` tied to the provided `SuperTEMContext`.
     - Queries the registry to locate the active hardware profile.
     - Ingests YAML data into `MicroscopeSettings` using STRICT mode to ensure
       control-plane safety before hardware handoff.

  2) Environment Preparation
     - Generates unique, timestamped session directories.
     - Bootstraps the logging subsystem (File + Console).

  3) Hardware Initialization (The Factory Pattern)
     - Maps the `manufacturer` identity to specific driver implementations
       (e.g., JEOL vs. DEMO).
     - Instantiates the `TemMicroscope` controller.

===============================================================================
II. Unit-Aware Persistence & Serialization
===============================================================================

The utility layer provides specialized I/O handlers that respect the
"Normalization vs. Serialization" contract defined in the base structures:
- `save_positions`: Strips Pint units -> Floats (for YAML compatibility).
- `get_saved_positions`: Re-hydrates Floats -> Pint units (for Safety).

===============================================================================
III. Media & Diagnostic Helpers
===============================================================================

- MicroscopeImage Integration: Media helpers (like `create_gif`) utilize the
  `MicroscopeImage` loading logic.
- Logging: Standardized formatting for cross-module traceability.

===============================================================================
IV. The Wiring Contract (Dependency Injection)
===============================================================================

All functions in this module are **Context-Dependent**. They do not assume
global state.

- **Rule:** If a function touches the disk (logging, loading YAML), it MUST
  accept `context: SuperTEMContext` as an argument.
- **Why?** This ensures that `setup_session()` is the *only* place where
  decisions about the environment (Prod vs Test) are made.
"""
import datetime
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Any

import yaml
from PIL import Image

from supertem.config import RegistryManager, SuperTEMContext
from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    MicroscopeImage,
    MicroscopeSettings,
    StagePosition,
    ParseMode
)

# =============================================================================
# Time & Directory Helpers
# =============================================================================

def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def make_logging_directory(context: SuperTEMContext, name="run") -> str:
    """Creates a subdirectory inside the context's log path."""
    directory = context.log_path / name
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)

def configure_logging(path: Path, log_filename="logfile", log_level=logging.DEBUG, _DEBUG: bool = False):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    logfile = path / f"{log_filename}.log"
    file_handler = logging.FileHandler(str(logfile))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO if not _DEBUG else logging.DEBUG)

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        handlers=[file_handler, stream_handler],
        force=True,
    )

# =============================================================================
# YAML I/O
# =============================================================================

def load_yaml(fname: Path, default=None):
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return default if data is None else data
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return default

def save_yaml(path: Path, data: Any) -> None:
    path = Path(path).with_suffix(".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, indent=4, sort_keys=False)

# =============================================================================
# Session & Microscope Setup
# =============================================================================

def setup_session(
    context: SuperTEMContext,          # <--- REQUIRE CONTEXT
    session_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    protocol_path: Optional[Path] = None,
    setup_logging: bool = True,
    ip_address: Optional[str] = None,
    manufacturer: Optional[str] = None,
    debug: bool = False,
) -> Tuple[TemMicroscope, MicroscopeSettings]:
    """Setup microscope session using registry-aware loading."""

    # 1. Instantiate Registry for this context
    registry = RegistryManager(context)

    # 2. Load settings (Uses base.py STRICT mode for control-plane safety)
    settings = load_microscope(registry, config_path, protocol_path, mode=ParseMode.STRICT)

    # 3. Create session directories
    session_name = f'{settings.protocol.get("name", "supertem")}_{current_timestamp()}'

    # Use context log_path if no override provided
    root_log = Path(session_path) if session_path else context.log_path
    session_dir = root_log / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    if setup_logging:
        configure_logging(session_dir, _DEBUG=debug)

    # 4. Overload System Info if provided
    # Note: We update via object attributes; normalization is handled in drivers or via re-validation
    if ip_address:
        settings.system.info.ip_address = ip_address
    if manufacturer:
        settings.system.info.manufacturer = manufacturer

    # Update dynamic output path
    settings.image.path = str(session_dir / "images")

    # 5. Final Validation before hardware handoff
    if not settings.validate():
        logging.error(f"Settings validation failed: {settings.extra.notes}")
        raise ValueError("Cannot initialize microscope with invalid settings.")

    # 6. Factory Selection
    mfg = settings.system.info.manufacturer.upper()
    if mfg == "DEMO":
        from supertem.microscopes.demo_microscope import DemoMicroscope
        microscope = DemoMicroscope(settings)
        microscope.connect(ip_address, port=7520)
    elif mfg == "JEOL":
        from supertem.microscopes.jeol_microscope import JeolMicroscope
        microscope = JeolMicroscope(settings)
        microscope.connect(ip_address, port=7520)

    else:
        raise NotImplementedError(f"Manufacturer {mfg} not supported.")

    logging.info(f"Finished setup for session: {session_name}")

    return microscope, settings

def load_microscope(
    registry: RegistryManager,
    config_path: Optional[Path] = None,
    protocol_path: Optional[Path] = None,
    mode: ParseMode = ParseMode.STRICT
) -> MicroscopeSettings:
    """Orchestrates loading using the provided registry."""

    # Use registry paths if not explicitly provided
    c_path = config_path or registry.get_active_config_path()
    p_path = protocol_path or registry.get_active_protocol_path()

    # Default dicts are stateless, so importing from config is safe
    from supertem.config import DEFAULT_MICROSCOPE_CONFIGURATION_YAML, DEFAULT_PROTOCOL_YAML

    config_dict = load_yaml(Path(c_path), default=DEFAULT_MICROSCOPE_CONFIGURATION_YAML)
    protocol_dict = load_yaml(Path(p_path), default=DEFAULT_PROTOCOL_YAML)

    # Ingest using base.py logic
    if isinstance(config_dict, dict):
        embedded_protocol = config_dict.get("protocol")
        if protocol_dict not in (None, {}, [], ""):
            config_dict["protocol"] = protocol_dict
        elif embedded_protocol not in (None, {}, [], ""):
            # legacy: protocol embedded in microscope config
            config_dict["protocol"] = embedded_protocol
        else:
            config_dict["protocol"] = protocol_dict

    return MicroscopeSettings.from_dict(config_dict, mode=mode)

# =============================================================================
# Stage Position Management
# =============================================================================

def get_saved_positions(context: SuperTEMContext) -> List[StagePosition]:
    """Returns list of StagePosition objects from storage."""
    # Resolve path via context
    path = context.config_path / "positions.yaml"

    data = load_yaml(path, default=[])
    # Ingest as LENIENT for storage reading
    return [StagePosition.from_dict(d, mode=ParseMode.LENIENT) for d in data]

def get_position_by_name(context: SuperTEMContext, name: str) -> Optional[StagePosition]:
    positions = get_saved_positions(context)
    for p in positions:
        if p.name == name:
            return p
    return None

def save_positions(context: SuperTEMContext, positions: Any, overwrite: bool = False) -> None:
    """Saves StagePosition objects, ensuring unit-safe serialization."""
    target_path = context.config_path / "positions.yaml"

    # Convert single input to list
    if not isinstance(positions, list):
        positions = [positions]

    # Load existing if not overwriting
    current_data = [] if overwrite else load_yaml(target_path, default=[])

    # Add new positions using base.py serialization (handles units -> floats)
    for pos in positions:
        if isinstance(pos, StagePosition):
            current_data.append(pos.to_dict())
        elif isinstance(pos, dict):
            current_data.append(pos)

    save_yaml(target_path, current_data)

# =============================================================================
# Media Helpers
# =============================================================================

def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    """Creates a GIF from a set of images using MicroscopeImage loading."""
    filenames = sorted(glob.glob(str(Path(path) / search)))
    if not filenames:
        print("No images found for GIF.")
        return

    imgs = [Image.fromarray(MicroscopeImage.load(fname).data) for fname in filenames]

    out_path = Path(path) / f"{gif_fname}.gif"
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        loop=loop,
    )
    print(f"{len(filenames)} images saved to {out_path}")