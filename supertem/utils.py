"""
supertem.utils

Operational Utilities, Session Lifecycle Management, and Unit-Aware I/O.

This module acts as the "Glue Layer" between the abstract data structures defined
in base_structures.py and the physical microscope hardware. It provides the high-level
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
     - **Profile Override:** Optionally accepts a specific `profile_name` to
       bypass the system default (e.g., for temporary testing of new settings).
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
II. State Persistence (The Snapshot Lifecycle)
===============================================================================

This module enables the "Clone & Patch" pattern for configuration management
via `save_live_config()`. This allows operators to use the microscope as a
GUI editor for the configuration files:

  1) Tune: Adjust microscope parameters (Voltage, Spotsize, etc.) interactively.
  2) Clone: The system deep-copies the active static limits (Validation Layer).
  3) Patch: The system queries the hardware for current values (Hardware Layer)
     and writes them into the `default_` fields of the settings object.
  4) Save: The result is serialized to a new YAML profile, ready for immediate
     use via `setup_session(..., profile_name="new_profile")`.

===============================================================================
III. Unit-Aware Persistence & Serialization
===============================================================================

The utility layer provides specialized I/O handlers that respect the
"Normalization vs. Serialization" contract defined in the base structures:
- `save_positions`: Strips Pint units -> Floats (for YAML compatibility).
- `get_saved_positions`: Re-hydrates Floats -> Pint units (for Safety).

===============================================================================
IV. Media & Diagnostic Helpers
===============================================================================

- MicroscopeImage Integration: Media helpers (like `create_gif`) utilize the
  `MicroscopeImage` loading logic.
- Logging: Standardized formatting for cross-module traceability.

===============================================================================
V. The Wiring Contract (Dependency Injection)
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
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
from copy import deepcopy

import yaml
from PIL import Image

from supertem.config import RegistryManager, SuperTEMContext
from supertem.microscopes.base_microscope import TemMicroscope
from supertem.structures.base_structures import (
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
    profile_name: Optional[str] = None,
    offline: bool = False
) -> Tuple[TemMicroscope, MicroscopeSettings]:
    """Setup microscope session using registry-aware loading."""

    # 1. Instantiate Registry for this context
    registry = RegistryManager(context)

    if profile_name:
        if profile_name not in registry.microscope_index:
            raise ValueError(f"Profile '{profile_name}' not found in registry.")
        registry.active_config_name = profile_name

    # 2. Load settings (Uses base_structures.py STRICT mode for control-plane safety)
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

    if offline:
        # We mark the model as offline. The JeolMicroscope driver looks for this string.
        current_model = settings.system.info.model or "Unknown"
        if "OFFLINE" not in current_model.upper():
            settings.system.info.model = f"{current_model} (Offline)"

        # Optional: Set a safe dummy IP so connect() doesn't hang looking for real hardware
        if not ip_address:
            settings.system.info.ip_address = "127.0.0.1"

        logging.info(f"Session configured for EXPLICIT OFFLINE SIMULATION.")

    # Update dynamic output path
    image_dir = session_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    settings.image.path = str(image_dir)

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

    # Ingest using base_structures.py logic
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

def save_live_config(
        microscope: TemMicroscope,
        context: SuperTEMContext,
        name: str,
        update_defaults: bool = True
) -> None:
    """
    Saves the current microscope configuration and system limits to a new profile.

    Args:
        microscope: The active microscope instance.
        context: The runtime context (for path resolution).
        name: The name of the new profile (e.g. 'high-res-stem').
        update_defaults: If True, updates the startup defaults (Beam/Projection)
                         to match the microscope's current live state.
    """
    registry = RegistryManager(context)

    # 1. Clone the active settings (preserves limits/safety)
    new_settings = deepcopy(microscope._settings)

    # 2. Update the "Default" fields with live values
    if update_defaults:
        sys = new_settings.system

        # --- Beam ---
        try:
            live_beam = microscope.get_beam_settings()
            if live_beam:
                sys.beam_system.default_beam = live_beam
        except Exception as e:
            print(f"Warning: Could not capture live beam: {e}")

        # --- Projection ---
        try:
            live_proj = microscope.get_projection_settings()
            if live_proj:
                sys.projection_system.default_projection = live_proj
        except Exception as e:
            print(f"Warning: Could not capture live projection: {e}")

        # --- Scan ---
        try:
            live_scan = microscope.get_scan_settings()
            if live_scan:
                sys.scan_system.default_scan = live_scan
        except Exception as e:
            print(f"Warning: Could not capture live scan: {e}")

        # --- Apertures ---
        try:
            for apt_id in microscope.list_apertures():
                live_apt = microscope.get_aperture_settings(apt_id)
                # Only update if we have a registry entry for it
                if live_apt and sys.aperture_system.defaults_by_id is not None:
                    sys.aperture_system.defaults_by_id[apt_id] = live_apt
        except Exception as e:
            print(f"Warning: Could not capture live apertures: {e}")

        # --- Stage ---
        try:
            current_holder_id = microscope.system_settings.stage_system.active_holder_id

            if current_holder_id:
                sys.stage_system.active_holder_id = current_holder_id

        except Exception as e:
            print(f"Warning: Could not capture active holder from settings: {e}")

        # --- Detectors ---
        try:
            for det_id in microscope.list_detectors():
                live_det = microscope.get_detector_settings(det_id)
                if live_det and sys.detector_system.defaults_by_id:
                    sys.detector_system.defaults_by_id[det_id] = live_det
        except Exception as e:
            print(f"Warning: Could not capture live detectors: {e}")

    # 3. Save to Disk
    filename = f"{name}.yaml"
    full_path = context.config_path / filename

    # Use base_structures.py's serialization to handle units
    save_yaml(full_path, new_settings.to_dict())
    print(f"Configuration saved to {full_path}")

    # 4. Update the Index (Register the file)
    index_data = registry._load_yaml(registry.microscope_index_path, default={})
    if "configurations" not in index_data:
        index_data["configurations"] = {}

    index_data["configurations"][name] = {"path": filename}
    registry._atomic_dump(registry.microscope_index_path, index_data)
    print(f"Profile '{name}' registered in index.")

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

    # Add new positions using base_structures.py serialization (handles units -> floats)
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