from __future__ import annotations

import os
import yaml
import logging

import supertem

METADATA_VERSION = "v1.0.0"


SUPPORTED_COORDINATE_SYSTEMS = [
    "RAW",
    "SPECIMEN",
    "STAGE",
    "Raw",
    "raw",
    "specimen",
    "Specimen",
    "Stage",
    "stage",
]


REFERENCE_HFW_WIDE = 2750e-6
REFERENCE_HFW_LOW = 900e-6
REFERENCE_HFW_MEDIUM = 400e-6
REFERENCE_HFW_HIGH = 150e-6
REFERENCE_HFW_SUPER = 80e-6
REFERENCE_HFW_ULTRA = 50e-6

REFERENCE_RES_SQUARE = [1024, 1024]
REFERENCE_RES_LOW = [768, 512]
REFERENCE_RES_MEDIUM = [1536, 1024]
REFERENCE_RES_HIGH = [3072, 2048]
REFERENCE_RES_SUPER = [6144, 4096]

MILL_HFW_THRESHOLD = 0.005  # 0.5% of the image

BASE_PATH = os.path.dirname(
    supertem.__path__[0]
)  # TODO: figure out a more stable way to do this
CONFIG_PATH = os.path.join(BASE_PATH, "supertem", "config")

MICROSCOPE_CONFIG_INDEX_PATH = os.path.join(CONFIG_PATH, "microscope-config-index.yaml")
PROTOCOL_INDEX_PATH = os.path.join(CONFIG_PATH, "protocol-index.yaml")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(CONFIG_PATH, "microscope-configuration.yaml")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")

LOG_PATH = os.path.join(BASE_PATH, "supertem", "log")
DATA_PATH = os.path.join(BASE_PATH, "supertem", "log", "data")
DATA_ML_PATH: str = os.path.join(BASE_PATH, "supertem", "log", "data", "ml")
DATA_CC_PATH: str = os.path.join(BASE_PATH, "supertem", "log", "data", "crosscorrelation")
DATA_TILE_PATH: str = os.path.join(DATA_PATH, "tile")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "supertem", "segmentation", "models")


os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DATA_ML_PATH, exist_ok=True)
os.makedirs(DATA_CC_PATH, exist_ok=True)
os.makedirs(DATA_TILE_PATH, exist_ok=True)

DATABASE_PATH = os.path.join(BASE_PATH, "supertem", "db", "supertem.db")
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)



def load_yaml(fname: str, default=None):
    """Load YAML from `fname`. Return `default` if missing/invalid/empty."""
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return default if data is None else data
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return default


__SUPPORTED_MANUFACTURERS__ = ["JEOL", "Demo"]
__DEFAULT_MANUFACTURER__ = "JEOL"
__DEFAULT_IP_ADDRESS__ = "192.168.0.1"
__SUPPORTED_PLASMA_GASES__ = ["Argon", "Oxygen", "Nitrogen", "Xenon"]


DEFAULT_MICROSCOPE_CONFIGURATION_YAML = {
    "info": {
        "name": "default-configuration",
        "ip_address": __DEFAULT_IP_ADDRESS__,
        "manufacturer": __DEFAULT_MANUFACTURER__,
        "model": "Unknown",
        "serial_number": "Unknown",
        "hardware_version": "Unknown",
        "software_version": "Unknown",
        # supertem_version is optional (SystemInfo.from_dict has defaults)
    },
    "stage": {
        "rotation_reference": 0.0,
        "rotation_180": 180.0,
        "shuttle_pre_tilt": 35.0,
        "manipulator_height_limit": 0.0,
        "enabled": True,
        "rotation": True,
        "tilt": True,
    },
    "beam": {
        "enabled": True,
        "eucentric_height": 7.0e-3,
        "column_tilt": 0.0,
        "plasma": False,
        "plasma_gas": None,

        # required by BeamSettings.from_dict (they are indexed, not .get)
        "voltage": 200000.0,
        "hfw": 400e-6,
        "resolution": [1024, 1024],
        "dwell_time": 1.0,

        # detector settings are optional-ish but safe to include
        "detector_id": None,
        "exposure_ms": None,
        "binning_index": None,
        "binning_xy": None,
        "roi": None,
        "frame_integration": None,
        "gain_index": None,
        "offset_index": None,
        "digital_rotation_deg": None,
        "capabilities": None,
        "extra": {},
    },
    "image": {
        "binning": None,
        "exposure_ms": None,
        "dwell_us": None,
        "file_format": "tiff",
        "path": None,
        "roi": None,
    },
    # protocol can be omitted; MicroscopeSettings.from_dict handles it
}

DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML: dict = {
    "configurations": {"default-configuration": {"path": MICROSCOPE_CONFIGURATION_PATH}},
    "default": "default-configuration",
}

DEFAULT_PROTOCOL_YAML: dict = {
    "name": "demo",
    "description": "Default protocol",
    "steps": [],
}

DEFAULT_PROTOCOL_INDEX_YAML: dict = {
    "protocols": {"default-protocol": {"path": PROTOCOL_PATH}},
    "default": "default-protocol",
}

DEFAULT_POSITIONS_YAML = []  # MUST be a list (load_positions iterates a list)

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
    """Ensure required directories + default YAML files exist. Safe to call repeatedly."""
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

bootstrap_config_files()

# changed naming user configurations to microscope config index -> move to supertem.db eventually
# --------------------------------------------------------------------
# Load microscope config index (registry)
# --------------------------------------------------------------------

MICROSCOPE_CONFIG_INDEX_YAML = load_yaml(MICROSCOPE_CONFIG_INDEX_PATH, DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML)
if not isinstance(MICROSCOPE_CONFIG_INDEX_YAML, dict):
    MICROSCOPE_CONFIG_INDEX_YAML = DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML.copy()

MICROSCOPE_CONFIG_INDEX_YAML.setdefault("configurations", {})
MICROSCOPE_CONFIG_INDEX_YAML.setdefault("default", "default-configuration")
if not isinstance(MICROSCOPE_CONFIG_INDEX_YAML["configurations"], dict):
    MICROSCOPE_CONFIG_INDEX_YAML["configurations"] = {}

MICROSCOPE_CONFIG_INDEX = MICROSCOPE_CONFIG_INDEX_YAML["configurations"]
DEFAULT_CONFIGURATION_NAME = MICROSCOPE_CONFIG_INDEX_YAML["default"]

MICROSCOPE_CONFIG_INDEX.setdefault("default-configuration", {"path": MICROSCOPE_CONFIGURATION_PATH})
if DEFAULT_CONFIGURATION_NAME not in MICROSCOPE_CONFIG_INDEX:
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    MICROSCOPE_CONFIG_INDEX_YAML["default"] = DEFAULT_CONFIGURATION_NAME

DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIG_INDEX[DEFAULT_CONFIGURATION_NAME].get("path") or MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    MICROSCOPE_CONFIG_INDEX_YAML["default"] = DEFAULT_CONFIGURATION_NAME
    MICROSCOPE_CONFIG_INDEX[DEFAULT_CONFIGURATION_NAME]["path"] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

_atomic_dump_yaml(MICROSCOPE_CONFIG_INDEX_PATH, MICROSCOPE_CONFIG_INDEX_YAML)
logging.info("Default configuration: %s", DEFAULT_CONFIGURATION_NAME)
logging.info("Default configuration path: %s", DEFAULT_CONFIGURATION_PATH)


# --------------------------------------------------------------------
# Load protocol index (registry)
# --------------------------------------------------------------------

PROTOCOL_INDEX_YAML = load_yaml(PROTOCOL_INDEX_PATH, DEFAULT_PROTOCOL_INDEX_YAML)
if not isinstance(PROTOCOL_INDEX_YAML, dict):
    PROTOCOL_INDEX_YAML = DEFAULT_PROTOCOL_INDEX_YAML.copy()

PROTOCOL_INDEX_YAML.setdefault("protocols", {})
PROTOCOL_INDEX_YAML.setdefault("default", "default-protocol")
if not isinstance(PROTOCOL_INDEX_YAML["protocols"], dict):
    PROTOCOL_INDEX_YAML["protocols"] = {}

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
logging.info("Default protocol path: %s", DEFAULT_PROTOCOL_PATH)


# --------------------------------------------------------------------
# Helper functions: microscope configs
# --------------------------------------------------------------------

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

    # If you removed the default, fall back to a known-safe default
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


# --------------------------------------------------------------------
# Helper functions: protocols
# --------------------------------------------------------------------

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

    # If you removed the default, fall back to a known-safe default
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
