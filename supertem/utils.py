import datetime
import glob

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Callable


import yaml
from PIL import Image

from supertem import config as cfg
from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    TemImage,
    MicroscopeSettings,
)


def current_timestamp():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%I-%M-%S%p") #PM/AM doesnt work?


def current_timestamp_v2():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return str(time.time()).replace(".", "_")

def _format_time_seconds(seconds: float) -> str:
    """Format a time delta in seconds to proper string format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def make_logging_directory(path: Path = None, name="run"):
    """
    Create a logging directory with the specified name at the specified file path. 
    If no path is given, it creates the directory at the default base path.

    Args:
        path (Path, optional): The file path to create the logging directory at. If None, default base path is used. 
        name (str, optional): The name of the logging directory to create. Default is "run".

    Returns:
        str: The file path to the created logging directory.
        """
    
    if path is None:
        path = os.path.join(cfg.BASE_PATH, "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory

# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
# https://stackoverflow.com/questions/61483056/save-logging-debug-and-show-only-logging-info-python
def configure_logging(path: Path = None, log_filename="logfile", log_level=logging.DEBUG, _DEBUG: bool = False):
    """Log to the terminal and to file simultaneously.

    If `path` is None/empty, logs are written under cfg.LOG_PATH (see config.py).
    """
    if not path:
        path = cfg.LOG_PATH
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    logfile = path / f"{log_filename}.log"

    file_handler = logging.FileHandler(str(logfile))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO if _DEBUG is False else logging.DEBUG)

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[file_handler, stream_handler],
        force=True,
    )


def load_yaml(fname: Path, default=None):
    """Load YAML. Return `default` if missing/invalid/empty."""
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return default if data is None else data
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return default


def save_yaml(path: Path, data: dict) -> None:
    """Saves a python dictionary object to a yaml file

    Args:
        path (Path): path location to save yaml file
        data (dict): dictionary object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path = Path(path).with_suffix(".yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    """Creates a GIF from a set of images. Images must be in same folder

    Args:
        path (Path): Path to images folder
        search (str): search name
        gif_fname (str): name to save gif file
        loop (int, optional): _description_. Defaults to 0.
    """
    filenames = glob.glob(os.path.join(path, search))

    imgs = [Image.fromarray(TemImage.load(fname).data) for fname in filenames]

    print(f"{len(filenames)} images added to gif.")
    imgs[0].save(
        os.path.join(path, f"{gif_fname}.gif"),
        save_all=True,
        append_images=imgs[1:],
        loop=loop,
    )

VALID_DEMO = ["Demo"]
VALID_JEOL = ["JEOL"]

def setup_session(
    session_path: Path = None,
    config_path: Path = None,
    protocol_path: Path = None,
    setup_logging: bool = True,
    ip_address: str = None,
    manufacturer: str = None,
    debug: bool = False,
) -> Tuple[TemMicroscope, MicroscopeSettings]:
    """Setup microscope session

    Args:
        session_path (Path): path to logging directory
        config_path (Path): path to config directory 
        protocol_path (Path): path to protocol file

    Returns:
        tuple: microscope, settings
    """

    # load settings
    settings = load_microscope(config_path, protocol_path)

    # create session directories
    session = f'{settings.protocol.get("name", "opensupertem")}_{current_timestamp()}'

    # configure paths (root log dir -> per-session dir)
    root_log = Path(session_path) if session_path else Path(cfg.LOG_PATH)
    root_log.mkdir(parents=True, exist_ok=True)
    session_dir = root_log / session
    session_dir.mkdir(parents=True, exist_ok=True)


    # configure logging
    if setup_logging:
        configure_logging(session_dir, _DEBUG=debug)

    # cheap overloading
    if ip_address:
        settings.system.info.ip_address = ip_address
    
    if manufacturer:
        settings.system.info.manufacturer = manufacturer

    manufacturer = settings.system.info.manufacturer
    ip_address = settings.system.info.ip_address

    # set default image_settings path
    settings.image.path = session_dir

    if manufacturer in VALID_DEMO:
        from supertem.microscopes.demo_microscope import DemoMicroscope
        microscope = DemoMicroscope(settings)
        microscope.connect_to_microscope(ip_address, port=7520)
    
    elif manufacturer in VALID_JEOL:
        from supertem.microscopes.jeol_microscope import JeolMicroscope
        microscope = JeolMicroscope(settings)
        microscope.connect_to_microscope(ip_address, port=7520)

    else:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")

    logging.info(f"Finished setup for session: {session}")

    return microscope, settings

def load_microscope(config_path: Path = None, protocol_path: Path = None) -> MicroscopeSettings:
    """Load microscope settings + protocol.

    This is the thin orchestrator:
      - `load_microscope_configuration()` resolves the microscope configuration dict
      - `load_protocol()` resolves the protocol dict
      - `MicroscopeSettings.from_dict()` builds the strongly-typed settings object
    """
    config = load_microscope_configuration(config_path)
    protocol = load_protocol(protocol_path)
    return MicroscopeSettings.from_dict(config, protocol=protocol)

def load_microscope_configuration(config_path: Path = None) -> dict:
    """Load microscope configuration YAML (config only).

    - If `config_path` is None, uses cfg.DEFAULT_CONFIGURATION_PATH.
    - If `config_path` is a directory, expects 'microscope-configuration.yaml' inside it.
    - If config YAML is missing/invalid, falls back to cfg.DEFAULT_MICROSCOPE_CONFIGURATION_YAML.

    Protocol resolution is analogous (defaults come from cfg.DEFAULT_PROTOCOL_PATH).
    """
    # Resolve microscope config file
    if config_path is None:
        config_file = Path(getattr(cfg, "DEFAULT_CONFIGURATION_PATH", cfg.MICROSCOPE_CONFIGURATION_PATH))
    else:
        cp = Path(config_path)
        config_file = cp / "microscope-configuration.yaml" if cp.is_dir() else cp
    fallback = getattr(cfg, "DEFAULT_MICROSCOPE_CONFIGURATION_YAML", {})
    config = load_yaml(config_file, default=fallback)
    if not isinstance(config, dict):
        config = getattr(cfg, "DEFAULT_MICROSCOPE_CONFIGURATION_YAML", {}) or {}

    return config

def load_protocol(protocol_path: Path = None) -> dict:
    """Load protocol YAML.

    - If `protocol_path` is None, uses cfg.DEFAULT_PROTOCOL_PATH (or cfg.PROTOCOL_PATH).
    - If `protocol_path` is a directory, expects 'protocol.yaml' inside it.
    - If missing/invalid, falls back to cfg.DEFAULT_PROTOCOL_YAML.
    """
    # Normalize to a file path
    if protocol_path is None:
        protocol_file = Path(getattr(cfg, "DEFAULT_PROTOCOL_PATH", cfg.PROTOCOL_PATH))
    else:
        pp = Path(protocol_path)
        protocol_file = pp / "protocol.yaml" if pp.is_dir() else pp

    fallback = getattr(cfg, "DEFAULT_PROTOCOL_YAML", {"name": "demo"})
    protocol = load_yaml(protocol_file, default=fallback)
    if not isinstance(protocol, dict):
        protocol = dict(fallback) if isinstance(fallback, dict) else {"name": "demo"}

    return protocol

def _format_dictionary(dictionary: dict) -> dict:
    """Recursively traverse dictionary and covert all numeric values to flaot.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    -------
    dictionary
        The input dictionary, with all numeric values converted to float type.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _format_dictionary(item)
        elif isinstance(item, list):
            dictionary[key] = [
                _format_dictionary(i)
                for i in item
                if isinstance(i, list) or isinstance(i, dict)
            ]
        else:
            if item is not None:
                try:
                    dictionary[key] = float(dictionary[key])
                except ValueError:
                    pass
    return dictionary

def get_params(main_str: str) -> list:
    """Helper function to access relevant metadata parameters from sub field

    Args:
        main_str (str): Sub string of relevant metadata

    Returns:
        list: Parameters covered by metadata
    """
    cats = []
    cat_str = ""

    i = main_str.find("\n")
    i += 1
    while i < len(main_str):

        if main_str[i] == "=":
            cats.append(cat_str)
            cat_str = ""
            i += main_str[i:].find("\n")
        else:
            cat_str += main_str[i]

        i += 1
    return cats


def _get_position(name: str):
    
    import os

    from supertem import config as cfg
    from supertem.structures.base import TemStagePosition

    ddict = load_yaml(fname=os.path.join(cfg.CONFIG_PATH, "positions.yaml"))
    # get position from save positions?
    for d in ddict:
        if d["name"] == name:
            return TemStagePosition.from_dict(d)
    return None

def _get_positions(fname: str = None) -> List[str]:    
    
    import os

    from supertem import config as cfg

    if fname is None:
        fname = os.path.join(cfg.CONFIG_PATH, "positions.yaml")

    ddict = load_yaml(fname=fname)

    return [d["name"] for d in ddict]


def save_positions(positions: list, path: str = None, overwrite: bool = False) -> None:
    """save the list of positions to file"""

    from supertem import config as cfg

    # convert single position to list
    if not isinstance(positions, list):
        positions = [positions]

    # default path
    if path is None:
        path = cfg.POSITION_PATH

    # get existing positions    
    pdict = []
    if not overwrite:
        pdict = load_yaml(fname=path)

    
    # append new positions
    for position in positions:
        pdict.append(position.to_dict())
    
    # save
    save_yaml(path, pdict)




