import pytest
import yaml
import logging
import pint
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from your actual modules
from supertem.utils import (
    setup_session,
    load_microscope,
    save_positions,
    get_saved_positions,
    make_logging_directory,
    load_yaml,
    get_position_by_name
)
from supertem.structures.base import StagePosition, ParseMode, Q_, Units


# =============================================================================
# Session Setup Tests
# =============================================================================

def test_setup_session_wires_context_and_jeol_driver(mock_context):
    """
    Test that setup_session correctly uses the injected context to create
    directories and initializes the JEOL driver (mocked).
    """
    with patch("supertem.microscopes.jeol_microscope.JeolMicroscope") as MockJeolClass:
        mock_instance = MockJeolClass.return_value
        mock_instance.get_instrument_info.return_value.manufacturer = "JEOL"

        scope, settings = setup_session(
            context=mock_context,
            manufacturer="JEOL",
            setup_logging=False
        )

        MockJeolClass.assert_called_once()
        mock_instance.connect.assert_called_once()

        # Verify Image Directory Creation
        image_path = Path(settings.image.path)
        assert image_path.exists()
        assert mock_context.log_path in image_path.parents


def test_setup_session_demo_driver_fallback(mock_context):
    """
    Test that asking for 'DEMO' manufacturer loads the DemoMicroscope.
    """
    # Patch the DemoMicroscope instead of JEOL
    with patch("supertem.microscopes.demo_microscope.DemoMicroscope") as MockDemoClass:
        scope, settings = setup_session(
            context=mock_context,
            manufacturer="DEMO",
            setup_logging=False
        )

        MockDemoClass.assert_called_once()
        # Verify settings reflect the override
        assert settings.system.info.manufacturer == "DEMO"


def test_setup_session_strict_validation_failure(mock_context, mock_registry):
    """
    Test that setup_session fails if the config file is invalid.
    """
    config_path = mock_registry.default_microscope_config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Inject invalid type
    data["system"]["beam_system"]["voltage_limits_kv"] = ["NOT_A_NUMBER", "INVALID"]
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with patch("supertem.microscopes.jeol_microscope.JeolMicroscope"):
        with pytest.raises((ValueError, pint.errors.UndefinedUnitError)):
            setup_session(context=mock_context, manufacturer="JEOL", setup_logging=False)


def test_setup_session_creates_log_file(mock_context):
    """
    Verify that setup_session actually creates a log file when setup_logging=True.
    """
    # Use Demo mode to avoid complexity
    with patch("supertem.microscopes.demo_microscope.DemoMicroscope"):
        scope, settings = setup_session(
            context=mock_context,
            manufacturer="DEMO",
            setup_logging=True
        )

        # The session folder should contain a .log file
        session_dir = Path(settings.image.path).parent
        log_files = list(session_dir.glob("*.log"))
        assert len(log_files) > 0


# =============================================================================
# Loader Logic Tests
# =============================================================================

def test_load_microscope_resolves_paths_via_registry(mock_registry):
    """
    Test that load_microscope uses the registry to find files in the context.
    """
    proto_name = "custom_workflow.yaml"
    proto_path = mock_registry.context.config_path / proto_name
    proto_data = {"name": "test_protocol", "steps": ["step1"]}
    proto_path.write_text(yaml.safe_dump(proto_data), encoding="utf-8")

    settings = load_microscope(
        registry=mock_registry,
        protocol_path=proto_path,
        mode=ParseMode.STRICT
    )
    assert settings.protocol["name"] == "test_protocol"


def test_load_yaml_resilience(tmp_path):
    """
    Test that load_yaml returns default if file is missing or corrupt.
    """
    # Case 1: File missing
    res = load_yaml(tmp_path / "nonexistent.yaml", default={"ok": False})
    assert res == {"ok": False}

    # Case 2: File corrupt (invalid YAML)
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("key: value: what:", encoding="utf-8")
    res = load_yaml(bad_file, default="fallback")
    assert res == "fallback"


# =============================================================================
# IO & Persistence Tests
# =============================================================================

def test_save_and_load_positions(mock_context):
    """
    Test saving positions to the context's specific storage file.
    """
    p1 = StagePosition(x=Q_(10, "um"), y=Q_(20, "um"), name="Pos A")

    save_positions(mock_context, [p1])

    loaded = get_saved_positions(mock_context)
    assert len(loaded) == 1
    assert loaded[0].name == "Pos A"
    assert loaded[0].x.to("um").magnitude == pytest.approx(10.0)
    assert format(loaded[0].x.units, "~") == Units.NM


def test_save_positions_append_logic(mock_context):
    """
    Test that saving positions respects the overwrite flag.
    """
    p1 = StagePosition(name="Pos 1", x=Q_(1, "um"))
    save_positions(mock_context, [p1], overwrite=True)

    # Append p2
    p2 = StagePosition(name="Pos 2", x=Q_(2, "um"))
    save_positions(mock_context, [p2], overwrite=False)

    loaded = get_saved_positions(mock_context)
    assert len(loaded) == 2
    names = [p.name for p in loaded]
    assert "Pos 1" in names
    assert "Pos 2" in names


def test_make_logging_directory(mock_context):
    """Test creating a run-specific folder inside the context's log tree."""
    path_str = make_logging_directory(mock_context, name="experiment_1")
    path = Path(path_str)

    assert path.exists()
    assert path.name == "experiment_1"
    assert path.parent == mock_context.log_path


def test_get_position_by_name(mock_context):
    """
    Test the helper function that finds a specific position by its label.
    """
    # Setup: Save 2 positions
    p1 = StagePosition(name="Grid Center", x=Q_(0, "um"))
    p2 = StagePosition(name="Target A", x=Q_(10, "um"))
    save_positions(mock_context, [p1, p2])

    # 1. Test Finding Existing
    found = get_position_by_name(mock_context, "Target A")
    assert found is not None
    assert found.name == "Target A"
    assert found.x.to("um").magnitude == pytest.approx(10.0)

    # 2. Test Missing
    missing = get_position_by_name(mock_context, "NonExistent")
    assert missing is None


def test_load_microscope_protocol_priority(mock_registry):
    """
    Ensure that an explicit protocol file overrides any protocol
    embedded inside the microscope config.
    """
    # 1. Create a config that HAS an embedded protocol (The "Old" way)
    config_data = {
        "system": {"beam_system": {"voltage_limits_kv": [80, 200]}},  # Min valid data
        "protocol": {"name": "EMBEDDED_PROTOCOL"}
    }

    # 2. Create a separate protocol file (The "New" way)
    protocol_data = {"name": "OVERRIDE_PROTOCOL"}

    # Write them to temp files
    c_path = mock_registry.context.config_path / "conf.yaml"
    p_path = mock_registry.context.config_path / "proto.yaml"

    c_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    p_path.write_text(yaml.safe_dump(protocol_data), encoding="utf-8")

    # 3. Load with explicit protocol path
    settings = load_microscope(
        registry=mock_registry,
        config_path=c_path,
        protocol_path=p_path,
        mode=ParseMode.LENIENT  # Use Lenient to ignore missing required fields for this simple test
    )

    # 4. ASSERT: The external file should win
    assert settings.protocol["name"] == "OVERRIDE_PROTOCOL"