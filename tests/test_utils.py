import pytest
import yaml
import pint
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from your actual modules
from supertem.utils import (
    setup_session,
    load_microscope,
    save_live_config,
    save_positions,
    get_saved_positions,
    make_logging_directory,
    configure_logging,
    load_yaml,
    get_position_by_name,
    create_gif,
)
from supertem.structures.base import (
    StagePosition,
    ParseMode,
    Q_,
    Units,
    BeamSettings,
    ProjectionSettings,
    DetectorSettings,
)

import numpy as np


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


# =============================================================================
# Branch & Edge-Case Coverage
# =============================================================================

def test_configure_logging_creates_directory_and_logfile(tmp_path):
    """configure_logging should create its target directory when missing."""
    log_dir = tmp_path / "logs" / "session_a"
    assert not log_dir.exists()

    configure_logging(log_dir, log_filename="unit_test_log", _DEBUG=True)

    assert log_dir.exists()
    assert (log_dir / "unit_test_log.log").exists()


def test_setup_session_profile_name_not_found_raises(mock_context):
    """setup_session should reject unknown profile names before any loading."""
    # Patch the registry instance used inside setup_session so this test is fully isolated.
    with patch("supertem.utils.RegistryManager") as MockRegistry:
        reg = MockRegistry.return_value
        reg.microscope_index = {"default": {"path": "default.yaml"}}

        with pytest.raises(ValueError, match="Profile 'missing' not found"):
            setup_session(
                context=mock_context,
                profile_name="missing",
                setup_logging=False,
            )


def test_setup_session_profile_name_sets_active_config_name(mock_context):
    """A valid profile_name should set RegistryManager.active_config_name."""
    with patch("supertem.utils.RegistryManager") as MockRegistry:
        reg = MockRegistry.return_value
        reg.microscope_index = {"my_profile": {"path": "some.yaml"}}
        reg.active_config_name = "default"

        # We only care that the assignment branch runs (line 171 in utils.py).
        with patch("supertem.utils.load_microscope", side_effect=RuntimeError("stop")):
            with pytest.raises(RuntimeError, match="stop"):
                setup_session(
                    context=mock_context,
                    profile_name="my_profile",
                    setup_logging=False,
                )

        assert reg.active_config_name == "my_profile"


def test_setup_session_rejects_invalid_settings_via_validate(mock_context, tmp_path):
    """If settings.validate() returns False, setup_session must raise ValueError."""
    bad_settings = MagicMock()
    bad_settings.protocol = {"name": "bad"}

    # These nested attributes are accessed before validate() is called.
    bad_settings.system.info.manufacturer = "DEMO"
    bad_settings.image.path = str(tmp_path / "images")
    bad_settings.extra.notes = "nope"
    bad_settings.validate.return_value = False

    with patch("supertem.utils.load_microscope", return_value=bad_settings):
        with pytest.raises(ValueError, match="invalid settings"):
            setup_session(
                context=mock_context,
                manufacturer="DEMO",
                setup_logging=False,
            )


def test_setup_session_unsupported_manufacturer_raises(mock_context, tmp_path):
    """A manufacturer other than DEMO/JEOL should raise NotImplementedError."""
    settings = MagicMock()
    settings.protocol = {"name": "ok"}
    settings.system.info.manufacturer = "DEMO"
    settings.image.path = str(tmp_path / "images")
    settings.validate.return_value = True

    with patch("supertem.utils.load_microscope", return_value=settings):
        with pytest.raises(NotImplementedError, match="not supported"):
            setup_session(
                context=mock_context,
                manufacturer="ACME",
                setup_logging=False,
            )


def test_load_microscope_uses_embedded_protocol_when_external_is_empty(mock_registry):
    """When protocol file is empty but config embeds protocol, embedded wins."""
    config_data = {
        "system": {"beam_system": {"voltage_limits_kv": [80, 200]}},
        "protocol": {"name": "EMBEDDED_PROTOCOL"},
    }

    # External protocol is present but empty
    protocol_data = {}

    c_path = mock_registry.context.config_path / "conf_embedded.yaml"
    p_path = mock_registry.context.config_path / "proto_empty.yaml"

    c_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    p_path.write_text(yaml.safe_dump(protocol_data), encoding="utf-8")

    settings = load_microscope(
        registry=mock_registry,
        config_path=c_path,
        protocol_path=p_path,
        mode=ParseMode.LENIENT,
    )

    assert settings.protocol["name"] == "EMBEDDED_PROTOCOL"


def test_load_microscope_sets_protocol_to_empty_when_both_sources_empty(mock_registry):
    """When both embedded and external protocol are empty, protocol stays empty."""
    config_data = {
        "system": {"beam_system": {"voltage_limits_kv": [80, 200]}},
        "protocol": {},
    }
    protocol_data = {}

    c_path = mock_registry.context.config_path / "conf_empty_proto.yaml"
    p_path = mock_registry.context.config_path / "proto_empty2.yaml"

    c_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    p_path.write_text(yaml.safe_dump(protocol_data), encoding="utf-8")

    settings = load_microscope(
        registry=mock_registry,
        config_path=c_path,
        protocol_path=p_path,
        mode=ParseMode.LENIENT,
    )

    assert settings.protocol == {}


def test_save_positions_accepts_single_position_and_raw_dict(mock_context):
    """save_positions should accept a single StagePosition and raw dict payloads."""
    p1 = StagePosition(name="Solo", x=Q_(1, "um"))
    save_positions(mock_context, p1, overwrite=True)

    # Append a raw dict (e.g., from a GUI form)
    raw = {"name": "Raw", "x": 2.0, "y": 0.0, "z": 0.0}
    save_positions(mock_context, raw, overwrite=False)

    loaded = get_saved_positions(mock_context)
    names = [p.name for p in loaded]
    assert "Solo" in names
    assert "Raw" in names


def test_create_gif_no_images_prints_message(tmp_path, capsys):
    """create_gif should exit cleanly when no images match the search."""
    create_gif(tmp_path, search="*.png", gif_fname="out")
    captured = capsys.readouterr()
    assert "No images found for GIF." in captured.out


def test_create_gif_builds_gif_from_loaded_images(tmp_path):
    """create_gif should assemble a gif when images are present."""
    # Create placeholder files for the glob search to find.
    (tmp_path / "img_001.png").write_bytes(b"not used")
    (tmp_path / "img_002.png").write_bytes(b"not used")

    class _Loaded:
        def __init__(self, data):
            self.data = data

    # Provide deterministic arrays so PIL can build images.
    frames = [
        _Loaded((np.zeros((4, 4), dtype=np.uint8))),
        _Loaded((np.ones((4, 4), dtype=np.uint8) * 255)),
    ]
    it = iter(frames)

    with patch("supertem.utils.MicroscopeImage.load", side_effect=lambda _: next(it)):
        create_gif(tmp_path, search="*.png", gif_fname="movie")

    assert (tmp_path / "movie.gif").exists()


def test_save_live_config_happy_path_updates_defaults_and_writes_index(mock_registry, capsys):
    """save_live_config should snapshot live state, write yaml, and update the registry index."""
    # Load a real settings object (keeps this test aligned with base.py serialization).
    settings = load_microscope(mock_registry, mode=ParseMode.LENIENT)

    # Ensure detector defaults exist and are truthy so the assignment branch is executed.
    settings.system.detector_system.defaults_by_id = {
        "D1": DetectorSettings(detector_id="D1")
    }

    microscope = MagicMock()
    microscope._settings = settings
    microscope.get_beam_settings.return_value = BeamSettings(spot_size=3)
    microscope.get_projection_settings.return_value = ProjectionSettings(magnification=42000)
    microscope.list_detectors.return_value = ["D1"]
    microscope.get_detector_settings.return_value = DetectorSettings(detector_id="D1", exposure=Q_(100, "ms"))

    # Force the "configurations"-missing branch in index update.
    with patch("supertem.config.RegistryManager._load_yaml", return_value={}):
        save_live_config(
            microscope=microscope,
            context=mock_registry.context,
            name="snap_test",
            update_defaults=True,
        )

    out = capsys.readouterr().out
    assert "Configuration saved" in out
    assert "registered in index" in out

    assert (mock_registry.context.config_path / "snap_test.yaml").exists()


def test_save_live_config_survives_driver_exceptions(mock_registry, capsys):
    """save_live_config should never crash if live getters fail."""
    settings = load_microscope(mock_registry, mode=ParseMode.LENIENT)
    microscope = MagicMock()
    microscope._settings = settings

    microscope.get_beam_settings.side_effect = RuntimeError("boom")
    microscope.get_projection_settings.side_effect = RuntimeError("boom")
    microscope.list_detectors.side_effect = RuntimeError("boom")

    # Keep index writing simple.
    with patch("supertem.config.RegistryManager._load_yaml", return_value={}):
        save_live_config(
            microscope=microscope,
            context=mock_registry.context,
            name="snap_failures",
            update_defaults=True,
        )

    out = capsys.readouterr().out
    assert "Warning: Could not capture live beam" in out
    assert "Warning: Could not capture live projection" in out
    assert "Warning: Could not capture live detectors" in out

    assert (mock_registry.context.config_path / "snap_failures.yaml").exists()