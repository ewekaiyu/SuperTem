"""
tests.test_config

Tests for the Configuration Subsystem, ensuring:
1. Contexts correctly resolve paths (Windows/Linux compatible).
2. RegistryManager self-heals missing files and survives corruption.
3. Atomic writes prevent corruption.
4. Profile validation works as expected (Strict vs Lenient).
5. Secondary registries (Protocols) function correctly.
"""
import os
import yaml
import pytest
from pathlib import Path

from supertem.config import (
    RegistryManager,
    SuperTEMContext,
    DEFAULT_MICROSCOPE_CONFIGURATION_YAML,
    DEFAULT_PROTOCOL_YAML,
    DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML
)

# =============================================================================
# 1. Context Tests
# =============================================================================

def test_context_paths_isolation(tmp_path):
    """Ensure Testing context is truly isolated from Production paths."""
    # Production Context (Standard Linux paths)
    prod_ctx = SuperTEMContext.production()
    # FIX: Normalize to lowercase to support Windows paths (e.g. "SuperTem")
    assert "supertem" in str(prod_ctx.base_path).lower()

    # Testing Context (Temp paths)
    test_ctx = SuperTEMContext.testing(tmp_path)
    assert str(test_ctx.base_path) == str(tmp_path)
    assert str(test_ctx.config_path) == str(tmp_path / "config")


# =============================================================================
# 2. Bootstrap & Self-Healing Tests
# =============================================================================

def test_registry_bootstrap_creates_directory_tree(mock_registry):
    """Test that __init__ creates the full folder hierarchy."""
    ctx = mock_registry.context

    assert ctx.config_path.exists()
    assert ctx.log_path.exists()
    assert ctx.data_path.exists()
    assert ctx.data_ml_path.exists()
    # Check if DB parent folder was created
    assert ctx.db_path.parent.exists()


def test_registry_bootstrap_creates_default_files(mock_registry):
    """Test that missing YAMLs are auto-generated."""
    # 1. Check Microscope Configuration
    cfg_path = mock_registry.default_microscope_config_path
    assert cfg_path.exists()

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert data["system"]["info"]["manufacturer"] == "JEOL"  # Matches DEFAULT constant

    # 2. Check Protocol
    proto_path = mock_registry.default_protocol_path
    assert proto_path.exists()

    p_data = yaml.safe_load(proto_path.read_text(encoding="utf-8"))
    assert p_data["name"] == DEFAULT_PROTOCOL_YAML["name"]

    # 3. Check Index
    idx_path = mock_registry.microscope_index_path
    assert idx_path.exists()

    i_data = yaml.safe_load(idx_path.read_text(encoding="utf-8"))
    assert "default-configuration" in i_data["configurations"]


def test_registry_self_healing_recreates_deleted_config(mock_context):
    """
    Verify that if the config directory is deleted, the registry
    re-bootstraps it on initialization.
    """
    # 1. Initialize registry (creates files)
    reg = RegistryManager(mock_context)
    assert mock_context.config_path.exists()

    # 2. SIMULATE DISASTER: Delete the config folder
    import shutil
    shutil.rmtree(mock_context.config_path)
    assert not mock_context.config_path.exists()

    # 3. Re-Initialize Registry
    reg_new = RegistryManager(mock_context)

    # 4. Assert Healing
    assert mock_context.config_path.exists()
    assert (mock_context.config_path / "microscope.yaml").exists() or \
           (mock_context.config_path / "microscope-config-index.yaml").exists()


# =============================================================================
# 3. Logic & Persistence Tests
# =============================================================================

def test_registry_resolve_paths(mock_registry):
    """Test that registry correctly resolves relative paths in the index."""
    # The default index points to "microscope-configuration.yaml" (relative)
    # The registry should convert this to an absolute path inside the context.

    resolved_path = mock_registry.get_active_config_path()
    expected_path = mock_registry.context.config_path / "microscope-configuration.yaml"

    assert resolved_path == expected_path
    assert resolved_path.is_absolute()


def test_registry_resolves_absolute_paths(mock_registry, tmp_path):
    """
    Ensure the registry respects absolute paths if provided in the index,
    pointing outside the standard config folder.
    """
    # 1. Create a config file in a completely different location (outside ctx.config_path)
    # tmp_path is the root, mock_registry uses tmp_path/config.
    # We will put this file in tmp_path/external.
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_config = external_dir / "external_scope.yaml"

    external_config.write_text(
        yaml.safe_dump(DEFAULT_MICROSCOPE_CONFIGURATION_YAML),
        encoding="utf-8"
    )

    # 2. Register it using the ABSOLUTE path
    abs_path_str = str(external_config.resolve())
    mock_registry.microscope_index["external-scope"] = {"path": abs_path_str}

    # 3. Resolve
    # We manually set active config without validation for this specific path test
    mock_registry.active_config_name = "external-scope"
    resolved = mock_registry.get_active_config_path()

    # 4. Assert
    assert resolved == external_config
    assert resolved.exists()


def test_atomic_write_persists_active_config(mock_registry):
    """
    Proves that 'set_default_microscope_config' updates the index on disk correctly.
    """
    # 1. Verify initial state
    assert mock_registry.active_config_name == "default-configuration"

    # 2. Create a "New Microscope" file manually in the temp folder
    new_name = "test-scope"
    new_filename = "test_scope.yaml"
    new_file_path = mock_registry.context.config_path / new_filename

    # Write valid config content
    new_file_path.write_text(
        yaml.safe_dump(DEFAULT_MICROSCOPE_CONFIGURATION_YAML, sort_keys=False),
        encoding="utf-8"
    )

    # 3. Manually register it into the index file (simulating a user action)
    current_index_data = {
        "configurations": {
            "default-configuration": {"path": "microscope-configuration.yaml"},
            new_name: {"path": new_filename}
        },
        "default": "default-configuration"
    }

    mock_registry._atomic_dump(mock_registry.microscope_index_path, current_index_data)

    # Reload registry so it sees the manual file change
    mock_registry.reload()
    assert new_name in mock_registry.microscope_index

    # 4. Switch the active profile (The Action under test)
    mock_registry.set_default_microscope_config(new_name)

    # 5. Verify In-Memory State
    assert mock_registry.active_config_name == new_name
    assert mock_registry.get_active_config_path().name == new_filename

    # 6. Verify Persistence (Reload from disk again)
    mock_registry.reload()
    assert mock_registry.active_config_name == new_name


# =============================================================================
# 4. Validation & Resilience Tests
# =============================================================================

def test_validate_profile_rejects_broken_config(mock_registry):
    """Test that we cannot switch to a corrupted configuration file."""
    bad_name = "broken-scope"
    bad_file = mock_registry.context.config_path / "broken.yaml"

    bad_data = {
        "system": {
            "stage_system": {
                "settle_time_s": -5.0  # Invalid!
            }
        }
    }
    bad_file.write_text(yaml.safe_dump(bad_data), encoding="utf-8")

    # Register it
    mock_registry.microscope_index[bad_name] = {"path": "broken.yaml"}

    # Attempt to validate it
    is_valid = mock_registry.validate_profile(bad_name, mode="strict")
    assert is_valid is False


def test_validate_profile_modes_strict_vs_lenient(mock_registry):
    """
    Complex Scenario: A config file has an invalid value (negative exposure).
    - Strict Mode: Should REJECT it (Safe for hardware).
    - Lenient Mode: Should ACCEPT it (Heals it for data/logging).
    """
    healable_name = "healable-scope"
    healable_file = mock_registry.context.config_path / "healable.yaml"

    # Create a config with a "healable" error
    # Exposure time is negative. Base.py `check_gt_zero` can heal this in Lenient mode.
    bad_data = {
        "system": {
            "detector_system": {
                "defaults_by_id": {
                    "SimCam": {
                        "exposure_ms": -50.0  # Invalid!
                    }
                }
            }
        }
    }
    healable_file.write_text(yaml.safe_dump(bad_data), encoding="utf-8")
    mock_registry.microscope_index[healable_name] = {"path": "healable.yaml"}

    # 1. Strict Check (Control Plane) -> Must Fail
    assert mock_registry.validate_profile(healable_name, mode="strict") is False

    # 2. Lenient Check (Data Plane) -> Must Pass (Auto-healed)
    assert mock_registry.validate_profile(healable_name, mode="lenient") is True


def test_registry_survives_corrupt_index_file(mock_registry):
    """
    Ensure the system falls back to defaults if the index YAML is corrupted,
    rather than crashing the application.
    """
    # 1. Corrupt the index file with invalid YAML
    mock_registry.microscope_index_path.write_text("!!python/object/apply:os.system ['broken']", encoding="utf-8")

    # 2. Trigger a reload
    # The _load_yaml method has a try/except block that should catch this
    mock_registry.reload()

    # 3. Assert fallback behavior
    # It should revert to the hardcoded DEFAULT_MICROSCOPE_CONFIG_INDEX_YAML in config.py
    assert mock_registry.active_config_name == "default-configuration"
    assert "default-configuration" in mock_registry.microscope_index


def test_protocol_registry_resolution(mock_registry):
    """
    Verify the secondary registry (Protocols) works independently
    of the primary registry (Microscope).
    """
    # 1. Verify Default Bootstrap
    assert mock_registry.active_protocol_name == "default-protocol"
    default_path = mock_registry.get_active_protocol_path()
    assert default_path.name == "protocol.yaml"
    assert default_path.exists()

    # 2. Add a custom protocol
    new_proto_name = "experiment-A"
    new_proto_file = "experiment_a.yaml"
    (mock_registry.context.config_path / new_proto_file).touch()

    # 3. Manually update protocol index
    # (Simulating atomic dump behavior for protocols)
    new_index = {
        "protocols": {
            "default-protocol": {"path": "protocol.yaml"},
            new_proto_name: {"path": new_proto_file}
        },
        "default": new_proto_name
    }
    mock_registry._atomic_dump(mock_registry.protocol_index_path, new_index)
    mock_registry.reload()

    # 4. Assert switch
    assert mock_registry.active_protocol_name == new_proto_name
    assert mock_registry.get_active_protocol_path().name == new_proto_file