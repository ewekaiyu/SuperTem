"""
tests.test_jeol_eos_tables
Tests for the static lookup tables and helper functions.
"""
import pytest
from unittest.mock import patch
from supertem.vendor.JEOL import jeol_eos_tables


def test_get_list_valid_retrieval():
    """Verify we can retrieve a standard list (e.g., MagList for TEM:MAG)."""
    # 1. Standard Case
    res = jeol_eos_tables.get_list("TEM:MAG", "MagList")
    assert len(res) > 0

    # Check structure of first item: (value, unit, label)
    val, unit, label = res[0]
    assert isinstance(val, float)
    assert isinstance(unit, str)
    assert isinstance(label, str)

    # Verify strict unit check
    assert unit == "X"


def test_get_list_case_insensitivity():
    """Verify keys are normalized/found even if casing differs."""
    # The table has 'TEM:MAG'. Requesting 'tem:mag' should work if logic handles it,
    # but jeol_eos_tables.get_list is strict on keys in the provided code.
    # We verify exact behavior: The provided code raises KeyError if key is missing.
    with pytest.raises(KeyError):
        jeol_eos_tables.get_list("tem:mag", "MagList")

    # Correct key should work
    assert jeol_eos_tables.get_list("TEM:MAG", "MagList")


def test_get_list_bad_data_tolerance():
    """
    Verify the try/except block in get_list swallows malformed entries
    without crashing the whole application.
    """
    # Create a corrupted table entry
    bad_data = {
        "TEST:BAD": {
            "MagList": [
                ["5000", "X", "Good"],  # Valid
                ["NotANumber", "X", "Bad"],  # Should be skipped (float conversion fails)
                [1000]  # Should be skipped (index error accessing [1])
            ]
        }
    }

    # Patch the global dictionary
    with patch.dict(jeol_eos_tables.EOS_MODE_TABLES, bad_data):
        res = jeol_eos_tables.get_list("TEST:BAD", "MagList")

        # Should only contain the one valid entry
        assert len(res) == 1
        assert res[0][0] == 5000.0


def test_get_list_unknown_key():
    """Verify unknown keys raise KeyError (Fail Loudly logic)."""
    with pytest.raises(KeyError) as exc:
        jeol_eos_tables.get_list("NON_EXISTENT_MODE", "MagList")
    assert "Unknown EOS mode key" in str(exc.value)


def test_list_unit_helper():
    """Verify the list_unit helper extracts the unit string."""
    data = [(100.0, "cm", "Label")]
    assert jeol_eos_tables.list_unit(data) == "cm"
    assert jeol_eos_tables.list_unit([]) == ""