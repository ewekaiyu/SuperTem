"""
tests.conftest

Pytest configuration and shared fixtures.
This module handles the "Context Injection" for the test suite.
"""
import pytest
import shutil
from pathlib import Path

# Import from the refactored modules
from supertem.config import SuperTEMContext, RegistryManager


@pytest.fixture
def mock_context(tmp_path: Path) -> SuperTEMContext:
    """
    Creates a fresh, isolated testing context for each test function.
    'tmp_path' is a built-in pytest fixture that provides a unique temporary directory.

    The context will point all paths (config, log, data) to subfolders inside tmp_path.
    """
    return SuperTEMContext.testing(tmp_path)


@pytest.fixture
def mock_registry(mock_context: SuperTEMContext) -> RegistryManager:
    """
    Returns a RegistryManager initialized with the mock context.

    Side Effect:
        Instantiating RegistryManager triggers .bootstrap(), which automatically
        creates the folder structure and default YAML files inside mock_context.
    """
    return RegistryManager(mock_context)