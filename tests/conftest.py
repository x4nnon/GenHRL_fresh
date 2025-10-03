"""
Pytest configuration for GenHRL tests.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="genhrl_test_")
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_api_key():
    """Provide a test API key."""
    return "test_api_key_for_testing"


@pytest.fixture
def test_isaaclab_path(temp_dir):
    """Provide a test IsaacLab path."""
    isaaclab_path = temp_dir / "IsaacLab"
    isaaclab_path.mkdir(parents=True, exist_ok=True)
    return isaaclab_path


@pytest.fixture
def clean_environment():
    """Provide a clean environment for tests."""
    # Save original environment
    original_env = {}
    test_env_vars = ["GENHRL_API_KEY", "OBJECT_CONFIG_PATH"]
    
    for var in test_env_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]
    
    yield
    
    # Restore original environment
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]
        if var in original_env:
            os.environ[var] = original_env[var]


@pytest.fixture(autouse=True)
def suppress_api_calls():
    """Suppress actual API calls during testing."""
    # This fixture automatically runs for all tests
    # You could mock API calls here if needed
    pass


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "readme: mark test as README validation test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark README validation tests
        if "readme" in item.nodeid.lower() or "installation" in item.nodeid.lower():
            item.add_marker(pytest.mark.readme)
        
        # Mark integration tests
        if "workflow" in item.nodeid.lower() or "config" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_report_header(config):
    """Add information to pytest header."""
    return [
        "GenHRL Test Suite",
        "Validating README functionality and installation",
        "=" * 50
    ]