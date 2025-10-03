"""
GenHRL Test Suite

This package contains comprehensive tests for all GenHRL functionality
mentioned in the README and documentation.
"""

# Test configuration
import os
import tempfile
from pathlib import Path

# Test constants
TEST_API_KEY = "test_api_key_placeholder"
TEST_ISAACLAB_PATH = tempfile.mkdtemp(prefix="test_isaaclab_")
TEST_ROBOT = "G1"

# Cleanup helper
def cleanup_test_files():
    """Clean up test files and directories."""
    import shutil
    test_dirs = [
        TEST_ISAACLAB_PATH,
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except (OSError, PermissionError):
                pass  # Best effort cleanup