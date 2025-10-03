#!/usr/bin/env python3
"""
Quick test to verify that all required imports for testing scripts work correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import genhrl modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    
    print("🧪 Testing imports for GenHRL testing framework...")
    
    try:
        # Test core training import
        from genhrl.training import TrainingOrchestrator
        print("✅ TrainingOrchestrator import successful")
    except ImportError as e:
        print(f"❌ Failed to import TrainingOrchestrator: {e}")
        return False
    
    try:
        # Test generation import
        from genhrl.generation import TaskManager
        print("✅ TaskManager import successful")
    except ImportError as e:
        print(f"❌ Failed to import TaskManager: {e}")
        return False
    
    # Test standard library imports used in testing
    try:
        import argparse
        import os
        import json
        import time
        import subprocess
        from dataclasses import dataclass
        from typing import List, Dict, Optional, Tuple
        print("✅ Standard library imports successful")
    except ImportError as e:
        print(f"❌ Failed to import standard library module: {e}")
        return False
    
    print("🎉 All imports successful! Testing framework should work correctly.")
    return True

def test_paths():
    """Test that basic paths exist."""
    
    print("\n🗂️  Testing basic paths...")
    
    # Test that IsaacLab default path exists
    isaaclab_path = Path("./IsaacLab")
    if isaaclab_path.exists():
        print(f"✅ Default IsaacLab path exists: {isaaclab_path.resolve()}")
    else:
        print(f"⚠️  Default IsaacLab path not found: {isaaclab_path.resolve()}")
        print("   You may need to specify --isaaclab-path when running tests")
    
    # Test testing directory structure
    testing_dir = Path(__file__).parent
    print(f"✅ Testing directory: {testing_dir}")
    
    results_dir = testing_dir / "results"
    if results_dir.exists():
        print(f"✅ Results directory exists: {results_dir}")
    else:
        print(f"📁 Results directory will be created when needed: {results_dir}")
    
    return True

if __name__ == "__main__":
    print("GenHRL Testing Framework - Import Verification")
    print("=" * 50)
    
    imports_ok = test_imports()
    paths_ok = test_paths()
    
    if imports_ok and paths_ok:
        print("\n🎉 All checks passed! You're ready to use the testing framework.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please resolve the issues above.")
        sys.exit(1)