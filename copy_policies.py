#!/usr/bin/env python3
"""
Simple script to copy trained policies from logs to skills directory.
"""

import sys
from pathlib import Path

# Add the genhrl package to path
sys.path.append(str(Path(__file__).parent))

from genhrl.training.orchestrator import TrainingOrchestrator


def main():
    # Configuration - update these paths for your setup
    ISAACLAB_PATH = "/path/to/IsaacLab"  # Update this!
    TASK_NAME = "your_task_name"        # Update this!
    ROBOT = "G1"                        # Robot name
    
    print("="*80)
    print("Copy Trained Policies")
    print("="*80)
    print(f"Task: {TASK_NAME}")
    print(f"Robot: {ROBOT}")
    print(f"IsaacLab Path: {ISAACLAB_PATH}")
    print("="*80)
    
    try:
        # Check if paths exist
        if not Path(ISAACLAB_PATH).exists():
            print(f"❌ IsaacLab path does not exist: {ISAACLAB_PATH}")
            print("Please update ISAACLAB_PATH in this script")
            return 1
        
        # Initialize orchestrator
        print("\n📋 Initializing orchestrator...")
        orchestrator = TrainingOrchestrator(
            isaaclab_path=ISAACLAB_PATH,
            task_name=TASK_NAME,
            robot=ROBOT
        )
        
        # Copy all policies
        print("\n🔄 Searching for and copying trained policies...")
        copied_count = orchestrator.copy_all_policies()
        
        if copied_count > 0:
            print(f"\n✅ Successfully copied {copied_count} policies!")
            return 0
        else:
            print("\n❌ No policies were copied!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 