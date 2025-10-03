#!/usr/bin/env python3
"""
Simple training script for GenHRL.
Just trains each skill for exactly 10,000 steps in the correct order.
"""

import sys
import os
from pathlib import Path

# Add the genhrl package to path
sys.path.append(str(Path(__file__).parent))

from genhrl.training.orchestrator import TrainingOrchestrator


def main():
    # Configuration - update these paths for your setup
    ISAACLAB_PATH = "/path/to/IsaacLab"  # Update this!
    TASK_NAME = "your_task_name"        # Update this!
    ROBOT = "G1"                        # Robot name
    STEPS_PER_SKILL = 10000            # Steps to train each skill
    NEW_RUN = False                    # Set to True to clean previous training data
    RECORD_VIDEO = True                # Whether to record training videos
    VIDEO_INTERVAL = 2000              # How often to record videos (in steps)
    VIDEO_LENGTH = 200                 # Length of each video recording (in steps)
    COPY_POLICIES_ONLY = False         # Set to True to only copy policies from logs to skills directory
    COPY_POLICIES_IMMEDIATELY = True  # Set to True to copy policies after each skill (instead of at the end)
    
    print("="*80)
    print("Simple GenHRL Training")
    print("="*80)
    print(f"Task: {TASK_NAME}")
    print(f"Robot: {ROBOT}")
    print(f"IsaacLab Path: {ISAACLAB_PATH}")
    print(f"Steps per skill: {STEPS_PER_SKILL}")
    print(f"New run (clean data): {NEW_RUN}")
    print(f"Copy policies only: {COPY_POLICIES_ONLY}")
    print(f"Copy policies immediately: {COPY_POLICIES_IMMEDIATELY}")
    print(f"Record videos: {RECORD_VIDEO}")
    if RECORD_VIDEO:
        print(f"Video interval: {VIDEO_INTERVAL} steps")
        print(f"Video length: {VIDEO_LENGTH} steps")
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
        
        if COPY_POLICIES_ONLY:
            # Just copy policies from logs to skills directory
            print("\n🔄 Copying policies from logs to skills directory...")
            copied_count = orchestrator.copy_all_policies()
            
            if copied_count > 0:
                print(f"\n✅ Successfully copied {copied_count} policies!")
                return 0
            else:
                print("\n❌ No policies were copied!")
                return 1
        else:
            # Show training plan
            training_order = orchestrator.get_training_order()
            print(f"\n📝 Training plan ({len(training_order)} skills):")
            for i, skill in enumerate(training_order):
                skill_type = "[P]" if orchestrator.is_skill_primitive(skill) else "[C]"
                print(f"  {i+1}. {skill} {skill_type}")
            
            print(f"\n🚀 Starting training sequence...")
            print(f"Each skill will train for exactly {STEPS_PER_SKILL} steps")
            if NEW_RUN:
                print("🆕 New run mode: Will clean previous training data")
            if COPY_POLICIES_IMMEDIATELY:
                print("📋 Policy copying: After each skill")
            else:
                print("📋 Policy copying: At the end of all training")
            if RECORD_VIDEO:
                print(f"📹 Video recording: Every {VIDEO_INTERVAL} steps, {VIDEO_LENGTH} steps each")
            else:
                print("📹 Video recording: Disabled")
            
            # Run the training
            success = orchestrator.train_all_skills_simple(STEPS_PER_SKILL, new_run=NEW_RUN, record_video=RECORD_VIDEO, video_interval=VIDEO_INTERVAL, video_length=VIDEO_LENGTH, copy_policies_immediately=COPY_POLICIES_IMMEDIATELY)
            
            if success:
                print("\n✅ All skills trained successfully!")
                return 0
            else:
                print("\n❌ Training failed!")
                return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 