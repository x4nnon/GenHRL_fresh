#!/usr/bin/env python3
"""
Demo script showing how to use the simplified GenHRL training.

This demonstrates the new simple training mode that just runs each skill 
for exactly N steps without complex monitoring or subprocess management.
"""

import sys
import os
from pathlib import Path

# Add the genhrl package to path
sys.path.append(str(Path(__file__).parent))

from genhrl.training.orchestrator import TrainingOrchestrator


def main():
    print("=" * 80)
    print("GenHRL Simple Training Demo")
    print("=" * 80)
    
    # Configuration - update these for your setup
    ISAACLAB_PATH = "./IsaacLab"  # Update this to your IsaacLab path
    TASK_NAME = "your_task_name"  # Update this to your task name
    ROBOT = "G1"
    
    print(f"Task: {TASK_NAME}")
    print(f"Robot: {ROBOT}")
    print(f"IsaacLab: {ISAACLAB_PATH}")
    print()
    
    # Check if paths exist
    if not Path(ISAACLAB_PATH).exists():
        print("❌ Please update ISAACLAB_PATH in this script to point to your IsaacLab installation")
        return 1
    
    try:
        # Initialize orchestrator
        print("📋 Initializing orchestrator...")
        orchestrator = TrainingOrchestrator(
            isaaclab_path=ISAACLAB_PATH,
            task_name=TASK_NAME,
            robot=ROBOT
        )
        
        # Show training order
        training_order = orchestrator.get_training_order()
        print(f"\n📝 Training order ({len(training_order)} skills):")
        for i, skill in enumerate(training_order):
            skill_type = "[Primitive]" if orchestrator.is_skill_primitive(skill) else "[Composite]"
            print(f"  {i+1}. {skill} {skill_type}")
        
        print("\n" + "=" * 60)
        print("DEMO OPTIONS")
        print("=" * 60)
        
        print("\n1️⃣  PRINT COMMANDS ONLY")
        print("   See what commands would be run without actually training:")
        print(f"   python -c \"")
        print(f"from genhrl.training.orchestrator import TrainingOrchestrator;")
        print(f"o = TrainingOrchestrator('{ISAACLAB_PATH}', '{TASK_NAME}', '{ROBOT}');")
        print(f"o.print_training_commands(10000)\"")
        
        print("\n2️⃣  SIMPLE TRAINING (10,000 steps per skill)")
        print("   Train each skill for exactly 10,000 steps:")
        print(f"   python -c \"")
        print(f"from genhrl.training.orchestrator import TrainingOrchestrator;")
        print(f"o = TrainingOrchestrator('{ISAACLAB_PATH}', '{TASK_NAME}', '{ROBOT}');")
        print(f"o.train_all_skills_simple(10000)\"")
        
        print("\n3️⃣  CUSTOM STEPS")
        print("   Train each skill for custom number of steps:")
        print(f"   python -c \"")
        print(f"from genhrl.training.orchestrator import TrainingOrchestrator;")
        print(f"o = TrainingOrchestrator('{ISAACLAB_PATH}', '{TASK_NAME}', '{ROBOT}');")
        print(f"o.train_all_skills_simple(5000)\"")
        
        print("\n4️⃣  SINGLE SKILL")
        print("   Train just one skill:")
        print(f"   python -c \"")
        print(f"from genhrl.training.orchestrator import TrainingOrchestrator;")
        print(f"o = TrainingOrchestrator('{ISAACLAB_PATH}', '{TASK_NAME}', '{ROBOT}');")
        print(f"o.train_skill_simple('{training_order[0] if training_order else 'skill_name'}', 10000)\"")

        print("\n5️⃣  CUSTOM VIDEO SETTINGS")
        print("   Train with custom video recording:")
        print(f"   python -c \"")
        print(f"from genhrl.training.orchestrator import TrainingOrchestrator;")
        print(f"o = TrainingOrchestrator('{ISAACLAB_PATH}', '{TASK_NAME}', '{ROBOT}');")
        print(f"o.train_all_skills_simple(10000, record_video=True, video_interval=1000, video_length=300)\"")

        print("\n6️⃣  CLI COMMANDS")
        print("   Use the genhrl command line interface:")
        print(f"   genhrl train {TASK_NAME} --simple --steps 10000")
        print(f"   genhrl train {TASK_NAME} --simple --new-run  # Clean previous data")
        print(f"   genhrl train {TASK_NAME} --simple --video-interval 1000 --video-length 100  # Custom video settings")
        print(f"   genhrl train {TASK_NAME} --simple --no-video  # Disable video recording")
        print(f"   genhrl train {TASK_NAME} --simple --print-commands")
        
        print("\n" + "=" * 60)
        print("ACTUALLY RUN THE DEMO?")
        print("=" * 60)
        
        choice = input("\nWould you like to:\n[1] Print commands only\n[2] Run simple training\n[q] Quit\nChoice: ").strip().lower()
        
        if choice == '1':
            print("\n📋 Printing training commands...")
            orchestrator.print_training_commands(10000)
            
        elif choice == '2':
            print("\n🚀 Running simple training (10,000 steps per skill)...")
            confirm = input("This will actually start training. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                success = orchestrator.train_all_skills_simple(10000)
                if success:
                    print("\n✅ Training completed!")
                else:
                    print("\n❌ Training failed!")
            else:
                print("Training cancelled.")
        else:
            print("Demo cancelled.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("- Updated ISAACLAB_PATH to point to your IsaacLab installation")
        print("- Updated TASK_NAME to an existing task")
        print("- Generated the task files using 'genhrl generate'")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 