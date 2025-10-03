#!/usr/bin/env python3
"""
Simple script to run GenHRL training using the simplified orchestrator.

Usage:
    python run_training.py --isaaclab-path /path/to/isaaclab --task-name your_task --robot G1
"""

import argparse
import sys
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from genhrl.training.orchestrator import TrainingOrchestrator, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Run GenHRL training sequence")
    parser.add_argument("--isaaclab-path", required=True, help="Path to IsaacLab installation")
    parser.add_argument("--task-name", required=True, help="Name of the task to train")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--max-time", type=int, default=60, help="Max time per skill in minutes (default: 60)")
    parser.add_argument("--min-success-states", type=int, default=100, help="Minimum success states required (default: 100)")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of environments (default: 4096)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-headless", action="store_true", help="Run with GUI (default: headless)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GenHRL Training - Simplified Orchestrator")
    print("=" * 60)
    print(f"IsaacLab Path: {args.isaaclab_path}")
    print(f"Task: {args.task_name}")
    print(f"Robot: {args.robot}")
    print(f"Max time per skill: {args.max_time} minutes")
    print(f"Min success states: {args.min_success_states}")
    print(f"Environments: {args.num_envs}")
    print(f"Headless: {not args.no_headless}")
    print("=" * 60)
    
    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            isaaclab_path=args.isaaclab_path,
            task_name=args.task_name,
            robot=args.robot
        )
        
        # Create training config
        config = TrainingConfig(
            skill_name="training",  # Will be updated per skill
            max_time_minutes=args.max_time,
            min_success_states=args.min_success_states,
            num_envs=args.num_envs,
            seed=args.seed,
            headless=not args.no_headless
        )
        
        # Run training
        print("Starting training sequence...")
        success = orchestrator.train_all_skills(config)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Print final status
            status = orchestrator.get_training_status()
            for skill_name, skill_status in status.items():
                print(f"✓ {skill_name}: {skill_status['status']}")
        else:
            print("\n" + "=" * 60)
            print("❌ TRAINING FAILED")
            print("=" * 60)
            
            # Print status for debugging
            status = orchestrator.get_training_status()
            for skill_name, skill_status in status.items():
                symbol = "✓" if skill_status['status'] in ['completed', 'skipped'] else "❌"
                print(f"{symbol} {skill_name}: {skill_status['status']}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 