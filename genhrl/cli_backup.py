"""
Main CLI interface for GenHRL

Provides unified access to task generation and training functionality.
"""

import argparse
import os
import sys
import shutil
import glob
from pathlib import Path
from typing import Optional

from .generation import TaskGenerator, SkillLibrary, TaskManager
from .training import TrainingOrchestrator
from .training.orchestrator import TrainingConfig


def find_project_root() -> Optional[Path]:
    """Find the project root by looking for pyproject.toml."""
    # Start from the directory of this file
    current_path = Path(__file__).resolve().parent
    # Go up until we find pyproject.toml or hit the filesystem root
    while not (current_path / "pyproject.toml").exists():
        if current_path == current_path.parent:  # Reached the root
            return None
        current_path = current_path.parent
    return current_path


def main():
    """Main CLI entry point."""
    project_root = find_project_root()
    default_isaaclab_path = "./IsaacLab"
    if project_root:
        default_isaaclab_path = str(project_root / "IsaacLab")
    else:
        # This is a fallback, but it's better to inform the user.
        print("Warning: Could not determine project root. Using relative path for --isaaclab-path: ./IsaacLab")

    parser = argparse.ArgumentParser(
        description="GenHRL: Generative Hierarchical Reinforcement Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a new task (uses local ./IsaacLab by default)
  genhrl generate "pick up the red ball" --api-key YOUR_KEY
  
  # Generate a simple task with single level only
  genhrl generate "walk forward" --api-key YOUR_KEY --max-hierarchy-levels 1
  
  # Train all skills for a task (complex mode with monitoring)
  genhrl train Create_Steps
  
  # Train with simple mode - just 10,000 steps per skill
  # Uses IsaacLab scripts for primitives, GenHRL scripts for composites
  genhrl train Create_Steps --simple
  
  # Train with simple mode - custom number of steps
  genhrl train Create_Steps --simple --steps 5000
  
  # Train with simple mode and clean previous data
  genhrl train Create_Steps --simple --new-run

  # Train with custom video settings
  genhrl train Create_Steps --simple --video-interval 1000 --video-length 300

  # Train without video recording  
  genhrl train Create_Steps --simple --no-video

  # Print training commands without running them
  genhrl train Create_Steps --simple --print-commands
  
  # Train specific skills only (complex mode)
  genhrl train Create_Steps --skills skill1,skill2
  
  # Generate and train in one command
  genhrl auto "climb the stairs" --api-key YOUR_KEY
  
  # Generate with two-level hierarchy and train
  genhrl auto "pick and place" --api-key YOUR_KEY --max-hierarchy-levels 2
  
  # List all tasks and their status
  genhrl list
  genhrl list --task Create_Steps
  
  # Remove a complete task with all skills and logs
  genhrl remove Create_Steps
  
  # Remove a specific skill from a task
  genhrl remove Create_Steps --skill walk_to_box
  
  # Preview what would be removed (dry run)
  genhrl remove Create_Steps --dry-run
  
  # Use custom IsaacLab path if needed
  genhrl generate "pick up the red ball" --isaaclab-path /custom/path/to/IsaacLab --api-key YOUR_KEY
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a new hierarchical task")
    generate_parser.add_argument("description", help="Natural language description of the task")
    generate_parser.add_argument("--name", help="Task name (auto-generated if not provided)")
    generate_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    generate_parser.add_argument("--api-key", help="API key for LLM services (or set GENHRL_API_KEY env var)")
    generate_parser.add_argument("--provider", choices=["google", "anthropic"], default="google", help="LLM provider")
    generate_parser.add_argument("--max-hierarchy-levels", type=int, default=3, help="Maximum number of hierarchy levels (1-3, default: 3). 1: single task only, 2: task -> skill, 3: task -> skill -> sub_skill")
    generate_parser.add_argument("--robot", default="G1", help="Robot type to use (default: G1). Available: G1, H1, Anymal_B, Anymal_C, Anymal_D, A1, Go1, Go2, Spot, Digit, Franka, UR10")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train skills for a task")
    train_parser.add_argument("task_name", help="Name of the task to train")
    train_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    train_parser.add_argument("--robot", default="G1", help="Robot type to use (default: G1). Available: G1, H1, Anymal_B, Anymal_C, Anymal_D, A1, Go1, Go2, Spot, Digit, Franka, UR10")
    train_parser.add_argument("--skills", help="Comma-separated list of specific skills to train (default: all)")
    train_parser.add_argument("--skip-complete", action="store_true", default=True, help="Skip skills with sufficient success states (default: enabled)")
    train_parser.add_argument("--no-skip-complete", action="store_true", help="Force training of all skills, even those with sufficient success states")
    train_parser.add_argument("--new-run", action="store_true", help="Clean all previous training data")
    train_parser.add_argument("--max-time", type=int, default=180, help="Maximum training time per skill (minutes)")
    train_parser.add_argument("--min-success-states", type=int, default=100, help="Minimum success states required")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--num-envs", type=int, default=4096, help="Number of environments (default: 1024, use lower values like 512 or 256 if you encounter memory issues)")
    train_parser.add_argument("--simple", action="store_true", help="Use simple training mode - just train each skill for exact number of steps without monitoring")
    train_parser.add_argument("--steps", type=int, default=10000, help="Number of steps per skill in simple mode (default: 10000)")
    train_parser.add_argument("--min-primitive-steps", type=int, help="Minimum number of steps for primitive skills (overrides --steps for primitives)")
    train_parser.add_argument("--min-composite-steps", type=int, help="Minimum number of steps for composite skills (overrides --steps for composites)")
    train_parser.add_argument("--print-commands", action="store_true", help="Only print training commands without running them")
    train_parser.add_argument("--video", action="store_true", default=True, help="Enable video recording during training (default: enabled)")
    train_parser.add_argument("--no-video", action="store_true", help="Disable video recording during training")
    train_parser.add_argument("--video-interval", type=int, default=2000, help="Video recording interval in steps (default: 2000)")
    train_parser.add_argument("--video-length", type=int, default=200, help="Video recording length in steps (default: 200)")
    
    # Auto command (generate + train)
    auto_parser = subparsers.add_parser("auto", help="Generate and train a task automatically")
    auto_parser.add_argument("description", help="Natural language description of the task")
    auto_parser.add_argument("--name", help="Task name (auto-generated if not provided)")
    auto_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    auto_parser.add_argument("--api-key", help="API key for LLM services (or set GENHRL_API_KEY env var)")
    auto_parser.add_argument("--provider", choices=["google", "anthropic"], default="google", help="LLM provider")
    auto_parser.add_argument("--max-hierarchy-levels", type=int, default=3, help="Maximum number of hierarchy levels (1-3, default: 3). 1: single task only, 2: task -> skill, 3: task -> skill -> sub_skill")
    auto_parser.add_argument("--robot", default="G1", help="Robot type to use (default: G1). Available: G1, H1, Anymal_B, Anymal_C, Anymal_D, A1, Go1, Go2, Spot, Digit, Franka, UR10")
    auto_parser.add_argument("--skip-complete", action="store_true", default=True, help="Skip skills with sufficient success states")
    auto_parser.add_argument("--max-time", type=int, default=180, help="Maximum training time per skill (minutes)")
    auto_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    auto_parser.add_argument("--num-envs", type=int, default=4096, help="Number of environments (default: 4096, use lower values like 512 or 256 if you encounter memory issues)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List existing tasks and their status")
    list_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    list_parser.add_argument("--robot", help="Filter by robot type (optional)")
    list_parser.add_argument("--task", help="Show details for specific task")
    
    # Status command  
    status_parser = subparsers.add_parser("status", help="Show training status for a task")
    status_parser.add_argument("task_name", help="Name of the task")
    status_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    status_parser.add_argument("--robot", help="Robot type (optional, auto-detected from task folder)")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove tasks or skills with proper cleanup")
    remove_parser.add_argument("task_name", help="Name of the task")
    remove_parser.add_argument("--skill", help="Name of specific skill to remove (if not provided, removes entire task)")
    remove_parser.add_argument("--isaaclab-path", default=default_isaaclab_path, help=f"Path to IsaacLab installation (default: {default_isaaclab_path})")
    remove_parser.add_argument("--robot", help="Robot type (optional, auto-detected from task folder)")
    remove_parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    remove_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "generate":
        handle_generate(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "auto":
        handle_auto(args)
    elif args.command == "list":
        handle_list(args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "remove":
        handle_remove(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def _clean_previous_runs(orchestrator: TrainingOrchestrator) -> None:
    """Clean previous training data for the specific task only."""
    print(f"🧹 Cleaning previous training data for task: {orchestrator.task_name}...")
    
    # Use the orchestrator's built-in task-specific cleanup method
    # This will clean both skill data and task-specific logs
    orchestrator.clean_previous_runs()


def handle_generate(args):
    """Handle task generation command."""
    # Validate max_hierarchy_levels
    if args.max_hierarchy_levels < 1 or args.max_hierarchy_levels > 3:
        print("Error: --max-hierarchy-levels must be between 1 and 3")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv("GENHRL_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set GENHRL_API_KEY environment variable")
        sys.exit(1)
    
    # Verify IsaacLab path
    isaaclab_path = Path(args.isaaclab_path).resolve()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        sys.exit(1)
    
    # Generate task name if not provided
    task_name = args.name
    if not task_name:
        # Create a simple task name from description
        task_name = "_".join(args.description.lower().split()[:3]).replace(".", "").replace(",", "")
        task_name = "".join(c for c in task_name if c.isalnum() or c == "_")
        print(f"Generated task name: {task_name}")
    
    # Create task manager and generate task
    print(f"🚀 Initializing GenHRL with IsaacLab at: {isaaclab_path}")
    print(f"🤖 Using robot: {args.robot}")
    task_manager = TaskManager(str(isaaclab_path), api_key, robot=args.robot)
    
    print(f"📝 Generating task: {task_name}")
    print(f"💭 Description: {args.description}")
    
    task_config = task_manager.create_task_from_description(task_name, args.description, max_hierarchy_levels=args.max_hierarchy_levels, robot=args.robot)
    
    print(f"\n🎉 Task generation complete!")
    print(f"📁 Task files generated at: {task_config.get_task_path()}")
    print(f"🎯 Skills generated at: {task_config.get_skills_base_path() / 'skills'}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review generated files in IsaacLab")
    print(f"  2. Train skills with: genhrl train {task_name} --robot {args.robot} --isaaclab-path {isaaclab_path}")


def handle_train(args):
    """Handle training command."""
    # Verify IsaacLab path
    isaaclab_path = Path(args.isaaclab_path).resolve()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        sys.exit(1)
    
    print(f"🚀 Initializing training for task: {args.task_name}")
    print(f"🤖 Using robot: {args.robot}")
    
    # Create training orchestrator
    orchestrator = TrainingOrchestrator(str(isaaclab_path), args.task_name, robot=args.robot)
    
    # Handle print commands mode
    if args.print_commands:
        print("📋 Printing training commands...")
        if args.simple:
            # Get video settings from args
            video_interval = args.video_interval
            video_length = args.video_length
            record_video = args.video and not args.no_video
            orchestrator.print_training_commands(args.steps, record_video, video_interval, video_length, args.num_envs, args.seed)
        else:
            print("Print commands only available in simple mode. Use --simple flag.")
        return

    # Handle simple training mode
    if args.simple:
        print(f"🎯 Using simple training mode: {args.steps} steps per skill")
        if args.new_run:
            print("🆕 New run mode enabled - will clean previous training data")
        
        # Get video settings from args  
        video_interval = args.video_interval
        video_length = args.video_length
        record_video = args.video and not args.no_video
        
        # Handle skip logic: --no-skip-complete overrides --skip-complete
        skip_if_complete = args.skip_complete and not args.no_skip_complete
        
        success = orchestrator.train_all_skills_simple(
            steps_per_skill=args.steps, 
            new_run=args.new_run,
            record_video=record_video,
            video_interval=video_interval,
            video_length=video_length,
            min_success_states=args.min_success_states,
            skip_if_complete=skip_if_complete,
            num_envs=args.num_envs,
            seed=args.seed
        )
        
        if success:
            print(f"\n✅ Simple training completed successfully for {args.task_name}!")
        else:
            print(f"\n❌ Simple training failed for {args.task_name}")
            sys.exit(1)
        return
    
    # Handle complex training mode (original)
    print("⚙️ Using complex training mode with monitoring")
    
    # Handle new run - clean previous training data
    if args.new_run:
        print("🧹 Cleaning previous training data...")
        _clean_previous_runs(orchestrator)
    
    # Create training config  
    config = TrainingConfig(
        skill_name="",  # Will be set per skill
        max_time_minutes=args.max_time,
        min_success_states=args.min_success_states,
        seed=args.seed,
        num_envs=args.num_envs,
        headless=True  # CLI always runs headless
    )
    
    # Train specific skills or all skills
    if args.skills:
        skill_names = [s.strip() for s in args.skills.split(",")]
        print(f"Training specific skills: {skill_names}")
        # TODO: Implement selective skill training
        print("Selective skill training not yet implemented. Training all skills.")
    
    # Note: --skip-complete functionality is now built into the simplified orchestrator
    # Each skill automatically checks has_sufficient_success_states() before training
    
    # Train all skills
    success = orchestrator.train_all_skills(config=config)
    
    if success:
        print(f"\n✅ Training completed successfully for {args.task_name}!")
    else:
        print(f"\n❌ Training failed for {args.task_name}")
        sys.exit(1)


def handle_auto(args):
    """Handle automatic generate + train command."""
    # Generate the task first
    print("🔄 Auto mode: Generating task first...")
    generate_args = argparse.Namespace(
        description=args.description,
        name=args.name,
        isaaclab_path=args.isaaclab_path,
        api_key=args.api_key,
        provider=args.provider,
        max_hierarchy_levels=args.max_hierarchy_levels,
        robot=args.robot
    )
    handle_generate(generate_args)
    
    # Determine task name
    task_name = args.name
    if not task_name:
        task_name = "_".join(args.description.lower().split()[:3]).replace(".", "").replace(",", "")
        task_name = "".join(c for c in task_name if c.isalnum() or c == "_")
    
    print(f"\n🔄 Auto mode: Starting training for {task_name}...")
    
    # Train the task
    train_args = argparse.Namespace(
        task_name=task_name,
        isaaclab_path=args.isaaclab_path,
        robot=args.robot,
        skills=None,
        skip_complete=args.skip_complete,
        no_skip_complete=False,
        new_run=False,
        max_time=args.max_time,
        min_success_states=100,
        seed=args.seed,
        num_envs=args.num_envs,
        simple=False,
        steps=10000,
        print_commands=False,
        video=True,
        no_video=False,
        video_interval=2000,
        video_length=200
    )
    handle_train(train_args)


def handle_list(args):
    """Handle list tasks command."""
    isaaclab_path = Path(args.isaaclab_path).resolve()
    
    # If robot filter is specified, only look in that robot's directory
    if args.robot:
        robot_dirs = [f"{args.robot}_generated"]
    else:
        # Look for all robot directories
        base_path = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        robot_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_generated")]
        if not robot_dirs:
            # Fallback to G1_generated for backward compatibility
            robot_dirs = ["G1_generated"]
    
    all_tasks = {}
    for robot_dir in robot_dirs:
        tasks_dir = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_dir / "tasks"
        if tasks_dir.exists():
            robot_name = robot_dir.replace("_generated", "")
            tasks = [d.name for d in tasks_dir.iterdir() if d.is_dir()]
            all_tasks[robot_name] = tasks
    
    if not all_tasks:
        print("No tasks directory found.")
        return
    
    if args.task:
        # Show details for specific task - need to find which robot it belongs to
        task_found = False
        for robot_name, tasks in all_tasks.items():
            if args.task in tasks:
                task_found = True
                tasks_dir = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / f"{robot_name}_generated/tasks"
                
                print(f"📋 Task Details: {args.task} (Robot: {robot_name})")
                task_path = tasks_dir / args.task
                
                # Show description
                description_file = task_path / "description.txt"
                if description_file.exists():
                    with open(description_file, 'r') as f:
                        description = f.read().strip()
                    print(f"📝 Description: {description}")
                
                # Show skills
                try:
                    orchestrator = TrainingOrchestrator(str(isaaclab_path), args.task, robot=robot_name)
                    training_order = orchestrator.get_training_order()
                    status = orchestrator.get_training_status()
                    
                    print(f"\n🎯 Skills ({len(training_order)}):")
                    for i, skill in enumerate(training_order):
                        skill_status = status.get(skill, {})
                        skill_type = "[P]" if skill_status.get("is_primitive", False) else "[C]"
                        has_policy = "✅" if skill_status.get("policy_exists", False) else "❌"
                        print(f"  {i+1}. {skill} {skill_type} Policy: {has_policy}")
                except Exception as e:
                    print(f"Error loading task details: {e}")
                break
        
        if not task_found:
            print(f"Task '{args.task}' not found.")
            return
    else:
        # List all tasks by robot
        for robot_name, tasks in all_tasks.items():
            if not tasks:
                continue
                
            print(f"\n🤖 {robot_name} Tasks:")
            tasks_dir = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / f"{robot_name}_generated/tasks"
            
            for task in sorted(tasks):
                task_path = tasks_dir / task
                description_file = task_path / "description.txt"
                
                if description_file.exists():
                    with open(description_file, 'r') as f:
                        description = f.read().strip()[:100]
                        if len(description) > 100:
                            description += "..."
                else:
                    description = "No description available"
                
                print(f"  📝 {task}: {description}")


def handle_status(args):
    """Handle training status command."""
    isaaclab_path = Path(args.isaaclab_path).resolve()
    
    # Auto-detect robot if not provided
    robot = args.robot
    if not robot:
        # Try to find which robot directory contains this task
        base_path = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        for robot_dir in base_path.iterdir():
            if robot_dir.is_dir() and robot_dir.name.endswith("_generated"):
                tasks_dir = robot_dir / "tasks"
                if tasks_dir.exists() and (tasks_dir / args.task_name).exists():
                    robot = robot_dir.name.replace("_generated", "")
                    break
        
        if not robot:
            # Fallback to G1 for backward compatibility
            robot = "G1"
    
    try:
        orchestrator = TrainingOrchestrator(str(isaaclab_path), args.task_name, robot=robot)
        status = orchestrator.get_training_status()
        
        print(f"🎯 Training Status for: {args.task_name} (Robot: {robot})")
        print("=" * 50)
        
        for skill_name, skill_status in status.items():
            skill_type = "[P]" if skill_status["is_primitive"] else "[C]"
            status_emoji = {
                "completed": "✅",
                "training": "🔄", 
                "failed": "❌",
                "pending": "⏳",
                "skipped": "⏭️"
            }.get(skill_status["status"], "❓")
            
            policy_status = "✅" if skill_status["policy_exists"] else "❌"
            success_states = "✅" if skill_status["has_success_states"] else "❌"
            
            print(f"{status_emoji} {skill_name} {skill_type}")
            print(f"    Status: {skill_status['status']}")
            print(f"    Policy: {policy_status}")  
            print(f"    Success States: {success_states}")
            print()
            
    except FileNotFoundError as e:
        print(f"Task '{args.task_name}' not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error getting status: {e}")
        sys.exit(1)


def handle_remove(args):
    """Handle remove command for tasks and skills."""
    from .generation.removal_manager import RemovalManager
    
    isaaclab_path = Path(args.isaaclab_path).resolve()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        sys.exit(1)
    
    # Auto-detect robot if not provided
    robot = args.robot
    if not robot:
        # Try to find which robot directory contains this task
        base_path = isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        for robot_dir in base_path.iterdir():
            if robot_dir.is_dir() and robot_dir.name.endswith("_generated"):
                tasks_dir = robot_dir / "tasks"
                if tasks_dir.exists() and (tasks_dir / args.task_name).exists():
                    robot = robot_dir.name.replace("_generated", "")
                    break
        
        if not robot:
            # Fallback to G1 for backward compatibility
            robot = "G1"
    
    # Create removal manager
    removal_manager = RemovalManager(str(isaaclab_path), robot)
    
    # Handle dry run
    if args.dry_run:
        removal_manager.dry_run_removal(args.task_name, args.skill)
        return
    
    # Perform actual removal
    try:
        if args.skill:
            # Remove specific skill
            print(f"🎯 Removing skill '{args.skill}' from task '{args.task_name}' (Robot: {robot})")
            success = removal_manager.remove_skill(
                task_name=args.task_name,
                skill_name=args.skill,
                confirm=not args.force
            )
        else:
            # Remove entire task
            print(f"🗑️  Removing task '{args.task_name}' (Robot: {robot})")
            success = removal_manager.remove_task(
                task_name=args.task_name,
                confirm=not args.force
            )
        
        if success:
            print(f"\n🎉 Removal completed successfully!")
        else:
            print(f"\n❌ Removal failed or was cancelled")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ Not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during removal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()