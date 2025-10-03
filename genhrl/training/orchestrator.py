"""
Training Orchestrator for GenHRL

This module coordinates the entire training process for hierarchical RL tasks,
integrating with IsaacLab and managing skill sequences.
"""

import os
import json
import subprocess
import time
import glob
import sys
import shutil
import re
import yaml
import signal
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class TrainingStatus(Enum):
    """Training status for skills."""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TrainingConfig:
    """Configuration for training a skill."""
    skill_name: str
    max_time_minutes: int = 180
    min_success_states: int = 100
    num_envs: int = 4096
    seed: int = 42
    headless: bool = True


class TrainingOrchestrator:
    """
    Simple orchestrator for hierarchical RL training.
    
    Runs skills in sequence and monitors for completion.
    """
    
    def __init__(self, 
                 isaaclab_path: str,
                 task_name: str,
                 robot: str = "G1"):
        """
        Initialize the training orchestrator.
        
        Args:
            isaaclab_path: Path to IsaacLab installation
            task_name: Name of the task to train
            robot: Robot name (default: G1)
        """
        self.isaaclab_path = Path(isaaclab_path)
        self.task_name = task_name
        self.robot = robot
        
        # Determine GenHRL scripts path (where hierarchical training scripts are located)
        # This file is in genhrl/training/orchestrator.py, so scripts are at genhrl/scripts/
        self.genhrl_scripts_path = Path(__file__).parent.parent / "scripts"
        
        # Verify the GenHRL scripts path exists
        if not self.genhrl_scripts_path.exists():
            print(f"Warning: GenHRL scripts path not found: {self.genhrl_scripts_path}")
            print(f"Hierarchical training scripts (train_l1.py, train_l2.py) may not work correctly.")
        
        # Import robot configuration helpers
        try:
            sys.path.append(str(Path(__file__).parent.parent / "generation"))
            from robot_configs import get_robot_folder_name
            robot_folder = get_robot_folder_name(robot)
        except ImportError:
            # Fallback for backward compatibility
            robot_folder = f"{robot}_generated"
        
        # Set up robot-specific paths
        self.tasks_base_path = self.isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder
        self.task_path = self.tasks_base_path / "tasks" / task_name
        self.skills_base_path = self.tasks_base_path / "skills"
        self.skills_path = self.skills_base_path / task_name / "skills"
        self.skill_library_path = self.skills_base_path / task_name / "skill_library.json"
        self.object_config_path = self.task_path / "object_config.json"
        
        # Verify paths exist
        if not self.object_config_path.exists():
            raise FileNotFoundError(f"Object config not found: {self.object_config_path}")
        if not self.skill_library_path.exists():
            raise FileNotFoundError(f"Skill library not found: {self.skill_library_path}")
        
        # Set environment variable for IsaacLab scripts
        os.environ['OBJECT_CONFIG_PATH'] = str(self.object_config_path)
        os.environ['GENHRL_TASK_NAME'] = self.task_name
        os.environ['GENHRL_ROBOT'] = self.robot
        
        # Training state
        self.training_status: Dict[str, TrainingStatus] = {}
        self.skill_library: Dict[str, Any] = {}
        self.training_order: List[str] = []
        # Track active child process (IsaacLab) for reliable cleanup
        self._active_process: Optional[subprocess.Popen] = None
        self._shutdown_requested: bool = False
        
        # Load skill library
        self._load_skill_library()

        # Register signal handlers to ensure IsaacLab shuts down cleanly
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            # Some environments (e.g., threads) may not allow setting handlers
            pass

    def _handle_signal(self, signum, frame) -> None:
        """Handle Ctrl+C / termination by shutting down child cleanly."""
        print(f"\nReceived signal {signum}. Shutting down IsaacLab...")
        self._shutdown_requested = True
        self._terminate_active_process()

    def _terminate_active_process(self, grace_seconds: int = 5) -> None:
        """Terminate the active IsaacLab subprocess and its process group."""
        proc = self._active_process
        if not proc:
            return
        try:
            # Send SIGTERM to the whole group
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            # Wait for graceful shutdown
            end_time = time.time() + max(1, grace_seconds)
            while proc.poll() is None and time.time() < end_time:
                time.sleep(0.2)
            # Escalate if still alive
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            # Drain and close stdout if present
            try:
                if proc.stdout:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
            finally:
                self._active_process = None
        except Exception:
            self._active_process = None
    
    def _format_gym_task_name(self, skill_name: str) -> str:
        """Format skill name for gym registry with task-specific naming.
        
        Converts skill_name to TaskName_SkillName format in title case.
        For example: 'walk_to_lowWall' -> 'ObstacleCourseSeeded42_WalkToLowwall'
        """
        # Create unique identifier by combining task name and skill name
        # Use underscore separator to match registration format: TaskName_SkillName
        gym_id_base = f"{self.task_name}_{skill_name}"
        
        # Convert to title case, but preserve the underscore between task and skill
        # Split by underscores and capitalize each part, but keep the main separator
        task_parts = self.task_name.split('_')
        skill_parts = skill_name.split('_')
        
        # Format: TaskName_SkillName (without underscore - matches registration format)
        formatted_task = ''.join(part.capitalize() for part in task_parts)
        formatted_skill = ''.join(part.capitalize() for part in skill_parts)
        
        # Return the full formatted name without underscore separator
        return f"{formatted_task}{formatted_skill}"
    
    def _load_skill_library(self) -> None:
        """Load the skill library JSON file."""
        with open(self.skill_library_path, 'r') as f:
            self.skill_library = json.load(f)
        
        if "skills" not in self.skill_library or "hierarchies" not in self.skill_library:
            raise ValueError(f"Invalid skill library format: {self.skill_library_path}")
    
    def get_training_order(self) -> List[str]:
        """Get the hierarchical training order for the task."""
        if self.task_name not in self.skill_library.get("hierarchies", {}):
            raise ValueError(f"Task '{self.task_name}' not found in skill library hierarchies")
        
        hierarchy_structure = self.skill_library["hierarchies"][self.task_name]["structure"]
        return self._get_hierarchical_training_order(hierarchy_structure)
    
    def _get_hierarchical_training_order(self, hierarchy_node: Dict) -> List[str]:
        """Determine training order using dependency levels from the skill library.

        Primitives (level 0) first, then composites with increasing levels computed via
        get_skill_level (1 + max level of sub-skills). This ensures parent composites
        always come after their dependencies, including the root task last.
        """
        # Collect skill names in traversal order for stable tie-breaking
        ordered_names: List[str] = []
        seen: set = set()

        def _collect_names(node: Dict) -> None:
            if not isinstance(node, dict) or "name" not in node:
                return
            name = node["name"]
            if name not in seen:
                seen.add(name)
                ordered_names.append(name)
            for child in node.get("children", []) or []:
                if isinstance(child, dict):
                    _collect_names(child)

        _collect_names(hierarchy_node)

        # Compute levels using library-defined sub_skills relationships
        skills_with_levels = []
        for idx, name in enumerate(ordered_names):
            level = self.get_skill_level(name)
            skills_with_levels.append((level, idx, name))

        # Sort by level ascending, then by traversal order (idx) for stability
        skills_with_levels.sort(key=lambda t: (t[0], t[1]))

        return [name for _, _, name in skills_with_levels]
    
    def _collect_skills_by_level(self, node: Dict, level: int, skills_by_level: Dict[int, List[str]]) -> None:
        """Recursively collect skills organized by their hierarchy level."""
        if not isinstance(node, dict) or "name" not in node:
            return
            
        skill_name = node["name"]
        children = node.get("children", [])
        
        if not children:  # Primitive skill (leaf node)
            if 0 not in skills_by_level:
                skills_by_level[0] = []
            skills_by_level[0].append(skill_name)
        else:
            # Composite skill
            max_child_level = level
            for child in children:
                if isinstance(child, dict) and "name" in child:
                    self._collect_skills_by_level(child, level + 1, skills_by_level)
                    child_children = child.get("children", [])
                    if child_children:
                        max_child_level = max(max_child_level, level + 1)
            
            # Add this composite skill at its appropriate level
            composite_level = max_child_level
            if composite_level not in skills_by_level:
                skills_by_level[composite_level] = []
            skills_by_level[composite_level].append(skill_name)
    
    def is_skill_primitive(self, skill_name: str) -> bool:
        """Check if a skill is primitive."""
        skill_info = self.skill_library.get("skills", {}).get(skill_name)
        if not skill_info:
            return False
        return skill_info.get("is_primitive", False)
    
    def get_skill_level(self, skill_name: str) -> int:
        """Get the hierarchy level of a skill."""
        if self.is_skill_primitive(skill_name):
            return 0
        
        skill_info = self.skill_library.get("skills", {}).get(skill_name)
        if not skill_info:
            return 0
        
        sub_skills = skill_info.get("sub_skills", [])
        if not sub_skills:
            return 0
        
        # Level is 1 + max level of sub-skills
        max_sub_level = 0
        for sub_skill in sub_skills:
            sub_level = self.get_skill_level(sub_skill)
            max_sub_level = max(max_sub_level, sub_level)
        
        return max_sub_level + 1
    
    def has_sufficient_success_states(self, skill_name: str, min_states: int = 100) -> bool:
        """Check if a skill has enough success state files."""
        skill_dir = self.skills_path / skill_name
        success_states_dir = skill_dir / "success_states"

        if not success_states_dir.exists():
            return False

        state_files = glob.glob(str(success_states_dir / "success_states_*.pt"))
        num_files = len(state_files)
        print(f"Found {num_files} success state files for {skill_name}")
        return num_files >= min_states
    
    def build_training_command(self, skill_name: str, config: TrainingConfig) -> str:
        """Build the Isaac Lab training command for a skill."""
        is_primitive = self.is_skill_primitive(skill_name)
        skill_level = self.get_skill_level(skill_name)
        
        # Format skill name for gym registry with task-specific naming
        # Convert to title case and include task name to match new registration format
        gym_task_name_suffix = self._format_gym_task_name(skill_name)
        
        if is_primitive:
            # Use IsaacLab's standard training script for primitive skills
            script_path = "scripts/reinforcement_learning/skrl/train.py"
            task_prefix = "Isaac-RobotFlat"
        else:
            # Use GenHRL's hierarchical training scripts for composite skills
            if skill_level > 0:
                script_name = f"train_l{skill_level}.py"
            else:
                script_name = "train.py"
            script_path = str(self.genhrl_scripts_path / script_name)
            task_prefix = "Isaac-RobotComposite"
        
        base_command = (
            f"./isaaclab.sh -p {script_path} "
            f"--task {task_prefix}{gym_task_name_suffix}-v0 "
            f"--num_envs {config.num_envs} --seed {config.seed}"
        )
        
        if not is_primitive:
            base_command += f" --skill_name {skill_name}"
        
        if config.headless:
            base_command += " --headless"
        
        # Check for existing policy
        policy_file = self.skills_path / skill_name / "policy" / "agent.pt"
        if policy_file.exists():
            print(f"Found existing policy for {skill_name}: {policy_file}")
            base_command += f" --checkpoint \"{policy_file}\""
        
        # Apply memory and CUDA fixes for full training
        env_setup = (
            "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 && "
            "export CARB_LOG_LEVEL=FATAL && "
            "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048 && "
            "export CUDA_LAUNCH_BLOCKING=0 && "
            f"export OBJECT_CONFIG_PATH='{self.object_config_path}' && "
            f"export GENHRL_TASK_NAME='{self.task_name}' && "
            f"export GENHRL_ROBOT='{self.robot}' && "
            f"echo OBJECT_CONFIG_PATH is now set to $OBJECT_CONFIG_PATH && "
            f"echo GENHRL_TASK_NAME is now set to $GENHRL_TASK_NAME && "
        )
        
        # Build full command
        full_command = f"{env_setup}{base_command}"
        
        return full_command
    
    def train_skill(self, skill_name: str, config: TrainingConfig) -> bool:
        """Train a single skill and monitor for completion."""
        print(f"\n{'='*60}")
        print(f"Training skill: {skill_name}")
        print(f"Type: {'Primitive' if self.is_skill_primitive(skill_name) else 'Composite'}")
        print(f"{'='*60}")
        
        # Check if already complete
        if self.has_sufficient_success_states(skill_name, config.min_success_states):
            print(f"Skill {skill_name} already has sufficient success states. Skipping.")
            self.training_status[skill_name] = TrainingStatus.SKIPPED
            return True
        
        # Build command
        command = self.build_training_command(skill_name, config)
        print(f"Command: {command}")
        
        # Set training status
        self.training_status[skill_name] = TrainingStatus.TRAINING
        
        try:
            # Run the training command
            process = subprocess.Popen(
                command,
                shell=True,
                executable='/bin/bash',
                cwd=str(self.isaaclab_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=True  # put child in its own process group for clean signal handling
            )
            # Track active process so signal handlers can terminate it
            self._active_process = process
            
            # Monitor the process and output
            start_time = time.time()
            max_time_seconds = config.max_time_minutes * 60
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break

                # Handle external shutdown requests
                if self._shutdown_requested:
                    print("\nShutdown requested. Terminating IsaacLab...")
                    self._terminate_active_process()
                    break
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time_seconds:
                    print(f"\nTimeout reached ({config.max_time_minutes} minutes). Terminating training.")
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except Exception:
                        process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except Exception:
                            process.kill()
                    break
                
                # Check for completion periodically
                if elapsed_time > 60 and int(elapsed_time) % 30 == 0:  # Check every 30 seconds after first minute
                    if self.has_sufficient_success_states(skill_name, config.min_success_states):
                        print(f"\nSufficient success states found for {skill_name}. Stopping training.")
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except Exception:
                            process.terminate()
                        time.sleep(5)
                        if process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except Exception:
                                process.kill()
                        break
                
                # Read and display output
                try:
                    if process.stdout:
                        line = process.stdout.readline()
                        if line:
                            print(line.rstrip())
                except:
                    pass
                
                time.sleep(0.1)
            
            # Final check for success
            success = self.has_sufficient_success_states(skill_name, config.min_success_states)
            
            if success:
                print(f"\nTraining completed successfully for {skill_name}")
                self.training_status[skill_name] = TrainingStatus.COMPLETED
                return True
            else:
                print(f"\nTraining failed for {skill_name} - insufficient success states")
                self.training_status[skill_name] = TrainingStatus.FAILED
                return False
                
        except KeyboardInterrupt:
            print(f"\nTraining interrupted for {skill_name}")
            self._terminate_active_process()
            self.training_status[skill_name] = TrainingStatus.FAILED
            return False
        except Exception as e:
            print(f"\nError training {skill_name}: {e}")
            self._terminate_active_process()
            self.training_status[skill_name] = TrainingStatus.FAILED
            return False
        finally:
            # Ensure reference is cleared
            self._active_process = None
    
    def build_simple_training_command(self, skill_name: str, steps: int = 10000, record_video: bool = True, video_interval: int = 2000, video_length: int = 200, num_envs: int = 4096, seed: int = 42, use_random_policies: bool = False) -> str:
        """Build a simple Isaac Lab training command for exactly N steps.
        
        Args:
            skill_name: Name of the skill to train
            steps: Number of steps to train (per environment)
            record_video: Whether to record videos
            video_interval: How often to record videos (in steps)
            video_length: Length of each video recording (in steps)
            num_envs: Number of environments to use
            seed: Random seed for training
            use_random_policies: Whether to use random policies when L0 checkpoints are missing
        """
        is_primitive = self.is_skill_primitive(skill_name)
        skill_level = self.get_skill_level(skill_name)
        
        # Format skill name for gym registry with task-specific naming
        gym_task_name_suffix = self._format_gym_task_name(skill_name)
        
        if is_primitive:
            # Use IsaacLab's standard training script for primitive skills
            script_path = "scripts/reinforcement_learning/skrl/train.py"
            task_prefix = "Isaac-RobotFlat"
            # For primitive skills, use max_iterations (divided by rollouts)
            rollouts = 24
            actual_iterations = max(1, steps // rollouts)
            steps_arg = f"--max_iterations {actual_iterations}"
        else:
            # Use GenHRL's hierarchical training scripts for composite skills
            if skill_level > 0:
                script_name = f"train_l{skill_level}.py"
            else:
                script_name = "train.py"
            script_path = str(self.genhrl_scripts_path / script_name)
            task_prefix = "Isaac-RobotComposite"
            # For composite skills, use --steps directly for exact control
            steps_arg = f"--steps {steps}"
        
        base_command = (
            f"./isaaclab.sh -p {script_path} "
            f"--task {task_prefix}{gym_task_name_suffix}-v0 "
            f"--num_envs {num_envs} --seed {seed} --headless "
            f"{steps_arg}"
        )
        
        # Add video recording if enabled
        if record_video:
            base_command += f" --video --video_interval {video_interval} --video_length {video_length}"
        
        if not is_primitive:
            base_command += f" --skill_name {skill_name}"
            if use_random_policies:
                base_command += " --use-random-policies"
        
        # Apply memory and CUDA fixes
        env_setup = (
            "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 && "
            "export CARB_LOG_LEVEL=FATAL && "
            "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048 && "
            "export CUDA_LAUNCH_BLOCKING=0 && "
            f"export OBJECT_CONFIG_PATH='{self.object_config_path}' && "
            f"export GENHRL_TASK_NAME='{self.task_name}' && "
            f"export GENHRL_ROBOT='{self.robot}' && "
        )
        
        # Build full command
        full_command = f"{env_setup}{base_command}"
        
        return full_command

    def train_skill_simple(self, skill_name: str, steps: int = 10000, record_video: bool = True, video_interval: int = 2000, video_length: int = 200, min_success_states: int = 100, skip_if_complete: bool = True, num_envs: int = 4096, seed: int = 42, use_random_policies: bool = False) -> bool:
        """Train a single skill for exactly N steps - simple version.
        
        Args:
            skill_name: Name of the skill to train
            steps: Number of steps to train (per environment)
            record_video: Whether to record videos
            video_interval: How often to record videos (in steps)
            video_length: Length of each video recording (in steps)
            min_success_states: Minimum number of success states required to consider skill complete
            skip_if_complete: Whether to skip training if skill already has sufficient success states
            num_envs: Number of environments to use
            seed: Random seed for training
            use_random_policies: Whether to use random policies when L0 checkpoints are missing
        """
        print(f"\n{'='*60}")
        print(f"Training skill: {skill_name} for {steps} steps")
        print(f"Type: {'Primitive' if self.is_skill_primitive(skill_name) else 'Composite'}")
        if record_video:
            print(f"Video recording: Every {video_interval} steps, {video_length} steps each")
        print(f"{'='*60}")
        
        # Check if already complete (if skip_if_complete is enabled)
        if skip_if_complete and self.has_sufficient_success_states(skill_name, min_success_states):
            print(f"✅ Skill {skill_name} already has sufficient success states ({min_success_states}+). Skipping.")
            return True
        
        # Build command
        command = self.build_simple_training_command(skill_name, steps, record_video, video_interval, video_length, num_envs, seed, use_random_policies)
        print(f"Running: {command}")
        
        try:
            # Simple subprocess run with independent process group for clean Ctrl+C handling
            process = subprocess.Popen(
                command,
                shell=True,
                executable='/bin/bash',
                cwd=str(self.isaaclab_path),
                start_new_session=True
            )
            self._active_process = process
            result = None
            try:
                while True:
                    if process.poll() is not None:
                        result_code = process.returncode
                        break
                    if self._shutdown_requested:
                        print("\nShutdown requested. Terminating IsaacLab...")
                        self._terminate_active_process()
                        # Simulate non-zero exit for failure path
                        result_code = 1
                        break
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nInterrupted by user (Ctrl+C). Stopping current skill training...")
                self._terminate_active_process()
                return False
            # Wrap a simple object to reuse existing status logic below
            class _Result:
                def __init__(self, code):
                    self.returncode = code
            result = _Result(result_code)
            
            if result.returncode == 0:
                print(f"Training completed for {skill_name}")
                
                # Show where the policy was saved
                self._show_policy_location(skill_name)
                
                return True
            else:
                print(f"Training failed for {skill_name} (exit code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"Error training {skill_name}: {e}")
            self._terminate_active_process()
            return False
        finally:
            self._active_process = None

    def train_all_skills(self, config: Optional[TrainingConfig] = None) -> bool:
        """Train all skills in the hierarchy in proper order."""
        # Get training order
        self.training_order = self.get_training_order()
        
        if not self.training_order:
            print("Error: Could not determine training order")
            return False
        
        print(f"\nTraining order for {self.task_name}:")
        for i, skill in enumerate(self.training_order):
            skill_type = "[P]" if self.is_skill_primitive(skill) else "[C]"
            print(f"{i+1}. {skill} {skill_type}")
        
        # Use default config if none provided
        if config is None:
            config = TrainingConfig(skill_name="default")
        
        # Train each skill in sequence
        for skill_name in self.training_order:
            # Update config for current skill
            skill_config = TrainingConfig(
                skill_name=skill_name,
                max_time_minutes=config.max_time_minutes,
                min_success_states=config.min_success_states,
                num_envs=config.num_envs,
                seed=config.seed,
                headless=config.headless
            )
            
            success = self.train_skill(skill_name, skill_config)
            
            if not success:
                print(f"Training failed for {skill_name}. Stopping sequence.")
                return False
            
            # Brief pause between skills
            time.sleep(2)
        
        print("\n" + "="*60)
        print("All skills trained successfully!")
        print("="*60)
        
        # Copy all trained policies to the correct locations
        print("\n🔄 Copying all trained policies to skills directories...")
        copied_count = self.copy_all_policies()
        
        if copied_count > 0:
            print(f"✅ Successfully copied {copied_count} policies!")
        else:
            print("⚠️  Warning: No policies were copied!")
        
        return True
    
    def train_all_skills_simple(self, steps_per_skill: int = 10000, new_run: bool = False, record_video: bool = True, video_interval: int = 2000, video_length: int = 200, min_success_states: int = 100, skip_if_complete: bool = True, copy_policies_immediately: bool = True, num_envs: int = 4096, seed: int = 42, use_random_policies: bool = False, min_primitive_steps: Optional[int] = None, min_composite_steps: Optional[int] = None) -> bool:
        """Train all skills in hierarchy order for exactly N steps each - simple version.
        
        Args:
            steps_per_skill: Number of steps to train each skill (default for all skills)
            new_run: Whether to clean previous training data before starting
            record_video: Whether to record videos during training
            video_interval: How often to record videos (in steps)
            video_length: Length of each video recording (in steps)
            min_success_states: Minimum number of success states required to consider skill complete
            skip_if_complete: Whether to skip training skills that already have sufficient success states
            copy_policies_immediately: Whether to copy policies after each skill (instead of at the end)
            num_envs: Number of environments to use
            seed: Random seed for training
            use_random_policies: Whether to use random policies when L0 checkpoints are missing
            min_primitive_steps: Override steps for primitive skills (overrides steps_per_skill for primitives)
            min_composite_steps: Override steps for composite skills (overrides steps_per_skill for composites)
        """
        # Handle cleanup if requested
        if new_run:
            self.clean_previous_runs()
        
        # Get training order
        self.training_order = self.get_training_order()
        
        if not self.training_order:
            print("Error: Could not determine training order")
            return False
        
        print(f"\nSimple training plan for {self.task_name}:")
        print(f"Each skill will train for exactly {steps_per_skill} steps")
        print(f"🏭 Environments: {num_envs}, Seed: {seed}")
        if record_video:
            print(f"📹 Video recording enabled: Every {video_interval} steps, {video_length} steps each")
        if skip_if_complete:
            print(f"⏭️  Skip mode enabled: Skills with {min_success_states}+ success states will be skipped")
        if new_run:
            print("🆕 New run mode: Previous training data has been cleaned")
        print("\nTraining order:")
        for i, skill in enumerate(self.training_order):
            skill_type = "[Primitive]" if self.is_skill_primitive(skill) else "[Composite]"
            print(f"{i+1}. {skill} {skill_type}")
        
        print(f"\nStarting training sequence...")
        
        # Train each skill in sequence
        for i, skill_name in enumerate(self.training_order):
            print(f"\n[{i+1}/{len(self.training_order)}] Training {skill_name}...")
            
            # Determine steps for this skill
            is_primitive = self.is_skill_primitive(skill_name)
            if is_primitive and min_primitive_steps is not None:
                skill_steps = min_primitive_steps
                print(f"   🏃 Using primitive steps: {skill_steps}")
            elif not is_primitive and min_composite_steps is not None:
                skill_steps = min_composite_steps
                print(f"   🧩 Using composite steps: {skill_steps}")
            else:
                skill_steps = steps_per_skill
                print(f"   📊 Using default steps: {skill_steps}")
            
            
            success = self.train_skill_simple(
                skill_name, 
                skill_steps, 
                record_video, 
                video_interval, 
                video_length,
                min_success_states,
                skip_if_complete,
                num_envs,
                seed,
                use_random_policies
            )
            
            if not success:
                print(f"Training failed for {skill_name}. Stopping sequence.")
                return False
            
            # Copy policy immediately if requested
            if copy_policies_immediately:
                print(f"\n🔄 Copying policy for {skill_name}...")
                policy_path = self.find_and_copy_latest_policy(skill_name)
                if policy_path:
                    print(f"✅ Policy copied to: {policy_path.relative_to(self.skills_path.parent)}")
                else:
                    print(f"⚠️  Warning: Could not copy policy for {skill_name}")

            # Propagate success states to the next skill in the hierarchy
            if i + 1 < len(self.training_order):
                next_skill = self.training_order[i + 1]
                try:
                    copied_count = self._propagate_success_states(skill_name, next_skill)
                    if copied_count > 0:
                        print(f"✅ Transferred {copied_count} success states from {skill_name} -> {next_skill}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not propagate success states from {skill_name} to {next_skill}: {e}")
        
        print(f"\n" + "="*60)
        print(f"All {len(self.training_order)} skills trained for {steps_per_skill} steps each!")
        print("="*60)
        
        # Copy all trained policies to the correct locations (unless already copied immediately)
        if not copy_policies_immediately:
            print("\n🔄 Copying all trained policies to skills directories...")
            copied_count = self.copy_all_policies()
            
            if copied_count > 0:
                print(f"✅ Successfully copied {copied_count} policies!")
            else:
                print("⚠️  Warning: No policies were copied!")
        else:
            print("\n✅ All policies were copied immediately after each skill training!")
        
        return True

    def print_training_commands(self, steps_per_skill: int = 10000, record_video: bool = True, video_interval: int = 2000, video_length: int = 200, num_envs: int = 4096, seed: int = 42, use_random_policies: bool = False, min_primitive_steps: Optional[int] = None, min_composite_steps: Optional[int] = None) -> None:
        """Print all training commands that would be run - for manual execution."""
        # Get training order
        training_order = self.get_training_order()
        
        if not training_order:
            print("Error: Could not determine training order")
            return
        
        print(f"\nTraining commands for {self.task_name}:")
        print(f"Each skill will train for exactly {steps_per_skill} steps")
        print(f"🏭 Environments: {num_envs}, Seed: {seed}")
        if record_video:
            print(f"Video recording: Every {video_interval} steps, {video_length} steps each")
        print("="*80)
        
        for i, skill_name in enumerate(training_order):
            skill_type = "[Primitive]" if self.is_skill_primitive(skill_name) else "[Composite]"
            print(f"\n# {i+1}. {skill_name} {skill_type}")
            command = self.build_simple_training_command(skill_name, steps_per_skill, record_video, video_interval, video_length, num_envs, seed, use_random_policies)
            print(f"cd {self.isaaclab_path}")
            print(command)
            print()

    def _propagate_success_states(self, from_skill: str, to_skill: str, max_files: int = 100, clear_next_start_states: bool = True) -> int:
        """Copy success state files from one skill to the next skill's start-states folder.
        
        Args:
            from_skill: The skill that just completed (source of success states)
            to_skill: The next skill to train (destination for start states)
            max_files: Maximum number of success-state files to copy
            clear_next_start_states: Whether to clear destination start-states before copying
        
        Returns:
            Number of files copied
        """
        source_dir = self.skills_path / from_skill / "success_states"
        dest_dir = self.skills_path / to_skill / "current_task_start_states"
        if not source_dir.exists():
            return 0
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Optionally clear destination to avoid mixing old states
        if clear_next_start_states and dest_dir.exists():
            for existing in dest_dir.glob("*.pt"):
                try:
                    existing.unlink()
                except Exception:
                    pass
        
        # Gather source files and copy with standardized names for the next skill
        source_files = [p for p in source_dir.glob("*.pt")]
        copied = 0
        for idx, src in enumerate(source_files[:max_files]):
            dst = dest_dir / f"start_state_{idx:06d}.pt"
            shutil.copy2(src, dst)
            copied += 1
        return copied
    
    def get_training_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current training status."""
        status = {}
        for skill_name in self.training_order:
            status[skill_name] = {
                "status": self.training_status.get(skill_name, TrainingStatus.PENDING).value,
                "is_primitive": self.is_skill_primitive(skill_name),
                "has_success_states": self.has_sufficient_success_states(skill_name, 50),
                "policy_exists": (self.skills_path / skill_name / "policy" / "agent.pt").exists(),
            }
        return status

    def clean_previous_runs(self) -> None:
        """Clean previous training data for skills belonging to this task only."""
        print(f"🧹 Cleaning previous training data for task: {self.task_name}...")
        
        # Get training order to know which skills to clean
        training_order = self.get_training_order()
        
        for skill_name in training_order:
            skill_dir = self.skills_path / skill_name
            print(f"  Cleaning skill data for {skill_name}...")
            
            # Clean success states
            success_states_dir = skill_dir / "success_states" 
            if success_states_dir.exists():
                for file in success_states_dir.iterdir():
                    if file.is_file():
                        file.unlink()
            
            # Clean start states
            start_states_dir = skill_dir / "current_task_start_states"
            if start_states_dir.exists():
                for file in start_states_dir.iterdir():
                    if file.is_file():
                        file.unlink()
            
            # Clean policy
            policy_dir = skill_dir / "policy"
            if policy_dir.exists():
                import shutil
                shutil.rmtree(policy_dir)
        
        # Clean logs for this task's skills only (not all logs)
        self._clean_task_specific_logs(training_order)
        
        print("✅ Task-specific cleanup completed!")

    def find_and_copy_latest_policy(self, skill_name: str) -> Optional[Path]:
        """
        Finds the most recent policy checkpoint for a specific skill in the logs directory
        and copies it to the skills directory.
        
        Args:
            skill_name: Name of the skill to find policy for
            
        Returns:
            Path to the copied policy file, or None if not found
        """
        # Look for logs in Isaac Lab logs directory (could be relative to current dir or absolute)
        possible_log_dirs = [
            Path("logs/skrl"),  # Relative to current directory
            self.isaaclab_path / "logs/skrl",  # Relative to IsaacLab path
        ]
        
        logs_dir = None
        for log_dir in possible_log_dirs:
            if log_dir.exists():
                logs_dir = log_dir
                break
        
        if logs_dir is None:
            print(f"No logs directory found. Tried: {[str(d) for d in possible_log_dirs]}")
            return None

        # Look for skill-specific log directories (be case-robust)
        # Pattern: logs/skrl/{skill_name_variations}/{timestamp}_ppo_torch/checkpoints/agent_{step}.pt
        patterns = [
            skill_name.lower(),
            skill_name,
            f"*{skill_name.lower()}*",
            f"*{skill_name}*",
        ]
        skill_log_dirs = []
        for pat in patterns:
            # If pattern contains no wildcard, treat as direct path
            if '*' not in pat and '?' not in pat and '[' not in pat:
                cand = logs_dir / pat
                if cand.exists() and cand.is_dir():
                    skill_log_dirs.append(cand)
                continue
            skill_log_dirs.extend([p for p in logs_dir.glob(pat) if p.is_dir()])
        # Deduplicate
        skill_log_dirs = list(dict.fromkeys(skill_log_dirs))
        
        if not skill_log_dirs:
            return None
        
        # Find all checkpoint files for this skill
        checkpoint_files = []
        for skill_log_dir in skill_log_dirs:
            # Look for checkpoints in subdirectories
            checkpoints = list(skill_log_dir.glob("**/checkpoints/agent_*.pt"))
            checkpoint_files.extend(checkpoints)
        
        if not checkpoint_files:
            return None
        
        # Sort by step number (extract from filename like agent_6000.pt)
        def extract_step_number(filepath: Path) -> int:
            match = re.search(r'agent_(\d+)\.pt$', filepath.name)
            return int(match.group(1)) if match else 0
        
        # Get the checkpoint with the highest step number from the most recent run
        checkpoint_files.sort(key=lambda p: (p.stat().st_mtime, extract_step_number(p)), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        # Ensure target directory exists
        target_skill_dir = self.skills_path / skill_name
        target_policy_dir = target_skill_dir / "policy"
        target_policy_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the policy file
        target_policy_path = target_policy_dir / "agent.pt"
        shutil.copy2(latest_checkpoint, target_policy_path)
        
        # Also create a simplified neural network export for faster loading
        try:
            self._export_simplified_policy(latest_checkpoint, target_policy_dir)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create simplified policy export: {e}")
        
        # Save training parameters and config for better policy reconstruction
        try:
            self._save_training_parameters(latest_checkpoint, target_policy_dir, skill_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not save training parameters: {e}")
        
        return target_policy_path

    def _export_simplified_policy(self, checkpoint_path: Path, target_dir: Path) -> None:
        """
        Export a simplified version of the policy for faster loading.
        Extracts just the neural network weights from SKRL checkpoint.
        """
        import torch
        
        try:
            # Load the SKRL checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if not isinstance(checkpoint, dict):
                return
            
            # Extract policy network weights
            policy_state_dict = None
            if 'policy' in checkpoint and isinstance(checkpoint['policy'], dict):
                policy_state_dict = checkpoint['policy']
            elif 'models' in checkpoint and 'policy' in checkpoint['models']:
                policy_state_dict = checkpoint['models']['policy']
            
            if policy_state_dict is None:
                return
            
            # Filter to get only network weights (exclude log_std etc.)
            network_weights = {}
            for key, value in policy_state_dict.items():
                # Include both legacy 'net.' and newer 'net_container.' formats
                if (
                    ('net.' in key or 'net_container.' in key) and
                    (key.endswith('.weight') or key.endswith('.bias'))
                ):
                    network_weights[key] = value
                # Include policy_layer explicitly for final action output
                elif key.startswith('policy_layer.') and (key.endswith('.weight') or key.endswith('.bias')):
                    network_weights[key] = value
            
            if network_weights:
                # Save simplified format
                simplified_path = target_dir / "network_weights.pt"
                torch.save(network_weights, simplified_path)
                print(f"   📦 Created simplified policy export: {simplified_path.name}")
                
        except Exception as e:
            # Don't fail the main process if this fails
            print(f"   ⚠️  Simplified export failed: {e}")

    def _save_training_parameters(self, checkpoint_path: Path, target_dir: Path, skill_name: str) -> None:
        """
        Save training parameters and configuration for policy reconstruction.
        This includes model architecture, training config, and environment info.
        """
        import torch
        
        try:
            # Load the SKRL checkpoint to extract metadata
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            training_info: Dict[str, Any] = {
                "skill_name": skill_name,
                "checkpoint_path": str(checkpoint_path),
                "saved_at": str(checkpoint_path.stat().st_mtime),
            }
            
            # Extract model architecture info from checkpoint
            if isinstance(checkpoint, dict):
                if 'policy' in checkpoint:
                    policy_state = checkpoint['policy']
                    
                    # Infer architecture from state dict
                    architecture_info = self._extract_architecture_info(policy_state)
                    training_info["model_architecture"] = architecture_info
                    
                # Extract any other useful info from checkpoint
                for key in ['step', 'episode', 'learning_rate', 'timestep']:
                    if key in checkpoint:
                        # Convert to basic types to avoid serialization issues
                        value = checkpoint[key]
                        if hasattr(value, 'item'):  # torch tensor
                            training_info[key] = value.item()
                        elif isinstance(value, (int, float, str, bool)):
                            training_info[key] = value
                        else:
                            training_info[key] = str(value)
            
            # Look for additional config files in the log directory
            log_dir = checkpoint_path.parent.parent  # Go up from checkpoints/ to the run directory
            config_info = self._extract_log_configs(log_dir)
            training_info.update(config_info)
            
            # Save training info as JSON
            training_params_path = target_dir / "training_params.json"
            with open(training_params_path, 'w') as f:
                json.dump(training_info, f, indent=2, default=str)
            
            print(f"   📋 Saved training parameters: {training_params_path.name}")
            
        except Exception as e:
            print(f"   ⚠️  Training parameters save failed: {e}")

    def _extract_architecture_info(self, policy_state_dict: dict) -> dict:
        """Extract neural network architecture info from policy state dict."""
        architecture = {
            "layers": [],
            "activation": "elu",  # Default SKRL activation
            "total_params": 0
        }
        
        try:
            # Find linear layers and their sizes - support both net_container.X and net.X formats
            layer_weights = {}
            policy_layer_weight = None
            
            for key, tensor in policy_state_dict.items():
                # Handle policy_layer.weight (final action output layer)
                if key == 'policy_layer.weight':
                    policy_layer_weight = tensor
                    continue
                # Handle net_container.X.weight format (newer SKRL)
                elif 'net_container.' in key and '.weight' in key:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[0] == 'net_container':
                        try:
                            layer_num = int(parts[1])
                            layer_weights[layer_num] = tensor
                        except ValueError:
                            continue
                # Handle legacy net.X.weight format
                elif 'net.' in key and '.weight' in key:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[0] == 'net':
                        try:
                            layer_num = int(parts[1])
                            layer_weights[layer_num] = tensor
                        except ValueError:
                            continue
            
            # Add policy_layer as the final layer if it exists
            final_layer_num = None
            if policy_layer_weight is not None:
                if layer_weights:
                    max_layer_num = max(layer_weights.keys())
                    final_layer_num = max_layer_num + 2  # Skip one number to maintain pattern
                else:
                    final_layer_num = 0
                layer_weights[final_layer_num] = policy_layer_weight
            
            # Sort layers and extract dimensions
            for layer_num in sorted(layer_weights.keys()):
                weight = layer_weights[layer_num]
                input_size = weight.shape[1]
                output_size = weight.shape[0]
                param_count = weight.numel()
                
                # Count bias parameters if they exist - check multiple formats
                bias_key_container = f"net_container.{layer_num}.bias"
                bias_key_legacy = f"net.{layer_num}.bias"
                bias_key_policy = "policy_layer.bias"
                
                if bias_key_container in policy_state_dict:
                    param_count += policy_state_dict[bias_key_container].numel()
                elif bias_key_legacy in policy_state_dict:
                    param_count += policy_state_dict[bias_key_legacy].numel()
                elif layer_num == final_layer_num and bias_key_policy in policy_state_dict:
                    # This is the policy layer
                    param_count += policy_state_dict[bias_key_policy].numel()
                
                architecture["layers"].append({
                    "layer_num": layer_num,
                    "input_size": input_size,
                    "output_size": output_size,
                    "param_count": param_count
                })
                architecture["total_params"] += param_count
            
            # Determine input and output dimensions
            if architecture["layers"]:
                architecture["input_dim"] = architecture["layers"][0]["input_size"]
                architecture["output_dim"] = architecture["layers"][-1]["output_size"]
                
        except Exception as e:
            print(f"   ⚠️  Architecture extraction failed: {e}")
            
        return architecture

    def _extract_log_configs(self, log_dir: Path) -> dict:
        """Extract configuration files from the log directory."""
        config_info = {}
        
        try:
            # Look for params directory with config files
            params_dir = log_dir / "params"
            if params_dir.exists():
                # Load agent config
                agent_yaml = params_dir / "agent.yaml"
                if agent_yaml.exists():
                    with open(agent_yaml, 'r') as f:
                        agent_config = yaml.safe_load(f)
                        config_info["agent_config"] = agent_config
                
                # Load environment config  
                env_yaml = params_dir / "env.yaml"
                if env_yaml.exists():
                    with open(env_yaml, 'r') as f:
                        env_config = yaml.safe_load(f)
                        # Extract key environment info (not the full massive config)
                        env_summary = {
                            "num_envs": env_config.get("scene", {}).get("num_envs"),
                            "episode_length_s": env_config.get("episode_length_s"),
                            "decimation": env_config.get("decimation"),
                            "sim_dt": env_config.get("sim", {}).get("dt"),
                        }
                        config_info["env_config"] = env_summary
                        
        except Exception as e:
            print(f"   ⚠️  Log config extraction failed: {e}")
            
        return config_info

    def _show_policy_location(self, skill_name: str) -> None:
        """Show where the policy was saved and where it will be copied to."""
        # Look for the policy in logs
        possible_log_dirs = [
            Path("logs/skrl"),  # Relative to current directory
            self.isaaclab_path / "logs/skrl",  # Relative to IsaacLab path
        ]
        
        logs_dir = None
        for log_dir in possible_log_dirs:
            if log_dir.exists():
                logs_dir = log_dir
                break
        
        if logs_dir is None:
            print(f"   📁 Could not find logs directory")
            return

        # Look for skill-specific log directories
        skill_log_pattern = f"*{skill_name.lower()}*"
        skill_log_dirs = list(logs_dir.glob(skill_log_pattern))
        
        if not skill_log_dirs:
            print(f"   📁 No training logs found for {skill_name}")
            return
        
        # Find the most recent checkpoint
        checkpoint_files = []
        for skill_log_dir in skill_log_dirs:
            checkpoints = list(skill_log_dir.glob("**/checkpoints/agent_*.pt"))
            checkpoint_files.extend(checkpoints)
        
        if checkpoint_files:
            # Sort by modification time and step number
            def extract_step_number(filepath: Path) -> int:
                match = re.search(r'agent_(\d+)\.pt$', filepath.name)
                return int(match.group(1)) if match else 0
            
            checkpoint_files.sort(key=lambda p: (p.stat().st_mtime, extract_step_number(p)), reverse=True)
            latest_checkpoint = checkpoint_files[0]
            step_num = extract_step_number(latest_checkpoint)
            
            print(f"   📁 Policy saved to: {latest_checkpoint}")
            print(f"   📈 Latest checkpoint: step {step_num}")
            print(f"   📍 Will be copied to: {self.task_name}/skills/{skill_name}/policy/agent.pt (at end of training)")
        else:
            print(f"   📁 No checkpoint files found for {skill_name}")

    def copy_all_policies(self) -> int:
        """
        Find and copy the latest policies for all skills that have been trained.
        
        Returns:
            Number of policies successfully copied
        """
        training_order = self.get_training_order()
        copied_count = 0
        
        for i, skill_name in enumerate(training_order):
            print(f"📁 [{i+1}/{len(training_order)}] Processing {skill_name}...")
            policy_path = self.find_and_copy_latest_policy(skill_name)
            if policy_path:
                print(f"   ✅ Policy copied to: {policy_path.relative_to(self.skills_path.parent)}")
                copied_count += 1
            else:
                print(f"   ❌ No policy found for {skill_name}")
        
        print(f"\n📊 Summary: {copied_count}/{len(training_order)} policies copied successfully")
        return copied_count

    def _clean_task_specific_logs(self, skill_names: List[str]) -> None:
        """Clean logs for specific skills only, preserving logs from other tasks/skills."""
        import shutil
        
        # Look for logs in both possible locations
        possible_log_dirs = [
            Path("logs/skrl"),  # Relative to current directory
            self.isaaclab_path / "logs/skrl",  # Relative to IsaacLab path
        ]
        
        logs_dir = None
        for log_dir in possible_log_dirs:
            if log_dir.exists():
                logs_dir = log_dir
                break
        
        if logs_dir is None:
            print("  📁 No logs directory found to clean")
            return
        
        print(f"  📁 Cleaning task-specific logs in: {logs_dir}")
        
        # Clean logs for each skill belonging to this task
        for skill_name in skill_names:
            print(f"    Cleaning logs for skill: {skill_name}")
            
            # Try different log directory patterns for this skill
            log_patterns = [
                skill_name.lower(),  # Hierarchical skills use lowercase  
                skill_name,          # Some skills might use original case
                f"*{skill_name.lower()}*",  # Fuzzy match with lowercase
                f"*{skill_name}*"    # Fuzzy match with original case
            ]
            
            removed_any = False
            for pattern in log_patterns:
                # Direct directory match
                if '*' not in pattern:
                    log_dir = logs_dir / pattern
                    if log_dir.exists() and log_dir.is_dir():
                        print(f"      🗂️  Removing log directory: {log_dir.relative_to(logs_dir.parent)}")
                        shutil.rmtree(log_dir)
                        removed_any = True
                
                # Glob pattern match
                else:
                    matching_dirs = list(logs_dir.glob(pattern))
                    for log_dir in matching_dirs:
                        if log_dir.is_dir():
                            print(f"      🗂️  Removing log directory: {log_dir.relative_to(logs_dir.parent)}")
                            shutil.rmtree(log_dir)
                            removed_any = True
            
            if not removed_any:
                print(f"      📂 No existing logs found for {skill_name}")
        
        print(f"  ✅ Task-specific log cleanup completed for {len(skill_names)} skills")


def main():
    """
    Simple example of how to use the training orchestrator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple GenHRL Training Orchestrator")
    parser.add_argument("--isaaclab_path", required=True, help="Path to IsaacLab installation")
    parser.add_argument("--task_name", required=True, help="Name of the task to train")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--steps", type=int, default=10000, help="Steps per skill (default: 10000)")
    parser.add_argument("--print_only", action="store_true", help="Only print commands, don't run them")
    parser.add_argument("--copy_policies", action="store_true", help="Only copy policies from logs to skills directory")
    parser.add_argument("--copy_immediately", action="store_true", default=True, help="Copy policies after each skill (instead of at the end)")
    
    args = parser.parse_args()
    
    print(f"Setting up orchestrator for task: {args.task_name}")
    print(f"Robot: {args.robot}")
    print(f"IsaacLab path: {args.isaaclab_path}")
    print(f"Steps per skill: {args.steps}")
    
    try:
        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(
            isaaclab_path=args.isaaclab_path,
            task_name=args.task_name,
            robot=args.robot
        )
        
        if args.copy_policies:
            # Just copy policies from logs to skills directory
            print("\n" + "="*80)
            print("COPYING POLICIES FROM LOGS TO SKILLS DIRECTORY")
            print("="*80)
            copied_count = orchestrator.copy_all_policies()
            if copied_count > 0:
                print(f"✅ Successfully copied {copied_count} policies!")
            else:
                print("❌ No policies were copied!")
                return 1
        elif args.print_only:
            # Just print the commands
            print("\n" + "="*80)
            print("COMMANDS TO RUN MANUALLY:")
            print("="*80)
            orchestrator.print_training_commands(args.steps)
        else:
            # Run the training
            success = orchestrator.train_all_skills_simple(args.steps, copy_policies_immediately=args.copy_immediately)
            if success:
                print("✅ All skills trained successfully!")
            else:
                print("❌ Training failed!")
                return 1
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())