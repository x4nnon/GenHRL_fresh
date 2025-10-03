"""
Task Manager for iGen - Code Generation for IsaacLab Integration

This module coordinates the generation of hierarchical RL tasks for IsaacLab,
creating the necessary configuration files and skill definitions in the 
appropriate IsaacLab directory structure.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import re
import yaml

from .code_generator import HierarchicalCodeGenerator
from .skill_library import SkillLibrary
from .robot_configs import get_robot_config, get_robot_folder_name
from genhrl.generation.object_utils import get_object_name_mapping


@dataclass
class TaskConfig:
    """Configuration for a hierarchical RL task."""
    name: str
    description: str
    isaaclab_path: str  # Path to IsaacLab installation
    robot: str  # Robot name (e.g., 'G1', 'H1', etc.)
    objects_config: Dict[str, Any]
    skills_hierarchy: Union[Dict[str, Any], List[Dict[str, Any]]]
    objects_mapping: Dict[str, Any] = field(default_factory=dict)
    
    def get_task_path(self) -> Path:
        """Get the path where task files should be generated."""
        robot_folder = get_robot_folder_name(self.robot)
        return Path(self.isaaclab_path) / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "tasks" / self.name
    
    def get_skills_base_path(self) -> Path:
        """Get the base path for this task's skills in IsaacLab."""
        robot_folder = get_robot_folder_name(self.robot)
        return Path(self.isaaclab_path) / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "skills" / self.name

    def get_skill_library_path(self) -> Path:
        """Get the path to this task's skill library JSON file."""
        return self.get_skills_base_path() / "skill_library.json"

    def get_task_skills_path(self) -> Path:
        """Get the path where individual skill directories are stored for this task."""
        return self.get_skills_base_path() / "skills"


class TaskManager:
    """
    Manages the creation and generation of hierarchical RL tasks for IsaacLab.
    
    This class coordinates the entire process of converting natural language task
    descriptions into complete IsaacLab skill hierarchies with proper configuration files.
    """
    
    def __init__(self, 
                 isaaclab_path: str, 
                 api_key: str, 
                 robot: str = "G1",
                 provider: str = "google",
                 model: str = "gemini-2.5-flash",
                 backup_model: str = "gemini-2.5-flash"):
        """
        Initialize the task manager.
        
        Args:
            isaaclab_path: Path to IsaacLab installation
            api_key: API key for LLM services
            robot: Robot name (default: G1)
            provider: LLM provider (default: google)
            model: LLM model (default: gemini-2.5-flash)
            backup_model: Backup LLM model (default: gemini-2.5-flash)
        """
        self.isaaclab_path = Path(isaaclab_path)
        self.robot = robot
        self.robot_config = get_robot_config(robot)
        
        # Create robot-specific paths
        robot_folder = get_robot_folder_name(robot)
        self.skills_base_path = self.isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "skills"
        self.tasks_base_path = self.isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "tasks"
        
        # Initialize code generator (don't pass skills_path yet - it's task-specific)
        self.code_generator = HierarchicalCodeGenerator(
            api_key,
            provider=provider,
            model=model,
            backup_model=backup_model,
        )
        # Store backup model for reference
        self.backup_model = backup_model
        
        # Create directories if they don't exist
        self.skills_base_path.mkdir(parents=True, exist_ok=True)
        self.tasks_base_path.mkdir(parents=True, exist_ok=True)
        
        # Copy template files to robot-specific directories if they don't exist
        self._ensure_robot_directories_setup()
    
    def create_task_from_description(self, task_name: str, task_description: str, max_hierarchy_levels: int = 3, robot: Optional[str] = None) -> TaskConfig:
        """
        Create a complete hierarchical task from a natural language description.
        
        Args:
            task_name: Name of the task
            task_description: Natural language description of the task
            max_hierarchy_levels: Maximum number of hierarchy levels to generate (default: 3)
            robot: Robot name (if None, uses the robot from TaskManager init)
            
        Returns:
            TaskConfig object containing all generated configurations
        """
        # Use robot from parameter or TaskManager default
        task_robot = robot or self.robot
        print(f"🎯 Creating task: {task_name}")
        print(f"🤖 Robot: {task_robot}")
        print(f"📝 Description: {task_description}")
        print(f"🔢 Max hierarchy levels: {max_hierarchy_levels}")
        
        # Set robot environment variable for skill config template
        os.environ['ROBOT_NAME'] = task_robot
        
        # 1. Generate object configuration
        print("\n🏗️ Generating scene objects...")
        objects_config = self.code_generator.generate_objects_config(task_description)
        
        # 2a. Generate mapping from ObjectN -> human description so prompts can
        #     reference semantic names without needing the full object config.
        try:
            objects_cfg_dict = json.loads(objects_config)
        except Exception:
            objects_cfg_dict = {}

        object_name_mapping = get_object_name_mapping(objects_cfg_dict)

        # 2. Decompose task into hierarchical skills
        print("\n🧩 Decomposing task into skills hierarchy...")
        # Use skills directory path for object_config.json to match training orchestrator expectations
        robot_folder = get_robot_folder_name(task_robot)
        skills_base_path = Path(self.isaaclab_path) / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "skills" / task_name
        object_config_path = str(skills_base_path / "object_config.json")
        skills_hierarchy = self.code_generator.decompose_task(
            task_name,
            task_description,
            object_config_path,
            objects_config,  # Full object configuration
            max_hierarchy_levels=max_hierarchy_levels,
            object_name_mapping=json.dumps(object_name_mapping, indent=2),  # Object mapping for prompts
        )
        
        # 3. Create task configuration
        task_config = TaskConfig(
            name=task_name,
            description=task_description,
            isaaclab_path=str(self.isaaclab_path),
            robot=task_robot,
            objects_config=json.loads(objects_config),
            skills_hierarchy=skills_hierarchy,
            objects_mapping=object_name_mapping
        )
        
        # 4. Generate all files in IsaacLab structure
        self._generate_task_files(task_config)
        
        print(f"\n✅ Task '{task_name}' created successfully!")
        print(f"📁 Task files: {task_config.get_task_path()}")
        print(f"🎯 Skills: {self.skills_base_path / 'skills'}")
        
        return task_config
    
    def modify_task_rewards(self, task_name: str, skill_name: str, modifications: Dict[str, Any]) -> None:
        """
        Modify the reward structure for a specific skill in a task.
        
        Args:
            task_name: Name of the task
            skill_name: Name of the skill to modify
            modifications: Dictionary of modifications to apply
        """
        skill_path = self.skills_base_path / "skills" / skill_name
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill not found: {skill_name}")
        
        # Implementation for modifying rewards
        # This would regenerate the TaskRewardsCfg.py file
        pass
    
    def visualize_task(self, task_name: str) -> Dict[str, Any]:
        """
        Create visualization data for a task (for potential UI integration).
        
        Args:
            task_name: Name of the task to visualize
            
        Returns:
            Dictionary containing visualization data
        """
        task_path = self.tasks_base_path / task_name
        if not task_path.exists():
            raise FileNotFoundError(f"Task not found: {task_name}")
        
        # Load task configuration - object_config is now in skills directory
        skills_path = self.skills_base_path / task_name
        with open(skills_path / "object_config.json", 'r') as f:
            objects_config = json.load(f)
        
        with open(task_path / "skills_hierarchy.json", 'r') as f:
            skills_hierarchy = json.load(f)
        
        return {
            "objects": objects_config,
            "hierarchy": skills_hierarchy,
            "task_path": str(task_path),
            "skills_available": self._get_available_skills()
        }
    
    def start_training(self, task_name: str) -> None:
        """
        Prepare for training by ensuring all necessary files are in place.
        
        This method verifies that the task is ready for training with
        the existing skill_chain_training scripts.
        """
        task_path = self.tasks_base_path / task_name
        if not task_path.exists():
            raise FileNotFoundError(f"Task not found: {task_name}")
        
        # Verify all required files exist
        skills_path = self.skills_base_path / task_name
        required_files = [
            skills_path / "object_config.json",  # object_config is now in skills directory
            task_path / "skills_hierarchy.json",
            task_path / "description.txt"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file missing: {file_path}")
        
        print(f"✅ Task '{task_name}' is ready for training!")
        print(f"💡 Run training with: skill_chain_training/Example.py")
        print(f"📝 Remember to set TASK_NAME = '{task_name}' in the training script")
    
    def _generate_task_files(self, task_config: TaskConfig) -> None:
        """Generate all task and skill files in IsaacLab structure."""
        # Create task directory
        task_path = task_config.get_task_path()
        task_path.mkdir(parents=True, exist_ok=True)
        
        # Write task description
        with open(task_path / "description.txt", 'w') as f:
            f.write(task_config.description)
        
        # Write skills hierarchy
        with open(task_path / "skills_hierarchy.json", 'w') as f:
            json.dump(task_config.skills_hierarchy, f, indent=2)
        
        # Create task-specific skill directories
        task_skills_base = task_config.get_skills_base_path()
        task_skills_base.mkdir(parents=True, exist_ok=True)
        
        # Write object configuration to skills directory (where training orchestrator expects it)
        with open(task_skills_base / "object_config.json", 'w') as f:
            json.dump(task_config.objects_config, f, indent=2)
        
        # Also write object configuration to tasks directory (for training script compatibility)
        task_objects_path = task_path / "object_config.json"
        with open(task_objects_path, 'w') as f:
            json.dump(task_config.objects_config, f, indent=2)
        
        print(f"📄 Object config written to both:")
        print(f"   - Skills: {task_skills_base / 'object_config.json'}")
        print(f"   - Tasks: {task_objects_path}")
        
        task_skills_dir = task_config.get_task_skills_path()
        task_skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize skill library with task-specific JSON file
        skill_library_path = task_config.get_skill_library_path()
        skill_library = SkillLibrary(str(skill_library_path))
        skill_library.add_hierarchy(
            task_config.name, 
            task_config.skills_hierarchy, 
            task_config.description
        )
        
        # Generate skill files
        self._generate_skill_files(task_config, skill_library)
        
        # Add gym registrations
        self._add_gym_registrations(task_config, skill_library)
    
    def _generate_skill_files(self, task_config: TaskConfig, skill_library: SkillLibrary) -> None:
        """Generate individual skill files in IsaacLab skills directory."""
        skills_dir = task_config.get_task_skills_path()
        
        # Get all skills in execution order
        full_skill_sequence = skill_library.get_full_skill_sequence(task_config.name)
        
        # Create skill descriptions mapping
        all_skill_descriptions = []
        for skill_name in full_skill_sequence:
            if skill_name in skill_library.skills["skills"]:
                description = skill_library.skills["skills"][skill_name].get("description", "N/A")
                all_skill_descriptions.append(description)
            else:
                all_skill_descriptions.append("N/A")
        
        # Generate files for each skill
        for skill_name in full_skill_sequence:
            skill_info = skill_library.skills["skills"].get(skill_name)
            if not skill_info:
                continue
            
            skill_path = skills_dir / skill_name
            if skill_path.exists():
                print(f"⚠️ Skill directory already exists: {skill_name}")
                continue
            
            # Create skill directory and files
            self._create_skill_directory(
                skill_name=skill_name,
                skill_info=skill_info,
                skill_path=skill_path,
                task_config=task_config,
                all_skills=full_skill_sequence,
                all_skills_descriptions=all_skill_descriptions,
                use_start_template=(skill_name in self._collect_first_skills_by_level(task_config.skills_hierarchy))
            )
    
    def _create_skill_directory(self, skill_name: str, skill_info: Dict, skill_path: Path,
                               task_config: TaskConfig, all_skills: List[str], 
                               all_skills_descriptions: List[str],
                               use_start_template: bool = False) -> None:
        """Create a complete skill directory with all necessary files."""

        print("DEBUG \n\n", use_start_template, "\n\n")

        # Create directory structure
        skill_path.mkdir(parents=True, exist_ok=True)
        (skill_path / "success_states").mkdir(exist_ok=True)
        
        # Copy agents configuration from Isaac Lab templates
        agents_dst = skill_path / "agents"
        if not agents_dst.exists():
            self._copy_agents_templates(agents_dst, skill_name, skill_info, task_config)
        
        # Generate reward functions
        print(f"🎯 Generating rewards for: {skill_name}")
        mapping_json = json.dumps(task_config.objects_mapping, indent=2)

        rewards_code = self.code_generator.generate_task_rewards(
            task_config.description,
            mapping_json,
            skill_name,
            skill_info["description"],
            all_skills,
            all_skills_descriptions,
            mapping_json,
        )
        
        with open(skill_path / "TaskRewardsCfg.py", 'w') as f:
            f.write(self._strip_markdown(rewards_code))
        
        # Generate success criteria
        print(f"✅ Generating success criteria for: {skill_name}")
        success_code = self.code_generator.generate_success_criteria(
            task_config.description,
            mapping_json,
            skill_name,
            skill_info["description"],
            rewards_code,
            all_skills,
            all_skills_descriptions,
            mapping_json,
        )
        
        # Create base success file if needed
        base_success_path = skill_path / "base_success.py"
        if not base_success_path.exists():
            # Import the base success code
            try:
                from .prompts.success_pre_code import get_success_pre_code
                with open(base_success_path, 'w') as f:
                    f.write(get_success_pre_code())
            except ImportError:
                print("⚠️ Could not import success pre-code")
        
        # Write success termination configuration
        success_content = f"""
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

{self._strip_markdown(success_code)}
"""
        
        with open(skill_path / "SuccessTerminationCfg.py", 'w') as f:
            f.write(success_content)
        
        # # Generate collapse termination
        # print(f"⚠️ Generating collapse termination for: {skill_name}")
        # collapse_code = self.code_generator.generate_collapse_termination(
        #     task_config.description,
        #     json.dumps(task_config.objects_config),
        #     skill_name,
        #     skill_info["description"],
        #     rewards_code,
        #     all_skills
        # )
        
        # with open(skill_path / "CollapseTerminationCfg.py", 'w') as f:
        #     f.write(self._strip_markdown(collapse_code))
        
        # Copy skill config template (it's a complete working config, no placeholders needed)
        # Resolve template paths relative to this source file so that the code works
        # regardless of the current working directory. We support two naming variants
        # for the *first*-skill template ("_first" is preferred, "_start" kept for
        # backward compatibility).
        base_generation_dir = Path(__file__).parent  # …/genhrl/generation

        if use_start_template:
            # Preferred file name.
            candidate_first = base_generation_dir / "skill_config_template_first.py"
            # Legacy/older name kept for compatibility.
            candidate_start = base_generation_dir / "skill_config_template_start.py"

            if candidate_first.exists():
                template_path = candidate_first
            elif candidate_start.exists():
                template_path = candidate_start
            else:
                # Fallback to the generic template if neither specialised file exists.
                template_path = base_generation_dir / "skill_config_template.py"
        else:
            template_path = base_generation_dir / "skill_config_template.py"

        print("debug – selected template for", skill_name, "is", template_path)

        config_path = skill_path / f"{skill_name.lower()}_cfg.py"
        if template_path.exists() and not config_path.exists():
            # Simply copy the template - it's already a complete working config
            shutil.copy2(template_path, config_path)
            print(f"Created skill config from template: {template_path}")
        elif config_path.exists():
            print(f"Config already exists: {config_path}")
        else:
            print(f"⚠️ Warning: Template not found at {template_path}")
        
        # Ensure all necessary __init__.py files exist for Python imports
        self._ensure_init_files(skill_path)
    
    def _ensure_init_files(self, skill_path: Path) -> None:
        """Ensure all necessary __init__.py files exist for Python imports."""
        # Get the skills base directory  
        skills_base = skill_path.parent.parent
        
        # List of directories that need __init__.py files
        init_dirs = [
            skills_base,  # skills/
            skills_base / skill_path.name,  # skills/skill_name/
            skill_path.parent,  # skills/skill_name/skills/
            skill_path,  # skills/skill_name/skills/skill_name/
        ]
        
        for init_dir in init_dirs:
            if init_dir.exists():
                init_file = init_dir / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"Created: {init_file}")
    
    def _copy_agents_templates(self, agents_dst: Path, skill_name: str, skill_info: Dict, task_config: TaskConfig) -> None:
        """Copy agent templates from GenHRL agents directory and configure them for the skill."""
        # Create agents directory
        agents_dst.mkdir(parents=True, exist_ok=True)
        
        # Find GenHRL agents templates directory
        genhrl_agents_path = Path(__file__).parent.parent / "agents"
        
        if not genhrl_agents_path.exists():
            print(f"⚠️ Warning: GenHRL agents templates not found at {genhrl_agents_path}")
            # Fallback to creating basic configs
            self._create_basic_agent_configs(agents_dst, skill_name, skill_info)
            return
        
        # Copy all template files
        for template_file in genhrl_agents_path.glob("*.yaml"):
            if template_file.is_file():
                dst_file = agents_dst / template_file.name
                shutil.copy2(template_file, dst_file)
                print(f"📋 Copied GenHRL agent template: {template_file.name}")
        
        # Create __init__.py file to make agents a proper Python module
        agents_init = agents_dst / "__init__.py"
        if not agents_init.exists():
            agents_init.touch()
            print(f"📋 Created __init__.py in agents directory")
        
        # Configure the copied agents
        self._configure_agents(agents_dst, skill_name, skill_info)
    
    def _create_basic_agent_configs(self, agents_dst: Path, skill_name: str, skill_info: Dict) -> None:
        """Create basic agent configurations when templates are not available."""
        skill_name = skill_name.lower()
        
        # Create a basic skrl_ppo_cfg file
        basic_config = f"""seed: 42

models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: STATES
        layers: [32, 32]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: STATES
        layers: [32, 32]
        activations: elu
    output: ONE

memory:
  class: RandomMemory
  memory_size: -1

agent:
  class: PPO
  rollouts: 32
  learning_epochs: 8
  mini_batches: 8
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 5.0e-04
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008
  state_preprocessor: RunningStandardScaler
  state_preprocessor_kwargs: null
  value_preprocessor: RunningStandardScaler
  value_preprocessor_kwargs: null
  random_timesteps: 0
  learning_starts: 0
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: True
  entropy_loss_scale: 0.0
  value_loss_scale: 2.0
  kl_threshold: 0.0
  rewards_shaper_scale: 0.1
  time_limit_bootstrap: False
  experiment:
    directory: "{skill_name}"
    experiment_name: ""
    write_interval: auto
    checkpoint_interval: auto

trainer:
  class: SequentialTrainer
  timesteps: 4800
  environment_info: log
"""
        
        with open(agents_dst / "skrl_ppo_cfg.yaml", 'w') as f:
            f.write(basic_config)
        
        print(f"📋 Created basic agent config for {skill_name}")

    def _configure_agents(self, agents_path: Path, skill_name: str, skill_info: Dict) -> None:
        """Configure agent settings based on skill type (primitive vs composite)."""
        is_primitive = skill_info.get("is_primitive", False)
        skill_name_lower = skill_name.lower()
        
        # Update all YAML files in the agents directory
        for config_file in agents_path.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Replace common experiment name patterns
                replacements = [
                    ('cartpole_direct', skill_name_lower),
                    ('cart_double_pendulum_direct', skill_name_lower),
                    ('humanoid_amp_run', skill_name_lower),
                    ('example_experiment', skill_name_lower),
                    ('"cartpole_direct"', f'"{skill_name_lower}"'),
                    ('"cart_double_pendulum_direct"', f'"{skill_name_lower}"'),
                    ('"humanoid_amp_run"', f'"{skill_name_lower}"'),
                    ('"example_experiment"', f'"{skill_name_lower}"'),
                ]
                
                for old_name, new_name in replacements:
                    content = content.replace(old_name, new_name)
                
                # For composite skills, adjust network layers based on sub-skills
                if not is_primitive:
                    num_sub_skills = len(skill_info.get("sub_skills", []))
                    if num_sub_skills > 0:
                        # Update network layers for hierarchical policy
                        layers_pattern = r'(\s*layers:\s*\[)([^\]]+)(\])'
                        match = re.search(layers_pattern, content)
                        if match:
                            prefix = match.group(1)
                            layers_str = match.group(2)
                            suffix = match.group(3)
                            try:
                                # Parse existing layers
                                layers = [int(x.strip()) for x in layers_str.split(',') if x.strip()]
                                if layers:
                                    # Set final layer to number of sub-skills
                                    layers[-1] = num_sub_skills
                                    new_layers_str = ', '.join(map(str, layers))
                                    content = re.sub(layers_pattern, f"{prefix}{new_layers_str}{suffix}", content)
                                    print(f"Updated layers to {layers} for composite skill {skill_name}")
                            except ValueError as e:
                                print(f"Warning: Could not parse layers for {skill_name}: {e}")
                    
                    # Adjust hyperparameters for composite skills
                    hyperparameter_updates = [
                        (r'(\s*entropy_loss_scale:\s*)([\d.]+)', r'\g<1>0.005'),
                        (r'(\s*learning_rate:\s*)([\d.eE+-]+)', r'\g<1>5.0e-04'),
                        (r'(\s*kl_threshold:\s*)([\d.]+)', r'\g<1>0.02'),
                        (r'(\s*discount_factor:\s*)([\d.]+)', r'\g<1>0.9'),
                        (r'(\s*entropy_coef:\s*)([\d.]+)', r'\g<1>0.005'),
                    ]
                    
                    for pattern, replacement in hyperparameter_updates:
                        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                
                # Write updated content back
                with open(config_file, 'w') as f:
                    f.write(content)
                    
                print(f"✅ Updated agent config: {config_file.name} for skill {skill_name}")
                
            except Exception as e:
                print(f"⚠️ Error updating {config_file.name}: {e}")
        
        # Ensure we have the primary config file needed for registration
        primary_config = agents_path / "skrl_ppo_cfg.yaml"
        if not primary_config.exists():
            # Look for any PPO config to rename as primary
            ppo_configs = list(agents_path.glob("*ppo*"))
            if ppo_configs:
                shutil.copy2(ppo_configs[0], primary_config)
                print(f"📋 Created primary config from {ppo_configs[0].name}")
    
    def _clean_stale_registrations(self, init_path: Path) -> None:
        """Remove registrations for skills/tasks that no longer exist."""
        if not init_path.exists():
            return
            
        try:
            with open(init_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return
        
        # Get all existing tasks and their skills
        valid_skills = set()
        for task_dir in self.tasks_base_path.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                skills_dir = self.skills_base_path / task_name / "skills"
                if skills_dir.exists():
                    for skill_dir in skills_dir.iterdir():
                        if skill_dir.is_dir():
                            valid_skills.add(skill_dir.name)
        
        # Clean up the registrations
        lines = content.split('\n')
        cleaned_lines = []
        skip_until_empty = False
        
        for line in lines:
            # If we're in a registration block for a non-existent skill, skip it
            if skip_until_empty:
                if line.strip() == '' or line.startswith('#'):
                    skip_until_empty = False
                    cleaned_lines.append(line)
                continue
            
            # Check if this line starts a registration for a non-existent skill
            if 'gym.register(' in line:
                # Look for skill names in the upcoming lines
                remaining_lines = '\n'.join(lines[lines.index(line):])
                registration_block = remaining_lines.split('}')[0] if '}' in remaining_lines else remaining_lines[:200]
                
                # Check if any valid skill path is referenced
                has_valid_skill = False
                for skill_name in valid_skills:
                    if f".skills.{skill_name}." in registration_block:
                        has_valid_skill = True
                        break
                
                if not has_valid_skill:
                    # Skip this registration and continue until we hit an empty line
                    skip_until_empty = True
                    continue
            
            cleaned_lines.append(line)
        
        # Write back the cleaned content
        cleaned_content = '\n'.join(cleaned_lines)
        # Remove any multiple consecutive empty lines
        import re
        cleaned_content = re.sub(r'\n\n\n+', '\n\n', cleaned_content)
        
        with open(init_path, 'w') as f:
            f.write(cleaned_content)
        
        if len(cleaned_lines) < len(lines):
            removed_count = len(lines) - len(cleaned_lines)
            print(f"🧹 Cleaned up {removed_count} lines of stale registrations")

    def _add_gym_registrations(self, task_config: TaskConfig, skill_library: SkillLibrary) -> None:
        """Add gym registrations for all skills in the hierarchy."""
        robot_folder = get_robot_folder_name(task_config.robot)
        init_path = self.isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "__init__.py"
        
        # Ensure the __init__.py exists and has necessary imports
        if not init_path.exists():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            with open(init_path, 'w') as f:
                f.write("import gymnasium as gym\n\n")
        
        # Clean up any stale registrations before adding new ones
        self._clean_stale_registrations(init_path)
        
        # Read existing registrations to avoid duplicates
        existing_content = ""
        try:
            with open(init_path, 'r') as f:
                existing_content = f.read()
        except FileNotFoundError:
            pass
        
        # Get the full sequence of ALL skills for the hierarchy
        all_skills_in_hierarchy = skill_library.get_full_skill_sequence(task_config.name)
        
        registrations_to_add = ""
        # Add registrations for each skill if not already present
        for skill_name in all_skills_in_hierarchy:
            # Check if skill info exists in the main library dict before proceeding
            if skill_name not in skill_library.skills["skills"]:
                print(f"Warning: Skill '{skill_name}' info not found in library during registration. Skipping.")
                continue
            
            skill_info = skill_library.skills["skills"][skill_name]
            is_primitive = skill_info["is_primitive"]
            
            # Use the name directly from the skill dictionary for consistency
            skill_name_lower = skill_name.lower()
            skill_name_title = skill_name.title()  # Use title case for registration ID consistency
            
            registration_block = ""
            
            if is_primitive:
                # --- Primitive Skill Registration ---
                skill_id_rough = f"Isaac-RobotRough{skill_name_title}-v0"
                skill_id_flat = f"Isaac-RobotFlat{skill_name_title}-v0"
                
                # Check if registration already exists
                if f'id="{skill_id_rough}"' in existing_content or f'id="{skill_id_flat}"' in existing_content:
                    print(f"Registration for primitive skill '{skill_name}' already exists. Skipping.")
                    continue
                
                print(f"Adding gym registration for primitive skill: {skill_name}")
                env_cfg_rough_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.{skill_name_lower}_cfg:RobotRoughEnvCfg"
                skrl_rough_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.agents:skrl_flat_ppo_cfg.yaml"
                env_cfg_flat_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.{skill_name_lower}_cfg:RobotFlatEnvCfg"
                skrl_flat_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.agents:skrl_flat_ppo_cfg.yaml"
                
                registration_block = f"""
# Primitive Skill: {skill_name}
gym.register(
    id="{skill_id_rough}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{env_cfg_rough_entry}",
        "skrl_cfg_entry_point": f"{skrl_rough_entry}",
    }},
)

gym.register(
    id="{skill_id_flat}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{env_cfg_flat_entry}",
        "skrl_cfg_entry_point": f"{skrl_flat_entry}",
    }},
)
"""
            else:
                # --- Composite Skill Registration ---
                skill_id_composite = f"Isaac-RobotComposite{skill_name_title}-v0"
                
                # Check if registration already exists
                if f'id="{skill_id_composite}"' in existing_content:
                    print(f"Registration for composite skill '{skill_name}' already exists. Skipping.")
                    continue
                
                print(f"Adding gym registration for composite skill: {skill_name}")
                env_cfg_composite_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.{skill_name_lower}_cfg:RobotFlatEnvCfg"
                skrl_composite_entry = f"{{__name__}}.skills.{task_config.name}.skills.{skill_name}.agents:skrl_ppo_cfg.yaml"
                
                registration_block = f"""
# Composite Skill: {skill_name}
gym.register(
    id="{skill_id_composite}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{env_cfg_composite_entry}",
        "skrl_cfg_entry_point": f"{skrl_composite_entry}",
    }},
)
"""
            
            registrations_to_add += registration_block
        
        if registrations_to_add:
            with open(init_path, 'a') as f:
                f.write(registrations_to_add)
            print(f"✅ Added gym registrations to {init_path}")
    
    def _get_available_skills(self) -> List[str]:
        """Get list of all available skills in the skills directory."""
        skills_dir = self.skills_base_path / "skills"
        if not skills_dir.exists():
            return []
        
        return [d.name for d in skills_dir.iterdir() if d.is_dir()]
    
    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting from generated code."""
        lines = text.split('\n')
        if lines and '```' in lines[0]:
            lines = lines[1:]
        if lines and '```' in lines[-1]:
            lines = lines[:-1]
        elif len(lines) > 1 and '```' in lines[-2] and not lines[-1].strip():
            lines = lines[:-2] + [lines[-1]]
        
        # Remove any other markdown code blocks
        cleaned_lines = []
        for line in lines:
            if line.strip() == '```' or line.startswith('```'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _ensure_robot_directories_setup(self) -> None:
        """Ensure robot-specific directories have all necessary template files."""
        # Copy skill config template if it doesn't exist
        template_src = Path(__file__).parent / "skill_config_template.py"
        template_dst = self.skills_base_path / "skill_config_template.py"
        
        if template_src.exists() and not template_dst.exists():
            shutil.copy2(template_src, template_dst)
            print(f"📋 Copied skill config template to {self.robot} directory")
        
        # Create agents directory if it doesn't exist
        agents_dir = self.skills_base_path / "agents"
        if not agents_dir.exists():
            agents_dir.mkdir(exist_ok=True)
            
            # Create default agent configurations based on robot type
            self._create_default_agent_configs(agents_dir)
    
    def _create_default_agent_configs(self, agents_dir: Path) -> None:
        """Create default agent configuration files for the robot."""
        # This would create robot-specific agent YAML files
        # For now, we'll create a basic structure
        default_config = {
            "defaults": [f"_self_", f"task/{self.robot.lower()}_rough"]
        }
        
        # Create a simple YAML-like structure (could use actual YAML library)
        config_content = f"""defaults:
  - _self_
  - task: {self.robot.lower()}_rough

seed: 42
"""
        
        config_file = agents_dir / "rsl_rl_ppo_cfg.yaml"
        if not config_file.exists():
            with open(config_file, 'w') as f:
                f.write(config_content)
            print(f"⚙️ Created default agent config for {self.robot}")

    def _collect_first_skills_by_level(self, hierarchy) -> set[str]:
        """Return a set containing the first skill found at each depth level."""
        first_by_level: dict[int, str] = {}

        def recurse(node, level: int):
            # Node can be dict (skill) or list (top-level children array)
            if isinstance(node, dict):
                # Register the first skill at this level if not yet seen
                if level not in first_by_level:
                    name = node.get("name")
                    if isinstance(name, str):
                        first_by_level[level] = name
                # Recurse into children if present
                for child in node.get("children", []):
                    recurse(child, level + 1)
            elif isinstance(node, list):
                for item in node:
                    recurse(item, level)

        recurse(hierarchy, 1)
        print("DEBUG \n\n ", set(first_by_level.values()), "\n\n")
        return set(first_by_level.values())