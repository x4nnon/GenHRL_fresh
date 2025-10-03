# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of a Level 2 (L2) hierarchical RL agent with skrl.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import os
import time
import torch
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

# Configure paths dynamically based on environment and project structure
L2_TASK_NAME = os.environ.get("GENHRL_TASK_NAME", "Create_Steps")
ROBOT_NAME = os.environ.get("GENHRL_ROBOT", "G1")

# Get Isaac Lab path - can be set via environment variable or auto-detected
ISAACLAB_ROOT = os.environ.get("ISAACLAB_PATH")
if not ISAACLAB_ROOT:
    # Try to auto-detect Isaac Lab relative to project root
    project_root = Path(__file__).parent.parent.parent
    isaaclab_candidate = project_root / "IsaacLab"
    if isaaclab_candidate.exists():
        ISAACLAB_ROOT = str(isaaclab_candidate)
    else:
        raise RuntimeError("ISAACLAB_PATH environment variable not set and IsaacLab not found in expected location. "
                         "Please set ISAACLAB_PATH to point to your Isaac Lab installation.")

# Build paths dynamically
ROBOT_FOLDER = f"{ROBOT_NAME}_generated"
ISAACLAB_TASKS_BASE = os.path.join(ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "manager_based", ROBOT_FOLDER)

TASK_PATH = os.path.join(ISAACLAB_TASKS_BASE, "tasks", L2_TASK_NAME)
SKILLS_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", L2_TASK_NAME, "skills")
SKILL_LIBRARY_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", L2_TASK_NAME, "skill_library.json")

# Set the object config path for the L2 task
if os.path.exists(TASK_PATH):
    os.environ['OBJECT_CONFIG_PATH'] = os.path.join(TASK_PATH, 'object_config.json')
    print(f"[INFO] Using Isaac Lab at: {ISAACLAB_ROOT}")
    print(f"[INFO] Task: {L2_TASK_NAME}, Robot: {ROBOT_NAME}")
    print(f"[INFO] Skills path: {SKILLS_PATH}")
    print(f"[INFO] Skill library: {SKILL_LIBRARY_PATH}")
else:
    print(f"Warning: Task path does not exist: {TASK_PATH}. OBJECT_CONFIG_PATH not set.")


# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an L2 RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=True, help="Record video during playback.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
# Default task now refers to the high-level L2 task registration name
parser.add_argument("--task", type=str, default=f"Isaac-G1CompositeCreate_Steps-v0", help="Name of the L2 task environment.")
parser.add_argument("--checkpoint", type=str, default="/home/tomcannon/Documents/IsaacLab/logs/skrl/create_steps/2025-06-19_12-38-53_L2_Create_Steps_ppo_torch/checkpoints/agent_10000.pt", help="Path to the L2 model checkpoint to load.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO"], # L2 wrapper assumes PPO for loaded agents
    help="The RL algorithm used for the skrl agent (must match checkpoint).",
)
# Skill name now refers to the L2 skill being played (determines sub-skills)
parser.add_argument("--skill_name", type=str, default=L2_TASK_NAME, help="Name of the L2 skill to play (from skill library)")
# Add arguments for hierarchical frequencies (must match training)
parser.add_argument("--steps_per_l1", type=int, default=300, help="Steps the selected L1 policy runs before L2 selects again.")
parser.add_argument("--steps_per_l0", type=int, default=100, help="Steps the selected L0 policy runs before L1 selects again.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless false for play unless specified
args_cli.headless = True # Override for debugging/consistency
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Add sys.path modification HERE ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

"""Rest everything follows."""

import gymnasium as gym

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# Import the L2 wrapper
from genhrl.hppo.l2_wrapper import HierarchicalVecActionWrapperL2

# config shortcuts
algorithm = args_cli.algorithm.lower()

def is_skill_primitive(skill_name, skill_library):
    """Checks if a skill is primitive based on the library."""
    skill_info = skill_library.get("skills", {}).get(skill_name)
    if not skill_info:
        print(f"Warning: Skill '{skill_name}' not found in library skills dictionary.")
        return False
    return skill_info.get("is_primitive", False)

# Function to recursively find all primitive (L0) sub-skills for a given skill
def find_all_l0_sub_skills(skill_name, skill_library, all_l0_skills=None):
    if all_l0_skills is None:
        all_l0_skills = set()

    skill_info = skill_library.get("skills", {}).get(skill_name)
    if not skill_info:
        print(f"Warning: Skill '{skill_name}' not found during L0 search.")
        return all_l0_skills

    if skill_info.get("is_primitive", False):
        all_l0_skills.add(skill_name)
    else:
        sub_skills = skill_info.get("sub_skills", [])
        for sub_skill in sub_skills:
            find_all_l0_sub_skills(sub_skill, skill_library, all_l0_skills)

    return all_l0_skills


def main():
    """Play L2 agent with skrl."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Load base agent config (might need adjustments based on checkpoint specifics)
    try:
        # L2 uses standard PPO config structure
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    except ValueError as e:
         print(f"Error loading base agent config for task {args_cli.task}: {e}")
         # Fallback or alternative loading might be needed if checkpoint structure differs significantly
         # For now, exit if base config cannot be loaded.
         sys.exit(1)

    # Checkpoint path
    resume_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(resume_path):
        print(f"[ERROR] Checkpoint not found at: {resume_path}")
        sys.exit(1)
    log_dir = os.path.dirname(os.path.dirname(resume_path)) # Base directory for video saving
    log_dir = os.path.join(log_dir)
    # create isaac environment (base L2 task environment)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt # type: ignore

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_l2"), # Save in separate subfolder
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during L2 playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- Apply SkrlVecEnvWrapper FIRST ---
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # --- Apply HierarchicalVecActionWrapperL2 AFTER SkrlVecEnvWrapper ---
    skill_library = json.load(open(SKILL_LIBRARY_PATH))

    # 1. Find L1 sub-skills for the target L2 skill
    l1_sub_skill_names = []
    if args_cli.skill_name and args_cli.skill_name in skill_library.get("skills", {}):
        l2_skill_info = skill_library["skills"][args_cli.skill_name]
        if l2_skill_info.get("is_primitive", False):
             print(f"Error: Skill '{args_cli.skill_name}' is marked as primitive. Cannot use as L2 skill.")
             sys.exit(1)
        l1_sub_skill_names = l2_skill_info.get("sub_skills", [])
    else:
        print(f"Error: L2 skill name '{args_cli.skill_name}' not provided or not found in skill library.")
        sys.exit(1)

    if not l1_sub_skill_names:
        print(f"Error: L2 skill '{args_cli.skill_name}' has no defined sub_skills (L1 skills) in library.")
        sys.exit(1)

    # 2. Collect L1 checkpoint paths and registered names
    l1_policy_checkpoint_paths = []
    registered_l1_skill_names = []
    l1_skill_l0_dependencies = defaultdict(set) # Store L0 dependencies as sets

    for l1_name in l1_sub_skill_names:
        l1_info = skill_library.get("skills", {}).get(l1_name)
        if not l1_info:
            print(f"Warning: L1 skill '{l1_name}' (sub-skill of '{args_cli.skill_name}') not found in library. Skipping.")
            continue

        checkpoint_file = os.path.join(SKILLS_PATH, l1_name, "policy", "agent.pt")
        if not os.path.exists(checkpoint_file):
            print(f"Warning: L1 Checkpoint file not found for sub-skill '{l1_name}' at '{checkpoint_file}'. Skipping.")
            continue

        l1_policy_checkpoint_paths.append(checkpoint_file)
        # Determine registered name (assuming L1 skills are composite)
        if l1_info.get("is_primitive", False):
             registered_name = f"Isaac-RobotFlat{l1_name.title()}-v0"
        else:
             registered_name = f"Isaac-G1Composite{l1_name.title()}-v0"
        registered_l1_skill_names.append(registered_name)

        # Find L0 dependencies for this L1 skill
        l0_deps = find_all_l0_sub_skills(l1_name, skill_library)
        l1_skill_l0_dependencies[l1_name] = l0_deps
        # print(f"L1 skill '{l1_name}' uses L0 skills: {l0_deps}")


    if not l1_policy_checkpoint_paths:
        print(f"Error: No valid L1 checkpoints found for L2 skill '{args_cli.skill_name}'. Exiting.")
        sys.exit(1)

    # 3. Collect unique L0 checkpoint paths and registered names across all needed L1 skills
    unique_l0_skill_names = set()
    for l0_set in l1_skill_l0_dependencies.values():
        unique_l0_skill_names.update(l0_set)

    if not unique_l0_skill_names:
         print("Error: L2 wrapper requires L0 policies. None found as dependencies.")
         sys.exit(1)

    l0_policy_checkpoint_paths = []
    registered_l0_skill_names = []
    for l0_name in sorted(list(unique_l0_skill_names)): # Sort for consistent order
        l0_info = skill_library.get("skills", {}).get(l0_name)
        if not l0_info or not l0_info.get("is_primitive", False):
             print(f"Warning: L0 skill '{l0_name}' invalid or not primitive. Skipping.")
             continue

        checkpoint_file = os.path.join(SKILLS_PATH, l0_name, "policy", "agent.pt")
        if not os.path.exists(checkpoint_file):
            print(f"Warning: L0 Checkpoint file not found for primitive skill '{l0_name}' at '{checkpoint_file}'. Skipping.")
            continue

        l0_policy_checkpoint_paths.append(checkpoint_file)
        registered_name = f"Isaac-G1Flat{l0_name.title()}-v0"
        registered_l0_skill_names.append(registered_name)

    if not l0_policy_checkpoint_paths:
        print(f"Error: No valid L0 checkpoints found for L2 skill '{args_cli.skill_name}'. Exiting.")
        sys.exit(1)

    # 4. Instantiate the L2 wrapper
    print(f"Applying HierarchicalVecActionWrapperL2 with {len(l1_policy_checkpoint_paths)} L1 policies and {len(l0_policy_checkpoint_paths)} unique L0 policies.")
    env = HierarchicalVecActionWrapperL2( # type: ignore
        env, # Pass the SkrlVecEnvWrapper instance
        l1_policy_checkpoint_paths=l1_policy_checkpoint_paths,
        l1_policy_registered_names=registered_l1_skill_names,
        skill_library_names=l1_sub_skill_names,
        l0_policy_checkpoint_paths=l0_policy_checkpoint_paths,
        l0_policy_registered_names=registered_l0_skill_names,
        skill_library_path=SKILL_LIBRARY_PATH,
        skills_path=SKILLS_PATH,
        steps_per_l1_policy=args_cli.steps_per_l1,
        l2_action_frequency=args_cli.steps_per_l1,
        steps_per_l0_policy=args_cli.steps_per_l0,
        l1_action_frequency=args_cli.steps_per_l0,
    )
    # Now 'env' is the HierarchicalVecActionWrapperL2 instance

    # configure and instantiate the skrl runner for evaluation
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    # The Runner receives the HierarchicalVecActionWrapperL2 instance
    # Add type: ignore because HierarchicalVecActionWrapperL2 doesn't inherit from skrl base wrappers,
    # but provides the necessary interface (duck typing).
    runner = Runner(env, experiment_cfg) # type: ignore



    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # L2 agent stepping (selects an L1 skill/policy)
            # The wrapper handles running the selected L1/L0 policies internally

            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0]) # PPO typical output structure
            # Env stepping (executes one step of the currently active L0 policy)

            fixed_actions = torch.ones_like(actions)

            if timestep < 500:
                actions = fixed_actions*0
            elif timestep < 1000:
                actions = fixed_actions*1
            else:
                actions = fixed_actions*2

            obs, _, _, _, info = env.step(actions)

        # Check if L2 made a decision in this step (only relevant if needing L2-specific info)
        # l2_decision_made = info.get("l2_decision_made", False) # Example of accessing info

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                print("Done")
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 