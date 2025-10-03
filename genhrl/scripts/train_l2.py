# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl at the L2 hierarchy level.

This script trains an L2 skill to select between L1 sub-skills.
The new clean implementation directly coordinates L1→L0 policies without wrapper nesting.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import os
from pathlib import Path

# Fix CUDA linear algebra backend issues (must be before Isaac Lab imports)
import torch
from isaaclab.app import AppLauncher

torch.backends.cuda.preferred_linalg_library("cusolver")  # Try cusolver first

# Configure paths dynamically based on environment and project structure
TASK_NAME = os.environ.get("GENHRL_TASK_NAME", "doorway_and_goal_seed42")
ROBOT_NAME = os.environ.get("GENHRL_ROBOT", "G1")

# Get Isaac Lab path - can be set via environment variable or auto-detected
ISAACLAB_ROOT = os.environ.get("ISAACLAB_PATH")
if not ISAACLAB_ROOT:
    project_root = Path(__file__).parent.parent.parent
    isaaclab_candidate = project_root / "IsaacLab"
    if isaaclab_candidate.exists():
        ISAACLAB_ROOT = str(isaaclab_candidate)
    else:
        raise RuntimeError(
            "ISAACLAB_PATH environment variable not set and IsaacLab not found in expected location. "
            "Please set ISAACLAB_PATH to point to your Isaac Lab installation."
        )

# Build paths dynamically
ROBOT_FOLDER = f"{ROBOT_NAME}_generated"
ISAACLAB_TASKS_BASE = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "manager_based", ROBOT_FOLDER
)

TASK_PATH = os.path.join(ISAACLAB_TASKS_BASE, "tasks", TASK_NAME)
SKILLS_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", TASK_NAME, "skills")
SKILL_LIBRARY_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", TASK_NAME, "skill_library.json")

# Set the object config path
os.environ["OBJECT_CONFIG_PATH"] = os.path.join(TASK_PATH, "object_config.json")

print(f"[INFO] Using Isaac Lab at: {ISAACLAB_ROOT}")
print(f"[INFO] Task: {TASK_NAME}, Robot: {ROBOT_NAME}")
print(f"[INFO] Skills path: {SKILLS_PATH}")
print(f"[INFO] Skill library: {SKILL_LIBRARY_PATH}")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-G1CompositeDoorwayAndGoalSeed42-v0",
    help="Name of the task.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument("--steps_per_l1", type=int, default=400, help="Number of steps per L1 policy.")
parser.add_argument(
    "--early_terminate_on_success",
    action="store_true",
    default=True,
    help="End a skill early if its success condition is met.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training."
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--skill_name",
    type=str,
    default="doorway_and_goal_seed42",
    help="Name of the skill to train, same as the skill library reference",
)
parser.add_argument("--steps", type=int, default=None, help="Total number of environment steps to train.")
parser.add_argument(
    "--use-random-policies",
    action="store_true",
    default=False,
    help="Use random policies when L1 policy checkpoints are missing (for testing builds)",
)
parser.add_argument(
    "--test-in-order-only",
    action="store_true",
    default=False,
    help="Test mode: L2 wrapper selects L1 skills in order to verify policy loading",
)

# L0 and L1 adaptation arguments
parser.add_argument("--adapt_l0", action="store_true", default=False, help="Enable L0 policy adaptation during L2 training.")
parser.add_argument("--adapt_l1", action="store_true", default=False, help="Enable L1 policy adaptation during L2 training.")
parser.add_argument("--l0_adapt_lr", type=float, default=1e-9, help="Learning rate for L0 adaptation (very small).")
parser.add_argument("--l1_adapt_lr", type=float, default=1e-8, help="Learning rate for L1 adaptation (very small).")
parser.add_argument("--l0_adapt_std", type=float, default=0.2, help="Assumed action std for L0 adaptation.")
parser.add_argument("--l1_adapt_std", type=float, default=0.2, help="Assumed action std for L1 adaptation.")
parser.add_argument("--l0_adapt_every_n_updates", type=int, default=1, help="Apply L0 adaptation every N PPO updates.")
parser.add_argument("--l1_adapt_every_n_updates", type=int, default=1, help="Apply L1 adaptation every N PPO updates.")
parser.add_argument(
    "--l0_adapt_signal",
    type=str,
    default="success",
    choices=["reward", "success", "both"],
    help="Signal used for L0 adaptation: reward-based PG, success imitation, or both."
)
parser.add_argument(
    "--l1_adapt_signal",
    type=str,
    default="success",
    choices=["reward", "success", "both"],
    help="Signal used for L1 adaptation: reward-based PG, success imitation, or both."
)

# Convenience negation flags for booleans with store_true defaults
parser.add_argument("--no_early_terminate_on_success", action="store_true", default=False, help="Override to disable early termination on success")
parser.add_argument("--no_adapt_l0", action="store_true", default=False, help="Override to disable L0 adaptation")
parser.add_argument("--no_adapt_l1", action="store_true", default=False, help="Override to disable L1 adaptation")

# Ablation flags
parser.add_argument(
    "--ablate_hierarchical_gae",
    action="store_true",
    default=False,
    help="Disable hierarchical GAE in HPPO",
)
parser.add_argument(
    "--ablate_pass_decision_mask",
    action="store_true",
    default=False,
    help="Do not pass decision_mask (behaves like flat PPO)",
)
parser.add_argument(
    "--ablate_advantage_norm",
    action="store_true",
    default=False,
    help="Disable advantage normalization at decision points",
)

# WandB overrides for ablation sweeps
parser.add_argument(
    "--wandb_project_override",
    type=str,
    default=None,
    help="Override WandB project name (e.g., 'HPPO ablations')",
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default=None,
    help="WandB group for this run (e.g., ablation id)",
)
parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name override")
parser.add_argument(
    "--ablation_name",
    type=str,
    default=None,
    help="Optional ablation name (used as WandB group if group not provided)",
)



# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

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
import os
import random
from datetime import datetime

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

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# Add Isaac Lab source directory to Python path for isaaclab_tasks imports
import sys
from pathlib import Path

workspace_root = Path(__file__).parent.parent.parent
isaaclab_source_path = workspace_root / "IsaacLab" / "source"
if str(isaaclab_source_path) not in sys.path:
    sys.path.insert(0, str(isaaclab_source_path))

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

# Add import for the custom memory and agent
from genhrl.hppo.decision_point_memory import DecisionPointMemory
from genhrl.hppo.hppo import HPPO
from genhrl.hppo.register_components import register_hppo_components

def is_skill_primitive(skill_name, skill_library):
    """Checks if a skill is primitive based on the library."""
    skill_info = skill_library.get("skills", {}).get(skill_name)
    if not skill_info:
        print(f"Warning: Skill '{skill_name}' not found in library skills dictionary.")
        return False
    return skill_info.get("is_primitive", False)

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Train with skrl agent at L2 level."""
    
    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.steps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.steps
    elif args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Set up logging directories
    log_root_path = os.path.join("logs", "skrl", f"{args_cli.skill_name}_L2")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    
    # WandB configuration
    if "wandb_kwargs" in agent_cfg["agent"]["experiment"]:
        if args_cli.wandb_project_override:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"] = args_cli.wandb_project_override
        else:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"] = f"genhrl_l2_{args_cli.skill_name}"
        if args_cli.wandb_group or args_cli.ablation_name:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["group"] = args_cli.wandb_group or args_cli.ablation_name
        agent_cfg["agent"]["experiment"]["wandb_kwargs"]["name"] = args_cli.wandb_name or log_dir
    
    log_dir = os.path.join(log_root_path, log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Load L1 skills from skill library
    print(f"[INFO] Loading skill library: {SKILL_LIBRARY_PATH}")
    
    if not os.path.exists(SKILL_LIBRARY_PATH):
        print(f"[ERROR] Skill library not found: {SKILL_LIBRARY_PATH}")
        sys.exit(1)
    
    skill_library = json.load(open(SKILL_LIBRARY_PATH))
    if args_cli.skill_name not in skill_library.get("skills", {}):
        print(f"[ERROR] Skill '{args_cli.skill_name}' not found in skill library")
        print(f"Available skills: {list(skill_library.get('skills', {}).keys())}")
        sys.exit(1)
    
    # Get L1 sub-skills for this L2 skill
    l1_sub_skills = skill_library["skills"][args_cli.skill_name].get("sub_skills", [])
    if not l1_sub_skills:
        print(f"[ERROR] No L1 sub-skills found for L2 skill '{args_cli.skill_name}'")
        sys.exit(1)
    
    print(f"[INFO] Found {len(l1_sub_skills)} L1 sub-skills: {l1_sub_skills}")

    # Build L1 policy paths and names
    l1_policy_paths = []
    l1_skill_names = []
    l1_folder_names = []
    
    for l1_skill in l1_sub_skills:
        l1_checkpoint = os.path.join(SKILLS_PATH, l1_skill, "policy", "agent.pt")
        l1_folder_names.append(l1_skill)
        
        # Generate registered name (follow train_l1.py pattern)
        if l1_skill in skill_library["skills"]:
            is_primitive = skill_library["skills"][l1_skill].get("is_primitive", False)
            if is_primitive:
                registered_name = f"Isaac-RobotFlat{l1_skill.title()}-v0"
            else:
                registered_name = f"Isaac-RobotComposite{l1_skill.title()}-v0"
        else:
            registered_name = f"Isaac-RobotComposite{l1_skill.title()}-v0"  # Default
        
        l1_skill_names.append(registered_name)
        
        if os.path.exists(l1_checkpoint):
            l1_policy_paths.append(l1_checkpoint)
            print(f"✅ Found L1 policy: {l1_skill} -> {l1_checkpoint}")
        else:
            print(f"❌ L1 policy not found: {l1_checkpoint}")
            if not args_cli.use_random_policies:
                print(f"[ERROR] L1 policy missing and --use-random-policies not enabled")
                sys.exit(1)
            l1_policy_paths.append("")  # Empty path for random policy

    if not l1_policy_paths and not args_cli.use_random_policies:
        print("[ERROR] No L1 policies found and --use-random-policies not enabled")
        sys.exit(1)

    # Apply L2 wrapper
    print(f"[INFO] Applying L2 wrapper with {len(l1_policy_paths)} L1 policies")
    
    from genhrl.hppo.l2_wrapper import L2Wrapper
    
    env = L2Wrapper(
        env,
        sub_policy_checkpoint_paths=l1_policy_paths,
        sub_policy_registered_names=l1_skill_names,
        sub_skill_folder_names=l1_folder_names,
        skills_root_path=SKILLS_PATH,
        steps_per_l1_policy=args_cli.steps_per_l1,
        l2_action_frequency=args_cli.steps_per_l1,
        base_env=base_env,
        early_terminate_on_success=args_cli.early_terminate_on_success and not args_cli.no_early_terminate_on_success,
        debug_mode=False,
        use_random_policies=args_cli.use_random_policies,
        disable_success_state_saving=True,  # Don't save success states during L2 training
        test_in_order_only=args_cli.test_in_order_only,  # Test mode flag
        # L0 and L1 adaptation parameters
        adapt_l0=args_cli.adapt_l0 and not args_cli.no_adapt_l0,
        adapt_l1=args_cli.adapt_l1 and not args_cli.no_adapt_l1,
        l0_adapt_lr=args_cli.l0_adapt_lr,
        l1_adapt_lr=args_cli.l1_adapt_lr,
        l0_adapt_std=args_cli.l0_adapt_std,
        l1_adapt_std=args_cli.l1_adapt_std,
        l0_adapt_every_n_updates=args_cli.l0_adapt_every_n_updates,
        l1_adapt_every_n_updates=args_cli.l1_adapt_every_n_updates,
        l0_adapt_signal=args_cli.l0_adapt_signal,
        l1_adapt_signal=args_cli.l1_adapt_signal,
    )

    # Register HPPO components
    register_hppo_components()

    print("Using DecisionPointMemory to filter L2 non-decision steps")
    print("Using HPPO agent for hierarchical learning")
    
    # Log adaptation configuration
    if args_cli.adapt_l0 and not args_cli.no_adapt_l0:
        print(f"[ADAPT] L0 adaptation enabled: lr={args_cli.l0_adapt_lr}, every_n_updates={args_cli.l0_adapt_every_n_updates}, signal={args_cli.l0_adapt_signal}")
    if args_cli.adapt_l1 and not args_cli.no_adapt_l1:
        print(f"[ADAPT] L1 adaptation enabled: lr={args_cli.l1_adapt_lr}, every_n_updates={args_cli.l1_adapt_every_n_updates}, signal={args_cli.l1_adapt_signal}")

    # Update agent config for L2 training
    agent_cfg["memory"] = {
        "class": "DecisionPointMemory",
        "memory_size": agent_cfg["agent"]["rollouts"],
        "hierarchy_level": 2,  # L2 training
        "debug_mode": False,
        "discount_factor": agent_cfg["agent"]["discount_factor"],
        "skill_reward_discount": 1.0,
        "use_average_reward": False,
        "max_skill_duration": args_cli.steps_per_l1,
    }
    agent_cfg["agent"]["class"] = "HPPO"
    agent_cfg["agent"]["debug_mode"] = False
    
    # Apply ablation flags
    if args_cli.ablate_hierarchical_gae:
        agent_cfg["agent"]["use_hierarchical_gae"] = False
    if args_cli.ablate_pass_decision_mask:
        agent_cfg["agent"]["pass_decision_mask"] = False
    if args_cli.ablate_advantage_norm:
        agent_cfg["agent"]["normalize_advantages_at_decisions"] = False

    print("Updated agent configuration:")
    print(f"  - Agent class: {agent_cfg['agent']['class']}")
    print(f"  - Memory class: {agent_cfg['memory']['class']}")
    print(f"  - Memory hierarchy level: {agent_cfg['memory']['hierarchy_level']}")
    print(f"  - Debug mode enabled: {agent_cfg['agent']['debug_mode']}")
    print(f"  - L2 action frequency: {args_cli.steps_per_l1}")

    from skrl.utils.runner.torch import Runner

    runner = Runner(env, agent_cfg)

    # Provide the agent with a reference to the managed environment for L0/L1 adaptation on updates
    try:
        if hasattr(runner, 'agent') and hasattr(runner.agent, 'set_managed_env'):
            runner.agent.set_managed_env(env)
            if (args_cli.adapt_l0 and not args_cli.no_adapt_l0) or (args_cli.adapt_l1 and not args_cli.no_adapt_l1):
                print(f"[ADAPT] Environment reference set for adaptation")
    except Exception as e:
        print(f"[WARNING] Failed to set managed environment: {e}")

    # Load checkpoint if provided
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # Run training
    runner.run()

    # Save L2 agent
    try:
        policy_dir = os.path.join(SKILLS_PATH, args_cli.skill_name, "policy")
        os.makedirs(policy_dir, exist_ok=True)
        save_path = os.path.join(policy_dir, "agent.pt")
        runner.agent.save(save_path)
        print(f"[INFO] Saved L2 agent checkpoint to: {save_path}")

        # Export simplified network weights
        try:
            import torch
            ckpt = torch.load(save_path, map_location="cpu")
            policy_state = None
            if isinstance(ckpt, dict):
                if "policy" in ckpt and isinstance(ckpt["policy"], dict):
                    policy_state = ckpt["policy"]
                elif "models" in ckpt and "policy" in ckpt["models"]:
                    policy_state = ckpt["models"]["policy"]
                elif "state_dict" in ckpt:
                    policy_state = ckpt["state_dict"]
            if policy_state is not None:
                network_weights = {}
                for key, value in policy_state.items():
                    if (("net_container." in key or "net." in key) and not ("value_layer" in key or "log_std" in key)):
                        network_weights[key] = value
                    elif key.startswith("policy_layer.") and (key.endswith(".weight") or key.endswith(".bias")):
                        network_weights[key] = value
                if network_weights:
                    torch.save(network_weights, os.path.join(policy_dir, "network_weights.pt"))
                    print("[INFO] Exported simplified network weights")
            
            # Save training parameters
            try:
                training_info = {
                    "skill_name": args_cli.skill_name,
                    "checkpoint_path": save_path,
                }
                params_dir = os.path.join(log_dir, "params")
                agent_yaml = os.path.join(params_dir, "agent.yaml")
                if os.path.exists(agent_yaml):
                    import yaml
                    with open(agent_yaml, "r") as f:
                        agent_cfg_logged = yaml.safe_load(f)
                    training_info["agent_config"] = agent_cfg_logged
                with open(os.path.join(policy_dir, "training_params.json"), "w") as f:
                    json.dump(training_info, f, indent=2)
                    print("[INFO] Saved training_params.json")
            except Exception as e:
                print(f"[WARN] Failed to save training_params.json: {e}")
        except Exception as e:
            print(f"[WARN] Simplified export failed: {e}")
    except Exception as e:
        print(f"[WARN] Failed to save L2 agent checkpoint: {e}")

    env.close()

    # Clean up wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("[INFO] Wandb run finished and cleaned up")
    except ImportError:
        pass

if __name__ == "__main__":
    main() # type: ignore[call-arg] # Hydra decorator handles arguments injection
    simulation_app.close()