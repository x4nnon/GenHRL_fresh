# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import os
from pathlib import Path

# Fix CUDA linear algebra backend issues (must be before Isaac Lab imports)
import torch
torch.backends.cuda.preferred_linalg_library("cusolver")  # Try cusolver first
# Alternative backends if cusolver fails: "cublas", "magma"

# Add the project root directory to sys.path to allow finding the genhrl module
# Moved this section down after AppLauncher initialization
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher
# Keep import of HierarchicalVecActionWrapper deferred
# from genhrl.hppo.l1_wrapper import HierarchicalVecActionWrapper 

# Configure paths dynamically based on environment and project structure
TASK_NAME = os.environ.get("GENHRL_TASK_NAME", "obstacle_course")
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

TASK_PATH = os.path.join(ISAACLAB_TASKS_BASE, "tasks", TASK_NAME)
SKILLS_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", TASK_NAME, "skills")
SKILL_LIBRARY_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", TASK_NAME, "skill_library.json")

# Set the object config path
os.environ['OBJECT_CONFIG_PATH'] = os.path.join(TASK_PATH, 'object_config.json')

print(f"[INFO] Using Isaac Lab at: {ISAACLAB_ROOT}")
print(f"[INFO] Task: {TASK_NAME}, Robot: {ROBOT_NAME}")
print(f"[INFO] Skills path: {SKILLS_PATH}")
print(f"[INFO] Skill library: {SKILL_LIBRARY_PATH}")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-G1CompositeJumpoverlowwall-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--steps_per_l0", type=int, default=100, help="Number of steps per L0 policy.")
parser.add_argument("--early_terminate_on_success", action="store_true", default=True, help="End an L0 skill early if its success condition is met.")
parser.add_argument("--adapt_l0", action="store_true", default=False, help="Enable slow online adaptation of L0 skills during L1 training.")
parser.add_argument("--l0_adapt_lr", type=float, default=1e-9, help="Learning rate for L0 adaptation (very small).")
parser.add_argument("--l0_adapt_std", type=float, default=0.1, help="Assumed action std for L0 Gaussian log-prob during adaptation.")
parser.add_argument("--l0_adapt_every_n_updates", type=int, default=1, help="Apply L0 adaptation every N PPO updates.")
parser.add_argument(
    "--l0_adapt_signal",
    type=str,
    default="success",
    choices=["reward", "success", "both"],
    help="Signal used for L0 adaptation: reward-based PG, success imitation, or both."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
parser.add_argument("--skill_name", type=str, default="JumpOverLowWall", help="Name of the skill to train, same as the skill library refernce")
parser.add_argument("--steps", type=int, default=None, help="Total number of environment steps to train.")
parser.add_argument("--use-random-policies", action="store_true", default=False, help="Use random policies when L0 policy checkpoints are missing (for testing builds)")

# Ablation flags [Ensure are all false for usual running]
parser.add_argument("--ablate_hierarchical_gae", action="store_true", default=True, help="Disable hierarchical GAE in HPPO")
parser.add_argument("--ablate_pass_decision_mask", action="store_true", default=False, help="Do not pass decision_mask (behaves like flat PPO)")
parser.add_argument("--ablate_advantage_norm", action="store_true", default=False, help="Disable advantage normalization at decision points")

# WandB overrides for ablation sweeps
parser.add_argument("--wandb_project_override", type=str, default=None, help="Override WandB project name (e.g., 'HPPO ablations')")
parser.add_argument("--wandb_group", type=str, default="Standard GAE no adaptation", help="WandB group for this run (e.g., ablation id)")
parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name override")
parser.add_argument("--ablation_name", type=str, default=None, help="Optional ablation name (used as WandB group if group not provided)")

# Convenience negation flags for booleans with store_true defaults
parser.add_argument("--no_early_terminate_on_success", action="store_true", default=False, help="Override to disable early termination on success")
parser.add_argument("--no_adapt_l0", action="store_true", default=False, help="Override to disable L0 adaptation")

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
    DirectMARLEnvCfg,
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

# Add import for the custom memory
from genhrl.hppo.decision_point_memory import DecisionPointMemory
from genhrl.hppo.register_memory import register_decision_point_memory
# Add import for the HPPO agent registration
from genhrl.hppo.register_agent import register_hppo_agent

# Add import for the unified component registration
from genhrl.hppo.register_components import register_hppo_components
from genhrl.hppo.hppo import HPPO

def is_skill_primitive(skill_name, skill_library):
    """Checks if a skill is primitive based on the library."""
    # Add error handling for missing skill
    skill_info = skill_library.get("skills", {}).get(skill_name)
    if not skill_info:
        print(f"Warning: Skill '{skill_name}' not found in library skills dictionary.")
        return False # Assume not primitive if not found
    return skill_info.get("is_primitive", False)

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.steps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.steps
    elif args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", args_cli.skill_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # Update wandb configuration (allow override for ablations)
    if "wandb_kwargs" in agent_cfg["agent"]["experiment"]:
        # Use override project if provided; else per-skill project as before
        if args_cli.wandb_project_override:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"] = args_cli.wandb_project_override
        else:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"] = f"genhrl_{args_cli.skill_name}"
        # Use group for ablations
        if args_cli.wandb_group or args_cli.ablation_name:
            agent_cfg["agent"]["experiment"]["wandb_kwargs"]["group"] = args_cli.wandb_group or args_cli.ablation_name
        # Run name
        agent_cfg["agent"]["experiment"]["wandb_kwargs"]["name"] = args_cli.wandb_name or log_dir

    
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # --- Import the wrapper *after* the environment is created AND sys.path is set ---
    from genhrl.hppo.l1_wrapper import HierarchicalVecActionWrapper
    # --- End wrapper import ---

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # Capture the base Isaac Lab environment (post any conversions/wrappers) for success checks
    try:
        base_env = env.unwrapped
    except Exception:
        base_env = env

    # --- Apply SkrlVecEnvWrapper FIRST ---
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    # Now 'env' is the SkrlVecEnvWrapper instance
    # --- End SkrlVecEnvWrapper ---

    # --- Apply HierarchicalVecActionWrapper AFTER SkrlVecEnvWrapper ---
    # we need to find the sub_policy_checkpoint_paths
    skill_library = json.load(open(SKILL_LIBRARY_PATH))
    if args_cli.skill_name and args_cli.skill_name in skill_library.get("skills", {}):
        skills_sub_policies = skill_library["skills"][args_cli.skill_name].get("sub_skills", [])
    else:
        print(f"Warning: Skill name '{args_cli.skill_name}' not provided or not found in skill library. Cannot load L0 policies.")
        skills_sub_policies = []

    sub_policy_checkpoint_paths = []
    registered_subskill_names = []
    for ssp in skills_sub_policies:
        checkpoint_file = os.path.join(SKILLS_PATH, ssp, "policy", "agent.pt")
        if is_skill_primitive(ssp, skill_library):
            registered_subskill_name = f"Isaac-RobotFlat{ssp.title()}-v0"
        else:
            registered_subskill_name = f"Isaac-RobotComposite{ssp.title()}-v0"
        registered_subskill_names.append(registered_subskill_name)
        if os.path.exists(checkpoint_file):
             sub_policy_checkpoint_paths.append(checkpoint_file)
        else:
             print(f"Warning: Checkpoint file not found for sub-skill '{ssp}' at '{checkpoint_file}'. Skipping.")

    # Check if we should proceed with wrapper even without sub-policy paths (using random policies)
    if sub_policy_checkpoint_paths or args_cli.use_random_policies:
        if sub_policy_checkpoint_paths:
            print(f"Applying HierarchicalVecActionWrapper with {len(sub_policy_checkpoint_paths)} L0 policies.")
        elif args_cli.use_random_policies:
            print(f"Applying HierarchicalVecActionWrapper with random policies for {len(skills_sub_policies)} L0 sub-skills (testing mode).")
            # Create empty paths list for skills that need random policies
            sub_policy_checkpoint_paths = [""] * len(skills_sub_policies)
            
        # Pass the SkrlVecEnvWrapper instance (currently in 'env') to the custom wrapper
        env = HierarchicalVecActionWrapper(env,
                                           sub_policy_checkpoint_paths=sub_policy_checkpoint_paths,
                                           sub_policy_registered_names=registered_subskill_names,
                                           sub_skill_folder_names=skills_sub_policies,
                                           skills_root_path=SKILLS_PATH,
                                           l1_skill_folder_name=args_cli.skill_name,
                                           steps_per_l0_policy=args_cli.steps_per_l0,
                                           l1_action_frequency=args_cli.steps_per_l0,
                                           base_env=base_env,
                                           early_terminate_on_success=args_cli.early_terminate_on_success,
                                           adapt_l0=args_cli.adapt_l0,
                                           l0_adapt_lr=args_cli.l0_adapt_lr,
                                           l0_adapt_std=args_cli.l0_adapt_std,
                                           l0_adapt_every_n_updates=args_cli.l0_adapt_every_n_updates,
                                           l0_adapt_signal=args_cli.l0_adapt_signal,
                                           disable_success_state_saving=True,
                                           debug_mode=False,
                                           use_random_policies=args_cli.use_random_policies)  # Observation timing fix is complete
        # Now 'env' is the HierarchicalVecActionWrapper instance
        
        # Register and use our custom memory that filters out non-decision steps
        register_hppo_components()  # Register both memory and agent with SKRL
        # Explicitly register DecisionPointMemory as well to ensure availability by name
        register_decision_point_memory()

        print("Using DecisionPointMemory to filter L1 non-decision steps")
        print("Using HPPO agent for hierarchical learning")
        
        # Update agent config to use our custom memory class (use string; registered via register_hppo_components)
        agent_cfg["memory"] = {
            "class": "DecisionPointMemory",
            "memory_size": agent_cfg["agent"]["rollouts"],  # Same as default RandomMemory
            "hierarchy_level": 1,  # Explicitly specify we're using L1 decision points
            "debug_mode": False,  # Also enable debug mode in memory config
            "discount_factor": agent_cfg["agent"]["discount_factor"],  # For L1 policy learning between decisions
            "skill_reward_discount": 1.0,  # No discounting during skill execution (full reward accumulation)
            "use_average_reward": False,
            "max_skill_duration": args_cli.steps_per_l0 # Maximum L1 skill duration (should match L2 wrapper config)
        }
        
        # Update agent config to use HPPO instead of PPO (use string; registered via register_hppo_components)
        agent_cfg["agent"]["class"] = "HPPO"
        
        # ENABLE DEBUG MODE for hierarchical learning
        agent_cfg["agent"]["debug_mode"] = False

        # Apply ablation flags to agent memory/HPPO behavior
        # A1: disable hierarchical GAE
        if args_cli.ablate_hierarchical_gae:
            agent_cfg["agent"]["use_hierarchical_gae"] = False
        # A3: do not pass decision_mask (flat behavior)
        if args_cli.ablate_pass_decision_mask:
            agent_cfg["agent"]["pass_decision_mask"] = False
        # A4: disable advantage normalization at decision points
        if args_cli.ablate_advantage_norm:
            agent_cfg["agent"]["normalize_advantages_at_decisions"] = False

        # B2: disable early termination on success (via CLI override)
        if args_cli.no_early_terminate_on_success:
            args_cli.early_terminate_on_success = False
        # C1: disable L0 adaptation (via CLI override)
        if args_cli.no_adapt_l0:
            args_cli.adapt_l0 = False
        
        print(f"Updated agent configuration:")
        print(f"  - Agent class: {agent_cfg['agent']['class']}")
        print(f"  - Memory class: {agent_cfg['memory']['class']}")
        print(f"  - Memory hierarchy level: {agent_cfg['memory']['hierarchy_level']}")
        print(f"  - Debug mode enabled: {agent_cfg['agent']['debug_mode']}")
    else:
        print("Warning: No sub-policy checkpoints loaded and --use-random-policies not enabled.")
        print("HierarchicalVecActionWrapper will not be applied.")
        # Exit if the wrapper is critical for this training script
        sys.exit(1)
    # --- End HierarchicalVecActionWrapper ---

    # Use the standard SKRL Runner (HPPO and memory already registered by register_hppo_components)
    from skrl.utils.runner.torch import Runner
    runner = Runner(env, agent_cfg)

    # Provide the agent with a reference to the managed environment for L0 adaptation on updates
    try:
        if hasattr(runner, 'agent') and hasattr(runner.agent, 'set_managed_env'):
            runner.agent.set_managed_env(env)
            if args_cli.adapt_l0:
                print(f"[ADAPT] L0 adaptation enabled: lr={args_cli.l0_adapt_lr}, every_n_updates={args_cli.l0_adapt_every_n_updates}")
    except Exception:
        pass

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # Save the trained L1 agent checkpoint to the skill's policy directory
    try:
        policy_dir = os.path.join(SKILLS_PATH, args_cli.skill_name, "policy")
        os.makedirs(policy_dir, exist_ok=True)
        save_path = os.path.join(policy_dir, "agent.pt")
        # Save full agent checkpoint (policy + value + optimizer state if available)
        runner.agent.save(save_path)
        print(f"[INFO] Saved L1 agent checkpoint to: {save_path}")

        # Also export simplified policy weights and training params for fast/robust loading by wrappers
        try:
            # Extract policy state dict from the agent checkpoint
            import torch
            ckpt = torch.load(save_path, map_location='cpu')
            policy_state = None
            if isinstance(ckpt, dict):
                if 'policy' in ckpt and isinstance(ckpt['policy'], dict):
                    policy_state = ckpt['policy']
                elif 'models' in ckpt and 'policy' in ckpt['models']:
                    policy_state = ckpt['models']['policy']
                elif 'state_dict' in ckpt:
                    policy_state = ckpt['state_dict']
            if policy_state is not None:
                # Filter only network weights (including policy_layer), mirroring L1/L2 loaders
                network_weights = {}
                for key, value in policy_state.items():
                    if (('net_container.' in key or 'net.' in key) and not ('value_layer' in key or 'log_std' in key)):
                        network_weights[key] = value
                    elif key.startswith('policy_layer.') and (key.endswith('.weight') or key.endswith('.bias')):
                        network_weights[key] = value
                if network_weights:
                    torch.save(network_weights, os.path.join(policy_dir, 'network_weights.pt'))
                    print(f"[INFO] Exported simplified network weights")
            # Save training parameters snapshot (mirrors orchestrator export)
            try:
                training_info = {
                    "skill_name": args_cli.skill_name,
                    "checkpoint_path": save_path,
                }
                # Persist minimal arch info if available in logs params
                params_dir = os.path.join(log_dir, 'params')
                agent_yaml = os.path.join(params_dir, 'agent.yaml')
                if os.path.exists(agent_yaml):
                    import yaml
                    with open(agent_yaml, 'r') as f:
                        agent_cfg_logged = yaml.safe_load(f)
                    training_info["agent_config"] = agent_cfg_logged
                with open(os.path.join(policy_dir, 'training_params.json'), 'w') as f:
                    json.dump(training_info, f, indent=2)
                    print(f"[INFO] Saved training_params.json")
            except Exception as e:
                print(f"[WARN] Failed to save training_params.json: {e}")
        except Exception as e:
            print(f"[WARN] Simplified export failed: {e}")
    except Exception as e:
        print(f"[WARN] Failed to save L1 agent checkpoint: {e}")

    # close the simulator
    env.close()
    
    # Clean up wandb state to prevent conflicts with subsequent training runs
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("[INFO] Wandb run finished and cleaned up")
    except ImportError:
        pass  # wandb not installed


if __name__ == "__main__":
    # run the main function (hydra_task_config will handle arguments)
    main()  # type: ignore[call-arg] # Hydra decorator handles arguments injection
    # close sim app
    simulation_app.close()
