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

# Fix CUDA linear algebra backend issues (must be before Isaac Lab imports)
import torch
torch.backends.cuda.preferred_linalg_library("cusolver")  # Try cusolver first

import argparse
import sys
import os 
import warnings

# Suppress IsaacLab deprecation warnings before any imports
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
import logging
logging.getLogger("isaaclab.utils.math").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*quat_rotate.*deprecated.*")
warnings.filterwarnings("ignore", message=".*quat_apply.*")
warnings.filterwarnings("ignore", message=".*will be deprecated.*")

from isaaclab.app import AppLauncher


# for debugging:
TASK_NAME = "walk_to_football"  # Update this to match your generated task
TASK_PATH = f"/home/tomcannon/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/tasks/{TASK_NAME}"
SKILLS_PATH = f"/home/tomcannon/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/skills/{TASK_NAME}/skills"
SKILL_LIBRARY_PATH = f"/home/tomcannon/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/skills/{TASK_NAME}/skill_library.json"
# set the object config path - try skills directory first, fallback to tasks directory
OBJECT_CONFIG_SKILLS_PATH = f'/home/tomcannon/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/skills/{TASK_NAME}/object_config.json'
OBJECT_CONFIG_TASKS_PATH = f'/home/tomcannon/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/tasks/{TASK_NAME}/object_config.json'

# Check which object config exists and use it
if os.path.exists(OBJECT_CONFIG_SKILLS_PATH):
    os.environ['OBJECT_CONFIG_PATH'] = OBJECT_CONFIG_SKILLS_PATH
    print(f"Using object config from skills directory: {OBJECT_CONFIG_SKILLS_PATH}")
elif os.path.exists(OBJECT_CONFIG_TASKS_PATH):
    os.environ['OBJECT_CONFIG_PATH'] = OBJECT_CONFIG_TASKS_PATH
    print(f"Using object config from tasks directory: {OBJECT_CONFIG_TASKS_PATH}")
else:
    print(f"Warning: Object config not found in either location!")
    print(f"  Skills path: {OBJECT_CONFIG_SKILLS_PATH}")
    print(f"  Tasks path: {OBJECT_CONFIG_TASKS_PATH}")
    os.environ['OBJECT_CONFIG_PATH'] = OBJECT_CONFIG_SKILLS_PATH  # Default to skills path

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-G1FlatEnsurehighwallfalls-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
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

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime
import pickle

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

# Register generated environments before importing isaaclab_tasks
import gymnasium as gym
import os

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


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
    if args_cli.max_iterations:
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
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
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
    
    # Update wandb configuration to use unique names per skill
    if "wandb_kwargs" in agent_cfg["agent"]["experiment"]:
        # Use the log_dir name for wandb run name to keep consistency  
        agent_cfg["agent"]["experiment"]["wandb_kwargs"]["name"] = log_dir
        # Extract task name from task argument for project name
        task_name = args_cli.task.replace("Isaac-Robot", "").replace("-v0", "")
        agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"] = f"genhrl_{task_name}"
    
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # --- Create Log Dirs --- 
    params_log_dir = os.path.join(log_dir, "params")
    os.makedirs(params_log_dir, exist_ok=True)
    # metadata_log_dir = os.path.join(log_dir, "metadata") # No longer creating separate metadata dir
    # os.makedirs(metadata_log_dir, exist_ok=True)
    # --- End Create Log Dirs ---

    # dump the configuration into log-directory
    dump_yaml(os.path.join(params_log_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_log_dir, "agent.yaml"), agent_cfg)
    # Save agent config pickle
    agent_pkl_path = os.path.join(params_log_dir, "agent.pkl")
    dump_pickle(agent_pkl_path, agent_cfg)
    # Don't save env config pickle as it doesn't have the runtime spaces
    # dump_pickle(os.path.join(params_log_dir, "env.pkl"), env_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # --- Save Spaces Pickle --- 
    try:
        # Save spaces from the ACTUAL wrapped env instance
        spaces_pkl_path = os.path.join(params_log_dir, "spaces.pkl") # Save directly into params dir
        obs_space = env.observation_space
        act_space = env.action_space
        if obs_space is None or act_space is None:
             raise ValueError("Observation or Action space is None after wrapping.")

        with open(spaces_pkl_path, "wb") as f:
            pickle.dump({"observation_space": obs_space, "action_space": act_space}, f)
        print(f"[INFO] Saved runtime spaces to: {spaces_pkl_path}")

    except Exception as e:
        print(f"[ERROR] Failed to save runtime spaces pickle: {e}")
        # This is critical for L1 wrapper, so maybe raise?
        raise RuntimeError(f"Could not save spaces: {e}") from e
    # --- End Save Spaces Pickle ---

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
