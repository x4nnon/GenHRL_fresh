"""
Train an L3 hierarchical agent (L3 -> L2 -> L1 -> L0) with SKRL.
Mirrors train_l2.py but adds one more hierarchy level.
"""

import argparse
import json
import sys
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch
torch.backends.cuda.preferred_linalg_library("cusolver")

from isaaclab.app import AppLauncher

# Paths via env or project layout
L3_TASK_NAME = os.environ.get("GENHRL_TASK_NAME", "Create_Steps_L3")
ROBOT_NAME = os.environ.get("GENHRL_ROBOT", "G1")

ISAACLAB_ROOT = os.environ.get("ISAACLAB_PATH")
if not ISAACLAB_ROOT:
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    isaaclab_candidate = project_root / "IsaacLab"
    if isaaclab_candidate.exists():
        ISAACLAB_ROOT = str(isaaclab_candidate)
    else:
        raise RuntimeError("ISAACLAB_PATH not set and IsaacLab not found. Set ISAACLAB_PATH.")

ROBOT_FOLDER = f"{ROBOT_NAME}_generated"
ISAACLAB_TASKS_BASE = os.path.join(ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "manager_based", ROBOT_FOLDER)

TASK_PATH = os.path.join(ISAACLAB_TASKS_BASE, "tasks", L3_TASK_NAME)
SKILLS_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", L3_TASK_NAME, "skills")
SKILL_LIBRARY_PATH = os.path.join(ISAACLAB_TASKS_BASE, "skills", L3_TASK_NAME, "skill_library.json")

if os.path.exists(TASK_PATH):
    os.environ['OBJECT_CONFIG_PATH'] = os.path.join(TASK_PATH, 'object_config.json')

# CLI
parser = argparse.ArgumentParser(description="Train an L3 RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1100)
parser.add_argument("--video_interval", type=int, default=11000)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default=f"Isaac-G1Composite{L3_TASK_NAME}-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"])  # L3 custom runner uses torch
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"])  # L3: PPO baseline with custom wrapper
parser.add_argument("--skill_name", type=str, default=L3_TASK_NAME, help="Name of the L3 composite to train")
parser.add_argument("--steps", type=int, default=None, help="Total environment steps")
parser.add_argument("--steps_per_l2", type=int, default=300)
parser.add_argument("--steps_per_l1", type=int, default=100)
parser.add_argument("--steps_per_l0", type=int, default=50)
parser.add_argument("--use-random-policies", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Prepare Hydra argv
sys.argv = [sys.argv[0]] + hydra_args

# Launch sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Path setup for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import skrl
from packaging import version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}")
    exit()

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from pathlib import Path
workspace_root = Path(__file__).parent.parent.parent
isaaclab_source_path = workspace_root / "IsaacLab" / "source"
if str(isaaclab_source_path) not in sys.path:
    sys.path.insert(0, str(isaaclab_source_path))
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point"

# Register HPPO components for compatibility (reuse)
from genhrl.hppo.register_components import register_hppo_components
register_hppo_components()

from genhrl.hppo.l3_wrapper import HierarchicalVecActionWrapperL3


def is_skill_primitive(skill_name: str, library: Dict[str, Any]) -> bool:
    info = library.get("skills", {}).get(skill_name)
    if not info:
        return False
    return info.get("is_primitive", False)


def find_all_l1_sub_skills(skill_name: str, library: Dict[str, Any]) -> List[str]:
    """Return immediate sub-skills (expected to be L1) for the composite L2/L3 skill."""
    info = library.get("skills", {}).get(skill_name)
    if not info:
        return []
    return list(info.get("sub_skills", []))


def find_all_l0_sub_skills(skill_name: str, library: Dict[str, Any], out: Optional[set[str]] = None):
    if out is None:
        out = set()
    info = library.get("skills", {}).get(skill_name)
    if not info:
        return out
    if info.get("is_primitive", False):
        out.add(skill_name)
    else:
        for s in info.get("sub_skills", []):
            find_all_l0_sub_skills(s, library, out)
    return out


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    # Basic overrides
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # Convert steps to iterations if provided
    if args_cli.steps is not None:
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        iterations = max(1, args_cli.steps // num_envs)
        agent_cfg["trainer"]["timesteps"] = iterations

    # Logging dirs
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", f"L3_{args_cli.skill_name}"))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_L3_{args_cli.skill_name}_{algorithm}_{args_cli.ml_framework}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    from isaaclab.utils.io import dump_yaml, dump_pickle
    from isaaclab.utils.dict import print_dict
    os.makedirs(os.path.join(log_root_path, log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap with SKRL vec wrapper
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Load skill library for building mappings and checkpoint lists
    with open(SKILL_LIBRARY_PATH, 'r') as f:
        skill_library = json.load(f)

    # At L3, target skill has L2 sub-skills; each L2 has L1 sub-skills; each L1 has L0 primitives
    # 1) Collect L2 sub-skills for target L3
    l3_skill = args_cli.skill_name
    l2_sub_skill_names = find_all_l1_sub_skills(l3_skill, skill_library)
    if not l2_sub_skill_names:
        print(f"Error: L3 skill '{l3_skill}' has no L2 sub-skills defined.")
        sys.exit(1)

    # 2) For each L2 skill, collect its L1 sub-skills and build L2->L1 mapping
    l2_policy_checkpoint_paths: List[str] = []
    l2_registered_names: List[str] = []
    l2_to_l1_mapping: Dict[int, List[int]] = {}

    # We'll also collect all distinct L1 skills to prepare L1 checkpoints and indices
    all_l1_skills: List[str] = []

    for l2_idx, l2_name in enumerate(l2_sub_skill_names):
        l2_info = skill_library.get("skills", {}).get(l2_name)
        if not l2_info:
            print(f"Warning: L2 skill '{l2_name}' not found in library; skipping")
            continue
        # L2 checkpoint
        l2_ckpt = os.path.join(SKILLS_PATH, l2_name, "policy", "agent.pt")
        l2_policy_checkpoint_paths.append(l2_ckpt if os.path.exists(l2_ckpt) else ("" if args_cli.use_random_policies else ""))
        # Registered name (composite)
        l2_registered_names.append(f"Isaac-RobotComposite{l2_name.title()}-v0")
        # L1 sub-skills for this L2
        l1_subs = l2_info.get("sub_skills", [])
        # Map local indices to global L1 indices we will assign later
        local_to_global: List[int] = []
        for l1_name in l1_subs:
            if l1_name not in all_l1_skills:
                all_l1_skills.append(l1_name)
            local_to_global.append(all_l1_skills.index(l1_name))
        l2_to_l1_mapping[l2_idx] = local_to_global

    if not all_l1_skills:
        print("Error: No L1 skills found for L3.")
        sys.exit(1)

    # 3) For each unique L1 skill, gather checkpoint and registered name, and collect L0 deps
    l1_policy_checkpoint_paths: List[str] = []
    l1_registered_names: List[str] = []
    l1_to_l0_dependencies: Dict[str, List[str]] = {}
    unique_l0_skills: set[str] = set()

    for l1_name in all_l1_skills:
        l1_info = skill_library.get("skills", {}).get(l1_name)
        l1_ckpt = os.path.join(SKILLS_PATH, l1_name, "policy", "agent.pt")
        if not os.path.exists(l1_ckpt) and not args_cli.use_random_policies:
            print(f"Warning: L1 checkpoint missing for '{l1_name}' at '{l1_ckpt}'.")
        l1_policy_checkpoint_paths.append(l1_ckpt if os.path.exists(l1_ckpt) else ("" if args_cli.use_random_policies else ""))
        l1_registered_names.append(f"Isaac-RobotComposite{l1_name.title()}-v0")
        # L0 dependencies
        l0_deps = find_all_l0_sub_skills(l1_name, skill_library)
        l1_to_l0_dependencies[l1_name] = sorted(list(l0_deps))
        unique_l0_skills.update(l0_deps)

    # 4) Prepare L0 checkpoints/registered names in stable order
    l0_policy_checkpoint_paths: List[str] = []
    l0_registered_names: List[str] = []
    sorted_l0 = sorted(list(unique_l0_skills))
    for l0_name in sorted_l0:
        l0_ckpt = os.path.join(SKILLS_PATH, l0_name, "policy", "agent.pt")
        if not os.path.exists(l0_ckpt) and not args_cli.use_random_policies:
            print(f"Warning: L0 checkpoint missing for '{l0_name}' at '{l0_ckpt}'.")
        l0_policy_checkpoint_paths.append(l0_ckpt if os.path.exists(l0_ckpt) else ("" if args_cli.use_random_policies else ""))
        l0_registered_names.append(f"Isaac-RobotFlat{l0_name.title()}-v0")

    # 5) Instantiate L3 wrapper
    env = HierarchicalVecActionWrapperL3(
        env,
        l2_policy_checkpoint_paths=l2_policy_checkpoint_paths,
        l2_policy_registered_names=l2_registered_names,
        l2_skill_names=l2_sub_skill_names,
        l1_policy_checkpoint_paths=l1_policy_checkpoint_paths,
        l1_policy_registered_names=l1_registered_names,
        l0_policy_checkpoint_paths=l0_policy_checkpoint_paths,
        l0_policy_registered_names=l0_registered_names,
        skill_library_path=SKILL_LIBRARY_PATH,
        skills_path=SKILLS_PATH,
        steps_per_l2_policy=args_cli.steps_per_l2,
        l3_action_frequency=args_cli.steps_per_l2,
        steps_per_l1_policy=args_cli.steps_per_l1,
        l2_action_frequency=args_cli.steps_per_l1,
        steps_per_l0_policy=args_cli.steps_per_l0,
        l1_action_frequency=args_cli.steps_per_l0,
        use_random_policies=args_cli.use_random_policies,
    )

    # Build index mappings from names to indices for L1 and L0
    l1_name_to_global_idx = {name: idx for idx, name in enumerate(all_l1_skills)}
    l0_name_to_global_idx = {name: idx for idx, name in enumerate(sorted_l0)}

    # Convert previously built mappings into index-based dicts for the wrapper
    l2_to_l1_index_mapping = {}
    for l2_idx, l2_name in enumerate(l2_sub_skill_names):
        local_list = []
        subs = skill_library.get("skills", {}).get(l2_name, {}).get("sub_skills", [])
        for l1_name in subs:
            local_list.append(l1_name_to_global_idx.get(l1_name, 0))
        l2_to_l1_index_mapping[l2_idx] = local_list

    l1_to_l0_index_mapping = {}
    for l1_name, l0_list in l1_to_l0_dependencies.items():
        local = [l0_name_to_global_idx.get(n, 0) for n in l0_list]
        l1_to_l0_index_mapping[l1_name_to_global_idx[l1_name]] = local

    env.register_mappings(l2_to_l1_index_mapping, l1_to_l0_index_mapping)

    # Configure agent and runner (reuse PPO runner)
    from genhrl.hppo.hppo_runner import HPPORunner as Runner
    runner = Runner(env, agent_cfg)  # type: ignore

    # Resume checkpoint if any
    from isaaclab.utils.assets import retrieve_file_path
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if resume_path:
        runner.agent.load(resume_path)

    # Train
    runner.run()

    # Close
    env.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()


