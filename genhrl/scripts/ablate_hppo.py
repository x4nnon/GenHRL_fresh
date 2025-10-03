#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime


def run(cmd, env=None):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    # Resolve project root and train script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_script = os.path.join(project_root, "genhrl", "scripts", "train_l1.py")

    # Task and skill targeting (from user's provided registry entry)
    # You can override via environment variables if needed.
    task_id = os.environ.get(
        "GENHRL_TASK_ID",
        "Isaac-RobotCompositeKnockOverPillarsSeed42KnockOverPillarsSeed42-v0",
    )
    # Ensure skill library paths resolve under this task name
    # Set GENHRL_TASK_NAME to task folder name hosting skill_library.json
    task_name_for_skills = os.environ.get("GENHRL_TASK_NAME", "knock_over_pillars_seed42")
    # Skill name used to locate sub-skills and checkpoints
    skill_name = os.environ.get("GENHRL_SKILL_NAME", "knock_over_pillars_seed42")

    # Shared settings
    num_envs = int(os.environ.get("GENHRL_NUM_ENVS", "1024"))
    steps_per_l0_baseline = int(os.environ.get("GENHRL_STEPS_PER_L0", "200"))
    video = False

    # Three seeds per variant
    seeds = [42, 1337, 2024]

    # WandB project for ablations
    wandb_project = "HPPO ablations"

    # Define ablation variants
    variants = {
        "baseline": [],
        # A1: disable hierarchical GAE
        "A1_no_hierarchical_gae": ["--ablate_hierarchical_gae"],
        # A3: do not pass decision_mask
        "A3_no_decision_mask": ["--ablate_pass_decision_mask"],
        # A4: disable advantage normalization at decisions
        "A4_no_advantage_norm": ["--ablate_advantage_norm"],
        # B2: disable early termination on success
        "B2_no_early_terminate": ["--no_early_terminate_on_success"],
        # B3: adjust steps_per_l0 (skill horizon)
        "B3_steps_per_l0_50": ["--steps_per_l0", "50"],
        # C1: disable L0 adaptation
        "C1_no_l0_adapt": ["--no_adapt_l0"],
    }

    # Base command args (common)
    base_args = [
        sys.executable,
        train_script,
        "--task",
        task_id,
        "--skill_name",
        skill_name,
        "--num_envs",
        str(num_envs),
        "--steps",
        "20000",
    ]
    if video:
        base_args.append("--video")

    # Environment for subprocess
    child_env = os.environ.copy()
    # Ensure the skills path is resolved for this task
    child_env["GENHRL_TASK_NAME"] = task_name_for_skills

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for ablation_name, ablation_flags in variants.items():
        for seed in seeds:
            # Build command per run
            cmd = base_args + [
                "--seed",
                str(seed),
                # Force wandb project/group/name
                "--wandb_project_override",
                wandb_project,
                "--wandb_group",
                ablation_name,
                "--wandb_name",
                f"{ablation_name}-seed{seed}-{timestamp}",
                "--ablation_name",
                ablation_name,
            ]

            # Carry baseline steps_per_l0 for all variants except B3 override when provided
            if "--steps_per_l0" not in ablation_flags:
                cmd += ["--steps_per_l0", str(steps_per_l0_baseline)]

            cmd += ablation_flags

            run(cmd, env=child_env)


if __name__ == "__main__":
    main()


