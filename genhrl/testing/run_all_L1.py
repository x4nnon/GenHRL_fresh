import argparse
import sys
import subprocess
import shutil
import os
from pathlib import Path


def _import_orchestrator():
    """Import TrainingOrchestrator with a robust fallback for local runs."""
    try:
        from genhrl.training.orchestrator import TrainingOrchestrator  # type: ignore
        return TrainingOrchestrator
    except Exception:
        # Fallback: add repo root to sys.path and retry
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from genhrl.training.orchestrator import TrainingOrchestrator  # type: ignore
        return TrainingOrchestrator


def list_l1_skills(orchestrator) -> list:
    """Return all level-1 composite skills for the task in orchestrator."""
    order = orchestrator.get_training_order()
    return [s for s in order if (not orchestrator.is_skill_primitive(s)) and orchestrator.get_skill_level(s) == 1]


def _resolve_logs_dir(isaaclab_path: Path) -> Path:
    """Return the logs/skrl directory used by IsaacLab."""
    candidates = [isaaclab_path / "logs" / "skrl", Path("logs/skrl").resolve()]
    for c in candidates:
        if c.exists():
            return c
    # Default to inside isaaclab
    return candidates[0]


def _auto_detect_isaaclab_path() -> Path:
    """Best-effort detection of IsaacLab path.

    Priority:
    1) ENV ISAACLAB_PATH if it exists
    2) RepoRoot/IsaacLab (repo root inferred from this file)
    3) ./IsaacLab relative to current working directory
    """
    # 1) Environment variable
    env_path = os.environ.get("ISAACLAB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    # 2) Repo root / IsaacLab
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    candidate = repo_root / "IsaacLab"
    if candidate.exists():
        return candidate

    # 3) CWD / IsaacLab
    cwd_candidate = Path.cwd() / "IsaacLab"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    # Fallback to repo_root/IsaacLab even if missing (to surface a clear error later)
    return candidate


def _snapshot_top_level_runs(logs_dir: Path) -> set:
    """Snapshot top-level run directories under logs/skrl."""
    if not logs_dir.exists():
        return set()
    return {p for p in logs_dir.iterdir() if p.is_dir()}


def _cleanup_new_runs(before: set, after: set) -> int:
    """Remove any new run directories created between snapshots. Returns count removed."""
    new_dirs = [p for p in after if p not in before]
    removed = 0
    for d in new_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        except Exception:
            pass
    # If logs/skrl became empty, remove it as well for cleanliness
    if after and (after - set(new_dirs)) == set() and after:
        parent = next(iter(after)).parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
    return removed


def _format_gym_task_name(task_name: str, skill_name: str) -> str:
    """Match orchestrator's formatting for gym task suffix."""
    task_parts = task_name.split('_')
    skill_parts = skill_name.split('_')
    formatted_task = ''.join(part.capitalize() for part in task_parts)
    formatted_skill = ''.join(part.capitalize() for part in skill_parts)
    return f"{formatted_task}{formatted_skill}"


def _build_flat_l1_command(isaaclab_path: Path, task_name: str, skill_name: str, steps: int, num_envs: int, seed: int, flat_suffix: str) -> str:
    """Build a flat PPO command for an L1 composite skill using IsaacLab's skrl/train.py.

    We expect duplicate gym registrations for composites with '-{flat_suffix}' in the ID that
    point to agents/skrl_ppo_cfg.yaml via skrl_cfg_entry_point.
    """
    # Match orchestrator's exact env id naming
    gym_suffix = _format_gym_task_name(task_name, skill_name)
    # Flat env id uses suffix before -v0
    env_id = f"Isaac-RobotComposite{gym_suffix}-{flat_suffix}-v0" if flat_suffix else f"Isaac-RobotComposite{gym_suffix}-v0"

    # Flat training uses IsaacLab train.py and --max_iterations
    # Align with L0 logic: assume rollouts=24 to approximate iterations from steps
    rollouts = 24
    max_iterations = max(1, steps // rollouts)

    base_command = (
        f"./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py "
        f"--task {env_id} "
        f"--num_envs {num_envs} --seed {seed} --headless "
        f"--max_iterations {max_iterations}"
    )

    # Prepend env setup similar to orchestrator
    object_config_path = os.environ.get('OBJECT_CONFIG_PATH', '')
    genhrl_task = os.environ.get('GENHRL_TASK_NAME', task_name)
    genhrl_robot = os.environ.get('GENHRL_ROBOT', 'G1')
    env_setup = (
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 && "
        "export CARB_LOG_LEVEL=FATAL && "
        "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048 && "
        "export CUDA_LAUNCH_BLOCKING=0 && "
        f"export OBJECT_CONFIG_PATH='{object_config_path}' && "
        f"export GENHRL_TASK_NAME='{genhrl_task}' && "
        f"export GENHRL_ROBOT='{genhrl_robot}' && "
    )

    return f"{env_setup}{base_command}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or run commands to train all L1 skills (hierarchical and flat) across multiple seeds",
    )
    parser.add_argument("--isaaclab_path", required=False, default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--steps", type=int, default=20000, help="Steps per run (default: 20000)")
    parser.add_argument("--max_timesteps", type=int, default=None, help="Override steps per run: total timesteps. For flat PPO this is converted to --max_iterations using rollouts=24")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments (default: 4096)")
    parser.add_argument(
        "--seeds_l1",
        type=int,
        nargs="+",
        default=[42, 666, 777, 888, 999, 101010],
        help="Seed values for hierarchical L1 runs (default: 5 seeds)",
    )
    parser.add_argument(
        "--seeds_flat",
        type=int,
        nargs="+",
        default=[42, 666, 777, 888, 999, 101010],
        help="Seed values for flat PPO runs (default: 5 seeds)",
    )
    parser.add_argument("--flat_suffix", type=str, default="Flat", help="Env ID suffix for flat PPO composite registrations (default: Flat)")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing")
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="After each run, remove newly created log directories to avoid persisting files",
    )
    parser.add_argument(
        "--no_success_saving",
        action="store_true",
        help="Disable saving success_states by patching save_success_state at import time",
    )
    parser.add_argument("--video_interval", type=int, default=2000, help="Video interval steps (unused when not recording)")
    parser.add_argument("--video_length", type=int, default=200, help="Video length steps (unused when not recording)")
    parser.add_argument("--skip_first", type=int, default=0, help="Skip the first N L1 skills before running")
    parser.add_argument("--run_baseline", default=False, action="store_true", help="Run baseline PPO for all L1 skills")


    args = parser.parse_args()

    # Resolve IsaacLab path (use provided or auto-detect)
    isaaclab_path = Path(args.isaaclab_path).expanduser().resolve() if args.isaaclab_path else _auto_detect_isaaclab_path()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        print("Hint: Set ISAACLAB_PATH env var or pass --isaaclab_path /path/to/IsaacLab")
        return 1

    TrainingOrchestrator = _import_orchestrator()

    orch = TrainingOrchestrator(
        isaaclab_path=str(isaaclab_path),
        task_name=args.task_name,
        robot=args.robot,
    )

    l1_skills = list_l1_skills(orch)
    if not l1_skills:
        print(f"No L1 skills found for task: {args.task_name}")
        return 1

    # Optionally skip the first N skills
    if args.skip_first > 0:
        if args.skip_first >= len(l1_skills):
            print(f"--skip_first={args.skip_first} >= number of L1 skills ({len(l1_skills)}). Nothing to do.")
            return 0
        l1_skills = l1_skills[args.skip_first:]

    logs_dir = _resolve_logs_dir(isaaclab_path)

    if not args.run:
        print(f"\nTraining commands for L1 skills in task '{args.task_name}':")
        if args.skip_first > 0:
            print(f"(Skipping first {args.skip_first} L1 skills)")
        print("=" * 80)

    # Compute repo root to help PYTHONPATH when needed
    repo_root = Path(__file__).resolve().parents[2]

    # Determine effective timesteps
    effective_steps = args.max_timesteps if args.max_timesteps is not None else args.steps

    for skill in l1_skills:
        if not args.run:
            print(f"\n# Skill: {skill} [L1]")
        # Hierarchical HPPO runs via orchestrator (train_l1.py)
        for seed in args.seeds_l1:
            cmd_hier = orch.build_simple_training_command(
                skill_name=skill,
                steps=effective_steps,
                record_video=False,  # don't write videos
                video_interval=args.video_interval,
                video_length=args.video_length,
                num_envs=args.num_envs,
                seed=seed,
                use_random_policies=False,
            )
            # Flat PPO baseline runs via IsaacLab train.py using -Flat env ids
            cmd_flat = _build_flat_l1_command(
                isaaclab_path=isaaclab_path,
                task_name=args.task_name,
                skill_name=skill,
                steps=effective_steps,
                num_envs=args.num_envs,
                seed=seed,
                flat_suffix=args.flat_suffix,
            )
            if not args.run:
                print(f"cd {isaaclab_path}")
                if args.no_success_saving:
                    print("export GENHRL_DISABLE_SUCCESS_SAVING=1")
                print("# Hierarchical (HPPO)")
                print(cmd_hier)
                print("# Flat PPO baseline")
                print(cmd_flat)
                print()
            else:
                # Snapshot existing runs
                before = _snapshot_top_level_runs(logs_dir)
                # Build env
                env = os.environ.copy()
                if args.no_success_saving:
                    env["GENHRL_DISABLE_SUCCESS_SAVING"] = "1"
                    # Ensure sitecustomize is discoverable
                    existing = env.get("PYTHONPATH", "")
                    path_str = str(repo_root)
                    if path_str not in existing.split(":"):
                        env["PYTHONPATH"] = f"{path_str}:{existing}" if existing else path_str
                # Execute hierarchical
                print(f"\n[RUN][HPPO] {skill} (seed={seed})")
                proc = subprocess.run(
                    cmd_hier,
                    shell=True,
                    executable="/bin/bash",
                    cwd=str(isaaclab_path),
                    env=env,
                )
                if proc.returncode != 0:
                    print(f"Run failed (HPPO) for {skill} (seed={seed}) with exit code {proc.returncode}")
                # Cleanup newly created runs if requested
                if args.no_save:
                    after = _snapshot_top_level_runs(logs_dir)
                    removed = _cleanup_new_runs(before, after)
                    if removed > 0:
                        print(f"[CLEANUP] Removed {removed} new log run(s) for {skill} (seed={seed}) [HPPO]")

                if args.run_baseline:
                    # Snapshot again before flat
                    before = _snapshot_top_level_runs(logs_dir)
                    # Execute flat PPO
                    print(f"\n[RUN][Flat] {skill} (seed={seed})")
                    proc = subprocess.run(
                        cmd_flat,
                        shell=True,
                        executable="/bin/bash",
                        cwd=str(isaaclab_path),
                        env=env,
                    )
                    if proc.returncode != 0:
                        print(f"Run failed (Flat) for {skill} (seed={seed}) with exit code {proc.returncode}")
                    # Cleanup newly created runs if requested
                    if args.no_save:
                        after = _snapshot_top_level_runs(logs_dir)
                        removed = _cleanup_new_runs(before, after)
                        if removed > 0:
                            print(f"[CLEANUP] Removed {removed} new log run(s) for {skill} (seed={seed}) [Flat]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
example usage
python3 testing/run_all_L1.py --task_name obstacle_course_seed42 --no_save --no_success_saving --run
"""


