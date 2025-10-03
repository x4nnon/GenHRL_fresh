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


def list_l0_skills(orchestrator) -> list:
    """Return all primitive (L0) skills for the task in orchestrator."""
    order = orchestrator.get_training_order()
    return [s for s in order if orchestrator.is_skill_primitive(s)]


def _resolve_logs_dir(isaaclab_path: Path) -> Path:
    """Return the logs/skrl directory used by IsaacLab."""
    candidates = [isaaclab_path / "logs" / "skrl", Path("logs/skrl").resolve()]
    for c in candidates:
        if c.exists():
            return c
    # Default to inside isaaclab
    return candidates[0]


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


def _replace_script_path_and_append_normaliser(command: str, repo_root: Path, normaliser: str) -> str:
    """Rewrite base command to use our train_l0.py and append --normaliser flag.

    - Replaces IsaacLab primitive script path with absolute path to genhrl/scripts/train_l0.py
    - Appends --normaliser <mode>
    """
    # Absolute path to our train_l0.py
    train_l0_path = repo_root / "genhrl" / "scripts" / "train_l0.py"
    # Replace the script after "-p "
    # Original for primitives: "-p scripts/reinforcement_learning/skrl/train.py"
    replaced = command.replace("-p scripts/reinforcement_learning/skrl/train.py",
                               f"-p {train_l0_path}")
    # Append normaliser flag if not already present
    replaced = f"{replaced} --normaliser {normaliser}"
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run L0 skills across multiple seeds and normaliser modes for comparison",
    )
    parser.add_argument("--isaaclab_path", required=False, default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--steps", type=int, default=20000, help="Steps per run (default: 20000)")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments (default: 4096)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 234, 345, 456, 567], help="List of 5 seed values")
    parser.add_argument("--modes", type=str, nargs="+", default=["standard", "None"], choices=["custom", "standard", "None"], help="Normaliser modes to sweep")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing")
    parser.add_argument("--no_save", action="store_true", help="After each run, remove newly created log directories to avoid persisting files")
    parser.add_argument(
        "--no_success_saving",
        action="store_true",
        help="Disable saving success_states by patching save_success_state at import time",
    )
    parser.add_argument("--video_interval", type=int, default=2000, help="Video interval steps (unused when not recording)")
    parser.add_argument("--video_length", type=int, default=200, help="Video length steps (unused when not recording)")
    parser.add_argument("--skip_first", type=int, default=0, help="Skip the first N L0 skills before running")

    args = parser.parse_args()

    # Resolve IsaacLab path (use provided or auto-detect similar to run_all_L0)
    def _auto_detect_isaaclab_path() -> Path:
        env_path = os.environ.get("ISAACLAB_PATH")
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if p.exists():
                return p
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]
        candidate = repo_root / "IsaacLab"
        if candidate.exists():
            return candidate
        cwd_candidate = Path.cwd() / "IsaacLab"
        if cwd_candidate.exists():
            return cwd_candidate.resolve()
        return candidate

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

    l0_skills = list_l0_skills(orch)
    if not l0_skills:
        print(f"No L0 skills found for task: {args.task_name}")
        return 1

    if args.skip_first > 0:
        if args.skip_first >= len(l0_skills):
            print(f"--skip_first={args.skip_first} >= number of L0 skills ({len(l0_skills)}). Nothing to do.")
            return 0
        l0_skills = l0_skills[args.skip_first:]

    logs_dir = _resolve_logs_dir(isaaclab_path)

    # Compute repo root to build absolute path to train_l0.py for replacement
    repo_root = Path(__file__).resolve().parents[2]

    if not args.run:
        print(f"\nTraining commands for L0 skills in task '{args.task_name}' across seeds {args.seeds} and modes {args.modes}:")
        if args.skip_first > 0:
            print(f"(Skipping first {args.skip_first} L0 skills)")
        print("=" * 80)

    for skill in l0_skills:
        if not args.run:
            print(f"\n# Skill: {skill} [L0]")
        for mode in args.modes:
            for seed in args.seeds:
                base_cmd = orch.build_simple_training_command(
                    skill_name=skill,
                    steps=args.steps,
                    record_video=False,
                    video_interval=args.video_interval,
                    video_length=args.video_length,
                    num_envs=args.num_envs,
                    seed=seed,
                    use_random_policies=False,
                )
                cmd = _replace_script_path_and_append_normaliser(base_cmd, repo_root, mode)
                if not args.run:
                    print(f"cd {isaaclab_path}")
                    if args.no_success_saving:
                        print("export GENHRL_DISABLE_SUCCESS_SAVING=1")
                    # Also export normaliser mode to ensure detection even when Hydra clears sys.argv
                    print(f"export GENHRL_NORMALISER={mode}")
                    print(f"export GENHRL_REWARD_NORMALISER={mode}")
                    print(cmd)
                    print()
                else:
                    # Snapshot existing runs
                    before = _snapshot_top_level_runs(logs_dir)
                    # Build env
                    env = os.environ.copy()
                    # Pass normaliser via environment for robust detection in reward_normalizer
                    env["GENHRL_NORMALISER"] = mode
                    env["GENHRL_REWARD_NORMALISER"] = mode
                    if args.no_success_saving:
                        env["GENHRL_DISABLE_SUCCESS_SAVING"] = "1"
                        # Ensure sitecustomize is discoverable
                        existing = env.get("PYTHONPATH", "")
                        path_str = str(repo_root)
                        if path_str not in existing.split(":"):
                            env["PYTHONPATH"] = f"{path_str}:{existing}" if existing else path_str
                    # Execute
                    print(f"\n[RUN] {skill} (seed={seed}, mode={mode})")
                    proc = subprocess.run(
                        cmd,
                        shell=True,
                        executable="/bin/bash",
                        cwd=str(isaaclab_path),
                        env=env,
                    )
                    if proc.returncode != 0:
                        print(f"Run failed for {skill} (seed={seed}, mode={mode}) with exit code {proc.returncode}")
                    # Cleanup newly created runs if requested
                    if args.no_save:
                        after = _snapshot_top_level_runs(logs_dir)
                        removed = _cleanup_new_runs(before, after)
                        if removed > 0:
                            print(f"[CLEANUP] Removed {removed} new log run(s) for {skill} (seed={seed}, mode={mode})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


