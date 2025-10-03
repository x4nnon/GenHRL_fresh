import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime as dt


def _import_orchestrator():
    """Import TrainingOrchestrator with a robust fallback for local runs."""
    try:
        from genhrl.training.orchestrator import TrainingOrchestrator  # type: ignore
        return TrainingOrchestrator
    except Exception:
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from genhrl.training.orchestrator import TrainingOrchestrator  # type: ignore
        return TrainingOrchestrator


def _auto_detect_isaaclab_path() -> Path:
    """Best-effort detection of IsaacLab path.

    Priority:
    1) ENV ISAACLAB_PATH if it exists
    2) RepoRoot/IsaacLab (repo root inferred from this file)
    3) ./IsaacLab relative to current working directory
    """
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


def _format_gym_task_name(task_name: str, skill_name: str) -> str:
    """Mirror orchestrator's formatting to compute Gym task name suffix."""
    task_parts = task_name.split('_')
    skill_parts = skill_name.split('_')
    formatted_task = ''.join(part.capitalize() for part in task_parts)
    formatted_skill = ''.join(part.capitalize() for part in skill_parts)
    return f"{formatted_task}{formatted_skill}"


def _expected_env_id_for_l1_flat(task_name: str, skill_name: str, flat_suffix: str) -> str:
    suffix = _format_gym_task_name(task_name, skill_name)
    return f"Isaac-RobotComposite{suffix}-{flat_suffix}-v0" if flat_suffix else f"Isaac-RobotComposite{suffix}-v0"


def _compute_wandb_project_for_l1_flat(task_name: str, skill_name: str, flat_suffix: str) -> str:
    """Project naming as used by IsaacLab's skrl train.py for flat PPO.

    train.py computes: task_name = args_cli.task.replace("Isaac-Robot", "").replace("-v0", "")
    project = f"genhrl_{task_name}"
    """
    env_id = _expected_env_id_for_l1_flat(task_name, skill_name, flat_suffix)
    task_part = env_id.replace("Isaac-Robot", "").replace("-v0", "")
    return f"genhrl_{task_part}"


def _timestamp_from_run_name(name: str) -> Optional[float]:
    try:
        if not isinstance(name, str) or not name:
            return None
        m = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
        if not m:
            return None
        ts_str = m.group(1)
        dt_obj = dt.datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
        return float(dt_obj.timestamp())
    except Exception:
        return None


def _extract_series_from_wandb_run(run, metric_key: str, step_key_candidates: List[str]) -> Optional[Tuple[List[float], List[float]]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        print("Error: pandas is required. Install with: pip install pandas")
        return None

    try:
        for step_key in step_key_candidates:
            steps: List[float] = []
            values: List[float] = []
            has_any = False
            try:
                for row in run.scan_history(keys=[metric_key, step_key]):
                    if not isinstance(row, dict):
                        continue
                    val = row.get(metric_key)
                    if isinstance(val, (int, float)):
                        has_any = True
                        step_val = row.get(step_key)
                        steps.append(float(step_val if isinstance(step_val, (int, float)) else len(values)))
                        values.append(float(val))
            except Exception:
                try:
                    df = run.history(keys=[metric_key, step_key], pandas=True)
                    if df is not None and not df.empty and metric_key in df:
                        has_any = True
                        s = df[metric_key].astype(float).tolist()
                        if step_key in df:
                            t = df[step_key].astype(float).tolist()
                        else:
                            t = list(range(len(s)))
                        values.extend(s)
                        steps.extend(t)
                except Exception:
                    pass
            if has_any and values:
                return steps, values
        return None
    except Exception:
        return None


def _discover_metric_keys_for_reward(run, preferred_key: str, must_include_terms: List[str], max_rows: int = 500) -> List[str]:
    """Discover reward metric keys.

    Returns a list starting with preferred_key, optionally followed by discovered keys
    that contain all must_include_terms (case-insensitive) and contain 'reward'.
    """
    discovered: List[str] = []
    seen: set = set()
    try:
        rows_seen = 0
        for row in run.scan_history(page_size=1000):
            if not isinstance(row, dict):
                continue
            for k in row.keys():
                if not isinstance(k, str):
                    continue
                if k.startswith("_"):
                    continue
                if k in seen:
                    continue
                seen.add(k)
                kl = k.lower()
                if "reward" not in kl:
                    continue
                if all(term in kl for term in must_include_terms):
                    discovered.append(k)
            rows_seen += 1
            if rows_seen >= max_rows:
                break
    except Exception:
        try:
            summ = getattr(run, "summary", {}) or {}
            if isinstance(summ, dict):
                for k in summ.keys():
                    if isinstance(k, str):
                        kl = k.lower()
                        if "reward" in kl and all(term in kl for term in must_include_terms):
                            if k not in discovered:
                                discovered.append(k)
        except Exception:
            pass

    ordered: List[str] = []
    if preferred_key:
        ordered.append(preferred_key)
        for k in discovered:
            if k != preferred_key:
                ordered.append(k)
    else:
        ordered = discovered
    return ordered


def _pretty_skill_name(skill_name: str) -> str:
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", skill_name)
    spaced = spaced.replace("_", " ")
    return " ".join(w.capitalize() for w in spaced.split())


def list_l1_skills(orchestrator) -> List[str]:
    order = orchestrator.get_training_order()
    return [s for s in order if (not orchestrator.is_skill_primitive(s)) and orchestrator.get_skill_level(s) == 1]


def _plot_total_vs_instant_rewards(
    skill_name: str,
    steps_total: List[float], means_total: List[float], stds_total: Optional[List[float]],
    steps_instant: List[float], means_instant: List[float], stds_instant: Optional[List[float]],
    output_dir: Path, show: bool,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return None

    if not (steps_total and means_total) and not (steps_instant and means_instant):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax2 = ax.twinx()

    total_color = "#1f77b4"
    instant_color = "#d62728"

    handles = []
    labels = []

    if steps_total and means_total:
        h1, = ax.plot(steps_total, means_total, linewidth=3.0, color=total_color, label="Total reward (mean)")
        handles.append(h1)
        labels.append("Total reward (mean)")
        if stds_total and len(stds_total) == len(steps_total):
            lower = [m - s for m, s in zip(means_total, stds_total)]
            upper = [m + s for m, s in zip(means_total, stds_total)]
            ax.fill_between(steps_total, lower, upper, color=total_color, alpha=0.2)

    if steps_instant and means_instant:
        h2, = ax2.plot(steps_instant, means_instant, linewidth=3.0, color=instant_color, label="Instantaneous reward (mean)")
        handles.append(h2)
        labels.append("Instantaneous reward (mean)")
        if stds_instant and len(stds_instant) == len(steps_instant):
            lower = [m - s for m, s in zip(means_instant, stds_instant)]
            upper = [m + s for m, s in zip(means_instant, stds_instant)]
            ax2.fill_between(steps_instant, lower, upper, color=instant_color, alpha=0.18)

    ax.set_title(_pretty_skill_name(skill_name), fontsize=20, fontweight='bold')
    ax.set_xlabel("Global Step (x 4096)", fontsize=18)
    ax.set_ylabel("Total reward (mean)", fontsize=18, color=total_color)
    ax2.set_ylabel("Instantaneous reward (mean)", fontsize=18, color=instant_color)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16, colors=total_color)
    ax2.tick_params(axis='y', which='major', labelsize=16, colors=instant_color)
    if handles:
        ax.legend(handles, labels, fontsize=14, loc="best")
    fig.tight_layout()

    out_path = output_dir / f"{skill_name}_l1_flat_total_vs_instant_reward.png"
    try:
        fig.savefig(out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot W&B rewards for all L1 skills (Flat PPO baseline): Total vs Instant",
    )
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--isaaclab_path", default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--wandb_entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--wandb_api_key", default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (default: genhrl/analysis/plots_l1/<task>)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--step_keys",
        nargs="+",
        default=["global_step", "Global Step", "global-step", "_step", "step", "iteration"],
        help="Candidate step keys to try for x-axis (Global Step preferred)",
    )
    parser.add_argument("--num_recent_runs", type=int, default=5, help="Number of last (oldest) runs to use for averaging (default: 5)")
    parser.add_argument("--max_runs_per_project", type=int, default=20, help="Maximum number of runs to fetch per project (default: 20)")
    parser.add_argument("--flat_suffix", type=str, default="Flat", help="Env ID suffix for flat PPO composite registrations (default: Flat)")
    parser.add_argument("--skip_first", type=int, default=0, help="Skip the first N L1 skills before plotting")
    parser.add_argument(
        "--metric_total_key",
        default="Reward / Total reward (mean)",
        help="W&B metric key for total reward (mean)",
    )
    parser.add_argument(
        "--metric_instant_key",
        default="Reward / Instantaneous reward (mean)",
        help="W&B metric key for instantaneous reward (mean)",
    )

    args = parser.parse_args()

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
        return 0
    if args.skip_first > 0:
        if args.skip_first >= len(l1_skills):
            print(f"--skip_first={args.skip_first} >= number of L1 skills ({len(l1_skills)}). Nothing to do.")
            return 0
        l1_skills = l1_skills[args.skip_first:]

    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = Path(__file__).resolve().parents[0] / "plots_l1" / args.task_name

    try:
        import wandb  # type: ignore
    except Exception:
        print("Error: wandb is required. Install with: pip install wandb")
        return 1

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    try:
        api = wandb.Api(timeout=150)
    except Exception as e:
        print(f"Failed to initialize W&B API: {e}")
        return 1

    print(f"Found {len(l1_skills)} L1 skills")
    print(f"L1 skills: {l1_skills}")

    total_plots = 0
    for skill in l1_skills:
        print(f"\nProcessing skill: {skill}")

        # Resolve project for Flat (per-env project)
        flat_project = _compute_wandb_project_for_l1_flat(args.task_name, skill, args.flat_suffix)
        project_path = f"{args.wandb_entity}/{flat_project}"

        # Collect runs
        collected_total: List[Tuple[float, str, List[float], List[float]]] = []
        collected_instant: List[Tuple[float, str, List[float], List[float]]] = []

        try:
            runs = api.runs(project_path, per_page=args.max_runs_per_project)
            print(f"Flat: Trying project '{flat_project}' -> {len(runs)} run(s)")
        except Exception as e:
            print(f"Flat project not available '{flat_project}': {e}")
            runs = []

        for run in runs:
            print(f"  Flat run: {run.name}")
            # Prefer exact keys, but discover close matches too
            total_candidates = _discover_metric_keys_for_reward(
                run,
                preferred_key=args.metric_total_key,
                must_include_terms=["total"],
            )
            instant_candidates = _discover_metric_keys_for_reward(
                run,
                preferred_key=args.metric_instant_key,
                must_include_terms=["instant"],
            )

            # Extract series for total reward
            series_total = None
            for mk in total_candidates:
                series_total = _extract_series_from_wandb_run(run, mk, args.step_keys)
                if series_total:
                    break

            # Extract series for instantaneous reward
            series_instant = None
            for mk in instant_candidates:
                series_instant = _extract_series_from_wandb_run(run, mk, args.step_keys)
                if series_instant:
                    break

            # Timestamp for sorting (oldest first)
            label = getattr(run, "name", None) or getattr(run, "id", "run")
            ts = 0.0
            try:
                if isinstance(label, str):
                    ts_name = _timestamp_from_run_name(label)
                    if ts_name is not None:
                        ts = ts_name
                if ts == 0.0:
                    t = getattr(run, "updated_at", None) or getattr(run, "created_at", None)
                    if t is not None:
                        try:
                            ts = t.timestamp()
                        except Exception:
                            ts = 0.0
            except Exception:
                ts = 0.0

            if series_total:
                steps_t, values_t = series_total
                collected_total.append((ts, label, steps_t, values_t))
            if series_instant:
                steps_i, values_i = series_instant
                collected_instant.append((ts, label, steps_i, values_i))

        # Aggregate oldest K runs for total reward
        steps_total: List[float] = []
        means_total: List[float] = []
        stds_total: List[float] = []
        if collected_total:
            collected_total.sort(key=lambda t: t[0])
            top_k_total = collected_total[: args.num_recent_runs]
            step_to_vals_total: Dict[float, List[float]] = {}
            for _, _, steps, values in top_k_total:
                try:
                    for s, v in zip(steps, values):
                        if not isinstance(s, (int, float)) or not isinstance(v, (int, float)):
                            continue
                        step_to_vals_total.setdefault(float(s), []).append(float(v))
                except Exception:
                    continue
            if step_to_vals_total:
                steps_sorted = sorted(step_to_vals_total.keys())
                for s in steps_sorted:
                    vals = step_to_vals_total.get(s, [])
                    if not vals:
                        continue
                    means_total.append(sum(vals) / len(vals))
                    stds_total.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                steps_total = steps_sorted

        # Aggregate oldest K runs for instantaneous reward
        steps_instant: List[float] = []
        means_instant: List[float] = []
        stds_instant: List[float] = []
        if collected_instant:
            collected_instant.sort(key=lambda t: t[0])
            top_k_instant = collected_instant[: args.num_recent_runs]
            step_to_vals_instant: Dict[float, List[float]] = {}
            for _, _, steps, values in top_k_instant:
                try:
                    for s, v in zip(steps, values):
                        if not isinstance(s, (int, float)) or not isinstance(v, (int, float)):
                            continue
                        step_to_vals_instant.setdefault(float(s), []).append(float(v))
                except Exception:
                    continue
            if step_to_vals_instant:
                steps_sorted = sorted(step_to_vals_instant.keys())
                for s in steps_sorted:
                    vals = step_to_vals_instant.get(s, [])
                    if not vals:
                        continue
                    means_instant.append(sum(vals) / len(vals))
                    stds_instant.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                steps_instant = steps_sorted

        out_path = _plot_total_vs_instant_rewards(
            skill,
            steps_total, means_total, stds_total if stds_total else None,
            steps_instant, means_instant, stds_instant if stds_instant else None,
            output_base,
            args.show,
        )
        if out_path:
            print(f"Saved plot for {skill}: {out_path}")
            total_plots += 1
        else:
            print(f"No plot generated for {skill} (no valid series)")

    if total_plots == 0:
        print("No plots were generated. Ensure your W&B runs exist and contain the specified reward metrics.")
    else:
        print(f"Generated {total_plots} plot(s) under: {output_base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# how to run
# python3 genhrl/analysis/plot_wandb_flat_rewards_l1.py \
#   --task_name <YourTaskName> \
#   --robot G1 \
#   --wandb_entity <your_entity> \
#   --wandb_api_key <your_key> \
#   --num_recent_runs 5 \
#   --max_runs_per_project 20 \
#   --flat_suffix Flat \
#   --skip_first 0 \
#   --metric_total_key "Reward / Total reward (mean)" \
#   --metric_instant_key "Reward / Instantaneous reward (mean)"


