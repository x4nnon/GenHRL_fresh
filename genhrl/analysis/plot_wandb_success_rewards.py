import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import re
import datetime as dt


def _get_skill_library_path(isaaclab_path: Path, task_name: str, robot: str) -> Path:
    """Return path to the task's skill_library.json without importing orchestrator."""
    robot_folder = f"{robot}_generated"
    return (
        isaaclab_path
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        / robot_folder
        / "skills"
        / task_name
        / "skill_library.json"
    )


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


def _format_gym_task_name(task_name: str, skill_name: str) -> str:
    """Mirror orchestrator's formatting to compute Gym task name suffix."""
    task_parts = task_name.split('_')
    skill_parts = skill_name.split('_')
    formatted_task = ''.join(part.capitalize() for part in task_parts)
    formatted_skill = ''.join(part.capitalize() for part in skill_parts)
    return f"{formatted_task}{formatted_skill}"


def _expected_env_id_for_l0(task_name: str, skill_name: str) -> str:
    suffix = _format_gym_task_name(task_name, skill_name)
    return f"Isaac-RobotFlat{suffix}-v0"


def _compute_wandb_project_for_l0(task_name: str, skill_name: str) -> str:
    """Replicate the project naming used by training for L0 runs.

    IsaacLab's skrl train.py sets:
      task_name = args_cli.task.replace("Isaac-Robot", "").replace("-v0", "")
      project = f"genhrl_{task_name}"
    where args_cli.task is our gym id (Isaac-RobotFlat<Formatted>-v0).
    """
    env_id = _expected_env_id_for_l0(task_name, skill_name)
    task_part = env_id.replace("Isaac-Robot", "").replace("-v0", "")
    return f"genhrl_{task_part}"


def _load_l0_skills(isaaclab_path: Path, task_name: str, robot: str) -> List[str]:
    """Load L0 skills from the skill library JSON directly."""
    lib_path = _get_skill_library_path(isaaclab_path, task_name, robot)
    if not lib_path.exists():
        raise FileNotFoundError(f"Skill library not found: {lib_path}")
    with open(lib_path, 'r') as f:
        lib = json.load(f)
    skills = lib.get("skills", {})
    if not isinstance(skills, dict):
        return []
    l0 = [name for name, info in skills.items() if isinstance(info, dict) and info.get("is_primitive", False)]
    # Stable ordering
    return sorted(l0)


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("Error: PyYAML is required. Install with: pip install pyyaml")
        return None
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _resolve_wandb_project_for_skill(isaaclab_path: Path, task_name: str, robot: str, skill_name: str) -> Optional[str]:
    """Find the W&B project name for a specific skill by inspecting generated configs."""
    robot_folder = f"{robot}_generated"
    base = (
        isaaclab_path
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        / robot_folder
        / "skills"
        / task_name
        / "skills"
        / skill_name
    )
    # 1) training_params.json (created during/after training)
    tp = base / "policy" / "training_params.json"
    if tp.exists():
        obj = _read_json(tp)
        if isinstance(obj, dict):
            wkwargs = obj.get("wandb_kwargs")
            if isinstance(wkwargs, dict):
                proj = wkwargs.get("project")
                if isinstance(proj, str) and proj.strip():
                    return proj.strip()
    # 2) agents YAML (pre-training config)
    agents_dir = base / "agents"
    if agents_dir.exists() and agents_dir.is_dir():
        for cfg_file in agents_dir.glob("*.yaml"):
            y = _load_yaml(cfg_file)
            if not isinstance(y, dict):
                continue
            # Look under common keys
            training = y.get("training") if isinstance(y.get("training"), dict) else y
            if isinstance(training, dict):
                wkwargs = training.get("wandb_kwargs")
                if isinstance(wkwargs, dict):
                    proj = wkwargs.get("project")
                    if isinstance(proj, str) and proj.strip():
                        return proj.strip()
    return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _timestamp_from_run_name(name: str) -> Optional[float]:
    """Parse leading timestamp from run name like 'YYYY-MM-DD_HH-MM-SS_rest'.

    Returns epoch seconds if parsed, else None.
    """
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


def _wandb_run_matches(run, expected_env_id: str, skill_name: str, task_name: str) -> bool:
    """Heuristically determine if a W&B run corresponds to the given env/skill."""
    try:
        cfg = getattr(run, "config", {}) or {}
        name = getattr(run, "name", "") or ""
        tags = getattr(run, "tags", []) or []
        # 1) Config fields
        for key in ["env_id", "task", "env", "gym_task", "task_id"]:
            val = cfg.get(key)
            if isinstance(val, str) and expected_env_id in val:
                return True
        # 2) Tags include env id or skill name or task
        tag_str = " ".join(str(t) for t in tags)
        if expected_env_id in tag_str or skill_name.lower() in tag_str.lower() or task_name.lower() in tag_str.lower():
            return True
        # 3) Name hints
        if isinstance(name, str) and (expected_env_id in name or skill_name.lower() in name.lower()):
            return True
    except Exception:
        return False
    return False


def _extract_success_series_from_wandb_run(run, metric_key: str, step_key_candidates: List[str]) -> Optional[Tuple[List[float], List[float]]]:
    """Extract (steps, metric) from a W&B run history via API.

    Tries the provided step_key_candidates in order (e.g., ["global_step", "_step", "step", "iteration"]).
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        print("Error: pandas is required. Install with: pip install pandas")
        return None

    try:
        # Prefer scan_history for large runs
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
                # Fallback to history DataFrame
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


def _discover_metric_keys_from_run(run, preferred_key: str, max_rows: int = 500) -> List[str]:
    """Inspect a subset of history to discover candidate metric keys.

    Returns a list ordered by preference, starting with preferred_key if present,
    followed by discovered keys containing both 'success' and 'reward' (case-insensitive),
    and exact matches for 'success_reward' variants.
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
                if k not in seen:
                    seen.add(k)
                    kl = k.lower()
                    if ("success" in kl and "reward" in kl) or kl.endswith("success_reward"):
                        discovered.append(k)
            rows_seen += 1
            if rows_seen >= max_rows:
                break
    except Exception:
        # Fallback: try summary keys (less reliable for timeseries)
        try:
            summ = getattr(run, "summary", {}) or {}
            if isinstance(summ, dict):
                for k in summ.keys():
                    if isinstance(k, str):
                        kl = k.lower()
                        if ("success" in kl and "reward" in kl) or kl.endswith("success_reward"):
                            if k not in discovered:
                                discovered.append(k)
        except Exception:
            pass

    # Ensure preferred is first if discovered
    ordered: List[str] = []
    if preferred_key in discovered:
        ordered.append(preferred_key)
        discovered = [k for k in discovered if k != preferred_key]
    else:
        ordered.append(preferred_key)
    # Deduplicate while preserving order
    for k in discovered:
        if k not in ordered:
            ordered.append(k)
    return ordered


def _find_max_episode_timesteps(run) -> float:
    """Find the maximum episode length using common metric keys from history or summary."""
    max_val = 0.0
    try:
        possible_keys = [
            "Episode / Total timesteps (max)",
            "Episode/Total timesteps (max)",
            "episode_total_timesteps_max",
            "episode_timesteps_max",
        ]
        # History scan for each candidate key (uses key filtering when available)
        for key in possible_keys:
            try:
                for row in run.scan_history(keys=[key], page_size=1000):
                    if not isinstance(row, dict):
                        continue
                    val = row.get(key)
                    if isinstance(val, (int, float)) and val > max_val:
                        max_val = float(val)
            except Exception:
                continue
        # Fallback to run summary if history didn't contain a value
        if max_val == 0.0:
            try:
                summ = getattr(run, "summary", {}) or {}
                if isinstance(summ, dict):
                    for key in possible_keys:
                        val = summ.get(key)
                        if isinstance(val, (int, float)) and val > max_val:
                            max_val = float(val)
            except Exception:
                pass
    except Exception:
        pass
    return max_val


def _pretty_skill_name(skill_name: str) -> str:
    # Insert spaces before camel-case capitals, then replace underscores, then title-case
    import re
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", skill_name)
    spaced = spaced.replace("_", " ")
    return " ".join(w.capitalize() for w in spaced.split())


def _plot_skill_series(skill_name: str, steps: List[float], means: List[float], stds: Optional[List[float]], output_dir: Path, show: bool) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return None

    if not steps or not means:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, means, linewidth=3.5, color="#1f77b4")
    if stds and len(stds) == len(steps):
        import numpy as np  # type: ignore
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(steps, lower, upper, color="#1f77b4", alpha=0.2)
    ax.set_title(_pretty_skill_name(skill_name), fontsize=20, fontweight='bold')
    ax.set_xlabel("Global Step (x 4096)", fontsize=18)
    ax.set_ylabel("Success Rate (%)", fontsize=18)
    ax.grid(True, alpha=0.3)
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    # No legend
    fig.tight_layout()
    out_path = output_dir / f"{skill_name}_success_reward.png"
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
        description="Plot W&B success reward for all L0 skills in a task",
    )
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--isaaclab_path", default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--wandb_entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--wandb_project", default=None, help="W&B project name (optional; auto-detected per skill if omitted)")
    parser.add_argument("--wandb_api_key", default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (default: genhrl/analysis/plots/<task>)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--metric_key", default="Info/Episode_Reward/success_reward", help="W&B metric key to plot (y-axis)")
    parser.add_argument("--step_keys", nargs="+", default=["global_step", "Global Step", "global-step", "_step", "step", "iteration"], help="Candidate step keys to try for x-axis (Global Step preferred)")
    parser.add_argument("--num_recent_runs", type=int, default=5, help="Number of most recent runs to use for averaging (default: 5)")
    parser.add_argument("--max_runs_per_project", type=int, default=20, help="Maximum number of runs to fetch per project (default: 20)")

    args = parser.parse_args()

    isaaclab_path = Path(args.isaaclab_path).expanduser().resolve() if args.isaaclab_path else _auto_detect_isaaclab_path()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        print("Hint: Set ISAACLAB_PATH env var or pass --isaaclab_path /path/to/IsaacLab")
        return 1

    # Collect L0 skills only (from skill_library.json)
    try:
        l0_skills = _load_l0_skills(isaaclab_path, args.task_name, args.robot)
    except Exception as e:
        print(f"Error loading skills: {e}")
        return 1

    if not l0_skills:
        print(f"No L0 skills found for task: {args.task_name}")
        return 0

    # Determine where to save plots
    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = Path(__file__).resolve().parents[0] / "plots" / args.task_name

    # Initialize W&B API
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

    # Iterate skills and collect series from matching runs
    total_plots = 0
    print(f"Found {len(l0_skills)} L0 skills")
    print(f"L0 skills: {l0_skills}")
    for skill in l0_skills:
        # Find the maximum "Episode / Total timesteps (max)" value across all runs for this skill
        max_episode_timesteps = 0.0
        
        expected_env = _expected_env_id_for_l0(args.task_name, skill)
        # Collect candidate series with timestamps
        collected: List[Tuple[float, str, List[float], List[float]]] = []

        # Build candidate project names for this skill
        candidates: List[str] = []
        if args.wandb_project:
            candidates.append(args.wandb_project)
        detected = _resolve_wandb_project_for_skill(isaaclab_path, args.task_name, args.robot, skill)
        if detected and detected not in candidates:
            candidates.append(detected)
        computed = _compute_wandb_project_for_l0(args.task_name, skill)
        if computed and computed not in candidates:
            candidates.append(computed)
        # Additional common fallbacks
        formatted_skill = ''.join(part.capitalize() for part in skill.split('_'))
        candidates.extend([
            f"genhrl_{formatted_skill}",  # used by hierarchical scripts
            f"genhrl_{args.task_name}",   # rarely used fallback
        ])
        # Deduplicate
        seen_names = set()
        candidates = [c for c in candidates if not (c in seen_names or seen_names.add(c))]

        runs_collected = False
        for proj in candidates:
            project_path = f"{args.wandb_entity}/{proj}"
            try:
                runs = api.runs(project_path, per_page=args.max_runs_per_project)  # Limit runs per page
                print(f"Trying project '{proj}' -> {len(runs)} run(s) (limited to {args.max_runs_per_project})")
            except Exception as e:
                print(f"Project not available '{proj}': {e}")
                continue


            # Second pass: accept all runs in the project (project is already per-skill)
            for run in runs:
                print(f"  processing run: {run.name}")
                run_max = _find_max_episode_timesteps(run)
                if run_max > max_episode_timesteps:
                    max_episode_timesteps = run_max
                try:
                    metric_candidates = _discover_metric_keys_from_run(run, args.metric_key)
                    series = None
                    for mk in metric_candidates:
                        series = _extract_success_series_from_wandb_run(run, mk, args.step_keys)
                        if series:
                            break
                    if not series:
                        continue
                    steps, values = series
                    label = getattr(run, "name", None) or getattr(run, "id", "run")
                    ts = 0.0
                    try:
                        # Prefer timestamp encoded in run name if present
                        if isinstance(label, str):
                            ts_name = _timestamp_from_run_name(label)
                            if ts_name is not None:
                                ts = ts_name
                        # Fallback to W&B timestamps
                        if ts == 0.0:
                            t = getattr(run, "updated_at", None) or getattr(run, "created_at", None)
                            if t is not None:
                                try:
                                    ts = t.timestamp()
                                except Exception:
                                    ts = 0.0
                    except Exception:
                        ts = 0.0
                    collected.append((ts, label, steps, values))
                    print(f"  collected run: {label}")
                    print(f"  timestamp: {ts}")
                    runs_collected = True
                except Exception:
                    continue
            if runs_collected:
                break


        # Use only the most recent N runs (by timestamp), compute mean and std series on Global Step
        mean_steps: List[float] = []
        mean_values: List[float] = []
        std_values: List[float] = []
        if collected:
            collected.sort(key=lambda t: t[0], reverse=True)
            top_k = collected[:args.num_recent_runs]
            
            # Use default scaling if no max found
            if max_episode_timesteps == 0.0:
                max_episode_timesteps = 1000.0  # Default fallback
                print(f"Warning: No max episode timesteps found for {skill}, using default scaling")
            else:
                print(f"Found max episode timesteps for {skill}: {max_episode_timesteps}")
            
            # Aggregate by step using available values and apply scaling
            step_to_vals: Dict[float, List[float]] = {}
            for _, _, steps, values in top_k:
                try:
                    for s, v in zip(steps, values):
                        if not isinstance(s, (int, float)) or not isinstance(v, (int, float)):
                            continue
                        # Scale the success rate: success_rate * max_value/1000, then convert to percentage
                        scaled_v = float(v) * max_episode_timesteps / 1000.0 * 100.0
                        step_to_vals.setdefault(float(s), []).append(scaled_v)
                except Exception:
                    continue
            if step_to_vals:
                steps_sorted = sorted(step_to_vals.keys())
                for s in steps_sorted:
                    vals = step_to_vals.get(s, [])
                    if not vals:
                        continue
                    mean_values.append(sum(vals) / len(vals))
                    std_values.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                mean_steps = steps_sorted

        if mean_steps and mean_values:
            out_path = _plot_skill_series(skill, mean_steps, mean_values, std_values, output_base, args.show)
            if out_path:
                print(f"Saved plot for {skill}: {out_path}")
                total_plots += 1
            else:
                print(f"No plot generated for {skill} (no valid series)")
        else:
            print(f"No matching W&B runs found for skill: {skill}")

    if total_plots == 0:
        print("No plots were generated. Ensure your W&B runs are stored locally and contain the metric 'Info/Episode_Reward/success_reward'.")
    else:
        print(f"Generated {total_plots} plot(s) under: {output_base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# example usage:

# python3 genhrl/analysis/plot_wandb_success_rewards.py   --task_name move_three_objects_seed42   --robot G1   --wandb_api_key a949a2366eeaf95169770ca10c361c6dea1e03d7 --wandb_entity tpcannon --num_recent_runs 4