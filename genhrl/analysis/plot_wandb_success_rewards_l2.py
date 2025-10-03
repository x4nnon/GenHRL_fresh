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


def _expected_env_id_for_l2_flat(task_name: str, skill_name: str, flat_suffix: str) -> str:
    suffix = _format_gym_task_name(task_name, skill_name)
    return f"Isaac-RobotComposite{suffix}-{flat_suffix}-v0" if flat_suffix else f"Isaac-RobotComposite{suffix}-v0"


def _compute_wandb_project_for_l2_flat(task_name: str, skill_name: str, flat_suffix: str) -> str:
    """Project naming as used by IsaacLab's skrl train.py for flat PPO.

    train.py computes: task_name = args_cli.task.replace("Isaac-Robot", "").replace("-v0", "")
    project = f"genhrl_{task_name}"
    """
    env_id = _expected_env_id_for_l2_flat(task_name, skill_name, flat_suffix)
    task_part = env_id.replace("Isaac-Robot", "").replace("-v0", "")
    return f"genhrl_{task_part}"


def _get_skill_base_dir(isaaclab_path: Path, task_name: str, robot: str, skill_name: str) -> Path:
    robot_folder = f"{robot}_generated"
    return (
        isaaclab_path
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based"
        / robot_folder
        / "skills"
        / task_name
        / "skills"
        / skill_name
    )


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


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


def _resolve_wandb_project_for_l2_hppo(isaaclab_path: Path, task_name: str, robot: str, skill_name: str) -> Optional[str]:
    """Resolve the per-skill HPPO W&B project from generated configs if available."""
    base = _get_skill_base_dir(isaaclab_path, task_name, robot, skill_name)
    tp = base / "policy" / "training_params.json"
    if tp.exists():
        obj = _read_json(tp)
        if isinstance(obj, dict):
            if isinstance(obj.get("agent_config"), dict):
                agent_cfg = obj["agent_config"]
                try:
                    wk = agent_cfg.get("agent", {}).get("experiment", {}).get("wandb_kwargs", {})
                    proj = wk.get("project")
                    if isinstance(proj, str) and proj.strip():
                        return proj.strip()
                except Exception:
                    pass
            wk = obj.get("wandb_kwargs")
            if isinstance(wk, dict):
                proj = wk.get("project")
                if isinstance(proj, str) and proj.strip():
                    return proj.strip()
    agents_dir = base / "agents"
    if agents_dir.exists() and agents_dir.is_dir():
        for cfg_file in agents_dir.glob("*.yaml"):
            y = _load_yaml(cfg_file)
            if not isinstance(y, dict):
                continue
            training = y.get("training") if isinstance(y.get("training"), dict) else y
            if isinstance(training, dict):
                wkwargs = training.get("wandb_kwargs")
                if isinstance(wkwargs, dict):
                    proj = wkwargs.get("project")
                    if isinstance(proj, str) and proj.strip():
                        return proj.strip()
    return None


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


def _extract_success_series_from_wandb_run(run, metric_key: str, step_key_candidates: List[str]) -> Optional[Tuple[List[float], List[float]]]:
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


def _discover_metric_keys_from_run(run, preferred_key: str, max_rows: int = 500) -> List[str]:
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

    ordered: List[str] = []
    if preferred_key in discovered:
        ordered.append(preferred_key)
        discovered = [k for k in discovered if k != preferred_key]
    else:
        ordered.append(preferred_key)
    for k in discovered:
        if k not in ordered:
            ordered.append(k)
    return ordered


def _find_max_episode_timesteps(run) -> float:
    max_val = 0.0
    try:
        possible_keys = [
            "Episode / Total timesteps (max)",
            "Episode/Total timesteps (max)",
            "episode_total_timesteps_max",
            "episode_timesteps_max",
        ]
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
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", skill_name)
    spaced = spaced.replace("_", " ")
    return " ".join(w.capitalize() for w in spaced.split())


def _plot_skill_series_dual(skill_name: str,
                            steps_hppo: List[float], means_hppo: List[float], stds_hppo: Optional[List[float]],
                            steps_flat: List[float], means_flat: List[float], stds_flat: Optional[List[float]],
                            output_dir: Path, show: bool) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return None

    if not (steps_hppo and means_hppo) and not (steps_flat and means_flat):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    if steps_hppo and means_hppo:
        ax.plot(steps_hppo, means_hppo, linewidth=3.0, color="#1f77b4", label="GenHRL + HPPO")
        if stds_hppo and len(stds_hppo) == len(steps_hppo):
            lower = [m - s for m, s in zip(means_hppo, stds_hppo)]
            upper = [m + s for m, s in zip(means_hppo, stds_hppo)]
            ax.fill_between(steps_hppo, lower, upper, color="#1f77b4", alpha=0.2)

    if steps_flat and means_flat:
        ax.plot(steps_flat, means_flat, linewidth=3.0, color="#d62728", label="Flat PPO baseline")
        if stds_flat and len(stds_flat) == len(steps_flat):
            lower = [m - s for m, s in zip(means_flat, stds_flat)]
            upper = [m + s for m, s in zip(means_flat, stds_flat)]
            ax.fill_between(steps_flat, lower, upper, color="#d62728", alpha=0.18)

    ax.set_title(_pretty_skill_name(skill_name), fontsize=20, fontweight='bold')
    ax.set_xlabel("Global Step (x 4096)", fontsize=18)
    ax.set_ylabel("Success Rate (%)", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"{skill_name}_l2_hppo_vs_flat_success_reward.png"
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


def list_l2_skills(orchestrator) -> List[str]:
    order = orchestrator.get_training_order()
    return [s for s in order if (not orchestrator.is_skill_primitive(s)) and orchestrator.get_skill_level(s) == 2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot W&B success reward for all L2 skills (GenHRL+HPPO vs Flat baseline)",
    )
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--isaaclab_path", default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--wandb_entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--wandb_api_key", default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (default: genhrl/analysis/plots_l2/<task>)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--metric_key", default="Info/Episode_Reward/success_reward", help="W&B metric key to plot (y-axis)")
    parser.add_argument("--step_keys", nargs="+", default=["global_step", "Global Step", "global-step", "_step", "step", "iteration"], help="Candidate step keys to try for x-axis (Global Step preferred)")
    parser.add_argument("--num_recent_runs", type=int, default=5, help="Number of last (oldest) runs to use for averaging per method (default: 5)")
    parser.add_argument("--max_runs_per_project", type=int, default=20, help="Maximum number of runs to fetch per project (default: 20)")
    parser.add_argument("--flat_suffix", type=str, default="Flat", help="Env ID suffix for flat PPO composite registrations (default: Flat)")
    parser.add_argument("--skip_first", type=int, default=0, help="Skip the first N L2 skills before plotting")

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

    l2_skills = list_l2_skills(orch)
    if not l2_skills:
        print(f"No L2 skills found for task: {args.task_name}")
        return 0
    if args.skip_first > 0:
        if args.skip_first >= len(l2_skills):
            print(f"--skip_first={args.skip_first} >= number of L2 skills ({len(l2_skills)}). Nothing to do.")
            return 0
        l2_skills = l2_skills[args.skip_first:]

    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = Path(__file__).resolve().parents[0] / "plots_l2" / args.task_name

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

    print(f"Found {len(l2_skills)} L2 skills")
    print(f"L2 skills: {l2_skills}")

    total_plots = 0
    for skill in l2_skills:
        print(f"\nProcessing skill: {skill}")

        # Resolve project candidates for HPPO (per-skill project) and Flat (per-env project)
        hppo_candidates: List[str] = []
        detected_hppo = _resolve_wandb_project_for_l2_hppo(isaaclab_path, args.task_name, args.robot, skill)
        if detected_hppo:
            hppo_candidates.append(detected_hppo)
        formatted_skill = ''.join(part.capitalize() for part in skill.split('_'))
        hppo_candidates.extend([
            f"genhrl_{skill}",
            f"genhrl_{formatted_skill}",
        ])
        hppo_candidates = list(dict.fromkeys(hppo_candidates))

        flat_candidates: List[str] = []
        computed_flat = _compute_wandb_project_for_l2_flat(args.task_name, skill, args.flat_suffix)
        flat_candidates.append(computed_flat)
        flat_candidates = list(dict.fromkeys(flat_candidates))

        # Collect runs for HPPO and Flat
        collected_hppo: List[Tuple[float, str, List[float], List[float]]] = []
        collected_flat: List[Tuple[float, str, List[float], List[float]]] = []
        max_episode_timesteps = 0.0

        # Fetch HPPO runs from first available matching project
        for proj in hppo_candidates:
            project_path = f"{args.wandb_entity}/{proj}"
            try:
                runs = api.runs(project_path, per_page=args.max_runs_per_project)
                print(f"HPPO: Trying project '{proj}' -> {len(runs)} run(s)")
            except Exception as e:
                print(f"HPPO project not available '{proj}': {e}")
                continue
            for run in runs:
                print(f"  HPPO run: {run.name}")
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
                    collected_hppo.append((ts, label, steps, values))
                except Exception:
                    continue
            if collected_hppo:
                break

        # Fetch Flat runs
        for proj in flat_candidates:
            project_path = f"{args.wandb_entity}/{proj}"
            try:
                runs = api.runs(project_path, per_page=args.max_runs_per_project)
                print(f"Flat: Trying project '{proj}' -> {len(runs)} run(s)")
            except Exception as e:
                print(f"Flat project not available '{proj}': {e}")
                continue
            for run in runs:
                print(f"  Flat run: {run.name}")
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
                    collected_flat.append((ts, label, steps, values))
                except Exception:
                    continue
            if collected_flat:
                break

        # Use default scaling if none found
        if max_episode_timesteps == 0.0:
            max_episode_timesteps = 1000.0
            print(f"Warning: No max episode timesteps found for {skill}, using default scaling")
        else:
            print(f"Found max episode timesteps for {skill}: {max_episode_timesteps}")

        # Aggregate recent runs for HPPO
        steps_hppo: List[float] = []
        means_hppo: List[float] = []
        stds_hppo: List[float] = []
        if collected_hppo:
            collected_hppo.sort(key=lambda t: t[0])
            top_k = collected_hppo[: args.num_recent_runs]
            step_to_vals: Dict[float, List[float]] = {}
            for _, _, steps, values in top_k:
                try:
                    for s, v in zip(steps, values):
                        if not isinstance(s, (int, float)) or not isinstance(v, (int, float)):
                            continue
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
                    means_hppo.append(sum(vals) / len(vals))
                    stds_hppo.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                steps_hppo = steps_sorted

        # Aggregate recent runs for Flat
        steps_flat: List[float] = []
        means_flat: List[float] = []
        stds_flat: List[float] = []
        if collected_flat:
            collected_flat.sort(key=lambda t: t[0])
            top_k = collected_flat[: args.num_recent_runs]
            step_to_vals: Dict[float, List[float]] = {}
            for _, _, steps, values in top_k:
                try:
                    for s, v in zip(steps, values):
                        if not isinstance(s, (int, float)) or not isinstance(v, (int, float)):
                            continue
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
                    means_flat.append(sum(vals) / len(vals))
                    stds_flat.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                steps_flat = steps_sorted

        out_path = _plot_skill_series_dual(
            skill,
            steps_hppo, means_hppo, stds_hppo if stds_hppo else None,
            steps_flat, means_flat, stds_flat if stds_flat else None,
            output_base,
            args.show,
        )
        if out_path:
            print(f"Saved plot for {skill}: {out_path}")
            total_plots += 1
        else:
            print(f"No plot generated for {skill} (no valid series)")

    if total_plots == 0:
        print("No plots were generated. Ensure your W&B runs exist and contain the metric 'Info/Episode_Reward/success_reward'.")
    else:
        print(f"Generated {total_plots} plot(s) under: {output_base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# how to run
# python3 genhrl/analysis/plot_wandb_success_rewards_l2.py \
#   --task_name <YourTaskName> \
#   --robot G1 \
#   --wandb_entity <your_entity> \
#   --wandb_api_key <your_key> \
#   --num_recent_runs 5 \
#   --max_runs_per_project 20 \
#   --flat_suffix Flat \
#   --skip_first 0


