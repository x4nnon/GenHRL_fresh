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


def _resolve_wandb_project_for_l1_hppo(isaaclab_path: Path, task_name: str, robot: str, skill_name: str) -> Optional[str]:
    """Resolve the per-skill HPPO W&B project from generated configs if available."""
    base = _get_skill_base_dir(isaaclab_path, task_name, robot, skill_name)
    tp = base / "policy" / "training_params.json"
    if tp.exists():
        obj = _read_json(tp)
        if isinstance(obj, dict):
            # Some exports embed full agent config under 'agent_config'
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


def _discover_metric_keys_from_run(run, preferred_key: str, max_rows: int = 200, page_size: int = 400) -> List[str]:
    """Discover plausible success reward metric keys quickly.

    Strategy:
    1) Prefer preferred_key
    2) Check run.summary keys (fast)
    3) Light scan of history up to max_rows with limited page_size
    """
    discovered: List[str] = []
    # Step 1: Always try preferred first
    ordered: List[str] = [preferred_key]

    # Step 2: Summary keys
    try:
        summ = getattr(run, "summary", {}) or {}
        if isinstance(summ, dict):
            for k in summ.keys():
                if isinstance(k, str):
                    kl = k.lower()
                    if ("success" in kl and "reward" in kl) or kl.endswith("success_reward"):
                        if k not in discovered and k != preferred_key:
                            discovered.append(k)
    except Exception:
        pass

    # Step 3: Light scan of history
    try:
        rows_seen = 0
        for row in run.scan_history(page_size=page_size):
            if not isinstance(row, dict):
                continue
            for k in row.keys():
                if not isinstance(k, str):
                    continue
                if k.startswith("_"):
                    continue
                kl = k.lower()
                if ("success" in kl and "reward" in kl) or kl.endswith("success_reward"):
                    if k not in discovered and k != preferred_key:
                        discovered.append(k)
            rows_seen += 1
            if rows_seen >= max_rows:
                break
    except Exception:
        pass

    for k in discovered:
        if k not in ordered:
            ordered.append(k)
    return ordered


def _find_max_episode_timesteps(run, summary_only: bool = False) -> float:
    max_val = 0.0
    try:
        possible_keys = [
            "Episode / Total timesteps (max)",
            "Episode/Total timesteps (max)",
            "episode_total_timesteps_max",
            "episode_timesteps_max",
        ]
        # Fast path: summary only
        try:
            summ = getattr(run, "summary", {}) or {}
            if isinstance(summ, dict):
                for key in possible_keys:
                    val = summ.get(key)
                    if isinstance(val, (int, float)) and val > max_val:
                        max_val = float(val)
        except Exception:
            pass

        if not summary_only and max_val == 0.0:
            for key in possible_keys:
                try:
                    for row in run.scan_history(keys=[key], page_size=400):
                        if not isinstance(row, dict):
                            continue
                        val = row.get(key)
                        if isinstance(val, (int, float)) and val > max_val:
                            max_val = float(val)
                except Exception:
                    continue
    except Exception:
        pass
    return max_val


def _pretty_skill_name(skill_name: str) -> str:
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", skill_name)
    spaced = spaced.replace("_", " ")
    return " ".join(w.capitalize() for w in spaced.split())


def _detect_group_label(run) -> Optional[str]:
    """Return a displayable group label for a run.

    Priority:
    1) run.group if present
    2) tag like 'group:<name>'
    3) config fields that look like group descriptors
    """
    try:
        grp = getattr(run, 'group', None)
        if isinstance(grp, str) and grp.strip():
            return grp.strip()

        tags = getattr(run, 'tags', []) or []
        for t in tags:
            try:
                ts = str(t)
            except Exception:
                continue
            if not ts:
                continue
            m = re.match(r"^group[:=](.+)$", ts.strip(), re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val:
                    return val

        cfg = getattr(run, 'config', {}) or {}
        if isinstance(cfg, dict):
            for key in [
                'group', 'group_name', 'ablation', 'variant', 'label', 'setting',
                'experiment_group', 'exp_group', 'normalisation', 'normalization',
            ]:
                val = cfg.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, bool):
                    return f"{key}={val}"
                if isinstance(val, (int, float)):
                    return f"{key}={val}"
    except Exception:
        return None
    return None


def _plot_overlaid_groups(skill_name: str, series_by_group: Dict[str, Tuple[List[float], List[float], Optional[List[float]]]], output_dir: Path, show: bool) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return None

    if not series_by_group:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Assign colors dynamically from tab10/tab20 cycles
    import itertools
    try:
        from matplotlib import cm
        color_cycle = itertools.cycle(cm.tab10.colors + cm.tab20.colors)
    except Exception:
        color_cycle = itertools.cycle([None])

    color_map: Dict[str, Optional[Tuple[float, float, float]]] = {}

    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(12, 8))
        for group_name, (steps, means, ses) in series_by_group.items():
            if group_name not in color_map:
                color_map[group_name] = next(color_cycle)
            color = color_map[group_name]
            ax.plot(steps, means, linewidth=3.0, label=group_name, color=color)
            if ses and len(ses) == len(steps):
                lower = [m - s for m, s in zip(means, ses)]
                upper = [m + s for m, s in zip(means, ses)]
                ax.fill_between(steps, lower, upper, color=color, alpha=0.18)

        ax.set_title(_pretty_skill_name(skill_name), fontsize=20, fontweight='bold')
        ax.set_xlabel("Global Step (x 4096)", fontsize=18)
        ax.set_ylabel("Success Rate (%)", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=14)
        fig.tight_layout()

        out_path = output_dir / f"{skill_name}_l1_groups_success_reward.png"
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
    except Exception:
        return None


def list_l1_skills(orchestrator) -> List[str]:
    order = orchestrator.get_training_order()
    return [s for s in order if (not orchestrator.is_skill_primitive(s)) and orchestrator.get_skill_level(s) == 1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot W&B success reward for L1 skills, overlaid by W&B run groups (no flat baseline)",
    )
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--isaaclab_path", default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--wandb_entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--wandb_api_key", default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (default: genhrl/analysis/plots_l1_groups/<task>)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--metric_key", default="Info/Episode_Reward/success_reward", help="W&B metric key to plot (y-axis)")
    parser.add_argument("--step_keys", nargs="+", default=["global_step", "Global Step", "global-step", "_step", "step", "iteration"], help="Candidate step keys to try for x-axis (Global Step preferred)")
    parser.add_argument("--num_recent_runs_per_group", type=int, default=5, help="Number of most recent runs to average per group (default: 5)")
    parser.add_argument("--max_runs_per_project", type=int, default=30, help="Maximum number of runs to fetch per project (default: 30)")
    parser.add_argument("--skip_first", type=int, default=0, help="Skip the first N L1 skills before plotting")
    # Performance and verbosity controls
    parser.add_argument("--fast", action="store_true", help="Fast mode: use summary-only scans where possible")
    parser.add_argument("--history_page_size", type=int, default=400, help="History scan page size (default: 400)")
    parser.add_argument("--discover_max_rows", type=int, default=200, help="Max history rows scanned during metric discovery (default: 200)")
    parser.add_argument("--log_progress_every", type=int, default=1, help="Print a progress message every N runs (default: 5)")
    parser.add_argument("--include_groups", nargs="*", default=None, help="Only include groups matching any of these regex patterns (case-insensitive)")
    parser.add_argument("--exclude_groups", nargs="*", default=None, help="Exclude groups matching any of these regex patterns (case-insensitive)")
    parser.add_argument("--max_groups", type=int, default=10, help="Maximum number of distinct groups to plot per skill (default: 10)")

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
        output_base = Path(__file__).resolve().parents[0] / "plots_l1_groups" / args.task_name

    try:
        import wandb  # type: ignore
    except Exception:
        print("Error: wandb is required. Install with: pip install wandb")
        return 1

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    try:
        api = wandb.Api(timeout=180)
    except Exception as e:
        print(f"Failed to initialize W&B API: {e}")
        return 1

    print(f"Found {len(l1_skills)} L1 skills")
    print(f"L1 skills: {l1_skills}")

    total_plots = 0
    for skill in l1_skills:
        print(f"\nProcessing skill: {skill}")

        # Resolve project candidates for HPPO (per-skill project)
        hppo_candidates: List[str] = []
        detected_hppo = _resolve_wandb_project_for_l1_hppo(isaaclab_path, args.task_name, args.robot, skill)
        if detected_hppo:
            hppo_candidates.append(detected_hppo)
        # Common naming patterns (support snake_case and CamelCase)
        formatted_skill = ''.join(part.capitalize() for part in skill.split('_'))
        hppo_candidates.extend([
            f"genhrl_{skill}",
            f"genhrl_{formatted_skill}",
        ])
        hppo_candidates = list(dict.fromkeys(hppo_candidates))

        # Collect runs grouped by group label within first available project
        grouped: Dict[str, List[Tuple[float, str, List[float], List[float]]]] = {}
        max_episode_timesteps = 0.0

        for proj in hppo_candidates:
            project_path = f"{args.wandb_entity}/{proj}"
            try:
                runs = api.runs(project_path, per_page=args.max_runs_per_project)
                print(f"Trying project '{proj}' -> {len(runs)} run(s)")
            except Exception as e:
                print(f"Project not available '{proj}': {e}")
                continue

            processed = 0
            for run in runs:
                processed += 1
                if processed % max(1, int(args.log_progress_every)) == 0:
                    print(f"  Scanned {processed}/{len(runs)} runs...")
                try:
                    grp = _detect_group_label(run) or "Ungrouped"
                    # Filter groups if include/exclude provided
                    def _matches_any(patterns: Optional[List[str]], text: str) -> bool:
                        if not patterns:
                            return False
                        for p in patterns:
                            try:
                                if re.search(p, text, re.IGNORECASE):
                                    return True
                            except re.error:
                                # Invalid regex; fallback to substring
                                if p.lower() in text.lower():
                                    return True
                        return False

                    if args.include_groups and not _matches_any(args.include_groups, grp):
                        continue
                    if args.exclude_groups and _matches_any(args.exclude_groups, grp):
                        continue

                    run_max = _find_max_episode_timesteps(run, summary_only=bool(args.fast))
                    if run_max > max_episode_timesteps:
                        max_episode_timesteps = run_max

                    metric_candidates = _discover_metric_keys_from_run(
                        run,
                        args.metric_key,
                        max_rows=int(args.discover_max_rows),
                        page_size=int(args.history_page_size),
                    )
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
                    grouped.setdefault(grp, []).append((ts, str(label), steps, values))
                except Exception:
                    continue

            # If we collected anything from this project, stop scanning further candidates
            if any(grouped.values()):
                break

        # Use default scaling if none found
        if max_episode_timesteps == 0.0:
            max_episode_timesteps = 1000.0
            print(f"Warning: No max episode timesteps found for {skill}, using default scaling")
        else:
            print(f"Found max episode timesteps for {skill}: {max_episode_timesteps}")

        # Limit number of groups if requested, based on recency (most recent run timestamp per group)
        if args.max_groups and len(grouped) > args.max_groups:
            group_scores: List[Tuple[float, str]] = []
            for g, lst in grouped.items():
                if not lst:
                    continue
                most_recent_ts = max(ts for ts, _, _, _ in lst)
                group_scores.append((most_recent_ts, g))
            group_scores.sort(key=lambda t: t[0], reverse=True)
            allowed = set(g for _, g in group_scores[: args.max_groups])
            grouped = {g: lst for g, lst in grouped.items() if g in allowed}

        # Aggregate per group: select most recent N runs, align on step, compute mean and standard error
        series_by_group: Dict[str, Tuple[List[float], List[float], Optional[List[float]]]] = {}
        for grp_key, runs_list in grouped.items():
            if not runs_list:
                continue
            runs_list.sort(key=lambda t: t[0], reverse=True)
            top_k = runs_list[: max(1, int(args.num_recent_runs_per_group))]

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

            if not step_to_vals:
                continue

            steps_sorted = sorted(step_to_vals.keys())
            means: List[float] = []
            ses: List[float] = []
            for s in steps_sorted:
                vals = step_to_vals.get(s, [])
                if not vals:
                    continue
                mean_v = sum(vals) / len(vals)
                if len(vals) > 1:
                    try:
                        stdev = statistics.stdev(vals)
                    except Exception:
                        stdev = 0.0
                    se = stdev / (len(vals) ** 0.5)
                else:
                    se = 0.0
                means.append(mean_v)
                ses.append(se)

            if steps_sorted and means:
                series_by_group[grp_key] = (steps_sorted, means, ses)

        out_path = _plot_overlaid_groups(
            skill,
            series_by_group,
            output_base,
            args.show,
        )
        if out_path:
            print(f"Saved plot for {skill}: {out_path}")
            total_plots += 1
        else:
            print(f"No plot generated for {skill} (no valid series)")

    if total_plots == 0:
        print("No plots were generated. Ensure your W&B runs exist and contain the metric 'Info/Episode_Reward/success_reward', and that runs have W&B groups or detectable group labels.")
    else:
        print(f"Generated {total_plots} plot(s) under: {output_base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# how to run
#
# python3 genhrl/analysis/plot_wandb_groups_l1.py \
#   --task_name <YourTaskName> \
#   --robot G1 \
#   --wandb_entity <your_entity> \
#   --wandb_api_key <your_key> \
#   --num_recent_runs_per_group 5 \
#   --max_runs_per_project 30 \
#   --skip_first 0 \
#   --include_groups Custom Standard \
#   --exclude_groups "ablation-.*"


