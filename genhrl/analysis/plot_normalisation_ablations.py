import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import math


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
    task_parts = task_name.split('_')
    skill_parts = skill_name.split('_')
    formatted_task = ''.join(part.capitalize() for part in task_parts)
    formatted_skill = ''.join(part.capitalize() for part in skill_parts)
    return f"{formatted_task}{formatted_skill}"


def _expected_env_id_for_l0(task_name: str, skill_name: str) -> str:
    suffix = _format_gym_task_name(task_name, skill_name)
    return f"Isaac-RobotFlat{suffix}-v0"


def _compute_wandb_project_for_l0(task_name: str, skill_name: str) -> str:
    """Replicate the project naming used by training for L0 runs."""
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
    tp = base / "policy" / "training_params.json"
    if tp.exists():
        obj = _read_json(tp)
        if isinstance(obj, dict):
            wkwargs = obj.get("wandb_kwargs")
            if isinstance(wkwargs, dict):
                proj = wkwargs.get("project")
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


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _wandb_run_matches(run, expected_env_id: str, skill_name: str, task_name: str, project_name: str = "") -> bool:
    """Heuristically determine if a W&B run corresponds to the given env/skill."""
    try:
        cfg = getattr(run, "config", {}) or {}
        name = getattr(run, "name", "") or ""
        tags = getattr(run, "tags", []) or []
        
        # 1) Check config fields for environment ID
        for key in ["env_id", "task", "env", "gym_task", "task_id"]:
            val = cfg.get(key)
            if isinstance(val, str) and expected_env_id in val:
                return True
        
        # 2) Check tags
        tag_str = " ".join(str(t) for t in tags)
        if expected_env_id in tag_str or skill_name.lower() in tag_str.lower() or task_name.lower() in tag_str.lower():
            return True
        
        # 3) Check run name
        if isinstance(name, str) and (expected_env_id in name or skill_name.lower() in name.lower()):
            return True
            
        # 4) Fallback: Check if project name contains skill information
        if project_name and skill_name.lower() in project_name.lower():
            return True
            
        # 5) Final fallback: If we're in a project that matches the expected pattern, assume it's correct
        if project_name and expected_env_id.replace("Isaac-Robot", "").replace("-v0", "").lower() in project_name.lower():
            return True
            
    except Exception:
        return False
    return False


def _extract_success_series_from_wandb_run(run, metric_key: str, step_key_candidates: List[str]) -> Optional[Tuple[List[float], List[float]]]:
    """Extract (steps, metric) from a W&B run history via API."""
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
            summ = getattr(run, "summary", {}) or {}
            if isinstance(summ, dict):
                for key in possible_keys:
                    val = summ.get(key)
                    if isinstance(val, (int, float)) and val > max_val:
                        max_val = float(val)
    except Exception:
        pass
    return max_val


def _pretty_skill_name(skill_name: str) -> str:
    import re
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", skill_name)
    spaced = spaced.replace("_", " ")
    return " ".join(w.capitalize() for w in spaced.split())


def _detect_normalisation_group(run, group_labels: Dict[str, str]) -> Optional[str]:
    """Return one of the canonical group labels based on run.group/tags/config."""
    try:
        canonical = {
            'no': group_labels['no'],
            'custom': group_labels['custom'],
            'standard': group_labels['standard'],
        }

        # 1) Direct group name
        grp = getattr(run, 'group', None)
        if isinstance(grp, str) and grp.strip():
            gpl = grp.strip().lower()
            if 'no normal' in gpl or gpl in {canonical['no'].lower(), 'no', 'none', 'no_norm', 'no-normalisation', 'no-normalization'}:
                return canonical['no']
            if 'custom' in gpl or gpl in {canonical['custom'].lower(), 'ours'}:
                return canonical['custom']
            if 'standard' in gpl or gpl in {canonical['standard'].lower(), 'std', 'standardize', 'standardise'}:
                return canonical['standard']

        # 2) Tags
        tags = getattr(run, 'tags', []) or []
        tag_str = ' '.join(str(t).lower() for t in tags)
        if any(k in tag_str for k in ['no normalisation', 'no normalization', 'no-norm', 'no_norm', 'none']):
            return canonical['no']
        if any(k in tag_str for k in ['custom normalisation', 'custom normalization', 'custom']):
            return canonical['custom']
        if any(k in tag_str for k in ['standard normalisation', 'standard normalization', 'standard', 'std']):
            return canonical['standard']

        # 3) Config
        cfg = getattr(run, 'config', {}) or {}
        if isinstance(cfg, dict):
            for key in [
                'normalisation', 'normalization', 'normalize', 'normalise', 'standardization',
                'standardisation', 'reward_normalization', 'obs_normalization', 'obs_norm',
                'reward_norm', 'normalization_type', 'norm_type', 'use_normalization',
            ]:
                val = cfg.get(key)
                if isinstance(val, str):
                    vl = val.lower()
                    if vl in {'no', 'none', 'off', 'false', 'disabled'} or 'no norm' in vl:
                        return canonical['no']
                    if 'custom' in vl:
                        return canonical['custom']
                    if 'standard' in vl or vl in {'std', 'on', 'true', 'enabled'}:
                        return canonical['standard']
                if isinstance(val, bool):
                    if val is False:
                        return canonical['no']
                    if val is True:
                        return canonical['standard']
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

    colors = {
        'No normalisation': '#d62728',      # red
        'Custom normalisation': '#ff7f0e',  # orange
        'Standard normalisation': '#1f77b4' # blue
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for group_name, (steps, means, ses) in series_by_group.items():
        color = colors.get(group_name, None)
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

    out_path = output_dir / f"{skill_name}_normalisation_ablations.png"
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
        description="Plot normalisation ablation (No/Custom/Standard) per L0 skill from W&B",
    )
    parser.add_argument("--task_name", required=True, help="Task name, e.g. Build_Stairs")
    parser.add_argument("--robot", default="G1", help="Robot name (default: G1)")
    parser.add_argument("--isaaclab_path", default=None, help="Path to IsaacLab installation (auto-detected if omitted)")
    parser.add_argument("--wandb_entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--wandb_project", default=None, help="W&B project name (optional; auto-detected per skill if omitted)")
    parser.add_argument("--wandb_api_key", default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (default: genhrl/analysis/plots/<task>_normalisation_ablations)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--metric_key", default="Info/Episode_Reward/success_reward", help="W&B metric key to plot (y-axis)")
    parser.add_argument("--step_keys", nargs="+", default=["global_step", "Global Step", "global-step", "_step", "step", "iteration"], help="Candidate step keys to try for x-axis (Global Step preferred)")
    parser.add_argument("--num_recent_runs_per_group", type=int, default=5, help="Number of most recent runs to average per group (default: 5)")
    parser.add_argument("--max_runs_per_project", type=int, default=30, help="Maximum number of runs to fetch per project (default: 30)")
    parser.add_argument("--no_label", default="No normalisation", help="Canonical label for 'no' group")
    parser.add_argument("--custom_label", default="Custom normalisation", help="Canonical label for 'custom' group")
    parser.add_argument("--standard_label", default="Standard normalisation", help="Canonical label for 'standard' group")

    args = parser.parse_args()

    isaaclab_path = Path(args.isaaclab_path).expanduser().resolve() if args.isaaclab_path else _auto_detect_isaaclab_path()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        print("Hint: Set ISAACLAB_PATH env var or pass --isaaclab_path /path/to/IsaacLab")
        return 1

    try:
        l0_skills = _load_l0_skills(isaaclab_path, args.task_name, args.robot)
    except Exception as e:
        print(f"Error loading skills: {e}")
        return 1

    if not l0_skills:
        print(f"No L0 skills found for task: {args.task_name}")
        return 0

    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = Path(__file__).resolve().parents[0] / "plots" / f"{args.task_name}_normalisation_ablations"

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

    group_labels = {
        'no': args.no_label,
        'custom': args.custom_label,
        'standard': args.standard_label,
    }

    total_plots = 0
    print(f"Found {len(l0_skills)} L0 skills")
    print(f"L0 skills: {l0_skills}")

    for skill in l0_skills:
        expected_env = _expected_env_id_for_l0(args.task_name, skill)

        # Determine candidate projects
        candidates: List[str] = []
        if args.wandb_project:
            candidates.append(args.wandb_project)
        detected = _resolve_wandb_project_for_skill(isaaclab_path, args.task_name, args.robot, skill)
        if detected and detected not in candidates:
            candidates.append(detected)
        computed = _compute_wandb_project_for_l0(args.task_name, skill)
        if computed and computed not in candidates:
            candidates.append(computed)
        formatted_skill = ''.join(part.capitalize() for part in skill.split('_'))
        candidates.extend([
            f"genhrl_{formatted_skill}",
            f"genhrl_{args.task_name}",
        ])
        seen_names = set()
        candidates = [c for c in candidates if not (c in seen_names or seen_names.add(c))]

        # Collect runs grouped by normalisation label
        # Dict[group_label] -> List[(timestamp, label, steps, values, run_obj)]
        grouped: Dict[str, List[Tuple[float, str, List[float], List[float], object]]] = {
            group_labels['no']: [],
            group_labels['custom']: [],
            group_labels['standard']: [],
        }

        for proj in candidates:
            project_path = f"{args.wandb_entity}/{proj}"
            try:
                runs = api.runs(project_path, per_page=args.max_runs_per_project)
                print(f"Trying project '{proj}' -> {len(runs)} run(s) (limited to {args.max_runs_per_project})")
            except Exception as e:
                print(f"Project not available '{proj}': {e}")
                continue

            # Phase 1: scan for matching runs and bucket by group
            processed_count = 0
            matches_found = 0
            groups_found = 0
            series_found = 0
            for run in runs:
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"  Scanned {processed_count}/{len(runs)} runs for matches...")
                try:
                    if not _wandb_run_matches(run, expected_env, skill, args.task_name, proj):
                        if processed_count <= 3:  # Show details for first few runs
                            print(f"    Run {processed_count} ({getattr(run, 'name', 'unnamed')}) did not match")
                        continue
                    matches_found += 1
                    print(f"    Run {processed_count} ({getattr(run, 'name', 'unnamed')}) MATCHED!")
                    grp = _detect_normalisation_group(run, group_labels)
                    if grp is None:
                        print(f"    Run {getattr(run, 'name', 'unnamed')} matched but no normalisation group detected")
                        continue
                    groups_found += 1
                    print(f"    Run {getattr(run, 'name', 'unnamed')} grouped as: {grp}")
                    # discover metric keys for this run, then extract series
                    metric_candidates = _discover_metric_keys_from_run(run, args.metric_key)
                    print(f"    Found metric candidates: {metric_candidates}")
                    series = None
                    for mk in metric_candidates:
                        try:
                            series = _extract_success_series_from_wandb_run(run, mk, args.step_keys)
                            if series:
                                print(f"    Successfully extracted series for metric: {mk}")
                                break
                        except Exception as e:
                            print(f"    Failed to extract series for {mk}: {e}")
                            continue
                    if not series:
                        print(f"    Run {getattr(run, 'name', 'unnamed')} matched and grouped but no metric series found")
                        continue
                    series_found += 1
                    steps, values = series
                    print(f"    Run {getattr(run, 'name', 'unnamed')} SUCCESS! Series length: {len(values)}")
                    label = getattr(run, "name", None) or getattr(run, "id", "run")
                    ts = 0.0
                    try:
                        t = getattr(run, "updated_at", None) or getattr(run, "created_at", None)
                        if t is not None:
                            try:
                                ts = t.timestamp()
                            except Exception:
                                ts = 0.0
                    except Exception:
                        ts = 0.0
                    grouped.setdefault(grp, []).append((ts, label, steps, values, run))
                except Exception as e:
                    print(f"    Run {processed_count} failed with exception: {e}")
                    continue
            
            print(f"  Project '{proj}' summary: {matches_found} matches, {groups_found} grouped, {series_found} with series")

        # If nothing collected yet, fallback: scan all projects for entity
        if not any(grouped.values()):
            try:
                projects_iter = api.projects(args.wandb_entity)
                for pr in projects_iter:
                    proj_name = getattr(pr, 'name', None) or getattr(pr, 'project', None)
                    if not isinstance(proj_name, str):
                        continue
                    project_path = f"{args.wandb_entity}/{proj_name}"
                    try:
                        runs = api.runs(project_path)
                    except Exception:
                        continue
                    for run in runs:
                        try:
                            if not _wandb_run_matches(run, expected_env, skill, args.task_name, proj_name):
                                continue
                            grp = _detect_normalisation_group(run, group_labels)
                            if grp is None:
                                continue
                            series = _extract_success_series_from_wandb_run(run, args.metric_key, args.step_keys)
                            if not series:
                                continue
                            steps, values = series
                            label = getattr(run, "name", None) or getattr(run, "id", "run")
                            ts = 0.0
                            try:
                                t = getattr(run, "created_at", None) or getattr(run, "updated_at", None)
                                if t is not None:
                                    try:
                                        ts = t.timestamp()
                                    except Exception:
                                        ts = 0.0
                            except Exception:
                                ts = 0.0
                            grouped.setdefault(grp, []).append((ts, label, steps, values, run))
                        except Exception:
                            continue
                    if any(grouped.values()):
                        print(f"Found matching runs in fallback project: {proj_name}")
                        break
            except Exception as e:
                print(f"Project list fallback failed: {e}")

        # For scaling, find max episode timesteps across all runs considered
        max_episode_timesteps = 0.0
        try:
            for lst in grouped.values():
                for _, _, _, _, r in lst:
                    try:
                        run_max = _find_max_episode_timesteps(r)
                        if run_max > max_episode_timesteps:
                            max_episode_timesteps = run_max
                    except Exception:
                        continue
        except Exception:
            pass
        if max_episode_timesteps == 0.0:
            max_episode_timesteps = 1000.0
            print(f"Warning: No max episode timesteps found for {skill}, using default scaling")
        else:
            print(f"Found max episode timesteps for {skill}: {max_episode_timesteps}")

        # Aggregate per group: select most recent N runs, align on step, compute mean and standard error
        series_by_group: Dict[str, Tuple[List[float], List[float], Optional[List[float]]]] = {}
        for grp_key, runs_list in grouped.items():
            if not runs_list:
                continue
            runs_list.sort(key=lambda t: t[0], reverse=True)
            top_k = runs_list[: max(1, int(args.num_recent_runs_per_group))]

            step_to_vals: Dict[float, List[float]] = {}
            for _, _, steps, values, _ in top_k:
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
                    se = stdev / math.sqrt(len(vals))
                else:
                    se = 0.0
                means.append(mean_v)
                ses.append(se)

            if steps_sorted and means:
                series_by_group[grp_key] = (steps_sorted, means, ses)

        if series_by_group:
            out_path = _plot_overlaid_groups(skill, series_by_group, output_base, args.show)
            if out_path:
                print(f"Saved plot for {skill}: {out_path}")
                total_plots += 1
            else:
                print(f"No plot generated for {skill} (no valid series)")
        else:
            print(f"No matching grouped W&B runs found for skill: {skill}")

    if total_plots == 0:
        print("No plots were generated. Ensure your W&B runs contain the metric 'Info/Episode_Reward/success_reward' and are grouped by normalisation.")
    else:
        print(f"Generated {total_plots} plot(s) under: {output_base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# example usage:
# python3 genhrl/analysis/plot_normalisation_ablations.py \
#   --task_name move_three_objects_seed42 \
#   --robot G1 \
#   --wandb_entity tpcannon \
#   --wandb_api_key <YOUR_WANDB_API_KEY> \
#   --num_recent_runs_per_group 4


