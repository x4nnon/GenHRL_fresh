#!/usr/bin/env bash
set -euo pipefail

# Directory setup
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Defaults (can be overridden via environment variables before calling this script)
ROBOT=${ROBOT:-G1}
WANDB_ENTITY=${WANDB_ENTITY:-tpcannon}
WANDB_API_KEY=${WANDB_API_KEY:-a949a2366eeaf95169770ca10c361c6dea1e03d7}
NUM_RECENT_RUNS=${NUM_RECENT_RUNS:-5}
MAX_RUNS_PER_PROJECT=${MAX_RUNS_PER_PROJECT:-20}
FLAT_SUFFIX=${FLAT_SUFFIX:-Flat}
SKIP_FIRST=${SKIP_FIRST:-0}

# Tasks to process
TASKS=(
  knock_over_pillars_seed42
  doorway_and_goal_seed42
  move_three_objects_seed42
  build_stairs_seed456
  obstacle_course_seed42
)

PY_SCRIPT="${REPO_ROOT}/genhrl/analysis/plot_wandb_success_rewards_l1.py"

echo "Running L1 HPPO vs Flat plots for ${#TASKS[@]} tasks..."

for TASK in "${TASKS[@]}"; do
  echo "\n=== Task: ${TASK} ==="
  python3 "${PY_SCRIPT}" \
    --task_name "${TASK}" \
    --robot "${ROBOT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --num_recent_runs "${NUM_RECENT_RUNS}" \
    --max_runs_per_project "${MAX_RUNS_PER_PROJECT}" \
    --flat_suffix "${FLAT_SUFFIX}" \
    --skip_first "${SKIP_FIRST}"
done

echo "\nAll tasks completed. Plots saved under genhrl/analysis/plots_l1/<task>."


