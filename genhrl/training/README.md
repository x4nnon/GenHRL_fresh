# GenHRL Training - Simplified Orchestrator

This directory contains a simplified orchestrator for training GenHRL skills in the correct hierarchical sequence.

## Overview

The simplified orchestrator focuses on the core functionality:

1. **Skill Sequence**: Determines the correct training order (primitives first, then composite skills by level)
2. **Training Execution**: Runs Isaac Lab training commands for each skill
3. **Completion Monitoring**: Checks for sufficient success states to determine when training is complete
4. **Sequential Training**: Moves to the next skill once the current one is complete

## Usage

### Basic Command

```bash
python genhrl/training/run_training.py \
    --isaaclab-path /path/to/isaaclab \
    --task-name your_task_name \
    --robot G1
```

### Full Options

```bash
python genhrl/training/run_training.py \
    --isaaclab-path /path/to/isaaclab \
    --task-name your_task_name \
    --robot G1 \
    --max-time 60 \
    --min-success-states 100 \
    --num-envs 4096 \
    --seed 42 \
    --no-headless
```

### Arguments

- `--isaaclab-path`: Path to your IsaacLab installation (required)
- `--task-name`: Name of the task to train (required)
- `--robot`: Robot name (default: G1)
- `--max-time`: Maximum training time per skill in minutes (default: 60)
- `--min-success-states`: Minimum success states required for completion (default: 100)
- `--num-envs`: Number of simulation environments (default: 4096)
- `--seed`: Random seed (default: 42)
- `--no-headless`: Run with GUI instead of headless mode

## How It Works

### 1. Training Order
The orchestrator reads your skill library and determines the training order:
- **Primitives first**: All primitive skills (level 0)
- **Level 1 composite**: Skills that use only primitives
- **Level 2 composite**: Skills that use level 1 skills
- And so on...

### 2. Skill Training
For each skill in sequence:
- Checks if already complete (has sufficient success states)
- Builds the appropriate Isaac Lab command:
  - Primitives: `train.py` with `Isaac-RobotFlat{SkillName}-v0`
  - Composites: `train_l{level}.py` with `Isaac-RobotComposite{SkillName}-v0`
- Runs the training command and monitors output
- Stops when sufficient success states are found or timeout is reached

### 3. Completion Check
Training for a skill stops when:
- Minimum number of success states are generated, OR
- Maximum time limit is reached

## Example Output

```
============================================================
GenHRL Training - Simplified Orchestrator
============================================================
IsaacLab Path: /path/to/isaaclab
Task: my_task
Robot: G1
Max time per skill: 60 minutes
Min success states: 100
Environments: 4096
Headless: True
============================================================

Training order for my_task:
1. reach [P]
2. grasp [P]
3. lift [P]
4. place [P]
5. pick_and_place [C]
6. stack_objects [C]

============================================================
Training skill: reach
Type: Primitive
============================================================
Command: ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-RobotFlatReach-v0 --num_envs 4096 --seed 42 --headless
...
Found 100 success state files for reach

Training completed successfully for reach

============================================================
Training skill: grasp
Type: Primitive
============================================================
...
```

## Key Features

- **Simple and debuggable**: Minimal complexity, easy to understand what's happening
- **Real-time monitoring**: Shows training output and checks completion periodically
- **Automatic skill detection**: Determines primitive vs composite automatically
- **Robust completion checking**: Uses success state files to verify training completion
- **Keyboard interrupt handling**: Clean exit with Ctrl+C

## Troubleshooting

### Common Issues

1. **"Object config not found"**: Make sure your task has the required files in the skills directory
2. **"Skill library not found"**: Ensure `skill_library.json` exists for your task
3. **Training doesn't stop**: Check that success states are being generated in the correct directory
4. **GPU memory issues**: The orchestrator doesn't include complex cleanup - restart if needed

### Debug Tips

- Run with `--no-headless` to see the simulation
- Reduce `--num-envs` if running out of memory
- Check success state directories manually: `skills/{task_name}/skills/{skill_name}/success_states/`
- Monitor Isaac Lab logs in the `logs/skrl/` directory

## Environment Setup

Remember to activate the Isaac Lab conda environment [[memory:2944206]]:

```bash
conda activate env_isaaclab
``` 