# GenHRL Task and Skill Removal Guide

This guide explains how to safely remove tasks and skills from GenHRL, including all associated files, logs, and training data.

## Overview

The GenHRL removal system provides comprehensive cleanup functionality that ensures all related files are properly removed when you delete tasks or skills:

- **Task directories** and configuration files
- **Skill directories** and individual skill files  
- **Skill library** entries and hierarchy updates
- **Training logs** from `logs/skrl/`
- **Success states** and checkpoints
- **Policy files** and other training artifacts

## Command Line Interface

### Remove Entire Task

Remove a complete task with all its skills:

```bash
# Basic task removal
genhrl remove Create_Steps

# Force removal without confirmation
genhrl remove Create_Steps --force

# Preview what will be removed (dry run)
genhrl remove Create_Steps --dry-run

# Specify robot (auto-detected if not provided)
genhrl remove Create_Steps --robot G1
```

### Remove Individual Skill

Remove a specific skill from a task:

```bash
# Remove specific skill
genhrl remove Create_Steps --skill walk_to_box

# Force skill removal without confirmation
genhrl remove Create_Steps --skill walk_to_box --force

# Preview skill removal impact
genhrl remove Create_Steps --skill walk_to_box --dry-run
```

### Custom IsaacLab Path

If your IsaacLab installation is not in the default location:

```bash
genhrl remove Create_Steps --isaaclab-path /path/to/IsaacLab
```

## What Gets Removed

### For Complete Task Removal

When you remove an entire task, the following are cleaned up:

1. **Task Directory**: `{robot}_generated/tasks/{task_name}/`
   - `description.txt`
   - `skills_hierarchy.json`
   - `object_config.json`

2. **Skills Directory**: `{robot}_generated/skills/{task_name}/`
   - `skill_library.json`
   - `object_config.json`
   - `skills/{skill_name}/` (all skill directories)

3. **Individual Skill Files** (for each skill):
   - Configuration files (`{skill}_cfg.py`)
   - Reward definitions (`TaskRewardsCfg.py`)
   - Success criteria (`SuccessTerminationCfg.py`)
   - Agent configurations (`agents/`)
   - Training data (`success_states/`, `current_task_start_states/`)
   - Policy files (`policy/`)

4. **Training Logs**: `logs/skrl/{skill_name}/`
   - All experiment directories
   - Checkpoint files
   - Training logs and metrics

### For Individual Skill Removal

When you remove a specific skill:

1. **Skill Library Update**: Removes skill entry and updates dependencies
2. **Skill Directory**: Removes `skills/{skill_name}/` completely
3. **Training Logs**: Removes all logs for that specific skill
4. **Hierarchy Update**: Updates task hierarchy if removing root skill

## Safety Features

### Confirmation Prompts

By default, removal operations require confirmation:

```
⚠️  WARNING: This will permanently delete task 'Create_Steps' and ALL its skills!
   - Task directory: /path/to/tasks/Create_Steps
   - Skills directory: /path/to/skills/Create_Steps
   - All training logs and checkpoints
Are you sure you want to continue? (yes/no):
```

### Dependency Checking

When removing individual skills, the system checks for dependencies:

```
⚠️  Warning: Skill 'walk_to_box' is used by other skills: ['push_box', 'climb_on_box']
Remove anyway? This may break dependent skills. (yes/no):
```

### Dry Run Mode

Preview what will be removed without actually deleting anything:

```bash
genhrl remove Create_Steps --dry-run
```

Output example:
```
🔍 DRY RUN: Task Removal Impact
   Target: Create_Steps

📁 Directories to be removed (2):
   - /path/to/tasks/Create_Steps
   - /path/to/skills/Create_Steps

🗂️  Log directories to be removed (5):
   - logs/skrl/walk_to_box
   - logs/skrl/push_box
   - logs/skrl/climb_on_box
   - logs/skrl/Walk_To_Box
   - logs/skrl/Create_Steps

🎯 Skills affected (5):
   - Create_Steps
   - walk_to_box
   - push_box
   - climb_on_box
   - stand_on_box

📊 Total files to be removed: 247
```

## Web UI Integration

The removal functionality is also available through the GenHRL web interface:

1. **Task List**: Click the trash icon next to any task
2. **Task Details**: Use the delete button in the task detail view
3. **Confirmation**: Web UI shows removal impact before deletion
4. **Individual Skills**: Remove specific skills from the task detail view

### API Endpoints

- `DELETE /api/tasks/{task_name}` - Remove entire task
- `DELETE /api/tasks/{task_name}?skill={skill_name}` - Remove specific skill
- `GET /api/tasks/{task_name}/removal-impact` - Get removal impact info

## Programmatic Usage

You can also use the removal functionality programmatically:

```python
from genhrl.generation.removal_manager import RemovalManager

# Create removal manager
removal_manager = RemovalManager("/path/to/IsaacLab", robot="G1")

# Preview removal impact
impact = removal_manager.get_removal_impact("Create_Steps")
print(f"Will remove {impact['file_count']} files")

# Remove entire task
success = removal_manager.remove_task("Create_Steps", confirm=False)

# Remove specific skill
success = removal_manager.remove_skill("Create_Steps", "walk_to_box", confirm=False)

# Dry run
removal_manager.dry_run_removal("Create_Steps")
```

## Recovery

**⚠️ Important: Removal operations are permanent and cannot be undone.**

To recover removed tasks or skills:
1. **Regenerate**: Use `genhrl generate` to recreate the task
2. **Restore from backup**: If you have backups of the IsaacLab directory
3. **Version control**: If your IsaacLab directory is under version control

## Best Practices

1. **Always use dry run first** to preview what will be removed
2. **Backup important tasks** before removal if needed
3. **Check dependencies** when removing individual skills
4. **Use force flag sparingly** - confirmation prompts prevent accidents
5. **Clean up regularly** to avoid cluttering the workspace

## Troubleshooting

### Permission Errors
Ensure you have write permissions to the IsaacLab directory and logs directory.

### Robot Detection
If the wrong robot is detected, explicitly specify with `--robot`:
```bash
genhrl remove Create_Steps --robot H1
```

### Partial Removal
If removal fails partway through, you may need to:
1. Check file permissions
2. Ensure no processes are using the files
3. Manually clean up remaining files if necessary

## Examples

### Complete Workflow
```bash
# List available tasks
genhrl list

# Preview removal impact
genhrl remove Basketball_Practice --dry-run

# Remove the task
genhrl remove Basketball_Practice

# Verify removal
genhrl list
```

### Selective Skill Removal
```bash
# See skills in a task
genhrl list --task Create_Steps

# Preview skill removal
genhrl remove Create_Steps --skill walk_to_box --dry-run

# Remove the skill
genhrl remove Create_Steps --skill walk_to_box

# Check updated task
genhrl status Create_Steps
```