# Skill Registration Solution for Multi-Seeded Tasks

## Problem
When running scientific testing with multiple seeded tasks (e.g., `obstacle_course_seed42`, `obstacle_course_seed123`, `obstacle_course_seed456`), the gym registrations were conflicting because skills with the same names from different tasks would overwrite each other.

**Original Error:**
```
🔧 DEBUG: Looking for task_id: 'Isaac-RobotFlatWalk_To_Smallsphere-v0'
🔧 DEBUG: NameNotFound for Isaac-RobotFlatWalk_To_Smallsphere-v0: Environment `Isaac-RobotFlatWalk_To_Smallsphere` doesn't exist.
```

## Solution

We implemented a comprehensive solution with two main components:

### 1. Automatic Skill Registration Script

Created `genhrl/scripts/register_all_skills.py` that:
- Scans all generated task directories automatically
- Reads skill libraries to extract all skills
- Generates task-specific gym registrations to avoid conflicts
- Updates the `__init__.py` file with proper registrations

**New Naming Convention:**
- Primitive skills: `Isaac-RobotFlat{TaskName}_{SkillName}-v0`
- Composite skills: `Isaac-RobotComposite{TaskName}_{SkillName}-v0`

**Example:**
```python
# Old (conflicting):
Isaac-RobotFlatWalkToLowwall-v0

# New (task-specific):
Isaac-RobotFlatObstacleCourseSeed42WalkToLowwall-v0
Isaac-RobotFlatObstacleCourseSeed123WalkToLowwall-v0
```

### 2. Updated Training Orchestrator

Modified `genhrl/training/orchestrator.py` to:
- Use the new task-specific naming convention
- Generate proper gym IDs that include the task name
- Maintain compatibility with existing training workflows

**Key Changes:**
- Added `_format_gym_task_name()` method for consistent naming
- Updated both `build_training_command()` and `build_simple_training_command()`
- Fixed type annotations and error handling

## Usage

### Register All Skills
```bash
# Generate registrations for all discovered tasks (uses ./IsaacLab by default)
python -m genhrl.scripts.register_all_skills

# Dry run to see what would be registered
python -m genhrl.scripts.register_all_skills --dry-run

# Use custom IsaacLab path
python -m genhrl.scripts.register_all_skills --isaaclab-path /path/to/IsaacLab

# Don't clean existing registrations (append mode - not recommended)
python -m genhrl.scripts.register_all_skills --no-clean
```

**🧹 Automatic Cleanup**: By default, the script now cleans existing registrations before generating new ones and creates a backup file (`__init__.py.backup`). This ensures no stale registrations from deleted tasks remain.

### Training with New System
```bash
# Training now works correctly with task-specific names
genhrl train "obstacle_course_seed42" --simple --steps 6000 --video-interval 2000 --video-length 200 --new-run
```

### Testing Registration
```bash
# Verify the new system works
python -m genhrl.testing.test_registration --isaaclab-path ./IsaacLab --task obstacle_course_seed42
```

## Results

✅ **181 skills registered** across **16 tasks**:
- 127 primitive skills
- 54 composite skills

✅ **No more naming conflicts** between seeded tasks

✅ **Backward compatibility** maintained

✅ **Automatic discovery** of new tasks

## Generated Registrations Summary

- **build_stairs_seed42/123/456**: Stair building tasks with different approaches
- **obstacle_course_seed42/123/456**: Obstacle navigation with varied skill decompositions  
- **doorway_and_goal_seed42/123/456**: Doorway manipulation and navigation
- **move_three_objects_seed42/123/456**: Object manipulation tasks
- **swimming_diving_board_seed42/123/456**: Complex diving board scenarios
- **navigate_test**: Basic navigation testing

Each task's skills are now uniquely identified and won't conflict with other tasks.

## Technical Details

### Registration Format
```python
gym.register(
    id="Isaac-RobotFlat{TaskName}_{SkillName}-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.G1_generated.skills.{task_name}.skills.{skill_name}.{skill_name.lower()}_cfg:RobotFlatEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}.skills.{task_name}.skills.{skill_name}.agents:skrl_ppo_cfg.yaml",  # or skrl_ppo_l1_cfg.yaml for composite skills
    },
)
```

**Important Notes**: 
- **Config Files**: Skill directories maintain the original case (e.g., `walk_to_lowWall`) but config files are always lowercase (e.g., `walk_to_lowwall_cfg.py`). The registration script correctly handles this case conversion.
- **YAML Files**: Primitive skills use `skrl_ppo_cfg.yaml` while composite skills use `skrl_ppo_l1_cfg.yaml`. The registration script automatically selects the correct file based on skill type.

### Orchestrator Integration
The orchestrator now generates commands like:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-RobotFlatObstacleCourseSeed42WalkToLowwall-v0 \
  --num_envs 4096 --seed 42 --headless
```

This solution ensures that scientific testing with multiple seeded tasks works reliably without gym registration conflicts.