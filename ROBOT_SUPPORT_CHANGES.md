# Robot Support Changes for GenHRL

This document summarizes the changes made to support multiple robots in GenHRL instead of just the hardcoded G1 robot.

## Overview

GenHRL now supports multiple robots available in IsaacLab through a new robot configuration system and robot-specific folder structure.

## Key Changes

### 1. New Robot Configuration System

**File:** `genhrl/generation/robot_configs.py`
- Added robot configuration dataclass with robot-specific settings
- Configured support for 12+ robots: G1, H1, Anymal_B/C/D, A1, Go1, Go2, Spot, Digit, Franka, UR10
- Each robot config includes:
  - Asset configuration imports
  - Joint name patterns
  - Default joint positions  
  - Actuator configurations
  - Robot-specific paths

### 2. Command Line Interface Updates

**File:** `genhrl/cli.py`
- Added `--robot` argument to all commands (generate, train, auto, list, status)
- Default robot is G1 for backward compatibility
- Robot filter support in list command
- Auto-detection of robot from task folder in status command

### 3. Dynamic Skill Config Template

**File:** `genhrl/generation/skill_config_template.py`
- Made robot imports dynamic based on `ROBOT_NAME` environment variable
- Updated robot configuration to use robot-specific settings
- Generalized reward classes to work with any robot
- Robot-specific joint names and paths

### 4. Robot-Specific Folder Structure

**Before:**
```
isaaclab_tasks/manager_based/G1_generated/
├── tasks/
└── skills/
```

**After:**
```
isaaclab_tasks/manager_based/
├── G1_generated/
│   ├── tasks/
│   └── skills/
├── H1_generated/
│   ├── tasks/ 
│   └── skills/
├── Anymal_C_generated/
│   ├── tasks/
│   └── skills/
└── ... (other robots)
```

### 5. Updated Core Classes

**TaskManager** (`genhrl/generation/task_manager.py`):
- Added robot parameter to constructor
- Robot-specific path handling
- Automatic directory creation for new robots
- Environment variable setting for skill config template

**TrainingOrchestrator** (`genhrl/training/orchestrator.py`):
- Added robot parameter and robot-specific paths
- Updated training task names to use robot prefix
- Auto-detection of robot from task folder structure

**TaskConfig** (`genhrl/generation/task_manager.py`):
- Added robot field
- Robot-specific path methods

## Usage Examples

### Generate tasks for different robots:

```bash
# Generate for G1 (default)
genhrl generate "walk forward" --api-key YOUR_KEY

# Generate for H1 
genhrl generate "walk forward" --robot H1 --api-key YOUR_KEY

# Generate for Anymal-C
genhrl generate "walk forward" --robot Anymal_C --api-key YOUR_KEY

# Generate for Franka manipulator
genhrl generate "pick up box" --robot Franka --api-key YOUR_KEY
```

### Train tasks:

```bash
# Train G1 task
genhrl train walk_forward --robot G1

# Train H1 task  
genhrl train walk_forward --robot H1

# Auto-detect robot from task folder
genhrl train walk_forward  # will auto-detect robot
```

### List tasks by robot:

```bash
# List all tasks for all robots
genhrl list

# List tasks for specific robot
genhrl list --robot H1

# Show task details (auto-detects robot)
genhrl list --task walk_forward
```

## Available Robots

The following robots are currently supported:

**Locomotion Robots:**
- G1 (Unitree G1 humanoid)
- H1 (Unitree H1 humanoid) 
- Anymal_B, Anymal_C, Anymal_D (ANYmal quadrupeds)
- A1, Go1, Go2 (Unitree quadrupeds)
- Spot (Boston Dynamics)
- Digit (Agility Robotics)

**Manipulation Robots:**
- Franka (Franka Emika Panda)
- UR10 (Universal Robots)

## Backward Compatibility

- Existing G1 tasks continue to work unchanged
- Default robot is G1 when not specified
- Falls back to G1_generated folder if robot folders don't exist
- Old command syntax still works

## Technical Details

### Robot Configuration Structure

Each robot configuration includes:
- `asset_cfg`: IsaacLab asset import name
- `joint_names`: Dictionary of joint patterns by body part
- `default_joint_pos`: Default joint positions
- `actuator_configs`: Actuator parameters (effort limits, stiffness, damping)
- Path templates for robot-specific sensor placement

### Environment Variable System

The skill config template uses the `ROBOT_NAME` environment variable to dynamically:
- Import correct robot assets
- Set robot-specific joint configurations
- Configure sensor paths
- Apply robot-specific reward terms

### Directory Management

- Robot directories are created automatically when needed
- Template files are copied to robot-specific directories
- Skills are isolated by robot to prevent conflicts

## Future Enhancements

- Add more robots as they become available in IsaacLab
- Support for custom robot configurations
- Improved agent configurations per robot type
- Robot-specific task templates and skills