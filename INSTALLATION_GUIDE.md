# GenHRL Installation & Usage Guide

## 🎯 **What is GenHRL?**

GenHRL is a standalone package that transforms natural language descriptions into trained hierarchical RL policies using IsaacLab. It provides:

1. **Task Generation**: Convert "pick up the red ball" → complete skill hierarchies
2. **Training Orchestration**: Automatically train all skills in proper order
3. **IsaacLab Integration**: Generate files directly into IsaacLab structure

## 📋 **Prerequisites**

### 1. IsaacLab Installation
```bash
# Install IsaacLab first (follow official instructions)
# https://isaac-sim.github.io/IsaacLab/

# Verify IsaacLab is working
cd /path/to/IsaacLab
./isaaclab.sh -p source/standalone/demos/basic.py
```

### 2. API Key
Get an API key from:
- **Google Gemini**: https://ai.google.dev/
- **Anthropic Claude**: https://console.anthropic.com/

## 🚀 **Installation**

### Install GenHRL
```bash
# Clone and install GenHRL
git clone https://github.com/your-org/genhrl.git
cd genhrl
pip install -e .

# Set your API key
export GENHRL_API_KEY="your_api_key_here"
# Or add to ~/.bashrc for persistence
echo 'export GENHRL_API_KEY="your_api_key_here"' >> ~/.bashrc
```

### Verify Installation
```bash
# Test CLI is available
genhrl --help

# Should show:
# GenHRL: Generative Hierarchical Reinforcement Learning Framework
# Available commands: generate, train, auto, list, status
```

## 💡 **Quick Start Examples**

### Example 1: Simple Manipulation
```bash
# Generate task
genhrl generate "pick up the red ball and place it on the table" \
    --isaaclab-path ~/IsaacLab

# Train all skills
genhrl train pick_up_the_red --isaaclab-path ~/IsaacLab

# Check progress
genhrl status pick_up_the_red --isaaclab-path ~/IsaacLab
```

### Example 2: Complex Navigation
```bash
# Generate and train in one command
genhrl auto "navigate around obstacles to reach the goal" \
    --isaaclab-path ~/IsaacLab \
    --max-time 120
```

### Example 3: Hierarchical Task
```bash
# Generate stair climbing task
genhrl generate "climb 3 stairs by stepping on each step" \
    --name stair_climbing \
    --isaaclab-path ~/IsaacLab

# Train with custom settings
genhrl train stair_climbing \
    --isaaclab-path ~/IsaacLab \
    --min-success-states 50 \
    --seed 42 \
    --new-run
```

## 🔧 **Command Reference**

### `genhrl generate`
Create new tasks from natural language descriptions.

**Required:**
- `description`: Natural language task description
- `--isaaclab-path`: Path to IsaacLab installation

**Optional:**
- `--name`: Custom task name (auto-generated if not provided)
- `--api-key`: API key (or use GENHRL_API_KEY env var)
- `--provider`: LLM provider (google/anthropic, default: google)

### `genhrl train`
Train all skills for a task in proper hierarchical order.

**Required:**
- `task_name`: Name of the task to train
- `--isaaclab-path`: Path to IsaacLab installation

**Optional:**
- `--max-time INT`: Max training time per skill in minutes (default: 180)
- `--min-success-states INT`: Required success states (default: 100)
- `--seed INT`: Random seed (default: 42)
- `--num-envs INT`: Number of environments (default: 4096)
- `--skip-complete`: Skip skills with sufficient success states (default: true)
- `--new-run`: Clean all previous training data

### `genhrl auto`
Generate and train a task automatically.
Combines `generate` + `train` with sensible defaults.

### `genhrl list`
Show available tasks and their details.

**Optional:**
- `--task NAME`: Show details for specific task

### `genhrl status`
Monitor training progress for a task.

## 📁 **Generated File Structure**

GenHRL creates files in your IsaacLab installation:

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/
├── tasks/
│   └── task_name/
│       ├── description.txt                # Original description
│       ├── object_config.json            # Scene configuration
│       └── skills_hierarchy.json         # Complete hierarchy
├── skills/
│   ├── skill_library.json               # Global skill library
│   ├── skill_config_template.py         # Base template
│   └── skills/
│       ├── primitive_skill/              # Primitive skill
│       │   ├── TaskRewardsCfg.py        # Generated rewards
│       │   ├── SuccessTerminationCfg.py # Generated terminations
│       │   ├── primitive_skill_cfg.py   # Main config (inherits template)
│       │   ├── agents/                   # Training configs
│       │   ├── policy/agent.pt          # Trained policy
│       │   └── success_states/           # Training success states
│       └── composite_skill/              # Composite skill
│           ├── TaskRewardsCfg.py        # Generated rewards
│           ├── SuccessTerminationCfg.py # Generated terminations
│           └── composite_skill_cfg.py   # Main config (inherits template)
```

## 🎮 **Training Process**

### What Happens During Training:

1. **Dependency Analysis**: GenHRL reads the skill hierarchy and determines training order
2. **Sequential Training**: 
   - Trains primitive skills first (using `train.py`)
   - Then trains composite skills (using `train_l1.py`)
3. **State Transfer**: Success states from completed skills are transferred to next skills
4. **Progress Monitoring**: Automatically detects when skills are learned and moves to next
5. **Policy Management**: Saves trained policies to skill directories

### Training Commands Used:
```bash
# Primitive skills
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-RobotFlatGraspBall-v0 --num_envs 4096

# Composite skills  
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_l1.py \
    --task Isaac-G1CompositePlaceOnTable-v0 \
    --skill_name PlaceOnTable --num_envs 4096
```

## 🔍 **Monitoring and Debugging**

### Check Training Progress
```bash
# Overall status
genhrl status my_task --isaaclab-path ~/IsaacLab

# List all tasks
genhrl list --isaaclab-path ~/IsaacLab

# Show task details
genhrl list --task my_task --isaaclab-path ~/IsaacLab
```

### Training Logs
Check IsaacLab logs directory:
```bash
ls ~/IsaacLab/logs/skrl/
# Shows individual skill training logs
```

### Generated Files
Inspect generated files:
```bash
# Task description
cat ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/tasks/my_task/description.txt

# Skill hierarchy
cat ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/tasks/my_task/skills_hierarchy.json

# Generated rewards for a skill
cat ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/skills/skills/GraspBall/TaskRewardsCfg.py
```

## 🔧 **Advanced Usage**

### Custom Training Settings
```bash
# Long training with high success requirements
genhrl train my_task \
    --isaaclab-path ~/IsaacLab \
    --max-time 300 \
    --min-success-states 200 \
    --num-envs 8192

# Quick training for testing
genhrl train my_task \
    --isaaclab-path ~/IsaacLab \
    --max-time 60 \
    --min-success-states 25 \
    --num-envs 2048
```

### Programmatic Usage
```python
from genhrl import TaskManager, TrainingOrchestrator, TrainingConfig

# Generate task
task_manager = TaskManager("/path/to/IsaacLab", "your_api_key")
task_config = task_manager.create_task_from_description(
    "pick_ball", "Pick up the red ball"
)

# Train task
orchestrator = TrainingOrchestrator("/path/to/IsaacLab", "pick_ball")
config = TrainingConfig(max_time_minutes=120, min_success_states=50)
success = orchestrator.train_all_skills(config=config)
```

## ❗ **Troubleshooting**

### Common Issues

**1. Import Errors**
```bash
# Missing dependencies
pip install psutil anthropic google-generativeai

# IsaacLab not found
export PYTHONPATH="/path/to/IsaacLab/source:$PYTHONPATH"
```

**2. API Key Issues**
```bash
# Check environment variable
echo $GENHRL_API_KEY

# Test API key
python -c "import os; print('API key set:', bool(os.getenv('GENHRL_API_KEY')))"
```

**3. Training Failures**
```bash
# Check GPU memory
nvidia-smi

# Reduce environments if out of memory
genhrl train my_task --num-envs 2048 --isaaclab-path ~/IsaacLab

# Check logs
tail -f ~/IsaacLab/logs/skrl/skill_name/latest/train.log
```

**4. File Permission Issues**
```bash
# Ensure write permissions to IsaacLab
chmod -R u+w ~/IsaacLab/source/isaaclab_tasks/
```

### Getting Help

1. **Check generated files** in IsaacLab to validate task generation
2. **Use status command** to monitor training progress
3. **Review training logs** in `~/IsaacLab/logs/skrl/`
4. **Test with simpler tasks** first
5. **Verify IsaacLab installation** with basic demos

## 📚 **Next Steps**

After successful installation:

1. **Try simple examples** first (pick up ball, navigate to goal)
2. **Experiment with descriptions** to understand generation capabilities
3. **Monitor training** to understand skill learning progression
4. **Scale up to complex tasks** (multi-step manipulation, navigation)
5. **Customize reward functions** if needed by editing generated files

## 🤝 **Integration with Existing Workflows**

GenHRL is designed to work with your existing IsaacLab setup:

- **Uses existing environments**: Works with G1 robot configurations
- **Preserves training scripts**: Enhances rather than replaces IsaacLab training
- **Compatible with SKRL**: Uses standard SKRL training framework
- **Maintains file structure**: Follows IsaacLab conventions

The generated tasks can be trained manually using IsaacLab's standard training commands if preferred over the orchestrator.