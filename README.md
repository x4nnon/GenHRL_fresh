# GenHRL: Generative Hierarchical Reinforcement Learning Framework

GenHRL is a comprehensive framework for automatically generating and training hierarchical reinforcement learning tasks from natural language descriptions. It integrates with IsaacLab to create complete training pipelines for complex robotic tasks.

## 🎨 **New: Web User Interface**

GenHRL now includes a modern web-based user interface for creating and managing tasks visually!

### Features
- **Visual Task Creation**: Generate tasks through an intuitive web form
- **Hierarchy Visualization**: Interactive skill tree exploration
- **Task Management**: Browse, search, and filter all your generated tasks
- **Skill Inspection**: View rewards, success criteria, and training status
- **Real-time Monitoring**: Track task generation progress

### Quick Start with UI
```bash
cd ui
./start-ui.sh  # Linux/Mac
# or start-ui.bat on Windows
```

Then open http://localhost:3000 in your browser!

For detailed UI documentation, see [`ui/README.md`](ui/README.md) and [`ui/example-usage.md`](ui/example-usage.md).

## Features

### 🎯 **Task Generation**
- **Natural Language Input**: Describe tasks in plain English
- **Configurable Hierarchy Levels**: Control complexity with 1-3 hierarchy levels
  - Level 1: Single task only (no decomposition)
  - Level 2: Task → Skill decomposition
  - Level 3: Task → Skill → Sub-skill (full hierarchy)
- **Automatic Scene Generation**: Creates object configurations and environments
- **Hierarchical Skill Decomposition**: Breaks complex tasks into trainable skill hierarchies
- **Reward Function Generation**: Automatically creates reward functions for each skill
- **Success Criteria**: Generates termination conditions for skill completion

### 🚀 **Training Orchestration**
- **Sequential Training**: Trains skills in proper dependency order
- **State Transfer**: Transfers success states between skills for curriculum learning
- **Progress Monitoring**: Real-time training progress and success state tracking
- **Resume Training**: Continue from checkpoints and skip completed skills
- **Parallel Support**: Multiple training configurations and environments

### 🔧 **IsaacLab Integration**
- **Seamless Integration**: Works directly with existing IsaacLab installations
- **Custom Training Scripts**: Includes hierarchical PPO and composite skill training
- **File Generation**: Creates all necessary configuration files in IsaacLab structure
- **Gym Registration**: Automatic environment registration for all skills

### 🛠️ **Programmatic & CLI APIs**
- **Python API**: Full programmatic access for integration into research workflows
- **Command Line Interface**: Easy-to-use CLI for quick task generation and training
- **Web User Interface**: Visual task creation and management
- **Modular Design**: Use individual components (TaskManager, SkillLibrary, etc.) independently
- **Flexible Configuration**: Support for multiple LLM providers (Google Gemini, Anthropic Claude)

## Installation

### Prerequisites
- **Python 3.8+**: Required for GenHRL
- **NVIDIA GPU**: Required for IsaacLab simulation
- **API Key**: Get an API key from Google (Gemini) or Anthropic (Claude)
- **Node.js 16+**: Required for the web UI (optional)

### Complete Installation

```bash
# 1. Clone the repository
git clone https://github.com/x4nnon/GenHRL_v2.git
cd GenHRL_v2

# 2. Install GenHRL
pip install -e .

# 3. Clone IsaacLab into the GenHRL folder
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 4. Install IsaacLab following their instructions
# https://isaac-sim.github.io/IsaacLab/
./isaaclab.sh --install

# 5. Return to GenHRL root and set your API key
cd ..
export GENHRL_API_KEY="your_api_key_here"
# Or add to ~/.bashrc for persistence
echo 'export GENHRL_API_KEY="your_api_key_here"' >> ~/.bashrc

# 6. (Optional) Set up the Web UI
cd ui
npm run install-all
```

### Verify Installation
```bash
# Test IsaacLab works
cd IsaacLab
./isaaclab.sh -p source/standalone/demos/basic.py

# Test GenHRL CLI
cd ..
genhrl --help

# Test GenHRL Python API
python -c "from genhrl.generation import TaskManager; print('✅ GenHRL installed successfully')"

# Test Web UI (optional)
cd ui && ./start-ui.sh
```



### Test README Functionality

Validate that all README examples and functionality work correctly:

```bash
# Quick validation of core functionality
python -m tests.test_runner quick

# Complete validation of all README claims (recommended)
python -m tests.test_runner

# Or use pytest
pytest tests/ -v

# Test specific functionality
pytest tests/test_installation.py -v  # Installation verification
pytest tests/test_workflows.py -v     # Workflow examples
pytest tests/test_config_and_structure.py -v  # Configuration validation
```

The test suite validates:
- ✅ All installation instructions work
- ✅ All code examples in README are functional
- ✅ API signatures match documentation
- ✅ File structures are generated correctly
- ✅ Configuration options work as described

See `tests/README_TESTS.md` for detailed information about the test suite.

## Quick Start

### Option 1: Web User Interface (Recommended for Beginners)

1. **Start the UI**:
   ```bash
   cd ui
   ./start-ui.sh  # Linux/Mac or start-ui.bat on Windows
   ```

2. **Open your browser** to http://localhost:3000

3. **Create a task** using the visual interface with example descriptions and helpful tips

4. **Explore your tasks** with the interactive hierarchy viewer and skill inspector

### Option 2: Programmatic API (Recommended for Research)

For research and development, use the Python API for maximum control:

```python
from genhrl.generation import create_task_with_workflow

# Simple task creation with full workflow
task_config = create_task_with_workflow(
    task_name="Pick_Up_Ball",
    task_description="The robot should pick up the red ball and place it on the table",
    isaaclab_path="./IsaacLab",  # Path to your IsaacLab installation
    api_key="your_api_key_here",
    robot="G1",
    max_hierarchy_levels=3,  # 1-3 levels of hierarchy
    remove_previous=True,    # Clean up previous tasks
    verify_decompose=True,   # Enable verification steps
    verify_plan=False,
    verify_rewards=False,
    verify_success=False
)

print(f"✅ Task created at: {task_config.get_task_path()}")
print(f"🎯 Skills directory: {task_config.get_skills_base_path() / 'skills'}")
```

For the exact task from your previous implementation:

```python
from genhrl.generation import main_create_steps_example

# Direct replacement for your previous main() function
task_config = main_create_steps_example(
    isaaclab_path="./IsaacLab",
    api_key="your_api_key_here",
    remove_previous=True
)
```

### Option 3: Command Line Interface

**Note**: All commands use the local `./IsaacLab` installation by default. You only need to specify `--isaaclab-path` if using a different IsaacLab installation.

#### 1. Generate a Task

```bash
# Generate a new hierarchical task (uses local ./IsaacLab automatically)
genhrl generate "pick up the red ball and place it on the table" \
    --api-key YOUR_API_KEY

# Generate with specific hierarchy levels (1-3, default: 3)
# Level 1: Single task only (no decomposition)
genhrl generate "walk to box" \
    --api-key YOUR_API_KEY \
    --max-hierarchy-levels 1

# Level 2: Task -> Skill decomposition  
genhrl generate "pick up the ball" \
    --api-key YOUR_API_KEY \
    --max-hierarchy-levels 2

# Level 3: Task -> Skill -> Sub-skill (full hierarchy)
genhrl generate "climb stairs by stepping on each step" \
    --api-key YOUR_API_KEY \
    --max-hierarchy-levels 3
```

#### 2. Train the Skills

```bash
# Train all skills in the hierarchy
genhrl train pick_up_the_red
```

#### 3. One-Command Generation and Training

```bash
# Generate and train in one command (default: 3 hierarchy levels)
genhrl auto "robot climbs stairs by stepping on each step" \
    --api-key YOUR_API_KEY
```

## Programmatic API Reference

### Core Components

```python
from genhrl.generation import (
    TaskManager,           # Main orchestration class
    HierarchicalCodeGenerator,  # LLM-powered code generation
    SkillLibrary,         # Skill management and persistence
    create_task_with_workflow,  # Complete workflow function
    remove_all_previous_skills  # Cleanup utility
)
```

### Basic Usage Patterns

#### 1. Complete Workflow (Recommended)

```python
from genhrl.generation import create_task_with_workflow

# Create a complete task with one function call
task_config = create_task_with_workflow(
    task_name="Navigation_Task",
    task_description="Navigate to the target while avoiding obstacles",
    isaaclab_path="/path/to/IsaacLab",
    api_key="your_api_key",
    robot="G1",                    # Supports G1, H1, etc.
    max_hierarchy_levels=3,        # 1-3 levels
    remove_previous=False,         # Don't clean up existing tasks
    verify_decompose=True,         # Enable LLM verification
    verify_plan=False,
    verify_rewards=False,
    verify_success=False
)
```

#### 2. Step-by-Step Control

```python
from genhrl.generation import TaskManager

# Initialize task manager
task_manager = TaskManager(
    isaaclab_path="/path/to/IsaacLab",
    api_key="your_api_key",
    robot="G1"
)

# Create task with fine-grained control
task_config = task_manager.create_task_from_description(
    task_name="My_Task",
    task_description="Task description here...",
    max_hierarchy_levels=2,
    robot="G1"
)

# Access generated files
print(f"Object config: {task_config.get_task_path() / 'object_config.json'}")
print(f"Skills hierarchy: {task_config.get_task_path() / 'skills_hierarchy.json'}")
```

#### 3. Custom Code Generator

```python
from genhrl.generation import HierarchicalCodeGenerator

# Create code generator with custom settings
generator = HierarchicalCodeGenerator(
    api_key="your_api_key",
    provider="google",        # "google" or "anthropic"
    model="gemini-2.0-flash",
    backup_model="gemini-2.0-flash",
    model_big="gemini-2.5-pro-preview-05-06",
    verify_decompose=True,
    verify_plan=False,
    verify_rewards=True,
    verify_success=True
)

# Generate components individually
objects_config = generator.generate_objects_config("your task description")
skills_hierarchy = generator.decompose_task("task_name", "description", "config_path", objects_config)
```

#### 4. Skill Library Management

```python
from genhrl.generation import SkillLibrary

# Initialize skill library
skill_library = SkillLibrary("/path/to/skills/skill_library.json")

# Add skills and hierarchies
skill_library.add_hierarchy("task_name", skills_hierarchy, "task description")

# Query skills
primitive_skills = skill_library.list_primitive_skills()
composite_skills = skill_library.list_composite_skills()
skill_sequence = skill_library.get_full_skill_sequence("task_name")

# Export/import libraries
skill_library.export_library("/path/to/backup.json")
skill_library.import_library("/path/to/other_library.json", merge=True)
```

### Cleanup and Utilities

```python
from genhrl.generation import remove_all_previous_skills, strip_markdown_formatting

# Clean up previous tasks for a robot
remove_all_previous_skills(
    isaaclab_path="/path/to/IsaacLab",
    robot="G1"
)

# Utility functions
clean_code = strip_markdown_formatting(llm_response_with_markdown)
```

### Training Integration

```python
from genhrl.training import TrainingOrchestrator, TrainingConfig

# Set up training
orchestrator = TrainingOrchestrator(
    isaaclab_path="/path/to/IsaacLab",
    task_name="My_Task",
    robot="G1"
)

# Configure training
config = TrainingConfig(
    max_time_minutes=120,
    min_success_states=50,
    num_envs=4096,
    seed=42
)

# Train all skills in hierarchy
success = orchestrator.train_all_skills(config=config)
```

## Advanced Configuration

### LLM Provider Configuration

#### Google Gemini (Default)
```python
generator = HierarchicalCodeGenerator(
    api_key="your_google_api_key",
    provider="google",
    model="gemini-2.0-flash",
    backup_model="gemini-2.0-flash",
    model_big="gemini-2.5-pro-preview-05-06"
)
```

#### Anthropic Claude
```python
generator = HierarchicalCodeGenerator(
    api_key="your_anthropic_api_key",
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    backup_model="claude-sonnet-4-20250514",
    model_big="claude-opus-4-20250514"
)
```

### Verification Settings

Enable different types of LLM verification for higher quality generation:

```python
create_task_with_workflow(
    # ... other parameters ...
    verify_decompose=True,  # Verify task decomposition
    verify_plan=True,       # Verify execution plans
    verify_rewards=True,    # Verify reward functions
    verify_success=True     # Verify success criteria
)
```

### Robot Support

GenHRL supports multiple robots through the robot configuration system:

```python
from genhrl.generation import get_available_robots, get_robot_config

# See available robots
robots = get_available_robots()
print(f"Supported robots: {robots}")

# Get robot configuration
config = get_robot_config("G1")
print(f"Hip height: {config['hip_height']}")
```

## Commands (CLI)

### Generate Tasks
```bash
# Basic generation (uses local ./IsaacLab automatically)
genhrl generate "task description"

# Control hierarchy complexity (1-3 levels, default: 3)
genhrl generate "simple walking task" \
    --max-hierarchy-levels 1  # Single task only

genhrl generate "pick up object" \
    --max-hierarchy-levels 2  # Task -> Skill

genhrl generate "complex navigation with obstacles" \
    --max-hierarchy-levels 3  # Task -> Skill -> Sub-skill

# With custom name and provider
genhrl generate "complex manipulation task" \
    --name my_task \
    --provider anthropic \
    --max-hierarchy-levels 2

# Use custom IsaacLab path if needed
genhrl generate "task description" --isaaclab-path /custom/path/to/IsaacLab
```

### Train Skills
```bash
# Train all skills
genhrl train my_task

# Train with custom settings
genhrl train my_task \
    --max-time 120 \
    --min-success-states 50 \
    --seed 42 \
    --new-run  # Clean previous training data

# Use custom IsaacLab path if needed
genhrl train my_task --isaaclab-path /custom/path/to/IsaacLab
```

### Monitor Progress
```bash
# List all tasks
genhrl list

# Show task details
genhrl list --task my_task

# Check training status
genhrl status my_task

# Use custom IsaacLab path if needed
genhrl list --isaaclab-path /custom/path/to/IsaacLab
```

## Example Workflows

### Research Workflow (Programmatic)

```python
from genhrl.generation import create_task_with_workflow, remove_all_previous_skills

# Clean workspace
remove_all_previous_skills("./IsaacLab", "G1")

# Generate research task
task_config = create_task_with_workflow(
    task_name="Manipulation_Study",
    task_description="""The robot should pick up objects of different shapes and sizes,
    sort them by color, and place them in designated containers. The robot should 
    adapt its grasping strategy based on object properties.""",
    isaaclab_path="./IsaacLab",
    api_key=os.getenv("GENHRL_API_KEY"),
    robot="G1",
    max_hierarchy_levels=3,
    remove_previous=False,
    verify_decompose=True,
    verify_rewards=True
)

# Train with research configuration
from genhrl.training import TrainingOrchestrator, TrainingConfig

orchestrator = TrainingOrchestrator("./IsaacLab", "Manipulation_Study", "G1")
config = TrainingConfig(
    max_time_minutes=240,
    min_success_states=100,
    num_envs=8192,
    seed=42
)

results = orchestrator.train_all_skills(config=config)
print(f"Training completed: {results}")
```

### Production Workflow (CLI)

```bash
# Complete workflow for production use
export GENHRL_API_KEY="your_key_here"

# Generate and train complex task
genhrl auto "robot assembles furniture by following assembly instructions" \
    --name furniture_assembly \
    --api-key $GENHRL_API_KEY \
    --max-hierarchy-levels 3 \
    --max-time 180 \
    --min-success-states 75

# Monitor results
genhrl status furniture_assembly
genhrl list --task furniture_assembly
```

### UI Workflow (Visual)

1. **Open the Web UI**: http://localhost:3000
2. **Create Task**: Use the visual form with helpful examples and tips
3. **Explore Results**: Interactive hierarchy viewer and skill inspector
4. **Manage Tasks**: Search, filter, and organize your task library
5. **Monitor Training**: Visual progress tracking (coming soon)

### Debugging Workflow

```python
from genhrl.generation import TaskManager

# Create task manager with debugging
task_manager = TaskManager("./IsaacLab", api_key="your_key", robot="G1")

try:
    task_config = task_manager.create_task_from_description(
        task_name="Debug_Task",
        task_description="Simple test task for debugging",
        max_hierarchy_levels=1
    )
    print("✅ Task created successfully")
    
    # Inspect generated files
    task_path = task_config.get_task_path()
    print(f"📁 Task files: {list(task_path.iterdir())}")
    
    skills_path = task_config.get_skills_base_path() / "skills"
    if skills_path.exists():
        print(f"🎯 Generated skills: {list(skills_path.iterdir())}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    # Check logs for detailed error information
```

## Generated File Structure

GenHRL creates the following structure in your IsaacLab installation:

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/
├── tasks/
│   └── task_name/
│       ├── description.txt                 # Original task description
│       ├── skills_hierarchy.json          # Hierarchical skill structure
│       └── object_config.json            # Scene object configuration
└── skills/
    └── task_name/                         # Task-specific skills directory
        ├── skill_library.json             # Task's skill library
        ├── object_config.json            # Object config (training compatible location)
        └── skills/                        # Individual skill directories
            ├── skill_1/                   # Each skill gets its own directory
            │   ├── TaskRewardsCfg.py      # Reward function configuration
            │   ├── SuccessTerminationCfg.py # Success criteria configuration
            │   ├── skill_1_cfg.py         # Skill-specific configuration
            │   ├── agents/                # Agent configuration files
            │   │   ├── skrl_flat_ppo_cfg.yaml
            │   │   └── skrl_hierarchical_ppo_cfg.yaml
            │   └── success_states/        # Directory for success state storage
            ├── skill_2/
            └── ...
```

## Available Interface Options

GenHRL provides three ways to create and manage tasks:

### 1. **Web UI** (ui/)
- Visual task creation with guided forms
- Interactive hierarchy exploration
- Task management dashboard
- Real-time progress tracking
- **Best for**: Beginners, visual learners, quick prototyping

### 2. **Command Line Interface** (genhrl command)
- Quick task generation from terminal
- Scriptable automation
- Training orchestration
- **Best for**: Power users, automation, CI/CD

### 3. **Python API** (programmatic)
- Full programmatic control
- Integration with research workflows
- Custom verification and settings
- **Best for**: Research, custom integrations, advanced users

All three interfaces create identical results and can be used interchangeably!

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and conventions  
- Testing requirements
- Submitting pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GenHRL in your research, please cite:

```bibtex
@software{genhrl2024,
  title={GenHRL: Generative Hierarchical Reinforcement Learning Framework},
  author={GenHRL Team},
  year={2024},
  url={https://github.com/x4nnon/GenHRL_v2}
}
```

#
# Troubleshooting: Native Heap/Allocator Errors ("malloc(): invalid size (unsorted)")
#

If you encounter errors like:

```
malloc(): invalid size (unsorted)
Aborted (core dumped)
```
or other heap corruption/allocator errors when running IsaacLab or GenHRL training, this is usually due to a mismatch between the system's glibc/NVIDIA driver and the Isaac Sim/Omniverse binaries, or a conflict between memory allocators (jemalloc vs. glibc malloc).

**Workaround:**

Before running any GenHRL or IsaacLab training command, set the following environment variables in your shell:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

This forces the use of jemalloc for all native allocations and disables PyTorch's expandable segments, which can trigger allocator bugs in some Isaac Sim builds.

If you do not have jemalloc installed, you can add it with:

```bash
sudo apt install libjemalloc2
```

If the problem persists, consider:
- Running inside the official Isaac Sim Docker container (which ships a compatible glibc/driver stack)
- Downgrading your NVIDIA driver to version 550.x (the last officially tested with Isaac Sim 4.5)
- Ensuring you do not import PyTorch or other libraries before Isaac Sim/Kit is initialised

See the main documentation and [Isaac Sim troubleshooting guide](https://docs.omniverse.nvidia.com/isaacsim/latest/troubleshooting.html) for more details.
