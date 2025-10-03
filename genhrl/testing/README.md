# GenHRL Scientific Testing Framework

This directory contains tools for scientific evaluation of the GenHRL method. The framework allows you to systematically test task generation across multiple tasks and seeds.

## Overview

The testing framework generates 5 different tasks, each with 3 different seeds (15 total experiments by default), to evaluate:
- **Generation consistency**: How consistent is task generation across seeds?
- **Skill consistency**: Do the same tasks generate similar skills across seeds?
- **Performance metrics**: Generation time, success rates

## Files

- `scientific_testing.py` - Main testing script for generation consistency
- `test_skill_execution.py` - Script to test individual skill execution within a task
- `analyze_results.py` - Results analysis and reporting
- `visualize_skill_hierarchies.py` - Script to visualize generated skill hierarchies as flowchart images or text trees
- `install_visualization_deps.py` - Helper script to install image generation dependencies
- `test_image_generation.py` - Test script to verify image generation works
- `example_usage.py` - Example script showing how to use the testing tools
- `results/` - Directory where test results are saved
- `README.md` - This documentation

## Quick Start

### 0. Test Individual Skills in a Task

If you have an existing task and want to test which skills work vs fail:

```bash
# Test all skills in a task (250 steps each)
conda activate env_isaaclab
python -m genhrl.testing.test_skill_execution YOUR_TASK_NAME

# Test with more steps for thorough testing
python -m genhrl.testing.test_skill_execution YOUR_TASK_NAME --steps 500

# Test with different robot
python -m genhrl.testing.test_skill_execution YOUR_TASK_NAME --robot H1
```

### 1. Set Up Task Descriptions

Edit `scientific_testing.py` and fill in the task descriptions in the `define_test_tasks()` method:

```python
tasks = [
    TaskDefinition(
        name="locomotion_task",
        description="Walk forward for 10 meters at a steady pace",  # Add your description
        max_hierarchy_levels=3
    ),
    TaskDefinition(
        name="manipulation_task", 
        description="Pick up a red cube and place it on a blue platform",  # Add your description
        max_hierarchy_levels=3
    ),
    # ... etc
]
```

### 2. Run Generation Testing

```bash
# Set API key first
export GENHRL_API_KEY=your_key_here

# Basic generation testing
python -m genhrl.testing.scientific_testing
```

### 3. Analyze Results

```bash
# View detailed analysis
python -m genhrl.testing.analyze_results results/scientific_test_results_*.json

# Generate CSV report
python -m genhrl.testing.analyze_results results/scientific_test_results_*.json --csv
```

### 4. Install Visualization Dependencies (for image generation)

```bash
# Install image generation dependencies automatically
python genhrl/testing/install_visualization_deps.py

# Or install manually:
pip install graphviz
# Plus system Graphviz (see detailed instructions below)
```

### 5. Visualize Generated Skill Hierarchies

```bash
# Generate compact hierarchy images for all tasks (default - saves to writing_images/hierarchies/)
python genhrl/testing/visualize_skill_hierarchies.py

# Try different layout styles for better report formatting
python genhrl/testing/visualize_skill_hierarchies.py --layout compact     # Tight clusters, narrow
python genhrl/testing/visualize_skill_hierarchies.py --layout staggered  # Smart wave pattern (max 4 tiers)
python genhrl/testing/visualize_skill_hierarchies.py --layout wide       # Original wide layout

# Generate text output to console only
python genhrl/testing/visualize_skill_hierarchies.py --format text

# Generate both images and text files
python genhrl/testing/visualize_skill_hierarchies.py --format both

# Visualize a specific task with compact layout
python genhrl/testing/visualize_skill_hierarchies.py --task build_stairs_seed42 --layout compact
```

## Detailed Usage

### Skill Execution Testing Script

The skill execution testing script (`test_skill_execution.py`) allows you to test individual skills within a task to identify which ones work vs which ones fail:

```bash
python -m genhrl.testing.test_skill_execution TASK_NAME [OPTIONS]
```

**Arguments:**
- `TASK_NAME`: Name of the task to test (required)
- `--steps`: Number of steps to run each skill (default: 250)
- `--robot`: Robot type (default: G1)
- `--isaaclab-path`: Path to IsaacLab installation (default: ./IsaacLab)

**Examples:**

```bash
# Test all skills in Create_Steps task with 250 steps each
conda activate env_isaaclab
python -m genhrl.testing.test_skill_execution Create_Steps

# Test with more steps for more thorough evaluation
python -m genhrl.testing.test_skill_execution Create_Steps --steps 1000

# Test with H1 robot instead of G1
python -m genhrl.testing.test_skill_execution Create_Steps --robot H1

# Use custom IsaacLab path
python -m genhrl.testing.test_skill_execution Create_Steps --isaaclab-path /custom/path/to/IsaacLab
```

**What it does:**
1. Loads the specified task and gets all skills in hierarchical order
2. Tests each skill by running it for the specified number of steps
3. Uses minimal settings (no video, fewer environments) for faster testing
4. Tracks which skills succeed vs fail
5. Provides a detailed summary showing:
   - Success rate overall and by skill type (primitive vs composite)
   - List of working skills with execution times
   - List of failed skills with error messages
   - Specific skills that need attention

**Output:**
- Real-time progress as each skill is tested
- Final summary with success/failure counts
- Detailed breakdown of which skills work and which don't
- Exit code 0 if all skills pass, 1 if any fail

**Note:** This script requires the `env_isaaclab` conda environment to be activated.

### Scientific Testing Script

The main testing script (`scientific_testing.py`) supports various options:

```bash
python -m genhrl.testing.scientific_testing [OPTIONS]
```

**Arguments:**
- `--api-key`: Your LLM API key (or set `GENHRL_API_KEY` environment variable)
- `--isaaclab-path`: Path to IsaacLab installation (default: ./IsaacLab)
- `--robot`: Robot type (default: G1)
- `--provider`: LLM provider - google or anthropic (default: google)
- `--seeds`: Comma-separated seeds (default: 42,123,456)

**Examples:**

```bash
# Set API key as environment variable (recommended)
export GENHRL_API_KEY=your_key_here
python -m genhrl.testing.scientific_testing

# Test with different seeds
python -m genhrl.testing.scientific_testing --seeds 42,123,456,789,999

# Test with H1 robot using Anthropic
python -m genhrl.testing.scientific_testing --robot H1 --provider anthropic

# Or pass API key directly
python -m genhrl.testing.scientific_testing --api-key YOUR_KEY
```

### Results Analysis

The analysis script provides detailed insights:

```bash
python -m genhrl.testing.analyze_results results/scientific_test_results_TIMESTAMP.json [OPTIONS]
```

**Options:**
- `--csv`: Generate CSV report for further analysis
- `--csv-output`: Custom CSV filename

**Analysis includes:**
- Success rates (generation)
- Time statistics (mean, median, std dev)
- Skill consistency across seeds
- Error analysis
- Per-task breakdowns

### Skill Hierarchy Visualization

The visualization script creates both image diagrams and text representations of generated skill hierarchies:

```bash
python genhrl/testing/visualize_skill_hierarchies.py [OPTIONS]
```

**Dependencies for Image Generation:**
```bash
# Install Python package
pip install graphviz

# Install system Graphviz (required for image generation)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows: Download from https://graphviz.org/download/
```

**Options:**
- `--tasks-path`: Path to tasks directory (default: auto-detected)
- `--max-width`: Maximum width for text formatting (default: 100)
- `--task`: Visualize only a specific task by name (default: visualize all tasks)
- `--format`: Output format - text, image, or both (default: image)
- `--layout`: Layout style for images - compact, staggered, or wide (default: compact)
- `--save`: Save visualizations to files instead of printing to console
- `--output-dir`: Directory to save output files (default: writing_images/hierarchies)

**Examples:**

```bash
# Generate compact hierarchy images for all tasks (default)
python genhrl/testing/visualize_skill_hierarchies.py

# Try different layout styles for report formatting
python genhrl/testing/visualize_skill_hierarchies.py --layout compact     # Clustered, narrow
python genhrl/testing/visualize_skill_hierarchies.py --layout staggered  # Smart wave pattern (max 4 tiers)
python genhrl/testing/visualize_skill_hierarchies.py --layout wide       # Original wide

# Generate text output to console
python genhrl/testing/visualize_skill_hierarchies.py --format text

# Generate both images and text files with staggered layout
python genhrl/testing/visualize_skill_hierarchies.py --format both --layout staggered

# Focus on one task with compact layout
python genhrl/testing/visualize_skill_hierarchies.py --task obstacle_course_seed123 --layout compact

# Generate text files for specific task
python genhrl/testing/visualize_skill_hierarchies.py --task obstacle_course_seed123 --format text --save

# Wider text output for detailed descriptions
python genhrl/testing/visualize_skill_hierarchies.py --format text --max-width 150

# Custom output directory with specific layout
python genhrl/testing/visualize_skill_hierarchies.py --layout compact --output-dir /custom/path

# Custom tasks directory with all layout options
python genhrl/testing/visualize_skill_hierarchies.py --tasks-path /path/to/custom/tasks --layout staggered
```

**Output features:**
- 🖼️ **Image Generation**: Clean flowchart-style hierarchy diagrams (PNG + SVG)
- 📐 **Layout Options**: 3 layout styles optimized for different use cases:
  - **Compact**: Tight clusters, minimal width (ideal for reports)
  - **Staggered**: Smart wave pattern with limited depth (prevents infinite staircases)
  - **Wide**: Original horizontal layout (maximum readability)
- 🎨 **Color-coded levels**: Different colors for each hierarchy level
- 📦 **Rounded boxes**: Professional-looking node styling with skill names only
- ➡️ **Directional arrows**: Clear parent-child relationships
- 🌳 **Text mode**: Pretty tree structure with Unicode characters and full descriptions
- 📝 **Smart truncation**: Automatic text wrapping for long skill names
- 🔧 **Leaf node highlighting**: Bold styling for executable skills
- 💾 **Flexible output**: Save to `writing_images/hierarchies/` directory
- 📋 **Summary generation**: Overview file when processing multiple tasks
- 🔀 **Multiple formats**: Generate images, text, or both

## Workflow

### Typical Research Workflow

1. **Define Test Tasks**: Edit task descriptions in `define_test_tasks()`
2. **Test LLM Connectivity**: Run quick test to verify API setup
3. **Full Generation Test**: Run complete generation testing
4. **Analyze Results**: Use analysis script to understand patterns
5. **Iterate**: Modify tasks or parameters based on findings

### Output Structure

```
genhrl/testing/
├── results/
│   ├── scientific_test_results_20241231_143022.json
│   ├── scientific_test_results_20241231_143022.csv
│   └── ...
├── scientific_testing.py
├── analyze_results.py
└── README.md
```

### Results Format

JSON results contain:
- **Metadata**: Robot, provider, timestamp, experiment count, focus (generation_only)
- **Results**: Per-experiment data including:
  - Task name with seed
  - Generation success/failure status
  - Generation timing information
  - Generated skills list
  - Error messages (if any)

## Task Types

The framework tests 5 different task categories:

1. **Locomotion Task**: Basic movement and navigation
2. **Manipulation Task**: Object interaction and manipulation
3. **Navigation Task**: Spatial navigation and path planning
4. **Complex Interaction Task**: Multi-step interactions
5. **Simple Behavior Task**: Single-level behaviors

Each task type tests different aspects of the GenHRL system.

## Scientific Considerations

### Reproducibility
- Seeds ensure reproducible LLM outputs
- All parameters logged in results
- Timestamps for experiment tracking

### Statistical Validity
- Multiple seeds test consistency
- Success rates measured across conditions
- Time measurements for performance analysis

### Consistency Analysis
- Skill overlap analysis across seeds
- Error pattern identification
- Performance variance measurement

## Troubleshooting

### Testing Framework Verification

Before using the testing tools, you can verify that everything is set up correctly:

```bash
# Test that all required imports work
python genhrl/testing/test_imports.py

# Test the example usage script
python genhrl/testing/example_usage.py
```

### Common Issues

**Conda Environment for Skill Testing:**
The skill execution testing script requires the `env_isaaclab` conda environment:

```bash
# Activate the environment first
conda activate env_isaaclab

# Then run skill testing
python -m genhrl.testing.test_skill_execution YOUR_TASK_NAME
```

**Task Not Found:**
If you get "Task not found" errors, verify:
- The task name is correct (case-sensitive)
- The task exists in the robot's tasks directory
- You're using the correct robot type with `--robot`

**Setting up API Key:**

The `GENHRL_API_KEY` environment variable can be set in different ways depending on your operating system:

**Linux/macOS (Bash/Zsh):**
```bash
# Temporary (current session only)
export GENHRL_API_KEY=your_key_here

# Permanent (add to ~/.bashrc, ~/.zshrc, or ~/.profile)
echo 'export GENHRL_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

**Windows (Command Prompt):**
```cmd
# Temporary (current session only)
set GENHRL_API_KEY=your_key_here

# Permanent (system-wide)
setx GENHRL_API_KEY "your_key_here"
```

**Windows (PowerShell):**
```powershell
# Temporary (current session only)
$env:GENHRL_API_KEY="your_key_here"

# Permanent (current user)
[Environment]::SetEnvironmentVariable("GENHRL_API_KEY", "your_key_here", "User")
```

**Verify Environment Variable:**
```bash
# Linux/macOS/Windows (Git Bash)
echo $GENHRL_API_KEY

# Windows (Command Prompt)
echo %GENHRL_API_KEY%

# Windows (PowerShell)
echo $env:GENHRL_API_KEY
```

**API Key Issues:**
```bash
export GENHRL_API_KEY=your_key_here
python -m genhrl.testing.scientific_testing
```

**IsaacLab Path Issues:**
```bash
python -m genhrl.testing.scientific_testing --isaaclab-path /absolute/path/to/IsaacLab
```

**Empty Task Descriptions:**
The script will error if task descriptions are empty. Fill in all descriptions in `define_test_tasks()`.

## Results Interpretation

### Success Rates
- **High generation success (>90%)**: LLM integration working well
- **Low generation success (<50%)**: Check API key, network, task descriptions

### Consistency Metrics
- **High skill consistency**: Stable generation across seeds
- **Low skill consistency**: May indicate LLM temperature/randomness issues
- **Similar generation times**: Consistent complexity across tasks

### Time Analysis
- Generation time: Typically 30-300 seconds per task
- High variance: May indicate instability or LLM service issues

## Extension

### Adding New Task Types

Edit `define_test_tasks()` in `scientific_testing.py`:

```python
TaskDefinition(
    name="your_new_task",
    description="Your task description here",
    max_hierarchy_levels=2,  # 1, 2, or 3
    expected_skills=["skill1", "skill2"]  # Optional validation
)
```

### Custom Analysis

The JSON results can be loaded into Python for custom analysis:

```python
import json
with open('results/scientific_test_results_*.json', 'r') as f:
    data = json.load(f)
    
# Custom analysis here
results = data['results']
# ... your analysis code
```

## Citation

If you use this testing framework in research, please cite the GenHRL project and mention the scientific testing framework.