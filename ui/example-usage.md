# GenHRL UI Example Usage

This guide shows how to use the GenHRL UI to create and manage hierarchical RL tasks.

## Quick Start

1. **Start the UI**:
   ```bash
   cd ui
   ./start-ui.sh  # On Linux/Mac
   # or
   start-ui.bat   # On Windows
   ```

2. **Open your browser** to http://localhost:3000

3. **Get an API key** from:
   - Google AI Studio: https://makersuite.google.com/
   - Anthropic Console: https://console.anthropic.com/

## Example: Creating a Box Stacking Task

1. **Navigate to "Create Task"** in the sidebar

2. **Fill in the form**:
   - **Task Name**: `Box_Stacking_Challenge`
   - **Description**: 
     ```
     The environment should contain three boxes of different sizes. The small box should be 0.3m wide, 0.3m deep, and 0.2m tall, weighing 5kg. The medium box should be 0.5m wide, 0.5m deep, and 0.3m tall, weighing 10kg. The large box should be 0.7m wide, 0.7m deep, and 0.4m tall, weighing 15kg.
     
     The boxes should be initially placed randomly around the robot within a 3m radius. The robot should pick up each box and stack them in order of size, with the largest box on the bottom, medium box in the middle, and small box on top. The final stack should be stable and centered.
     ```
   - **Robot**: G1
   - **Hierarchy Levels**: 3 (full hierarchy)
   - **API Key**: Your API key

3. **Click "Create Task"** and wait for generation to complete

4. **View the results** in the task detail page

## Example Generated Structure

After creation, you'll see:

### Dashboard
- Task count and statistics
- Recent tasks overview
- Quick actions

### Task Details
- **Overview**: Task description and stats
- **Hierarchy**: Visual skill tree showing:
  ```
  Box_Stacking_Challenge
  ├── Locate_Objects
  │   ├── Find_Small_Box
  │   ├── Find_Medium_Box
  │   └── Find_Large_Box
  ├── Stack_Boxes
  │   ├── Place_Large_Box_Base
  │   ├── Stack_Medium_Box
  │   └── Stack_Small_Box_Top
  └── Verify_Stack_Stability
  ```
- **Skills**: Individual skill details with:
  - Reward functions
  - Success criteria
  - Training status
- **Objects**: Object configuration JSON
- **Library**: Skill library data

### Generated Files

The UI creates the same structure as the CLI:
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/G1_generated/
├── tasks/Box_Stacking_Challenge/
│   ├── description.txt
│   ├── skills_hierarchy.json
│   └── object_config.json
└── skills/Box_Stacking_Challenge/
    ├── skill_library.json
    ├── object_config.json
    └── skills/
        ├── Locate_Objects/
        ├── Find_Small_Box/
        ├── Find_Medium_Box/
        └── [other skills...]
```

## Comparing UI vs CLI

### Using the UI:
1. Open web interface
2. Fill out form
3. Click create
4. Browse and inspect results visually

### Using the CLI:
```python
from genhrl.generation import create_task_with_workflow

task_config = create_task_with_workflow(
    task_name="Box_Stacking_Challenge",
    task_description="...",  # Same description
    isaaclab_path="./IsaacLab",
    api_key="your_api_key",
    robot="G1",
    max_hierarchy_levels=3
)
```

Both methods create identical results - the UI is just a visual frontend to the same GenHRL API.

## Tips for Best Results

### Task Descriptions
- **Be specific** about object properties (size, weight, material)
- **Include spatial relationships** between objects
- **Describe the sequence** of required actions
- **Mention success criteria** explicitly

### Good Example:
```
The robot should navigate to a table containing three colored balls (red, blue, green) 
of 0.1m diameter. Each ball weighs 0.5kg. The robot should pick up the balls one by 
one and place them in matching colored containers located 2m away. The red container 
is on the left, blue in the center, and green on the right. Success is achieved when 
all balls are in their correct containers.
```

### Hierarchy Levels
- **Level 1**: Simple, single-task scenarios
- **Level 2**: Basic skill decomposition
- **Level 3**: Complex hierarchical breakdown (recommended)

## Troubleshooting

### Task Creation Fails
1. Check API key validity
2. Verify internet connection
3. Check console logs in browser developer tools

### No Tasks Showing
1. Verify IsaacLab path in `server/index.js`
2. Check file system permissions
3. Ensure tasks were created successfully

### UI Won't Start
1. Install Node.js 16+
2. Run `npm run install-all` in the ui directory
3. Check that ports 3000 and 5000 are available

## Next Steps

Once your task is created:

1. **Review the generated files** in IsaacLab
2. **Examine the skill hierarchy** to understand the decomposition
3. **Check individual skills** for reward and success functions
4. **Start training** using the GenHRL training system:
   ```bash
   genhrl train Box_Stacking_Challenge
   ```

The UI provides a visual way to understand and manage your hierarchical RL tasks, making it easier to iterate on task design and monitor progress.