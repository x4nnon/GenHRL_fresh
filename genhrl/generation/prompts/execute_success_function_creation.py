def execute_success_function_creation_prompt(skill_name: str, skill_description: str, task_description: str, objects_config: str, skill_rewards: str, success_plan: str, objects_mapping: str = "") -> str:
    """Generate a prompt for creating success function based on the success criteria plan."""
    
    objects_config = str(objects_config).replace('"', '\\"').replace('{', '{{').replace('}', '}}')
    skill_rewards = str(skill_rewards).replace('"', '\\"').replace('{', '{{').replace('}', '}}')

    prompt = f"""
# 🧩 SUCCESS FUNCTION IMPLEMENTATION

⚠️ ABSOLUTE REQUIREMENTS - VIOLATIONS WILL BE REJECTED ⚠️
1. SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
2. ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
3. ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
4. NEVER use hard-coded positions or arbitrary thresholds
5. Access objects directly - objects should always exist in the scene
6. ALWAYS use check_success_duration and save_success_state
7. Do not implement reward functions here, only success functions.
8. NEVER use previous_object_positions or any non-approved (examples below) attributes assigned to env.attribute
9. DO NOT INCLUDE ANY IMPORT STATEMENTS - All necessary imports are already provided by the template


## SKILL INFORMATION
**Skill Name:** {skill_name}
**Skill Description:** {skill_description}
**Task Description:** {task_description}

## 🧱 Object Configuration (full)
```json
{objects_config}
```

## 🔖 Object Name Mapping (compact)
```json
{objects_mapping}
```

⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
You MUST access objects using these exact names: env.scene['Object1'], env.scene['Object2'], etc.

## 🤔 STEP-BY-STEP THINKING PROCESS
Before implementing the success function, think through these steps carefully:
1. First, analyze the success criteria plan to understand the required measurements
2. Next, identify which robot parts and objects need to be accessed
3. Then, plan how to calculate the necessary distances using approved patterns
4. Consider edge cases and ensure all objects are properly accessed
5. Finally, verify that all measurements use only relative distances
6. Double-check that the implementation follows all critical rules

## SUCCESS CRITERIA PLAN -- You may change this if it does not align with your rules and requirements, it is just a guide.
```
{success_plan}
```

## 📝 Skill Rewards (for context) -- You should ensure your success criteria is aligned with the rewards.
```python
{skill_rewards}
```

## ⚠️ CRITICAL IMPLEMENTATION RULES - READ FIRST
    
### MOST CRITICAL RULES:
0. **All success criteria must only be based on relative distances between objects and robot parts, NO OTHER MEASUREMENTS MAY BE USED FOR SUCCESS CRITERIA**

1. **USE ONLY APPROVED ACCESS PATTERNS** - Only use the methods and attributes shown in the examples below
2. **NO ENVIRONMENT VARIABLES** - Never store variables on the env object (e.g., `env.initial_X = ...`)
3. **HANDLE TENSOR OPERATIONS CORRECTLY** - All operations must work with batched environments
4. **CHECK SUCCESS DURATION** - Always use check_success_duration function (already imported) 
5. **SAVE SUCCESS STATES** - Always use save_success_state for successful environments
6. ** THERE IS NO way to access the SIZE of an object** - if you need this. You must read this from the object config and hard code the value for it.
7. ** YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN** - if you need this.
          - INCORRECT: wall_x_location = 3.9 'this uses a hard coded value which is incorrect!'
          - INCORRECT: Target_x_location = 2.7 'this uses a hard coded value which is incorrect!'
          - CORRECT: wall_x_location = env.scene['Object3'].data.root_pos_w[:, 0] 'this uses the approved pattern to get the x location of the wall'
          - CORRECT: Target_x_location = env.scene['Object3'].data.root_pos_w[:, 0] 'this uses the approved pattern to get the x location of the wall'
8. **NO IMPORT STATEMENTS** - Do NOT include any import statements in your generated code. All necessary imports (DoneTerm, torch, etc.) are already provided by the template.
9. **ONLY USE Object1...Object5** - Objects are accessed using env.scene['Object1'], env.scene['Object2'], etc. based on the object configuration above.


### POSITION & MOTION RULES:
1. **NO ORIENTATION OR LINEAR VELOCITY CHECKS** - Velocity is too volatile to check reliably
2. **NO ROBOT DIMENSION ASSUMPTIONS** - Do not use arbitrary values for positions of limbs or pelvis
3. **USE RELATIVE DISTANCES** - Always calculate distances between objects/robot parts, not absolute positions. 
4. **NO INITIAL POSITIONS** - Success must be robust to variations in starting positions
5. **YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS. FOR INSTANCE, WALKING TO A WALL ONLY REQUIRES THE X COMPONENT TO BE CLOSE. THE OTHER COMPONENTS ARE IRRELEVANT IN THIS CASE.**
    
### SUCCESS CRITERIA RULES:
1. **USE LENIENT THRESHOLDS** - Primary conditions should be strict but secondary conditions should be lenient
2. **AVOID IMPOSSIBLE CONDITIONS** - Ensure success criteria are physically possible
3. **REASONABLE TOLERANCES** - Use appropriate thresholds (typically 0.05-0.1m for distances)

## 🚫 COMMON ERRORS TO AVOID
1. ❌ **NEVER USE data.size or data.root_size** - These attributes don't exist on RigidObjectData
       - INCORRECT: `low_wall_x_size = low_wall.data.size[0, 0]` 
       - ERROR: AttributeError: 'RigidObjectData' object has no attribute 'size'
    
2. ❌ **NEVER ACCESS OBJECT DIMENSIONS FROM RigidObject** - Use object configuration instead
       - INCORRECT: `football_radius = football.radius[0]`
       - INCORRECT: `wall_width = wall.size[1]` 
       - INCORRECT: `ball_diameter = ball.dimensions[0]`
       - ERROR: AttributeError: 'RigidObject' object has no attribute 'radius'/'size'/'dimensions'
       - CORRECT: Get dimensions from object configuration JSON and hardcode the values:
       ```python
       # From object config: "Object1": {{"type": "sphere", "radius": 0.1}}
       football_radius = 0.1  # Get from object config, not from object
       
       # From object config: "Object2": {{"type": "box", "size": [0.5, 2.0, 0.3]}} 
       wall_width = 2.0  # Get from object config, not from object
       ```
    
3. ❌ **NEVER USE dictionary-style scene access** 
       - INCORRECT: `if "Object1" in env.scene:` or `robot = env.scene.get("robot")`
       - CORRECT: `obj = env.scene["Object1"]`
    
4. ❌ **NEVER USE hard-coded body indices** 
       - INCORRECT: `robot.data.body_pos_w[:, 0]` 
       - CORRECT: `robot.data.body_pos_w[:, robot.body_names.index('pelvis')]`
    
5. ❌ **NEVER USE quat_rotate_vector or mdp.quat_rotate_vector** - These are undefined

6. ❌ **NEVER INCLUDE IMPORT STATEMENTS** - All imports are provided by the template
       - INCORRECT: `from isaaclab.utils.rewards import DoneTerm` (this path doesn't exist)
       - INCORRECT: `from isaaclab.managers import TerminationTermCfg as DoneTerm` (already imported)
       - CORRECT: Just use `DoneTerm` directly - it's already available

7. ❌ **NEVER USE incorrect object names** - You must use Object1, Object2, Object3, Object4, or Object5 based on the object configuration

## 🧱 HOW TO USE OBJECT DIMENSIONS

When you need object dimensions (radius, size, height, etc.), you MUST:

1. **Read the object configuration JSON** to understand the object properties
2. **Hardcode the dimension values** directly in your success functions
3. **NEVER try to access dimensions from the RigidObject** - they don't exist

### ✅ CORRECT Examples:
```python
# If object config shows: "Object1": {{"type": "sphere", "radius": 0.15}}
football = env.scene['Object1']
football_pos = football.data.root_pos_w
football_radius = 0.15  # CORRECT: Get from object config

# Check if robot is close enough to football (accounting for its size)
distance_to_football = torch.norm(pelvis_pos - football_pos, dim=1)
success_condition = distance_to_football < (football_radius + 0.1)

# If object config shows: "Object2": {{"type": "box", "size": [0.4, 1.8, 0.2]}}
wall = env.scene['Object2'] 
wall_pos = wall.data.root_pos_w
wall_width = 1.8   # CORRECT: Get from object config  
wall_height = 0.2  # CORRECT: Get from object config

# Check if robot is within wall width bounds
y_distance = torch.abs(pelvis_pos[:, 1] - wall_pos[:, 1])
within_wall_bounds = y_distance < (wall_width / 2)
```

### ❌ INCORRECT Examples:
```python
# WRONG - These attributes don't exist!
football_radius = football.radius[0]           # AttributeError!
wall_width = wall.size[1]                      # AttributeError!
ball_diameter = ball.dimensions[0]             # AttributeError!
object_height = obj.data.root_size[2]          # AttributeError!
```

## 🤖 APPROVED ACCESS PATTERNS - ONLY USE THESE PATTERNS

### Accessing Robot Body Parts:
```python
# Getting the robot object
robot = env.scene["robot"]
            
# Get indices for specific body parts
left_hand_idx = robot.body_names.index('left_palm_link')
right_hand_idx = robot.body_names.index('right_palm_link')
        
# Get positions, these are absolute positions in the world frame (not environment frame) so you must always make them relative to something else.
left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]  # Shape: [num_envs, 3]
right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    
must always be used as torch.norm(pos1 - pos2, dim=1) never used as an absolute value on it's own.

### Available Body Parts:
- Hands: 'left_palm_link', 'right_palm_link'
- Feet: 'left_ankle_roll_link', 'right_ankle_roll_link'
- Body: 'pelvis', 'head_link'
- Joints: 'left_knee_link', 'right_knee_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link'
- Fingers: 'left_two_link', 'right_two_link' (thumbs), 'left_five_link', 'right_five_link', 'left_six_link', 'right_six_link'


### Accessing Object Positions, these are absolute positions in the world frame (not environment frame) so you must always make them relative to something else.
```python
# Direct access to known objects - ONLY use Object1, Object2, Object3, Object4, Object5
object1 = env.scene['Object1']  # Based on object configuration above
object_pos = object1.data.root_pos_w  # Shape: [num_envs, 3]
    
# Direct access to multiple objects
for object_name in ["Object1", "Object2", "Object3", "Object4", "Object5"]:
    obj = env.scene[object_name]
    # Process obj...
```

### Distance Calculations:
```python
# Calculate distance between two positions, this is how you should calculate distances between objects.

pos1 = object1.data.root_pos_w
pos2 = object2.data.root_pos_w

distance = torch.norm(pos1 - pos2, dim=1)  # Euclidean distance in 3D
    
# Distance in specific dimensions
xy_distance = torch.norm(pos1[:, :2] - pos2[:, :2], dim=1)  # Only x,y plane
x_distance = torch.abs(pos1[:, 0] - pos2[:, 0])
y_distance = torch.abs(pos1[:, 1] - pos2[:, 1])
z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.

```


### Checking Success Duration and Saving States:
```python
# Where condition is your tensor of booleans
success = check_success_duration(env, condition, skill_name, duration=1.0)
    
# Save success states for environments that succeeded
if success.any():
    for env_id in torch.where(success)[0]:
        save_success_state(env, env_id, skill_name)
```

## 📋 ANNOTATED EXAMPLE - STUDY THIS CAREFULLY

```python
def grasp_ball_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the grasp_ball skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CORRECT: Direct indexed access to the robot
    robot = env.scene["robot"]
    
    # CORRECT: Using robot.body_names.index instead of hardcoded indices
    right_thumb_idx = robot.body_names.index('right_two_link')
    right_finger1_idx = robot.body_names.index('right_five_link')
    right_finger2_idx = robot.body_names.index('right_six_link')
    
    # CORRECT: Direct object access using Object1...Object5 names
    ball = env.scene['Object1']  # Assuming Object1 is the ball (checked from the object config)
    ball_pos = ball.data.root_pos_w  # [num_envs, 3]
    
    # CORRECT: Getting positions for all envs at once (handling batch operations)
    right_thumb_pos = robot.data.body_pos_w[:, right_thumb_idx]  # [num_envs, 3]
    right_finger1_pos = robot.data.body_pos_w[:, right_finger1_idx]
    right_finger2_pos = robot.data.body_pos_w[:, right_finger2_idx]
    
    # CORRECT: Using relative distances, not absolute positions
    thumb_distance = torch.norm(right_thumb_pos - ball_pos, dim=1)
    finger1_distance = torch.norm(right_finger1_pos - ball_pos, dim=1)
    finger2_distance = torch.norm(right_finger2_pos - ball_pos, dim=1)
    
    # CORRECT: Using reasonable threshold with appropriate tolerance
    contact_threshold = 0.05  # 5cm distance threshold for contact
    
    # CORRECT: Computing success condition as tensor operation (handles batched envs)
    num_contacts = (thumb_distance < contact_threshold).float() + \
                  (finger1_distance < contact_threshold).float() + \
                  (finger2_distance < contact_threshold).float()
    
    # CORRECT: Adding a second condition that's physically meaningful, z absolute position is the only absolute position allowed.
    ball_height = ball_pos[:, 2]
    ball_lifted = ball_height > 0.2  # Ball is lifted at least 20cm
    
    # CORRECT: Combining conditions with proper tensor operations
    grasp_success = (num_contacts >= 2) & ball_lifted
    
    # CORRECT: Using check_success_duration and appropriate duration
    success = check_success_duration(env, grasp_success, "grasp_ball", duration=1.0)
    
    # CORRECT: Saving success states for successful environments
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "grasp_ball")
    
    return success
    
class SuccessTerminationCfg:
    success = DoneTerm(func=grasp_ball_success)
```

## 🔍 YOUR TASK

Implement the success function for the "{skill_name}" skill following all requirements.
    
Your implementation MUST:
1. Include explicit reasoning as inline comments explaining how you're addressing each requirement
2. Use only approved access patterns shown above
3. Carefully handle tensor operations for batched environments
4. Use appropriate thresholds with reasonable tolerances
5. Avoid all common errors listed above
    
## 🧪 CODE TESTING SECTION
    
Before submitting your final implementation, test your code against these scenarios:
    
1. Does your code handle batch processing correctly? All operations should work on tensors of shape [num_envs, ...]
2. Does your code access all required objects directly?
3. Are your success thresholds physically reasonable?
4. Have you avoided all the common errors listed above?
5. Does your success function properly check duration and save success states?

## ✅ FINAL COMPLIANCE CHECKLIST
    
Before submitting, verify your implementation meets all these requirements:
    
- [ ] Function only uses approved access patterns
- [ ] No initial positions or environment variables used
- [ ] Success duration properly checked with appropriate duration value
- [ ] Success states saved for successful environments
- [ ] Reasonable thresholds with appropriate tolerances
- [ ] No velocity checks included in success criteria
- [ ] All tensor operations correctly handle batched environments
- [ ] Objects accessed directly using the approved patterns
- [ ] Detailed comments explain implementation reasoning

Create a function named "{skill_name}_success" and a SuccessTerminationCfg class following this template:

```python
def {skill_name}_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the {skill_name} skill has been successfully completed.'''
    # 1. Get robot parts
    robot = env.scene["robot"]
    part_idx = robot.body_names.index('part_name')  # Replace with actual part
    part_pos = robot.data.body_pos_w[:, part_idx]
    
    # 2. Get object position
    object = env.scene['ObjectName']  # Replace with actual object
    object_pos = object.data.root_pos_w
    
    # 3. Calculate distance (use specific components when appropriate)
    distance = torch.norm(part_pos - object_pos, dim=1)  # or x_distance = torch.abs(part_pos[:, 0] - object_pos[:, 0])
    
    # 4. Check success condition
    condition = distance < 0.1  # Example threshold, adjust as needed
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    success = check_success_duration(env, condition, "{skill_name}", duration=0.5)  # Adjust duration if needed
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "{skill_name}")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func={skill_name}_success)
```

## IMPLEMENTATION TEMPLATE - FOLLOW THIS STRUCTURE

```python
# Standard imports - DO NOT MODIFY keep these exactly as they are and do not add any other uneeded imports.
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer # this automatically sets up the RewNormalizer instance.

from genhrl.generation.objects import get_object_volume

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations  
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

def main_{skill_name}_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for {skill_name}.'''
    # 1. Get robot parts
    robot = env.scene["robot"]
    part_idx = robot.body_names.index('part_name')  # Replace with actual part
    part_pos = robot.data.body_pos_w[:, part_idx]
    
    # 2. Get object position
    object = env.scene['ObjectName']  # Replace with actual object
    object_pos = object.data.root_pos_w
    
    # 3. Calculate distance
    distance = torch.norm(part_pos - object_pos, dim=1)  # Or specific dimension
    
    # 4. Compute reward
    reward = 1.0 / (1.0 + distance)  # Example reward function
    
    # 5. Normalize reward - DO NOT MODIFY THIS SECTION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

# Add more reward functions following the same pattern

@configclass
class TaskRewardsCfg:
    # Main reward with weight 1.0
    MainReward = RewTerm(func=main_{skill_name}_reward, weight=1.0, 
                      params={{"normalise": True, "normaliser_name": "main_reward"}})
    
    # Supporting rewards with weights ≤ 0.6
    # SupportReward1 = RewTerm(...)
```

## DO'S AND DON'TS - REFERENCE EXAMPLES

### ✅ DO - Check specific distance components:
```python
# Success based on x-position only (e.g., past a wall)
x_pos_diff = robot_pos[:, 0] - wall_pos[:, 0]
condition = x_pos_diff > 0.2  # Robot is past the wall

# Success based on height (z-component)
z_distance = torch.abs(hand_pos[:, 2] - object_pos[:, 2])
height_condition = z_distance < 0.05  # Hand is at right height
```

### ✅ DO - Combine conditions correctly:
```python
# Multiple conditions must ALL be true
condition = (x_distance < 0.1) & (z_distance < 0.05)
```

### ❌ DON'T - Use velocities or orientations:
```python
# WRONG
velocity = robot.data.root_lin_vel_w
condition = velocity[:, 0] < 0.1  # Checking if robot is stopped
```

### ❌ DON'T - Use environment variables:
```python
# WRONG
if not hasattr(env, 'initial_positions'):
    env.initial_positions = object_pos.clone()
```

## ✅ FINAL CHECKLIST
Before submitting, verify ALL of these:
[ ] Success criteria use ONLY distances between objects and robot parts
[ ] All object positions are accessed with env.scene['ObjectName'].data.root_pos_w
[ ] All robot parts are accessed with robot.body_names.index('part_name')
[ ] No hard-coded positions or arbitrary values are used
[ ] Objects accessed directly using the approved patterns
[ ] check_success_duration and save_success_state are properly implemented

⚠️ FINAL INSTRUCTION: ⚠️
Generate ONLY the success function and SuccessTerminationCfg class.
DO NOT include any import statements - they are already provided by the template.
DO NOT generate reward functions - we are only generating success criteria.
Respond with only Python code, no explanations or markdown.
"""
    return prompt 