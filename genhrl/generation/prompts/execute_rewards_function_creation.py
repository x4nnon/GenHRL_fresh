def execute_rewards_function_creation_prompt(skill_name: str, skill_description: str, task_description: str, objects_config: str, reward_plan: str, objects_mapping: str = "") -> str:
    """Generate a prompt for creating reward functions based on the reward design plan."""
    
    # Escape the JSON string by replacing quotes and braces
    objects_config = str(objects_config).replace('"', '\\"').replace('{', '{{').replace('}', '}}')
    reward_plan = str(reward_plan).replace('"', '\\"').replace('{', '{{').replace('}', '}}')

    prompt = f"""
# 🏆 REWARD FUNCTION IMPLEMENTATION

⚠️ ABSOLUTE REQUIREMENTS - VIOLATIONS WILL BE REJECTED ⚠️
1. ALL rewards MUST ONLY use relative distances between objects and robot parts
2. ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
3. ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
4. NEVER use hard-coded positions or arbitrary thresholds.
5. Access objects directly - objects should always exist in the scene
6. ALWAYS implement proper reward normalization
7. NEVER use previous_object_positions or any non-approved (examples below) attributes assigned to env.attribute

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
Before implementing the reward functions, think through these steps carefully:
1. First, analyze the reward design plan to understand the required measurements
2. Next, identify which robot parts and objects need to be accessed
3. Then, plan how to calculate the necessary distances using approved patterns
4. Consider edge cases and ensure all objects are properly accessed
5. Finally, verify that all rewards use only relative distances and are continuous
6. Double-check that the implementation follows all critical rules

## REWARD DESIGN PLAN -- You may change this if it does not align with your rules and requirements, it is just a guide.
```
{reward_plan}
```

## ⚠️ CRITICAL IMPLEMENTATION RULES - READ FIRST

⚠️ **THIS IS ISAAC LAB - NOT GENERIC RL!** ⚠️
You MUST generate Isaac Lab compatible code with proper imports and normalization!

### MOST CRITICAL RULES:
1. **MANDATORY IMPORTS** - You MUST include these exact imports at the top:
   ```python
   from isaaclab.managers import RewardTermCfg as RewTerm
   from isaaclab.utils import configclass
   from isaaclab.managers import SceneEntityCfg
   from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
   from genhrl.generation.objects import get_object_volume
   from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
    import torch

   from isaaclab.envs import mdp
    # Import custom MDP functions from genhrl
    import genhrl.generation.mdp.rewards as custom_rewards
    import genhrl.generation.mdp.terminations as custom_terminations  
    import genhrl.generation.mdp.observations as custom_observations
    import genhrl.generation.mdp.events as custom_events
    import genhrl.generation.mdp.curriculums as custom_curriculums
   ```

2. **MANDATORY REWARD NORMALIZATION** - EVERY reward function MUST include normalization:
   ```python
   # Get normalizer instance is generated in the import.
   
   # At the end of EVERY reward function:
   if normaliser_name not in RewNormalizer.stats:
       RewNormalizer.stats[normaliser_name] = RewardStats()

   if normalise:
       scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
       RewNormalizer.update_stats(normaliser_name, reward)
       return scaled_reward
   return reward
   ```

3. **MANDATORY FUNCTION SIGNATURE** - Every reward function MUST have this signature:
   ```python
   def function_name(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "reward_name") -> torch.Tensor:
   ```

4. **USE ONLY APPROVED ACCESS PATTERNS** - Only use the methods and attributes shown in the examples below
5. **NO ENVIRONMENT VARIABLES** - Never store variables on the env object (e.g., `env.initial_X = ...`)
6. **HANDLE TENSOR OPERATIONS CORRECTLY** - All operations must work with batched environments
7. **CONTINUOUS REWARDS** - Use smooth, continuous rewards.
8. ** THERE IS NO way to access the SIZE of an object** - if you need this. You must read this from the object config and hard code the value for it.
9. ** YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN** - if you need this.
          - INCORRECT: wall_x_location = 3.9 'this uses a hard coded value which is incorrect!'
          - INCORRECT: Target_x_location = 2.7 'this uses a hard coded value which is incorrect!'
          - CORRECT: wall_x_location = env.scene['Object3'].data.root_pos_w[:, 0] 'this uses the approved pattern to get the x location of the wall'
          - CORRECT: Target_x_location = env.scene['Object3'].data.root_pos_w[:, 0] 'this uses the approved pattern to get the x location of the wall'
10. **Suitable for RL** - You must ensure that the reward space for each skill should be as linear as possible, local minima.
11. **ONLY USE Object1...Object5** - Objects are accessed using env.scene['Object1'], env.scene['Object2'], etc. based on the object configuration above.
    
### POSITION & MOTION RULES:
1. **USE RELATIVE DISTANCES** - Always calculate distances between objects/robot parts, not absolute positions
2. **NO INITIAL POSITIONS** - Success must be robust to variations in starting positions
3. **CAREFUL WITH SIGNS** - The robot can move in negative x/y directions; ensure reward signs are correct
    
### REWARD STRUCTURE RULES:
1. **IMPLEMENT ALL COMPONENTS** - Create primary and all supporting reward functions from the plan
2. **PROPER WEIGHTS** - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)
3. **DIRECT OBJECT ACCESS** - Objects should always exist in the scene, access them directly
4. Where possible, rewards should be continuous and positive.
5. **absolute distances** must be used for distances from objects. For example abs(pelvis_pos[:, 0] - low_wall_pos[:, 0]) 
6. All rewards must only be based on relative distances between objects and robot parts, NO OTHER MEASUREMENTS MAY BE USED FOR SHAPING REWARDS.
7. You must consider the x, y and z components of distances seperately, including their thresholds. However, in most cases you will only want one threshold to be small and all other can be very lenient!
    - For example, walking to a wall. we want a threshold in the x direction of being close. But the threshold in the y direction should be the width of the wall in the y direction. and z should not be considered. 

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
    
6. ❌ **DO NOT DEFINE or IMPORT RewNormalizer** - It's already imported elsewhere

7. ❌ **NEVER USE incorrect object names** - You must use Object1, Object2, Object3, Object4, or Object5 based on the object configuration

## 🧱 HOW TO USE OBJECT DIMENSIONS

When you need object dimensions (radius, size, height, etc.), you MUST:

1. **Read the object configuration JSON** to understand the object properties
2. **Hardcode the dimension values** directly in your reward functions
3. **NEVER try to access dimensions from the RigidObject** - they don't exist

### ✅ CORRECT Examples:
```python
# If object config shows: "Object1": {{"type": "sphere", "radius": 0.15}}
football = env.scene['Object1']
football_pos = football.data.root_pos_w
football_radius = 0.15  # CORRECT: Get from object config

# Calculate target position accounting for football size
target_x_pos = football_pos[:, 0] - football_radius - 0.1

# If object config shows: "Object2": {{"type": "box", "size": [0.4, 1.8, 0.2]}}
wall = env.scene['Object2'] 
wall_pos = wall.data.root_pos_w
wall_width = 1.8   # CORRECT: Get from object config  
wall_height = 0.2  # CORRECT: Get from object config

# Use wall dimensions in calculations
safe_distance_y = wall_width / 2 + 0.2  # Half width plus clearance
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

### Reward Normalization:
⚠️ **MANDATORY** - You must include this EXACT code in EVERY reward function!
```python
# Get normalizer instance (add at the top of each function)
RewNormalizer = get_normalizer(env.device)

# Initialize normalizer if needed (at the end of each function before return)
if normaliser_name not in RewNormalizer.stats:
    RewNormalizer.stats[normaliser_name] = RewardStats()

# Normalize and update stats (always at the end)
if normalise:
    scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
    RewNormalizer.update_stats(normaliser_name, reward)
    return scaled_reward
return reward
```

## 📋 ANNOTATED EXAMPLE - STUDY THIS CAREFULLY

```python
# CORRECT: Standard imports - do not modify these, do not add any other imports or import reward stats from anywhere else, it is handled in get_normalizer
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
import torch

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations  
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums


def main_jump_over_lowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for jump_over_lowWall.

    Reward for moving the robot's pelvis to a position 0.5 meters beyond the low wall in the x-direction.
    This encourages the robot to jump over and land past the wall, reaching the desired final position for this skill and preparing for the next skill.
    '''
    # RewNormalizer is generated in the import. no need to do again.
    
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    low_wall = env.scene['Object3'] # CORRECT: Accessing object using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing object position using approved pattern

    distance_x = torch.abs(pelvis_pos[:, 0] - low_wall_pos[:, 0]) # CORRECT: Relative distance in x-direction
    reward_x = -(distance_x - 0.5) # CORRECT: Reward based on relative distance and target of 0.5m past the wall, continuous reward

    pelvis_z = pelvis_pos[:, 2]
    activation_z = pelvis_pos[:,0] < low_wall_pos[:,0]
    reward_z = torch.where(activation_z, pelvis_z, torch.tensor(0.0, device=env.device))

    reward = reward_x + reward_z

    # CORRECT: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def distance_to_goal_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "facing_reward") -> torch.Tensor:
    '''Supporting reward being closer to the goal.
    '''
    # CORRECT: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)
    
    # CORRECT: Using approved access patterns for robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # CORRECT: Direct object access
    goal = env.scene['Object4'] # making sure that object4 is the goal in the object config
    goal_pos = goal.data.root_pos_w
    
    # CORRECT: Proper tensor operations that work with batch envs
    xy_distance_to_goal = torch.abs(goal_pos[:, :2] - pelvis_pos[:, :2])
    
    # CORRECT: Smooth continuous reward based on alignment
    reward = -xy_distance_to_goal
    
    # CORRECT: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward



@configclass
class TaskRewardsCfg:
    # CORRECT: Main reward with weight 1.0
    Main_JumpOverLowWallReward = RewTerm(func=main_jump_over_lowWall_reward, weight=1.0, 
                                params={{"normalise": True, "normaliser_name": "main_reward"}})
    
    # CORRECT: Supporting rewards with lower weights
    DistanceToGoalReward = RewTerm(func=distance_to_goal_reward, weight=0.3,
                              params={{"normalise": True, "normaliser_name": "distance_to_goal_reward"}})
    
```

## 🔍 YOUR TASK

⚠️ **GENERATE ISAAC LAB CODE - NOT GENERIC RL CODE!** ⚠️

Implement the reward functions for the "{skill_name}" skill following all requirements.
    
Your implementation MUST:
1. **START WITH THE MANDATORY IMPORTS** shown above
2. **USE ISAAC LAB FUNCTION SIGNATURES** with env: ManagerBasedRLEnv parameter
3. **INCLUDE REWARD NORMALIZATION** in every single function 
4. Include explicit reasoning as inline comments explaining how you're addressing each requirement
5. Use only approved access patterns shown above
6. Carefully handle tensor operations for batched environments
7. Use smooth, continuous rewards
8. **END WITH TaskRewardsCfg** using @configclass decorator
9. **Follow the exact example format** - DO NOT deviate!
    
## 🧪 CODE TESTING SECTION
    
Before submitting your final implementation, test your code against these scenarios:
    
1. Does your code handle batch processing correctly? All operations should work on tensors of shape [num_envs, ...]
2. Does your code access all required objects directly?
3. Are your reward functions smooth and continuous (not binary 0/1)?
4. Have you avoided all the common errors listed above?
5. Does your code properly normalize all rewards?
6. Are your reward weights appropriate (main reward ~1.0, supporting rewards typically 0.1-0.5)?

## ✅ FINAL COMPLIANCE CHECKLIST
    
Before submitting, verify your implementation meets all these requirements:
    
- [ ] All reward functions use only approved access patterns
- [ ] No initial positions or environment variables used
- [ ] All reward functions properly implement normalization
- [ ] All rewards are continuous (not binary 0/1)
- [ ] Objects accessed directly using the approved patterns
- [ ] TaskRewardsCfg includes all reward components with appropriate weights
- [ ] All tensor operations correctly handle batched environments
- [ ] Detailed comments explain implementation reasoning
- [ ] Rewards use relative positions/distances, not absolute coordinates


    """
    return prompt 