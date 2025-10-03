def planning_rewards_prompt(skill_name: str, skill_description: str, task_description: str, objects_config: str, next_skill_name: str, next_skill_description: str, objects_mapping: str = "") -> str:
    """
    Generate a prompt for planning reward functions for reinforcement learning.
    
    This focuses on identifying WHAT rewards are needed before implementing them.
    """
    return f"""
# 🎯 REINFORCEMENT LEARNING REWARD PLANNING

⚠️ ABSOLUTE REQUIREMENTS - VIOLATIONS WILL BE REJECTED ⚠️
1. Consider x, y, and z components of distances separately with appropriate thresholds.
2. NEVER use absolute or hard-coded positions. Everything should be relative to objects or robot parts.
3. You must ensure that the rewards will work with the likely start position of the next skill in the chain. e.g. if the current skill is to jump over a wall. and the next skill is to push a sphere. then you shouldn't jump past the sphere.
4. NEVER use previous_object_positions or any temporal measurements or attributes. They are not available.
5. You should plan collision avoidance shaping rewards for sensible objects and robot parts.

# MAIN OBJECTIVE
1. Plan a sequence of events that will complete the skill, ending with a final desired position
2. Design rewards for teaching a robot to perform the skill "{skill_name}" through reinforcement learning
3. Focus only on this specific skill, not the entire task
4. DO NOT IMPLEMENT CODE YET - just plan the reward structure
5. You must plan which robot part should be considered to be measured in each skill. For instance, feet, hands, pelvis. Consider sensibly depending on the skill.

## 🤔 STEP-BY-STEP THINKING PROCESS
Before designing the rewards, think through these steps carefully:
1. First, analyze the skill description to identify the key robot parts and objects involved
2. Next, break down the skill into logical phases (approach, interaction, completion)
3. Then, identify which distances need to be measured in each phase
4. Consider how rewards should transition between phases
5. Finally, verify that all rewards use only relative distances and are continuous
6. Double-check that rewards align with the next skill's likely starting position

## SKILL INFORMATION
**Skill Name:** {skill_name}
**Skill Description:** {skill_description}
**Task Description:** {task_description}

## OBJECT CONFIGURATION (full)
{objects_config}

## 🔖 Object Name Mapping (compact)
```json
{objects_mapping}
```

⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
When planning rewards, reference objects using these exact names: Object1, Object2, etc.

## 🧱 HOW TO USE OBJECT DIMENSIONS IN PLANNING

When planning rewards that need object dimensions (radius, size, height, etc.), you MUST:

1. **Read the object configuration JSON** to understand the object properties
2. **Plan to hardcode the dimension values** in the reward functions
3. **NEVER plan to access dimensions from RigidObject** - they don't exist

### ✅ CORRECT Planning Examples:
```
If object config shows: "Object1": {{"type": "sphere", "radius": 0.15}}
- Plan: "Use football radius of 0.15m from object config for distance calculations"
- Implementation: football_radius = 0.15  # Get from object config

If object config shows: "Object2": {{"type": "box", "size": [0.4, 1.8, 0.2]}}  
- Plan: "Use wall width of 1.8m from object config for clearance calculations"
- Implementation: wall_width = 1.8  # Get from object config
```

### ❌ INCORRECT Planning Examples:
```
- Plan: "Access football.radius for distance calculations"  # WRONG - doesn't exist!
- Plan: "Use wall.size[1] for width calculations"          # WRONG - doesn't exist!
- Plan: "Get object dimensions dynamically"                # WRONG - not possible!
```

## REWARD DESIGN STRUCTURE
Your reward system must include:

### 1️⃣ PRIMARY REWARD (weight = 1.0)
- Must directly measure progress toward completing the skill
- Based ONLY on position/distance metrics between robot parts and objects
- Should provide clear gradient toward the goal state (continuous and positive)
- Never use binary rewards (1 or 0) based on conditions. Use continuous rewards instead.

### 2️⃣ SHAPING REWARDS (weights ≤ 0.6 each)
- up to 2 additional task-specific rewards that guide learning
- Must ONLY be based on relative distances between objects and robot parts
- Must use conditional activation based on robot states (e.g., approaching vs. past object): torch.where(condition, reward, 0) # use this carefully, as will only work with positive rewards.
- Where possible, rewards should be continuous and positive. Continous rewards means it is always active.
- if using conditional rewards, you should be explicit about the condition. Do not use relu.


### Other points:
1) If using a pelvis default position, which you should encourage when you want the robot to be standing and stable, you should use pelvis_z = 0.7 as the default position if required.
2) Never use relu, always use torch.abs(). Or define custom activations. 
3) You must make sure the robot is stable before the next skill, the robot always gets a large negative reward for falling over.

# TEMPLATE FOR REWARD skeleton
All rewards must follow this format:
''' python

#access the required objects - USE ONLY Object1, Object2, Object3, Object4, Object5
object_name = env.scene['Object1'] # Access the required object using approved pattern
object_name2 = env.scene['Object2'] # Access the required object using approved pattern  
# continue for as many objects as needed

#access the required robot part(s)
robot_partX_idx = robot.body_names.index('part_name') # Getting the index of the required robot part using approved pattern
robot_partX_pos = robot.data.body_pos_w[:, robot_partX_idx] # Getting the position of the required robot part using approved pattern
robot_partX_pos_x = robot_partX_pos[:, 0]
robot_partX_pos_y = robot_partX_pos[:, 1]
robot_partX_pos_z = robot_partX_pos[:, 2]
# continue for as many robot parts as needed ...

#calculate the distance vector between the object and the robot part
distance_x = object_name.data.root_pos_w[:, 0] - robot_partX_pos[:, 0]
distance_y = object_name.data.root_pos_w[:, 1] - robot_partX_pos[:, 1]
distance_z = object_name.data.root_pos_w[:, 2] - robot_partX_pos[:, 2]

#example activation condition choose something sensible based on the skill and reward.
activation_condition = (robot_partX_pos_x < object_name.data.root_pos_w[:, 0]) & (robot_partX_pos_x > object_name.data.root_pos_w[:, 0] - 0.5)

# reward should be abs distance to the object to
reward = -torch.abs(robot_partX_pos_z - object_name.data.root_pos_w[:, 2])

reward = torch.where(activation_condition, reward, 0) # use this carefully, as will only work with positive rewards.

'''

### Available Body Parts:
- Hands: 'left_palm_link', 'right_palm_link'
- Feet: 'left_ankle_roll_link', 'right_ankle_roll_link'
- Body: 'pelvis', 'head_link'
- Joints: 'left_knee_link', 'right_knee_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link'
- Fingers: 'left_two_link', 'right_two_link' (thumbs), 'left_five_link', 'right_five_link', 'left_six_link', 'right_six_link'

## OUTPUT FORMAT

**PLAN:**
[Detailed plan of the skill execution, sequence of events, and final desired state]

**PRIMARY REWARD:**
- Description: [Clear explanation]
- Reward skeleton: [Reward skeleton]
- Measurements: [Specific distances needed]
- Objects tracked: [Objects/body parts involved]
- Weight: 1.0


**SHAPING REWARD 1:**
- Description: [Clear explanation]
- Reward skeleton: [Reward skeleton]
- Measurements: [Specific distances]
- Objects tracked: [Objects/body parts]
- Weight: [≤ 0.6]
- Activation: [When this reward applies]

[Additional shaping rewards...]

## ✅ SELF-VERIFICATION CHECKLIST
[ ] Have I separated x, y, z components of distances where appropriate?
[ ] Are there NO hard-coded positions or arbitrary values?
[ ] Does every reward contribute to learning the skill?
[ ] Does the reward structure work with the likely start position of the next skill in the chain, We do not want it to overshoot the next skill?
[ ] is the skeleton using the correct format provided?
[ ] Are all objects referenced using Object1, Object2, Object3, Object4, or Object5 names?

## EXAMPLES OF CORRECT & INCORRECT APPROACHES

### ✅ GOOD REWARD:
- Name: Main_reward_jump_over_lowWall
- Description: before the wall, the robot should be rewarded for increasing the height of the pelvis and the feet above the height of the wall (and some clearance).
         This reward should only be active before the wall. We should also be rewarding the robot for decreasing the distance between it's pelvis and an area between the low wall and the large sphere.
- Measurements: Distance from pelvis to the low wall x and z, Distance from pelvis to the large sphere x.
- Objects tracked: Pelvis, Object3 (low wall), Object1 (large sphere)
- Type: mixture of activation and dense continious rewards.
- Weight: 1.0

- (for jumping a wall) Increasing reward for increasing pelvis z height while approaching the wall, reducing reward for increasing pelvis z height after the wall.

### ❌ BAD REWARDS:
- Using time-based rewards: "Reward for completing the task quickly"
- Using orientation checks: "Reward for keeping torso upright"
- Using fixed positions: "Reward when pelvis reaches height of 0.3m"
- Using velocities: "Reward for moving at optimal speed"
- Increasing reward based on x distance forever, without limits of the next skill.
- using pelvis height as a measure of a robot standing ontop of an object. feet would be much better.
- Using incorrect object names: "Reward based on wall position" (should reference specific ObjectN)
""" 