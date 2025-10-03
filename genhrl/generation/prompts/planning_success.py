def planning_success_prompt(skill_name: str, skill_description: str, task_description: str, objects_config: str, skill_rewards: str, next_skill_name: str, next_skill_description: str, objects_mapping: str = "") -> str:
    """
    Generate a prompt for planning success criteria for a skill.
    
    This focuses on WHAT success criteria are needed before implementing them.
    """
    return f"""
# SUCCESS CRITERIA PLANNING

⚠️ ABSOLUTE REQUIREMENTS - VIOLATIONS WILL BE REJECTED ⚠️
1. SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
2. NO OTHER MEASUREMENTS MAY BE USED (no velocities, orientations, or time-based criteria)
3. Consider x, y, and z components of distances separately with appropriate thresholds
4. z distances should only be used when height is important to the skill.
5. Plan only ONE success criterion based on the FINAL STATE of the skill
6. You must ensure that the success criterion will work with the likely start position of the next skill in the chain. e.g. if the current skill is to jump over a wall. and the next skill is to push a sphere. then you shouldn't jump past the sphere.
7. NEVER use absolute or hard-coded positions. Everything should be relative to objects or robot parts.
7. NEVER use previous_object_positions or any temporal measurements or attributes. They are not available.


## SKILL INFORMATION
**Skill Name:** {skill_name}
**Skill Description:** {skill_description}
**Task Description:** {task_description}

## OBJECT CONFIGURATION (full)
```json
{objects_config}
```

## 🔖 Object Name Mapping (compact)
```json
{objects_mapping}
```

⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
When planning success criteria, reference objects using these exact names: Object1, Object2, etc.

## 🧱 HOW TO USE OBJECT DIMENSIONS IN PLANNING

When planning success criteria that need object dimensions (radius, size, height, etc.), you MUST:

1. **Read the object configuration JSON** to understand the object properties
2. **Plan to hardcode the dimension values** in the success functions
3. **NEVER plan to access dimensions from RigidObject** - they don't exist

### ✅ CORRECT Planning Examples:
```
If object config shows: "Object1": {{"type": "sphere", "radius": 0.15}}
- Plan: "Check if robot is within football radius (0.15m) plus clearance for contact"
- Implementation: football_radius = 0.15  # Get from object config

If object config shows: "Object2": {{"type": "box", "size": [0.4, 1.8, 0.2]}}  
- Plan: "Check if robot is past wall considering wall height (0.2m) for clearance"
- Implementation: wall_height = 0.2  # Get from object config
```

### ❌ INCORRECT Planning Examples:
```
- Plan: "Check robot distance using football.radius"       # WRONG - doesn't exist!
- Plan: "Use wall.size[2] for height threshold"           # WRONG - doesn't exist!
- Plan: "Dynamically determine object dimensions"         # WRONG - not possible!
```

## REWARD FUNCTIONS (FOR REFERENCE)
```python
{skill_rewards}
```

## PLANNING OBJECTIVE
1. Analyze the reward functions to understand what they encourage the robot to achieve
2. Plan a single, clear success criterion measuring when the skill is complete
3. DO NOT IMPLEMENT THE SUCCESS CRITERIA YET - just plan what it should be

## 🤔 STEP-BY-STEP THINKING PROCESS
Before designing the success criteria, think through these steps carefully:
1. First, analyze the reward functions to understand the desired end state
2. Next, identify which robot parts and objects are most relevant for success
3. Then, determine which distance measurements best capture the completion state
4. Consider appropriate thresholds that are achievable but not too lenient
5. Finally, verify that the success criteria align with the reward functions
6. Double-check that the success state provides a good starting point for the next skill

# TEMPLATE FOR SUCCESS CRITERIA skeleton
All success criteria must follow this format:
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

#example success condition choose something sensible based on the skill and reward.
success = distance_x < 0.5 & distance_y < 2

return success
'''

### Available Body Parts:
- Hands: 'left_palm_link', 'right_palm_link'
- Feet: 'left_ankle_roll_link', 'right_ankle_roll_link'
- Body: 'pelvis', 'head_link'
- Joints: 'left_knee_link', 'right_knee_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link'
- Fingers: 'left_two_link', 'right_two_link' (thumbs), 'left_five_link', 'right_five_link', 'left_six_link', 'right_six_link'



## OUTPUT FORMAT

**PLAN:**
[Detailed description of the skill execution, focusing on the final desired state]

**SUCCESS CRITERION:**
- Description: [What this criterion checks - the core end state]
- Measurements needed: [Specific distances to measure]
- Objects tracked: [Objects and robot parts involved]
- Success threshold: [Specific distance or condition with appropriate tolerance]
- Duration required: [How long this condition must be maintained, typically 0.5-1.0 seconds]
- Feasibility Comment: [Why this is measurable and achievable]

## ESSENTIAL RULES TO FOLLOW

1. **Final State Focus:** Success criteria must represent the FINAL STATE of the skill, not intermediate steps
2. **Lenient Thresholds:** Use reasonable thresholds with appropriate tolerance (typically 0.3-1m)
    - the success criterion should be based on the reward functions, but slightly more lenient.
3. **Relative Distances Only:** Only use relative distances between objects and robot parts
4. **No Velocities or Orientations:** These measurements are too volatile for reliable success checks
5. **Consider Object Shapes:** Success may vary based on shape (e.g., distance to any point along a wall)
6. **Use Separate Components:** Consider x, y, z components separately when appropriate

## EXAMPLES OF CORRECT & INCORRECT APPROACHES

### ✅ GOOD SUCCESS CRITERION:
- Description: Robot pelvis is on the other side of the wall
- Measurements: X-axis distance between pelvis and wall
- Objects tracked: Robot pelvis, Object3 (wall)
- Success threshold: Pelvis x-position > wall x-position + 0.2m
- Duration required: 0.3 seconds

### ❌ BAD SUCCESS CRITERIA:
- Using velocities: "Robot is moving at least 1m/s"
- Using orientations: "Robot torso is upright"
- Using fixed height: "Pelvis is at least 0.3m high"
- Using time: "Task completed in under 10 seconds"
- Using incorrect object names: "Robot near wall" (should reference specific ObjectN)

## ✅ SELF-VERIFICATION CHECKLIST
[ ] Is my success criterion based ONLY on distances between objects and robot parts?
[ ] Does it measure the FINAL STATE of the skill, not intermediate steps?
[ ] Have I considered x, y, z components separately where appropriate?
[ ] Is my threshold reasonable and achievable (not too strict, no unnecessary constraints)?
[ ] Have I specified a reasonable duration (typically 0.1-0.5 seconds)?
[ ] Does the success criterion align perfectly with the reward functions?
[ ] Will the success criterion be sensible for the start of the next skill?
[ ] Have I made sure that there are no absolute or hard-coded positions?
[ ] Are all objects referenced using Object1, Object2, Object3, Object4, or Object5 names?
""" 

