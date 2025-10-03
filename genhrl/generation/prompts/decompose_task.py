import json

def decompose_task_prompt(task_plan: str, task_name: str, task_description: str, object_config_path: str, objects_config: str, skills, max_hierarchy_levels: int = 3, object_name_mapping: str = "") -> str:
    
    # Generate dynamic hierarchy description
    if max_hierarchy_levels == 1:
        hierarchy_description = "Create a single task with no decomposition. This should be a simple, direct task that doesn't require breaking down into sub-skills."
        restrictions_text = f"""0. **Only one level - the task itself** - You must create only the top-level task with NO children or sub-skills.
         - This means NO decomposition should be performed! The task should be simple enough to be executed directly."""
    elif max_hierarchy_levels == 2:
        hierarchy_description = "Create a two layer hierarchical list of skills for a reinforcement learning agent to complete the task. The top level is the task, and the second level consists of primitive skills."
        restrictions_text = f"""0. **Only two levels INCLUDING the top level task** - You must create a two level hierarchy. Task --> primitive_skill
         - This means that only one level of skills may be produced! You must ensure that this happens and no more."""
    elif max_hierarchy_levels == 3:
        hierarchy_description = "Create a three layer hierarchical list of skills for a reinforcement learning agent to complete the task. This is the RECOMMENDED and OPTIMAL approach. The top level is the task, the second level consists of high-level skills, and the third level contains primitive sub-skills."
        restrictions_text = f"""0. **RECOMMENDED: Three levels INCLUDING the top level task** - You must create a three level hierarchy. Task --> skill --> sub_skill
         - This is the optimal structure for hierarchical reinforcement learning
         - Second level: High-level skills that accomplish major phases of the task
         - Third level: Primitive sub-skills that are reusable and focused
         - This means that only two levels of skills may be produced! You must ensure that this happens and no more."""
    elif max_hierarchy_levels == 4:
        hierarchy_description = "Create a four layer hierarchical list of skills for a reinforcement learning agent to complete the task. The top level is the task, followed by three levels of increasingly specific skills."
        restrictions_text = f"""0. **Only four levels INCLUDING the top level task** - You must create a four level hierarchy. Task --> skill --> sub_skill --> sub_sub_skill
         - This means that only three levels of skills may be produced! You must ensure that this happens and no more."""
    else:
        hierarchy_description = f"Create a {max_hierarchy_levels} layer hierarchical list of skills for a reinforcement learning agent to complete the task. The top level is the task, followed by {max_hierarchy_levels-1} levels of increasingly specific skills."
        restrictions_text = f"""0. **Only {max_hierarchy_levels} levels INCLUDING the top level task** - You must create a {max_hierarchy_levels} level hierarchy.
         - This means that only {max_hierarchy_levels-1} levels of skills may be produced! You must ensure that this happens and no more."""
    
    prompt = f"""
    # 🧩 HIERARCHICAL TASK SKILL DECOMPOSITION

    **🚫 NO ASCII ART**: Do NOT create any ASCII art, diagrams, visual representations, or layouts using dashes, lines, symbols, or characters. Provide only concise text descriptions and JSON.

    ## Task Context
    **Task Name:** {task_name}
    **Task Description:** {task_description}

    ## 🎯 Primary Objective
    {hierarchy_description} Each must:
    - Have a clear, measurable goal state. This goal state should tie into the next skill in the sequence and represent a distinct, stable configuration from which a new reinforcement learning policy can effectively begin. It must be well defined.
    - Focus on achieving a specific position or configuration.
    - Focus on skills that result in a significant change in the robot's position or the position of an object. Avoid fine-grained adjustments or manipulations that would be lost when a new policy takes over for the subsequent skill.
    - Include detailed description of exactly what the robot needs to acheive, what the goal state should be, how this ties into the next skill. You must make this descriptive enough for a robot to follow it.
    - Be reusable where possible across different tasks and within this task if possible.
    - Be as simple and short as possible, focusing on discrete, achievable goals. Complex maneuvers should be broken down into a sequence of simpler skills, each ending in a stable state suitable for transitioning to the next skill in an RL chain.
      - The smaller and easier the skill, the better the robot will learn. And since we are going to build these together, then this is preferable to more complex skills.
      - It is important that the skills are extremely well defined and can easily chain together.
    - These will be used in reinforcement learning tasks, where each skill is executed by a separate policy. Therefore, the end state of one skill becomes the initial state for the next. Skills must be robust enough that the exact precision of the previous skill's termination does not critically affect the success of the next. Focus on substantial changes in state (e.g., robot moved to a new location, object relocated) rather than precise alignments or delicate operations.
    - Each level of the hierarchy must make sense in a sequence to acheive the goal state. For instance, the middle level sequentially must make sense even without the lower level skills.

    ## 🚫 Skill Creation Restrictions
    {restrictions_text}
    1. **NO perception skills** - The robot already knows all object positions and properties
    2. **NO memory-dependent skills** - No "next" or "previous" references
    3. **NO search/scan skills** - No need to locate or identify objects
    4. **NO decision-making skills** - Focus on actions, not decisions
    5. **NO stabalization as a seperate skill** - Stabalization is a natural part of each skill
    6. **NO absolute positions** - You must use relative positions to objects and the robot. INCORRECT: walk to x=2.0m. CORRECT: walk to lowWall
    7. **Use descriptive object names** - Reference objects by their descriptive names from object configuration. PREFERRED: walk_to_lowWall. ACCEPTABLE: walk_to_Object1. INCORRECT: walk_to_nearestWall
    8. **NO orientation or fine alignment skills** - You should not have any skills that are solely about orientation or achieving very precise alignments. Gross positioning is acceptable, but fine-tuning is not, as it will be lost when transitioning to a new RL policy for the next skill.
    9. **NO fine motor skills that require delicate or sustained precision** - Skills should be robust and achievable with a degree of tolerance. For example, 'place_block_on_platform' is better than 'insert_peg_into_hole_with_millimeter_accuracy'.
    
    ## 📋 Skill Naming Requirements
    1. **Be explicit and precise** - Use clear action verbs
    2. **Use descriptive object names from object configuration** - Reference objects using their descriptive names from the object comments:
       - ✅ PREFERRED: "walk_to_lowWall", "push_sphere_towards_wall", "pick_up_redCube"
       - ✅ ACCEPTABLE: "walk_to_Object1", "push_Object2_towards_Object3" (if object names unclear)
       - ❌ NEVER USE: "walk_to_nearestWall", "push_largest_sphere", "pick_up_closest_cube"
    3. **NO relative or ambiguous terms** - Never use terms like "nearest", "largest", "closest", "leftmost", "smallest"
       - These terms are ambiguous and not suitable for reinforcement learning
    4. **Reference specific objects** - Each skill must target a specific, explicitly named object
    5. **Use underscore format** - Connect actions with "_to_", "_with_", "_on_", etc.
    6. **Extract names from object mapping** - Use the object comments to determine appropriate descriptive names

    ## 📊 Skills Checklist - Each skill must have:
    - ✓ Clear beginning and end states, where the end state is a stable configuration suitable for initiating a new RL policy for the subsequent skill.
    - ✓ Measurable success criteria
    - ✓ Each skill should be as simple as possible, representing a distinct phase of movement or object interaction. Can you break each skill down into even smaller skills that still result in a significant, stable change in state?
    - ✓ Logical sequence from previous skill
    - ✓ Realistic physical actions for a humanoid robot
    - ✓ Have we maximised the amount of skills that can be reused?
    - ✓ Focus on gross motor movements (robot or object displacement) rather than fine manipulation or precise alignment.

    ## 🤖 Robot Capabilities
    - Humanoid robot with standard locomotion and manipulation
    - Already knows position/properties of all objects
    - Can perform basic humanlike movements: walk, reach, grasp, climb, push, etc.
    - Cannot teleport or perform physically impossible actions


    ## Examples of good skills
    - walk_to_lowWall (using descriptive name from object configuration)
    - walk_to_cardboardBox (using descriptive name from object configuration)
    - place_feet_onto_top_of_lowWall
    - jump_down_from_lowWall
    - place_hands_onto_sphere
    - push_sphere_towards_wall
    - raise_feet_onto_platform
    - remain_on_platform
    - jump_while_remaining_on_platform
    - pick_up_redCube
    - carry_redCube_to_goalZone

    ## Examples of bad skills
    - Orient to object (not needed)
    - turn to face (not needed)
    - locate object (not needed)
    - walk_to_nearestObject (unclear and ambiguous)
    - walk_to_nearestWall (ambiguous - should be walk_to_lowWall)
    - pick_up_largest_cube (ambiguous - should be pick_up_redCube)


    ## 🗂️ Object Configuration
    ```json
    {objects_config}
    ```
    
    ## 🔗 Object Name Mapping
    The following shows what each ObjectN represents in this scene:
    {object_name_mapping if object_name_mapping else "Object mappings will be provided during generation"}
    
    **CRITICAL**: Extract descriptive names from the object mapping comments and use them in skill names.
    - PREFERRED: Use descriptive names like "lowWall", "sphere", "redCube" from the object comments
    - ACCEPTABLE: Use Object1, Object2, etc. if descriptive names are unclear
    - NEVER USE: Relative terms like "nearest", "largest", "closest"

    ## 📝 Task Execution Plan
    {task_plan}

    ## 📚 Available Skill Library (Optional reference)
    {skills}

    ## 📋 Output Format Requirements
    Return a JSON object with this structure, you must ensure that it is a valid JSON object!

    {"For level 1 (single task only):" if max_hierarchy_levels == 1 else "For multi-level hierarchy:"}
    
    {f'''{{
      "name": "walk_forward",
      "description": "Walk forward in a straight line."
    }}''' if max_hierarchy_levels == 1 else '''{{
      "name": "obstacle_course",
      "description": "Complete the obstacle course.",
      "children": [
        {{
          "name": "JumpOverLowWall",
          "description": "Approach and jump over the low wall.",
          "children": [
            {{"name": "walk_to_lowWall", "description": "Walk close to the low wall."}},
            {{"name": "jump_over_lowWall", "description": "Perform the jump action over the low wall."}},
            {{"name": "stabilize_after_jump", "description": "Regain balance after landing."}}
          ]
        }},
        {{
          "name": "PushSphere", 
          "description": "Push the sphere towards the wall.",
           "children": [
            {{"name": "walk_to_sphere", "description": "Walk to the sphere."}},
            {{"name": "push_sphere_towards_wall", "description": "Push the sphere in direction of the wall."}}
           ]
        }}
      ]
    }}'''}

    ## 🔍 Example Skill Structure
    GOOD SKILL:
    ```
    {{
      "name": "raise_feet_onto_top_of_lowWall",
      "description": "The robot should approach the low wall, place both hands on top of it, then use its arms to push upward while simultaneously lifting one leg over, followed by the other leg. The robot should maintain balance throughout and end in a standing position on the opposite side of the low wall.",
      "object_config_paths": ["{object_config_path}"]
    }}
    ```

    BAD SKILL:
    ```
    {{
      "name": "move_to_next_position",
      "description": "The robot should move to the next position in the sequence.",
      "object_config_paths": ["{object_config_path}"]
    }}
    ```
    (Bad because: uses vague "next" term, lacks specific direction or target, doesn't describe how the robot should move)

    BAD SKILL (RL context):
    ```
    {{
      "name": "align_fingers_perfectly_on_button",
      "description": "The robot should meticulously align its fingertips with the precise center of the button before pressing.",
      "object_config_paths": ["{object_config_path}"]
    }}
    ```
    (Bad because: focuses on fine alignment that is likely to be lost or irrelevant when a new policy starts for the 'press_button' skill. A better skill would be 'reach_for_button' ending when the hand is near the button, followed by 'press_button'.)

    CRITICAL: Return only the JSON object, no additional text.

    ## OUTPUT FORMAT
    Provide the skill hierarchy in the following JSON format. Use double curly braces `{{}}` for the outer structure and ensure all internal JSON uses double curly braces as well if you are including this example literally. **Only output the JSON object.**

    Example you must reply in a valid JSON object format such as below:

    {f'''{{
      "name": "walk_forward",
      "description": "Walk forward in a straight line."
    }}''' if max_hierarchy_levels == 1 else '''{{
      "name": "obstacle_course",
      "description": "Complete the obstacle course.",
      "children": [
        {{
          "name": "JumpOverLowWall",
          "description": "Approach and jump over the low wall.",
          "children": [
            {{"name": "walk_to_lowWall", "description": "Walk close to the low wall."}},
            {{"name": "jump_over_lowWall", "description": "Perform the jump action over the low wall."}},
            {{"name": "stabilize_after_jump", "description": "Regain balance after landing."}}
          ]
        }},
        {{
          "name": "PushSphere", 
          "description": "Push the sphere towards the wall.",
           "children": [
            {{"name": "walk_to_sphere", "description": "Walk to the sphere."}},
            {{"name": "push_sphere_towards_wall", "description": "Push the sphere in direction of the wall."}}
           ]
        }}
      ]
    }}'''}


    # Task Plan:
    # {task_plan}

    # Task Name:
    # {task_name}

    # Task Description:
    # {task_description}

    # Object Config Path:
    # {object_config_path}

    # Objects Config:
    # {objects_config}


    # INSTRUCTIONS:
    # Generate the hierarchical JSON skill structure for the task '{task_name}' based on the plan and description.
    # Ensure skill names are descriptive and follow snake_case_naming_convention.
    # Breakdown should be logical. Use existing skills only if they perfectly match a required primitive step.
    # Focus on creating a hierarchy that reflects the sub-task structure described in the task plan.
    # Output ONLY the JSON object.
    """
    
    return prompt
