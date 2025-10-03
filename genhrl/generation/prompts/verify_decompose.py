from .decompose_task import decompose_task_prompt
from typing import Dict


def verify_decompose_prompt(task_name: str, task_description: str, task_plan: str, skills: Dict, object_config_path: str, objects_config: str, max_hierarchy_levels: int = 3) -> str:
    """Generate a prompt to verify and refine the skill decomposition."""
    
    # Generate dynamic hierarchy validation text
    if max_hierarchy_levels == 1:
        hierarchy_validation = """- [ ] **SINGLE LEVEL VALIDATION**: There is ONLY one level - the task itself with NO children or sub-skills. No decomposition should be performed for a level 1 hierarchy.
        - [ ] The JSON should contain only "name" and "description" fields
        - [ ] There should be NO "children" field or it should be empty
        - [ ] The task should be simple enough to be executed by a single RL policy"""
        validation_focus = "Focus on ensuring this is a single, executable task without any decomposition."
    elif max_hierarchy_levels == 2:
        hierarchy_validation = """- [ ] **TWO LEVEL VALIDATION**: There are ONLY two levels INCLUDING the top level task. Structure should be: Task → Skills
        - [ ] Top level has "children" containing 2-5 primitive skills
        - [ ] Each child skill has NO further "children" (or empty children array)
        - [ ] Skills should be primitive, executable actions
        - [ ] No nested sub-skills beyond this level"""
        validation_focus = "Focus on ensuring each skill is primitive and directly executable, with clear boundaries between skills."
    elif max_hierarchy_levels == 3:
        hierarchy_validation = """- [ ] **THREE LEVEL VALIDATION**: There are ONLY three levels INCLUDING the top level task. Structure should be: Task → Skills → Sub-skills
        - [ ] Top level has "children" containing major skill groups
        - [ ] Each skill has "children" containing 2-4 sub-skills
        - [ ] Sub-skills have NO further "children" (or empty children array)
        - [ ] Sub-skills should be primitive, executable actions"""
        validation_focus = "Focus on ensuring the hierarchy has meaningful groupings at the skill level, with primitive actions at the sub-skill level."
    elif max_hierarchy_levels == 4:
        hierarchy_validation = """- [ ] **FOUR LEVEL VALIDATION**: There are ONLY four levels INCLUDING the top level task. Structure should be: Task → Skills → Sub-skills → Primitive Actions
        - [ ] Top level has "children" containing major skill phases
        - [ ] Each skill has "children" containing sub-skill groups
        - [ ] Each sub-skill has "children" containing primitive actions
        - [ ] Primitive actions have NO further "children" (or empty children array)"""
        validation_focus = "Focus on ensuring meaningful abstractions at each level, from high-level phases down to primitive actions."
    else:
        hierarchy_validation = f"""- [ ] **{max_hierarchy_levels} LEVEL VALIDATION**: There are ONLY {max_hierarchy_levels} levels INCLUDING the top level task.
        - [ ] The hierarchy should have exactly {max_hierarchy_levels-1} levels of skills below the task level
        - [ ] Each level should represent a meaningful abstraction
        - [ ] The deepest level should contain primitive, executable actions
        - [ ] No level should exceed the {max_hierarchy_levels} level limit"""
        validation_focus = f"Focus on ensuring {max_hierarchy_levels} meaningful levels of abstraction with primitive actions at the deepest level."

    prompt = f"""
    # 🔍 SKILL DECOMPOSITION VERIFICATION

    ## 📋 Task Information
    **Task Name:** {task_name}
    **Task Description:** {task_description}
    
    ## 📝 Current Hierarchical Skill Sequence
    {skills}

    ## 🎯 Verification Objective
    Perform a critical analysis of the skill decomposition to ensure it's logical, comprehensive, 
    and follows best practices for reinforcement learning skill design.
    
    **HIERARCHY LEVEL {max_hierarchy_levels} FOCUS:** {validation_focus}

    ## 🤔 STEP-BY-STEP THINKING PROCESS
    Before verifying the skill decomposition, think through these steps carefully:
    1. First, analyze each skill's name and description for clarity and specificity
    2. Next, check that skills follow the naming conventions and avoid forbidden terms
    3. Then, verify that each skill has a clear, measurable end state
    4. Consider how skills connect in sequence and support task completion
    5. Finally, ensure that skills are appropriately granular and reusable
    6. Double-check that no skills require search, memory, or absolute positions

    ## 🤖 Robot Capabilities & Limitations
    - Humanoid robot WITHOUT visual perception systems
    - Knows ALL object locations, sizes, and properties at all times
    - NO need for search, scan, or identification actions
    - NO memory of previous actions (can't use "next" or "previous")
    - CAN perform basic humanlike movements: walk, reach, grasp, climb, push, etc.
    - CANNOT perform physically impossible actions
    - CANNOT use absolute positions. Must be relative to robot or objects. INCORRECT: walk to x=2.0m. CORRECT: walk to the object

    ## ✅ Verification Checklist
    Evaluate each skill against these criteria:

    ### 1. Naming Convention
    - [ ] Uses clear action verbs
    - [ ] Includes target objects in name
    - [ ] Avoids temporal references ("next", "previous")
    - [ ] Avoids perception terminology ("find", "identify", "scan")
    - [ ] Uses consistent formatting (underscore_separated with camelCase objects)
    - [ ] Uses the object name directly in the skill name. INCORRECT: walk_to_object. CORRECT: walk_to_lowWall
    
    ### 2. Completeness
    - [ ] All necessary skills for task completion are included
    - [ ] No missing steps in the sequence
    - [ ] All object interactions are accounted for
    - [ ] Have we maximised the amount of skills that can be reused? If not, why not?
    - [ ] Have we included a task description which is descriptive enough for a robot to follow it, does it detail which robot parts should do what are what point in the skill?

    ### 3. Granularity
    - [ ] Skills have appropriate complexity, each skill should be as simple as possible **while still having sensible and crystal clear end-states**
    - [ ] Skills are not redundant or overlapping
    - [ ] Each skill represents a meaningful subgoal

    ### 4. Clarity & Specificity
    - [ ] Descriptions clearly explain HOW to perform the skill
    - [ ] End states are measurable and well-defined
    - [ ] Skills have clear success criteria
    - [ ] Skills are not preparation skills such as "prepare_for_jump" or "position_feet_for_running" or stablization skills


    ### 5. Physical Realism
    - [ ] Skills respect physical constraints and limitations
    - [ ] Sequence maintains proper balance and stability
    - [ ] Object interactions are physically plausible

    ### 6. Task Execution Plan Alignment
    - [ ] Skills align with the original task execution plan
    - [ ] Skills align with the intent of the task name and description. The execution is logical and coeherent.
    - [ ] No unnecessary deviations from plan
    - [ ] All plan elements are reflected in skills
    - [ ] **CRITICAL** The end state of each skill must tie into the next skill in the sequence.

    ### 7. Hierarchical Structure
    {hierarchy_validation}
    
    ### 8. Hierarchy-Specific Examples
    
    {f'''**GOOD Level 1 Examples:**
    - Single task: {{"name": "walk_to_box", "description": "Walk to the red box and stop in front of it"}}
    - Single task: {{"name": "pick_up_ball", "description": "Pick up the ball and hold it"}}
    
    **BAD Level 1 Examples:**
    - Has children: {{"name": "walk_to_box", "children": [...]}} ❌ (No children allowed)
    - Too complex: {{"name": "navigate_obstacle_course", "description": "..."}} ❌ (Too complex for single skill)''' if max_hierarchy_levels == 1 else f'''**GOOD Level {max_hierarchy_levels} Examples:**
    - Clear {max_hierarchy_levels}-level structure with appropriate abstraction at each level
    - Primitive actions only at the deepest level
    - Meaningful skill groupings at intermediate levels
    
    **BAD Level {max_hierarchy_levels} Examples:**
    - Wrong number of levels (less than {max_hierarchy_levels} or more than {max_hierarchy_levels})
    - Non-primitive actions at deeper levels than allowed
    - Meaningless intermediate groupings'''}
    
    
        ## Examples of good skills
    - walk_to_low_wall
    - walk_to_high_wall
    - walk_to_block
    - walk_to_sphere
    - walk_to_position_in_front_of_lowWall
    - raise_feet_onto_top_of_lowWall
    - remain_on_lowWall
    - step_down_from_lowWall_towards_sphere
    - place hands on sphere
    - push_sphere_towards_highwall
    - raise_feet_ont_block
    - remain_on_block
    - jump_while_remaining_on_block
    - etc

    ## Examples of bad skills.
    - prepare_for_jump
    - stabalize_robot
    - position_feet_for_running
    - jump_over_lowWall (This is too complex as a single skill, you should break this down into smaller skills)
    - orientate to object
    - face object 
    - find object

    ## 📝 Execution Plan for Reference
    ```
    {task_plan}
    ```

    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```

    ## 🚫 Common Decomposition Errors
    1. Missing intermediate steps (e.g., positioning before manipulation)
    2. Overly vague skill descriptions lacking HOW to perform
    3. Skills requiring perception/search (forbidden)
    4. Skills using temporal references (forbidden)
    5. Skills with inappropriate granularity (too broad or too narrow)
    6. Missing stabilization phases after position changes
    7. Physically unrealistic sequences
    8. Do not include preperation skills such as "prepare_for_jump" or "position_feet_for_running" or stablization skills

    ## 📊 RESPONSE FORMAT
    Provide your analysis in this structured format:

    ```
    VERIFICATION RESULT: [PASS/NEEDS REFINEMENT]

    ## Detailed Analysis
    [Provide a concise analysis of the skill sequence, focusing on strengths and weaknesses]

    ## Specific Issues
    1. [Skill Name] - [Issue Category] - [Issue Description] - [Recommended Fix]
    2. [Skill Name] - [Issue Category] - [Issue Description] - [Recommended Fix]
    ...

    ## Refined Linear Skill Structure (if needed):
    [Include complete JSON with refined skills if NEEDS REFINEMENT, otherwise omit this section]
    ```

    Be extremely thorough in your analysis. If the decomposition PASSES with only minor issues, 
    provide suggestions but mark it as PASS. If there are significant issues that would impair 
    learning or task completion, mark as NEEDS REFINEMENT and provide a complete refined skill structure.
    

    the return must be should be a valid JSON object.

    """

    return prompt 