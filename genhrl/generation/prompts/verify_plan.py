def verify_plan_prompt(task_name: str, task_description: str, current_plan: str, object_config: str) -> str:
    """
    Generate a prompt to verify and potentially refine the task execution plan.
    
    Args:
        task_name: Name of the robot task
        task_description: Detailed description of the task
        current_plan: The current high-level execution plan for the task
        object_config: JSON configuration of objects in the scene
        
    Returns:
        A prompt string for the LLM to verify and refine the task execution plan
    """
    
    prompt = f"""
    # 🔍 ROBOTICS TASK PLAN VERIFICATION

    You are an expert in robotics planning and humanoid robot control. Your job is to verify and refine an execution plan 
    for a complex robotic task, ensuring it's detailed, accurate, and feasible.

    ## 🤔 STEP-BY-STEP THINKING PROCESS
    Before verifying the task plan, think through these steps carefully:
    1. First, analyze each step for completeness and physical feasibility
    2. Next, check that steps follow a logical sequence and respect dependencies
    3. Then, verify that each step has clear success criteria
    4. Consider how steps handle object interactions and stability
    5. Finally, ensure that the plan uses relative positioning throughout
    6. Double-check that no steps require search, memory, or absolute positions

    ## 📋 TASK INFORMATION
    **Task Name:** {task_name}
    **Task Description:** {task_description}

    ## 📝 CURRENT EXECUTION PLAN
    ```
    {current_plan}
    ```

    ## 🧱 OBJECT CONFIGURATION
    ```json
    {object_config}
    ```

    ## 🤖 ROBOT CAPABILITIES & LIMITATIONS
    - Humanoid robot WITHOUT visual perception systems
    - Knows ALL object locations, sizes, and properties at all times
    - NO need for search, scan, or identification actions
    - NO memory of previous actions (can't use "next" or "previous")
    - CAN perform basic human-like movements (walk, reach, grab, push, climb)
    - CAN detect collisions and maintain balance
    - CANNOT perform physically impossible actions
    - CANNOT use absolute positions. Must be relative to robot or objects.

    ## ✅ VERIFICATION CHECKLIST
    Evaluate the plan against each criterion with specific examples or issues:

    ### 1. Completeness
    - [ ] Covers ALL necessary steps from start to goal
    - [ ] Includes required setup and preparation phases
    - [ ] Accounts for all objects that need manipulation
    - [ ] Have we maximised the amount of skills that can be reused? If not, why not?

    ### 2. Physical Feasibility
    - [ ] All actions are physically possible for a humanoid robot
    - [ ] Considers balance and stability during movements
    - [ ] Accounts for the robot's reach and strength limitations

    ### 3. Logical Sequencing
    - [ ] Steps follow a natural progression
    - [ ] Respects physics (can't move an object before reaching it)
    - [ ] Handles dependencies correctly (prerequisite actions)

    ### 4. Specificity
    - [ ] Each step has clear success criteria
    - [ ] Actions specify HOW movements should be performed
    - [ ] Object interactions are precisely described
    - [ ] The plan does not use absolute positions, such as walk to x=2.0m.

    ### 5. Object Interactions
    - [ ] Correctly accounts for object properties (size, weight)
    - [ ] Specifies hand usage for manipulations (one vs. both hands)
    - [ ] Includes necessary stabilization when handling objects

    ### 6. Robustness
    - [ ] Handles potential variations in object positions
    - [ ] Avoids brittle assumptions about exact positions
    - [ ] Focuses on relative positioning rather than absolute coordinates

    ## 🚫 COMMON PLAN ERRORS
    1. Missing locomotion steps between interactions
    2. Omitting specific approach or positioning before manipulation
    3. Failing to account for two-handed operations when needed
    4. Assuming perfect precision in movements
    5. Neglecting stabilization phases after position changes
    6. Using search/scan operations (unnecessary)
    7. Using temporal references like "next" (forbidden)

    ## 📝 PLAN REFINEMENT GUIDELINES
    When refining the execution plan:

    1. ✓ Use CLEAR, SEQUENTIAL, NUMBERED steps
    2. ✓ Start each step with a specific ACTION VERB
    3. ✓ SPECIFY body parts used for each action
    4. ✓ INCLUDE all movement phases (approach, position, execute, stabilize)
    5. ✓ SPECIFY hand usage when manipulating objects
    6. ✓ INCLUDE stabilization phases when needed
    7. ✓ USE relative positioning rather than absolute coordinates
    8. ✗ AVOID search, scan, or identification steps
    9. ✗ AVOID temporal references (next, previous, etc.)
    10. ✗ AVOID vague or ambiguous instructions

    ## 📊 RESPONSE FORMAT
    Provide your analysis in this structured format:

    ```
    VERIFICATION RESULT: [PASS/NEEDS REFINEMENT]

    ## Detailed Analysis
    [Provide a concise analysis of the current execution plan, focusing on strengths and weaknesses]

    ## Specific Issues
    1. [Issue category] - [Issue description] - [Recommended fix]
    2. [Issue category] - [Issue description] - [Recommended fix]
    ...

    ## Refined Execution Plan
    1. [Step 1 with specific action verb]
    2. [Step 2 with specific action verb]
    3. [Step 3 with specific action verb]
    ...
    ```

    If the plan PASSES, still provide minor suggestions for improvement.
    If the plan NEEDS REFINEMENT, provide a complete, detailed alternative plan.
    """
    
    return prompt