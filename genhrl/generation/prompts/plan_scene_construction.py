def plan_scene_construction_prompt(task_description) -> str:
    """Generate a prompt to plan scene construction for a robot task."""
    prompt = f"""
    # 🏗️ SCENE CONSTRUCTION PLANNING
    
    ## 📋 Task Information
    **Task Description:** {task_description}
    
    
    ## 🎯 Planning Objective
    Create a detailed, step-by-step scene construction plan that will set up an environment 
    optimized for the robot to complete the specified task.
    
    ## 🤔 STEP-BY-STEP THINKING PROCESS
    Before creating the scene plan, think through these steps carefully:
    1. First, analyze the task requirements to identify all necessary objects
    2. Next, consider the robot's capabilities and limitations for each object interaction
    3. Then, plan object placement to ensure physical stability and accessibility
    4. Consider the sequence of interactions and ensure objects are positioned appropriately
    5. Finally, verify that all object placements are physically realistic and support task completion
    6. Double-check that no object is placed in a way that would make the task impossible
    
    ## 🤖 Robot Capabilities & Constraints
    - small humanoid robot with human-like proportions and movement capabilities
    - Standing height of approximately 1.1 meters
    - The robot always starts at (0,0,0)
    
    ## 🌍 Scene Planning Requirements
    
    ### 1. Object Selection & Placement
    - Select appropriate objects from the available configurations
    - Position objects strategically for optimal task execution
    - Ensure object positions are physically realistic and align with the task description
    - Consider appropriate distances, heights, and orientations
    
    ### 2. Environmental Considerations
    - Consider stability of object placements
    - Plan for realistic physics interactions
    
    ### 3. Explicit Measurements
    - Specify exact coordinates for object placement (x, y, z)
    - Use metric measurements (meters)
    - Define precise distances between objects
    
    ### 4. Object Sizes
    - You must be careful with x,y,z coordinates and radius / diameter of the objects. Make sure these align with the task.

    ## 📏 Coordinate System
    - Origin (0,0,0) at floor level center of the scene and where the robot starts.
    - Positive X axis points forward
    - Positive Y axis points left
    - Positive Z axis points up
    - Measurements in meters
    
    ## 📊 Output Format Requirements
    
    Your scene construction plan must include:
    
    1. **Object Selection**: List of specific objects to include in the scene
    2. **Object Placement**: Exact coordinates and orientations for each object
    3. **Robot Positioning**: Precise starting position and orientation
    4. **Rationale**: Brief explanation for key placement decisions
    5. **Success Criteria**: How the scene setup facilitates task completion
    
    ## 📝 RESPONSE FORMAT
    
    ```
    # SCENE CONSTRUCTION PLAN
    
    ## Selected Objects
    [List all objects to be included in the scene]
    
    ## Object Placement Details
    1. [Object1]: position (x,y,z), orientation [degrees], [rationale]
    2. [Object2]: position (x,y,z), orientation [degrees], [rationale]
    ...
    
    ## Layout Summary
    [CONCISE text description of object spatial relationships - NO ASCII art or diagrams]
    
    ## Success Criteria
    [Explanation of how this layout enables successful task completion]
    ```
    
    **CRITICAL**: Use only concise text descriptions. Do NOT create ASCII art, diagrams, or visual representations with dashes, lines, or symbols. Keep all descriptions brief and focused.
    
    Create a plan that is realistic, precise, and optimized for the specific task requirements.
    Focus on creating an environment that will maximize the robot's chance of successfully 
    completing the described task.
    """
    return prompt