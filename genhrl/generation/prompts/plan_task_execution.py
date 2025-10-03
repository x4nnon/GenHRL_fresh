def plan_task_execution_prompt(task_name: str, task_description: str, max_hierarchy_levels: int = 3, object_name_mapping: str = "") -> str:
    """Generate a detailed sequence of events required for the robot to complete the task."""
    
    # Generate hierarchy-specific guidance
    if max_hierarchy_levels == 1:
        hierarchy_guidance = """
## 🎯 HIERARCHY LEVEL: 1 (Single Task Only)
Since this is a single-level task, your plan should be:
- **SIMPLE and DIRECT** - No complex sub-phases or detailed breakdowns
- **EXECUTABLE AS ONE SKILL** - The entire task should be achievable by a single RL policy
- **5-8 BASIC STEPS MAXIMUM** - Keep the plan concise and straightforward
- **NO SKILL DECOMPOSITION** - Don't break this into multiple distinct skills
- Focus on the core action needed to complete the task"""
        
    elif max_hierarchy_levels == 2:
        hierarchy_guidance = """
## 🎯 HIERARCHY LEVEL: 2 (Task → Skills)
Since this is a two-level hierarchy, your plan should:
- **IDENTIFY 2-5 MAJOR PHASES** - Each phase will become a primitive skill
- **EACH PHASE = ONE SKILL** - Each major phase should be a distinct, reusable skill
- **CLEAR SKILL BOUNDARIES** - Each phase should have a clear start/end state
- **8-12 STEPS TOTAL** - Detailed enough to identify distinct skills but not overly complex
- Think: "What are the main skills needed?" (e.g., approach, interact, complete)"""
        
    elif max_hierarchy_levels == 3:
        hierarchy_guidance = """
## 🎯 HIERARCHY LEVEL: 3 (Task → Skills → Sub-skills) - RECOMMENDED APPROACH
Since this is a three-level hierarchy (the OPTIMAL structure for hierarchical RL), your plan should:
- **IDENTIFY MAJOR PHASES** - Each will become a high-level skill (2-4 major skills)
- **BREAK PHASES INTO SUB-ACTIONS** - Each sub-action becomes a primitive sub-skill (2-4 sub-skills per skill)
- **DETAILED DECOMPOSITION** - 10-15 steps showing clear skill and sub-skill boundaries
- **NESTED STRUCTURE THINKING** - Consider how skills contain multiple sub-skills
- **USE EXPLICIT OBJECT REFERENCES** - Reference Object1, Object2, etc. throughout the plan
- Think: "What skills are needed, and what sub-skills make up each skill?\""""
        
    else:
        hierarchy_guidance = f"""
## 🎯 HIERARCHY LEVEL: {max_hierarchy_levels} (Multi-level)
Since this is a {max_hierarchy_levels}-level hierarchy, your plan should:
- **HIGHLY DETAILED BREAKDOWN** - Show {max_hierarchy_levels-1} levels of skill decomposition
- **NESTED SKILL STRUCTURE** - Each level should contain meaningful sub-levels
- **15+ STEPS** - Comprehensive enough to support deep hierarchy
- Think about multiple levels of abstraction from high-level goals to primitive actions"""

    prompt = f"""
    # 🤖 Task Planning for: {task_name}
    
    **🚫 NO ASCII ART**: Do NOT create any ASCII art, diagrams, visual representations, charts, or layouts using dashes, lines, symbols, or characters. Provide only concise text descriptions.
    
    ## Task Description
    {task_description}
    
    ## 🔗 Object Name Mapping
    The following shows what each ObjectN represents in this scene:
    {object_name_mapping if object_name_mapping else "Object mappings will be provided during generation"}
    
    **CRITICAL**: Extract descriptive names from the object mapping and use them in your plan.
    - PREFERRED: Use descriptive names like "lowWall", "sphere", "redCube" from the object comments
    - ACCEPTABLE: Use Object1, Object2, etc. if descriptive names are unclear
    - NEVER USE: Relative terms like "nearest", "largest", "closest"
    
    {hierarchy_guidance}
    
    ## Planning Objective
    Create a precise, sequential plan for a reinforcement learning robot to accomplish the task. This plan will guide the creation of reusable skills. You should attempt to explain completing the task with reusable skills where possible.

    ## 🤔 STEP-BY-STEP THINKING PROCESS
    Before creating the plan, think through these steps carefully:
    1. First, analyze the task description and object mapping to identify key objects and their descriptive names
    2. Next, break down the task into logical phases (approach, interaction, completion)
    3. Then, identify which robot parts will be needed for each phase
    4. Consider how each phase connects to the next, ensuring smooth transitions
    5. Finally, verify that each step uses descriptive object names and is physically possible
    6. Double-check that no steps require search, memory, absolute positions, or relative terms

    ## ⚠️ Critical Constraints
    - Robot training is expensive - prioritize reusable skills
    - Skills must be general enough to be used across tasks
    - The robot has NO visual search capabilities
    - The robot has NO memory of previous actions
    - The plan should not use absolute positions. INCORRECT: walk to x=2.0m. CORRECT: walk to lowWall
    - **Use descriptive object names** - Reference objects using their descriptive names from object configuration. PREFERRED: walk to lowWall. ACCEPTABLE: walk to Object1. INCORRECT: walk to nearestWall
    - **NO relative or ambiguous terms** - Never use "nearest", "largest", "closest", "leftmost" etc.

    
    ## ❌ ABSOLUTELY FORBIDDEN SKILLS
    DO NOT plan for skills that involve:
    1. Identifying objects
    2. Finding/searching for objects
    3. Determining object properties
    4. Scanning the environment
    5. Skills with specific object parameters
    6. Skills using temporal terms (next, previous)
    7. preperation skills such as "prepare_for_jump" or "position_feet_for_running"
    8. Orientation to objects
    
    ## 📝 Naming Convention Requirements
    - **Use descriptive object names from object configuration** (e.g., "walk_to_lowWall", "jump_over_sphere")
    - PREFERRED: Extract descriptive names from object comments (lowWall, sphere, redCube, etc.)
    - ACCEPTABLE: Use Object1-Object5 references if object names are unclear
    - **NO relative descriptors** - Never use "smaller", "larger", "nearest", "leftmost", etc.
    - **NO ambiguous terms** - Be explicit and specific about which object is referenced
    
    ## 👍 GOOD Skill Examples
    - "walk_to_lowWall" ✓ (descriptive object name, clear and reusable)
    - "pick_up_redCube" ✓ (descriptive object name, clear target)
    - "stack_redCube_on_platform" ✓ (descriptive object relationship)
    - "jump_over_lowWall" ✓ (clear action on specific object)
    - "push_sphere_towards_goalZone" ✓ (descriptive object interaction)
    - "walk_to_Object1" ✓ (acceptable if object description unclear)
    
    ## 👎 BAD Skill Examples
    - "identify_largest_object" ✗ (robot already knows object properties)
    - "find_next_block" ✗ (uses temporal term, no search needed)
    - "walk_to_nearest_sphere" ✗ (uses relative term "nearest", should be "walk_to_sphere")
    - "pick_up_smallest_cube" ✗ (uses relative term "smallest", should be "pick_up_redCube")
    - "walk_to_largest_wall" ✗ (uses relative term "largest", should be "walk_to_highWall")
    - "walk_to_specific_object" ✗ (too generic, not descriptive, and we can't pass the object to the skill)
    - "determine_stacking_location" ✗ (decision-making, not action)
    - "prepare_for_jump" ✗ (preparation, not action)
    - "position_feet_for_running" ✗ (preparation, not action)
    
    ## 🧠 Robot Knowledge Context
    The robot ALREADY KNOWS:
    - All object locations, sizes, and properties
    - Relative positions of everything in the scene
    - No need for search, scan, or identification steps
    
    ## Expected Plan Format
    Provide a numbered, sequential plan with 5-15 distinct steps. Each step should:
    1. Clearly describe a specific action or movement
    2. Reference specific objects by their properties
    3. Include necessary positioning and orientation
    4. Avoid any search or identification actions
    
    **CRITICAL**: Provide ONLY a concise numbered list. Do NOT create ASCII art, diagrams, charts, or visual representations with dashes, lines, or symbols. Keep all descriptions brief and text-only.
    
    ## Plan:
    """
    
    return prompt