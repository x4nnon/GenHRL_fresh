def verify_rewards_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    """Generate a prompt to verify reward function code."""
    prompt = f"""
    # 🔍 REWARD FUNCTION VERIFICATION
    
    ## 📋 Skill Information
    **Skill Name:** {skill_name}
    **Skill Description:** {skill_description}
    
    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```
    
    ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
    Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
    You MUST access objects using these exact names: env.scene['Object1'], env.scene['Object2'], etc.
    
    ## 🚫 COMMON ERRORS TO CHECK FOR:
    
    1. ❌ **Accessing object dimensions from RigidObject** - Check for these patterns:
       ```python
       # INCORRECT patterns to flag:
       object.radius[0]
       object.size[1] 
       object.dimensions[0]
       obj.data.root_size[2]
       ```
       
    2. ❌ **Missing object dimension hardcoding** - Look for missing hardcoded values:
       ```python
       # If object config shows radius: 0.15, verify code uses:
       football_radius = 0.15  # CORRECT: From object config
       
       # Not:
       football_radius = football.radius[0]  # INCORRECT: Doesn't exist
       ```
    
         3. ❌ **Dictionary-style scene access patterns**:
        ```python
        # INCORRECT patterns to flag:
        if "Object1" in env.scene:
        for obj_idx in range(1, 6):
            if f"Object{obj_idx}" in env.scene:
        ```
    
    ## 📝 Current Reward Implementation
    ```python
    {rewards_code}
    ```
    
    ## 🎯 Verification Objective
    Perform a comprehensive review of the reward function implementation to identify any issues, 
    errors, or potential improvements in the code. 

    CRITICAL verification check. 
    - you must check that NO accessing functions or data attributes that are not explicitly shown in the examples.
    
    ## ✅ Verification Checklist
    
    ### 1. Reward Design & Alignment
    - [ ] Rewards directly incentivize progress toward completing the skill
    - [ ] Main reward component clearly drives the primary task objective
    - [ ] Supporting rewards guide secondary aspects
    - [ ] Penalties discourage undesirable behaviors
    - [ ] Rewards are well-balanced (nothing dominating inappropriately)
    - [ ] Rewards are included to encourage human-like behaviours.
    
    ### 2. Code Quality & Safety
    - [ ] Is this complete python code?
    - [ ] No potential runtime errors (divide by zero, index errors, etc.)
    - [ ] Properly handles edge cases and exceptions
    - [ ] Uses direct object access patterns
    - [ ] No hardcoded indices or magic numbers without comments
    
    ### 3. Data Access Patterns
    - [ ] Only uses allowed methods and attributes. Those shown in the examples are the only ones used?
    - [ ] No storage of variables on env object
    - [ ] Correctly handles object existence/non-existence
    - [ ] Uses proper tensor operations with correct dimensions
    - [ ] Uses relative distances instead of absolute positions
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ### 4. Reward Characteristics
    - [ ] Rewards are smooth (not too sparse)
    - [ ] Reward magnitudes are appropriate (not too large or small)
    - [ ] No unnecessary normalization that could lead to instability
    - [ ] Rewards are additive with appropriate weights
    
    ## 🚫 Common Reward Function Errors
    
    1. **Reward Shaping Issues**
       - Sparse rewards (making learning difficult)
       - Deceptive rewards (leading to wrong behaviors)
       - Misaligned rewards (not promoting skill objective)
       - Imbalanced magnitudes (one component overwhelms others)
    
    2. **Logical Issues**
       - Incorrect distance calculations
       - Improper thresholds
       - Missing normalization where needed
       - Excessive normalization where unnecessary
    
    3. **Technical Issues**
       - Potential NaN or infinity values
       - Tensor dimension mismatches
       - Device inconsistencies
       - Missing broadcasting for batch environments
    
    4. **Access Issues**
       - Using non-existent object methods/attributes
       - Using env.variable to store state
       - Accessing objects using incorrect patterns
       - Using quat_rotate_vector or mdp.quat_rotate_vector ----> quat_rotate_vector and mdp.quat_rotate_vector is undefined.
       - Using incorrect object names (must be Object1, Object2, Object3, Object4, or Object5)
    
    ## 📊 RESPONSE FORMAT
    
    Provide your analysis in this structured format:
    
    ```
    VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Summary Assessment
    [Brief overall assessment of the reward function implementation]
    
    ## Specific Issues
    1. [Category] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Category] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Reward Balance Assessment
    [Analysis of how well the different reward components are balanced]
    
    ## Recommendations
    [List of specific improvements or changes needed]
    ```
    
    Be extremely thorough in your analysis. If the implementation has only minor issues,
    provide suggestions but mark it as PASS. If there are significant issues that would prevent 
    effective learning, mark as NEEDS IMPROVEMENT.
    """
    return prompt

def verify_attributes_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    prompt = f"""
            You are an expert in reinforcement learning and robotics. Your task is to critically analyze reward functions 
            for a humanoid robot skill, specifically focusing on UNDEFINED ATTRIBUTES AND METHODS.
            
            Skill Name: {skill_name}
            Skill Description: {skill_description}

            ## 🧱 Object Configuration
            ```json
            {objects_config}
            ```
            
            ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
            Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
            You MUST access objects using these exact names: env.scene['Object1'], env.scene['Object2'], etc.

            Reward Functions:
            ```python
            {rewards_code}
            ```
            
            ## CRITICAL: CHECK FOR UNDEFINED ATTRIBUTES AND METHODS
            When reviewing the code, specifically check that:
            - NO methods or functions from the env class (ManagerBasedRLEnv) are being called
            - NO attributes or methods are used on any object that aren't explicitly shown in the examples
            - There are NO invented attributes like robot.lower_bounds, robot.device, etc.
            - There are NO invented methods like env.get_X(), robot.compute_X(), etc.
            - Flag ANY instances where the code attempts to access potentially undefined attributes or methods
            - The code ONLY uses the EXACT data access patterns shown in the examples
            - Objects are accessed using ONLY Object1, Object2, Object3, Object4, Object5 names
            
            CORRECT DATA ACCESS PATTERNS:
            ## OBJECTS
            ```python
            object = env.scene['Object1']  # or Object2, Object3, Object4, Object5
            object_positions = object.data.root_pos_w  # shape (num_envs, 3) where 3 is x,y,z
            object_volume = get_object_volume(object)  # returns volume of object
            ```

            ## BODY POSITIONS
            ```python
            robot = env.scene["robot"]
            
            left_hand_idx = robot.body_names.index('left_palm_link')
            right_hand_idx = robot.body_names.index('right_palm_link')
                
            left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
            right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
            # shape (num_envs, 3) where 3 is x,y,z
            ```

            ## ROTATIONS
            ```python
            # For body part rotations:
            head_rot = robot.data.body_state_w[:, robot.body_names.index("head_link"), 3:7]
            ```

            ### Available Body Parts:
            - 'left_palm_link', 'right_palm_link' (hands)
            - 'left_ankle_roll_link', 'right_ankle_roll_link' (feet)
            - 'pelvis'
            - 'left_knee_link', 'right_knee_link'
            - 'left_elbow_pitch_link', 'right_elbow_pitch_link'
            - 'head_link'
            - 'left_two_link', 'right_two_link' (thumbs)
            - 'left_five_link', 'right_five_link' (finger one)
            - 'left_six_link', 'right_six_link' (finger two)

            For each issue identified, please suggest a specific fix or improvement.

            ## CRITICAL: DO NOT ALLOW ACCESS TO ROBOT TORQUES, you may remove this reward or part of the reward is needed.

            ### CRITICAL: You should not assign any variables to the environment such as below:
               # we cannot do this because 1) the environment is shared across multiple environments and 2) we cannot reset this everytime the reset command is called. 
               # please try to create rewards without storing variables.

               INCORRECT -- THIS MUST NEVER HAPPEN: 
                if not hasattr(env, 'initial_pelvis_x'):
                    env.initial_pelvis_x = robot.data.root_pos_w[:, 0].clone().detach()  # Store initial x position at the first call

            
            You MUST check that we do not assign any variable to the environment using the above method. 

            Format your response as:
            ```
            ATTRIBUTE VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
            
            ISSUES:
            1. [Issue description] - [Suggested fix]
            2. [Issue description] - [Suggested fix]
            ...
            ```
            """
    return prompt

def verify_reward_design_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    prompt = f"""
            You are an expert in reinforcement learning and robotics. Your task is to critically analyze reward functions 
            for a humanoid robot skill, specifically focusing on REWARD DESIGN PRINCIPLES.
            
            Skill Name: {skill_name}
            Skill Description: {skill_description}

            ## 🧱 Object Configuration
            ```json
            {objects_config}
            ```
            
            ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
            Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).

            Reward Functions:
            ```python
            {rewards_code}
            ```
            
            ## CRITICAL: ANALYZE REWARD DESIGN
            Please analyze:
            1. Is the main reward function (weight 1.0) focused on a SINGLE metric that directly measures progress toward the primary goal?
            2. Are the shaping rewards (weight 0.3 or less) appropriate and helpful for the task?
            3. Are there any loopholes or shortcuts the agent might exploit?
            4. Are the reward weights appropriate and correctly signed?
            5. Is there any potential for reward hacking?
            6. Do the rewards make sense throughout the entire task sequence?
            7. Is the main reward sufficient to guide the agent toward task success?
            8. Do the shaping rewards work well together or might they conflict?
            9. Does the main reward function reward the robot for the exact task, and not any other shaping behaviour?
            10. Are objects accessed using the correct Object1...Object5 names based on the object configuration?
            
            CRITICAL GUIDELINES:
            - The main reward should be only one reward based on the exact task being completed
            - The shaping rewards should encourage the robot to do the task, but are not required for task success
            - Rewards should not be clamped to 0 or 1 values, continuous rewards are better
            - Rewards should not use arbitrary distance or height thresholds
            - Rewards should not reference body parts that can easily move, which could hack the reward
            - There should be sufficient penalties to discourage the robot from certain behaviors
            - You must never use initial positions in the reward functions. Find another way..
            
            If the task does not require the robot to move in the xy plane, recommend adding:
            ```python
            stay_still_reward = RewTerm(func=mdp.lin_vel_xy_l2, weight=-0.3, params=("asset_cfg": SceneEntityCfg("robot"), "normalise": True, "normaliser_name": "stay_still_reward"))
            ```

            For each issue identified, please suggest a specific fix or improvement.
            
            Format your response as:
            ```
            REWARD DESIGN VERIFICATION: [PASS/NEEDS IMPROVEMENT]
            
            ISSUES:
            1. [Issue description] - [Suggested fix]
            2. [Issue description] - [Suggested fix]
            ...
            ```
            """
    return prompt

def verify_data_access_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    prompt = f"""
            You are an expert in reinforcement learning and robotics. Your task is to critically analyze reward functions 
            for a humanoid robot skill, specifically focusing on CORRECT DATA ACCESS PATTERNS.
            
            Skill Name: {skill_name}
            Skill Description: {skill_description}

            ## 🧱 Object Configuration
            ```json
            {objects_config}
            ```
            
            ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
            Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).

            Reward Functions:
            ```python
            {rewards_code}
            ```
            
            ## CRITICAL: VERIFY CORRECT DATA ACCESS PATTERNS
            
            Check that all object and robot data access follows these exact patterns:
            
            ## OBJECTS
            ```python
            object = env.scene['Object1']  # ONLY Object1, Object2, Object3, Object4, Object5 are valid
            object_positions = object.data.root_pos_w  # shape (num_envs, 3) where 3 is x,y,z
            object_volume = get_object_volume(object)  # returns volume of object
            ```

            ## BODY POSITIONS
            ```python
            robot = env.scene["robot"]
            
            left_hand_idx = robot.body_names.index('left_palm_link')
            right_hand_idx = robot.body_names.index('right_palm_link')
                
            left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
            right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
            # shape (num_envs, 3) where 3 is x,y,z
            ```

            ### Available Body Parts:
            - 'left_palm_link', 'right_palm_link' (hands)
            - 'left_ankle_roll_link', 'right_ankle_roll_link' (feet)
            - 'pelvis'
            - 'left_knee_link', 'right_knee_link'
            - 'left_elbow_pitch_link', 'right_elbow_pitch_link'
            - 'head_link'
            - 'left_two_link', 'right_two_link' (thumbs)
            - 'left_five_link', 'right_five_link' (finger one)
            - 'left_six_link', 'right_six_link' (finger two)

            ## ROTATIONS
            ```python
            # For body part rotations:
            head_rot = robot.data.body_state_w[:, robot.body_names.index("head_link"), 3:7]
            ```

            ## CORRECT OBJECT ACCESS IN SCENE
            Objects in the scene MUST be accessed using direct indexing with the known object names:
            - CORRECT: `env.scene["Object1"]`, `env.scene["Object2"]`, etc.
            - INCORRECT: `if obj_name in env.scene:` or `if "Object1" in env.scene:`
            
            The env.scene is NOT a dictionary and does NOT support the "in" operator to check if objects exist.
            
            Access objects directly with known names:
            
            CORRECT:
            ```python
            # Direct access to objects
            obj1 = env.scene["Object1"]
            # Process obj1...
                
            # Direct access with valid objects list
            objects = []
            volumes = []
            
                         # Process all existing objects
             for i in range(1, 6):
                 obj = env.scene[f"Object{{i}}"]
                 vol = get_object_volume(obj)
                 objects.append(obj)
                 volumes.append(vol)
            ```
            
            INCORRECT:
            ```python
            # DO NOT use dictionary-style checks  
            for i in range(1, 6):
                if f"Object{i}" in env.scene:  # This will cause an error
                    obj = env.scene[f"Object{i}"]
            ```

            ALSO AVOID THIS ERROR:
            ```python
            env.initial_pelvis_x = robot.data.root_pos_w[:, 0, 0].clone().detach()  # Store initial x position
            # IndexError: too many indices for tensor of dimension 2
            ```

            For each issue identified, please suggest a specific fix or improvement.
            
            Format your response as:
            ```
            DATA ACCESS VERIFICATION: [PASS/NEEDS IMPROVEMENT]
            
            ISSUES:
            1. [Issue description] - [Suggested fix]
            2. [Issue description] - [Suggested fix]
            ...
            ```
            """
    return prompt

def verify_calculations_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    prompt = f"""
            You are an expert in reinforcement learning and robotics. Your task is to critically analyze reward functions 
            for a humanoid robot skill, specifically focusing on REWARD CALCULATIONS.
            
            Skill Name: {skill_name}
            Skill Description: {skill_description}

            ## 🧱 Object Configuration
            ```json
            {objects_config}
            ```
            
            ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
            Use the object configuration above to understand what each ObjectN represents.

            Reward Functions:
            ```python
            {rewards_code}
            ```
            
            ## CRITICAL: VERIFY REWARD CALCULATIONS
            
            Please analyze all calculations in the reward functions for:
            1. Correctness - do they calculate what they intend to? Are you sure there isn't any confusion about orientation? 
            2. Simplicity - are they as simple as possible to avoid errors? Even a single error in a reward can cause the skill to fail.
            3. Continuity - are they continuous rather than clamped to 0/1 values?
            4. Thresholds - do they avoid arbitrary distance or height thresholds?
            5. Potential bugs - are there any potential division by zero, NaN issues, etc.?
            6. Tensor shapes - are all tensor operations using the correct dimensions?
            7. Weights - are the weights appropriate and correctly signed?
            8. Generalisation - are the rewards general enough to encourage the robot to learn general behaviours of the skill?
            9. Object naming - are objects accessed using correct Object1...Object5 names based on the object configuration?
            
            Specific checks:
            - Reward calculations should avoid clamping to 0 or 1 values, continuous rewards are better
            - Rewards should not use arbitrary distance or height thresholds (you may assume ground is at z = 0) Rewards should use velocitys or relative positions to objects or iteself.
                DO NOT USE ARBITRARY THRESHOLDS, you do not know the size of things and this may cause the skill to fail, 
            - Tensor operations should use correct dimensions and avoid indexing errors

            CRITICAL: You do not need to initialize or define the RewNormalizer, it is already defined elsewhere.
            - Please explicitly check that the RewNormalizer is not being defined or initialized in this code. You must give specific instructions to remove it if so.
            
            For each issue identified, please suggest a specific fix or improvement.
            
            Format your response as:
            ```
            CALCULATION VERIFICATION: [PASS/NEEDS IMPROVEMENT]
            
            ISSUES:
            1. [Issue description] - [Suggested fix]
            2. [Issue description] - [Suggested fix]
            ...
            ```
            """
    return prompt

def verify_rewards_design_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying the reward design."""
    prompt = f"""
    # 🔍 REWARD DESIGN VERIFICATION
    
    ## 📋 Skill Information
    **Skill Name:** {skill_name}
    **Skill Description:** {skill_description}
    
    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```
    
    ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
    Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
    
    ## 📝 Current Reward Implementation
    ```python
    {rewards_code}
    ```
    
    ## 🎯 Verification Focus: REWARD DESIGN PRINCIPLES
    Your task is to specifically verify that the reward function follows solid reward 
    design principles for reinforcement learning.
    
    ## ✅ Reward Design Checklist
    
    ### 1. Reward Alignment
    - [ ] Main reward directly incentivizes the primary skill objective
    - [ ] All reward components contribute to the skill goal
    - [ ] No contradictory rewards that work against each other
    - [ ] Rewards reflect proper progress toward skill completion
    
    ### 2. Reward Shaping
    - [ ] Rewards provide useful gradients throughout learning
    - [ ] Not too sparse (learning occurs constantly, not just at completion)
    - [ ] Not too dense (avoiding reward hacking)
    - [ ] Appropriate balance between proximity and completion rewards
    
    ### 3. Penalties & Constraints
    - [ ] Appropriate penalties for undesirable behaviors
    - [ ] Penalties are not too severe to prevent learning
    - [ ] Constraints on robot movement are reasonable
    - [ ] Balance between exploration and constraint enforcement
    
    ### 4. Reward Composition
    - [ ] Rewards are properly combined (addition, multiplication, etc.)
    - [ ] Weights between components are balanced
    - [ ] Success reward is significant enough to drive behavior
    - [ ] Environment-specific rewards are appropriate
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ## 🧠 Consider These Key Questions:
    
    1. Will this reward function guide the agent toward completing the skill?
    2. Are there any "reward hacking" opportunities that could lead to unintended behaviors?
    3. Is the reward sufficiently smooth to provide learning signals?
    4. Do the reward components have appropriate relative magnitudes?
    5. Are objects accessed using the correct Object1...Object5 names from the object configuration?
    
    ## 📊 RESPONSE FORMAT
    
    ```
    REWARD DESIGN VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Reward Design Analysis
    [Brief assessment of the overall reward design]
    
    ## Reward Component Assessment
    - Main Objective Component: [Effectiveness: Strong/Medium/Weak] - [Explanation]
    - Supporting Components: [Effectiveness: Strong/Medium/Weak] - [Explanation]
    - Penalty Components: [Effectiveness: Strong/Medium/Weak] - [Explanation]
    
    ## Design Issues
    1. [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Reward Balance Recommendation
    [Concrete suggestions for improving balance between reward components]
    ```
    """
    return prompt

def verify_rewards_attributes_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying the attributes and data access in reward function code."""
    prompt = f"""
    # 🔍 REWARD FUNCTION ATTRIBUTES VERIFICATION
    
    ## 📋 Skill Information
    **Skill Name:** {skill_name}
    **Skill Description:** {skill_description}
    
    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```
    
    ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
    Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
    
    ## 📝 Current Reward Implementation
    ```python
    {rewards_code}
    ```
    
    ## 🎯 Verification Focus: DATA ACCESS PATTERNS
    Your task is to specifically verify that the reward function implementation uses 
    ONLY allowed attributes and methods to access data from the environment and objects.
    
    ## ✅ Data Access Verification Checklist
    
    ### 1. Robot Data Access
    - [ ] Only uses allowed robot body access patterns:
        - `robot = env.scene["robot"]`
        - `body_idx = robot.body_names.index('<body_name>')`
        - `body_pos = robot.data.body_pos_w[:, body_idx]`
        - `body_rot = robot.data.body_state_w[:, body_idx, 3:7]`
    
    ### 2. Object Data Access
    - [ ] Only uses allowed object access patterns:
        - `object = env.scene['ObjectN']` (where N is 1,2,3,4,5)
        - `object_pos = object.data.root_pos_w`
    - [ ] Uses direct access patterns for all objects
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ### 3. Env Variable Access
    - [ ] Does NOT store state on env object (e.g., `env.variable = ...`)
    - [ ] Only uses allowed env attributes: 
        - `env.num_envs`
        - `env.device`
        - `env.common_step_counter`
    
    ### 4. Tensor Operations
    - [ ] Correctly handles tensor dimensions
    - [ ] Performs operations on the correct device
    - [ ] Uses proper broadcasting for batched environments
    
    ## 🔍 Examine EVERY SINGLE line of code that accesses:
    1. The robot or its parts
    2. Any object in the scene
    3. Any environment attribute
    4. Any tensor operation
    
    ## 📊 RESPONSE FORMAT
    
    ```
    ATTRIBUTE VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Data Access Analysis
    [Brief assessment of the data access patterns]
    
    ## Attribute Access Issues
    1. [Line X] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Line Y] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Potentially Problematic Patterns
    [List any usage patterns that might cause runtime errors]
    ```
    """
    return prompt

def verify_rewards_calculations_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying calculations in reward functions."""
    prompt = f"""
    # 🔍 REWARD CALCULATIONS VERIFICATION
    
    ## 📋 Skill Information
    **Skill Name:** {skill_name}
    **Skill Description:** {skill_description}
    
    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```
    
    ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
    Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
    
    ## 📝 Current Reward Implementation
    ```python
    {rewards_code}
    ```
    
    ## 🎯 Verification Focus: CALCULATIONS & NUMERICAL OPERATIONS
    Your task is to specifically verify that all calculations, normalizations, and numerical
    operations in the reward function are correct and stable.
    
    ## ✅ Calculations Verification Checklist
    
    ### 1. Distance Calculations
    - [ ] Correctly calculates distances (Euclidean vs. Manhattan)
    - [ ] Properly handles dimensions (x,y only vs. x,y,z)
    - [ ] Uses appropriate norms for vector distances
    
    ### 2. Reward Scaling
    - [ ] Reward magnitudes are appropriate (not too large or small)
    - [ ] Properly normalizes rewards where appropriate
    - [ ] Avoids unnecessary normalization that could cause instability
    - [ ] Uses clipping appropriately to prevent extreme values
    
    ### 3. Mathematical Operations
    - [ ] No potential division by zero
    - [ ] No negative square roots
    - [ ] No inappropriate normalization
    - [ ] Correct tensor dimensions in calculations
    
    ### 4. Numerical Stability
    - [ ] Uses stable mathematical operations
    - [ ] Handles edge cases without producing NaN or Inf
    - [ ] Properly initializes variables
    - [ ] Avoids unnecessarily complex calculations
    
    ### 5. Object Access
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    - [ ] Properly handles cases where objects might not exist
    
    ## 🔍 Analyze EVERY numerical operation:
    
    1. Is the calculation mathematically correct?
    2. Could the operation cause numerical instability?
    3. Are tensor dimensions handled correctly?
    4. Is normalization used appropriately?
    5. Are objects accessed using the correct Object1...Object5 names?
    
    ## 📊 RESPONSE FORMAT
    
    ```
    CALCULATIONS VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Calculations Analysis
    [Brief assessment of the calculations and numerical operations]
    
    ## Calculation Issues
    1. [Line X] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Line Y] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Numerical Stability Assessment
    [Analysis of potential stability issues in calculations]
    
    ## Recommended Improvements
    [Specific suggestions for improving calculations]
    ```
    """
    return prompt

def verify_rewards_env_variables_prompt(skill_name: str, skill_description: str, rewards_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying no persistent variables are stored on the env object."""
    prompt = f"""
    # 🔍 REWARD FUNCTION ENV VARIABLES VERIFICATION
    
    ## 📋 Skill Information
    **Skill Name:** {skill_name}
    **Skill Description:** {skill_description}
    
    ## 🧱 Object Configuration
    ```json
    {objects_config}
    ```
    
    ⚠️ **CRITICAL OBJECT NAMING**: Objects in the scene are named Object1, Object2, Object3, Object4, Object5 only.
    Use the object configuration above to understand what each ObjectN represents (e.g., Object1 might be a ball, Object3 might be a wall).
    
    ## 📝 Current Reward Implementation
    ```python
    {rewards_code}
    ```
    
    ## 🎯 Verification Focus: ENV VARIABLE USAGE
    Your task is to specifically verify that the reward function implementation does NOT
    store any persistent variables on the env object, which is a critical error.
    
    ## ✅ Env Variable Verification Checklist
    
    ### 1. Forbidden Patterns
    - [ ] NO instances of `env.variable_name = ...`
    - [ ] NO instances of `setattr(env, 'variable_name', ...)`
    - [ ] NO instances of `if hasattr(env, 'variable_name'):`
    - [ ] NO storage of initial positions or reference values
    
    ### 2. Allowed Env Access
    - [ ] ONLY uses documented env attributes:
        - `env.num_envs`
        - `env.device`
        - `env.common_step_counter`
        - `env.scene`
    
    ### 3. Stateless Calculations
    - [ ] Rewards are calculated without stored state
    - [ ] Uses only current positions/orientations
    - [ ] Does not compare to previous values
    
    ### 4. Object Access
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    - [ ] Accesses objects properly without storing references on env
    
    ## 🚫 SPECIFICALLY SEARCH FOR THESE PATTERNS:
    
    ```python
    # FORBIDDEN:
    if not hasattr(env, 'initial_pos'):
        env.initial_pos = robot.data.root_pos_w.clone()
    
    # FORBIDDEN:
    env.variable_name = value
    
    # FORBIDDEN:
    if hasattr(env, 'step_count'):
        env.step_count += 1
    else:
        env.step_count = 0
    ```
    
    ## 📊 RESPONSE FORMAT
    
    ```
    ENV VARIABLES VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Env Variable Analysis
    [Brief assessment of any env variable usage]
    
    ## Env Variable Issues
    1. [Line X] - [Issue Description] - [Severity: Critical] - [Fix Recommendation]
    2. [Line Y] - [Issue Description] - [Severity: Critical] - [Fix Recommendation]
    ...
    
    ## Alternative Implementation
    [If issues found, recommend a stateless alternative approach]
    ```
    
    If you find ANY instance of storing variables on the env object, this is a CRITICAL issue
    that must be fixed, as it will cause errors in the multi-environment setting.
    """
    return prompt
