def verify_success_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt to verify success criteria code."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Objective
    Perform a comprehensive review of the success criteria implementation to identify any issues, 
    errors, or potential improvements in the code.

    CRITICAL verification check. 
    - you must check that NO accessing functions or data attributes that are not explicitly shown in the examples.
    - you must check that none of the common errors have been made.

    ## ✅ Verification Checklist
    
    ### 1. Correctness & Alignment
    - [ ] Success criteria correctly match the skill description
    - [ ] Logic correctly determines successful skill completion
    - [ ] Thresholds are reasonable and justified
    - [ ] Duration requirements are appropriate
    - [ ] Success criteria aligns with the rewards functions
    
    ### 2. Code Quality & Safety
    - [ ] No potential runtime errors (divide by zero, index errors, etc.)
    - [ ] Properly handles edge cases and exceptions
    - [ ] Uses direct object access patterns
    - [ ] No hardcoded indices or magic numbers without comments
    
    ### 3. Data Access Patterns
    - [ ] Only uses allowed methods and attributes. Those shown in the examples are the only ones used?
    - [ ] No storage of variables on env object
    - [ ] Correctly handles object existence/non-existence
    - [ ] Uses proper tensor operations
    - [ ] Uses relative distances instead of absolute positions
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ### 4. Implementation Completeness
    - [ ] Includes necessary primary success conditions
    - [ ] Includes appropriate secondary conditions (if needed)
    - [ ] Properly checks duration with check_success_duration
    - [ ] Saves success states when environments succeed
    - [ ] Success criteria are not too restrictive or permissive. 
    - [ ] Instantaneous velocities can be high on robot body parts. These should be avoided.
    - [ ] You do not know the dimensions of the robot. You should NOT use arbitrary values for positions of limbs or pelvis. You do not know if this is correct.
    
    ## 🚫 Common Implementation Errors
    
    1. **Access Pattern Errors**
       - Using non-existent object methods/attributes
       - Using env.variable to store state
       - Accessing objects using incorrect patterns
       - Using incorrect tensor dimensions
       - Using quat_rotate_vector or mdp.quat_rotate_vector ----> quat_rotate_vector and mdp.quat_rotate_vector is undefined.
        - using AttributeError: object.cfg.size ---> 'RigidObjectCfg' object has no attribute 'size'
       - Using incorrect object names (must be Object1, Object2, Object3, Object4, or Object5)
       
    2. **Logical Errors**
       - Too restrictive criteria (making success impossible)
       - Too permissive criteria (allowing "fake" success)
       - Missing necessary conditions
       - Incorrect mathematical calculations
    
    3. **Physical Errors**
       - Unrealistic thresholds
       - Physically impossible requirements
       - Inconsistent units
       - Missing stability conditions
    
    4. **Technical Errors**
       - Missing duration checks
       - Missing success state saving
       - Tensor device mismatches
       - Failing to handle multiple environments
    
    ## 📊 RESPONSE FORMAT
    
    Provide your analysis in this structured format:
    
    ```
    VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Summary Assessment
    [Brief overall assessment of the success criteria implementation]
    
    ## Specific Issues
    1. [Category] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Category] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Recommendations
    [List of specific improvements or changes needed]
    ```
    
    Be extremely thorough in your analysis. If the implementation has only minor issues,
    provide suggestions but mark it as PASS. If there are significant issues that would prevent 
    proper skill success detection, mark as NEEDS IMPROVEMENT.
    """
    return prompt

def verify_success_attributes_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying the attributes and data access in success criteria code."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA ATTRIBUTES VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Focus: DATA ACCESS PATTERNS
    Your task is to specifically verify that the success criteria implementation uses 
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

def verify_success_design_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying the logical design of success criteria."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA LOGIC VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Focus: LOGICAL CORRECTNESS
    Your task is to specifically verify that the success criteria correctly and 
    comprehensively identifies when the skill has been successfully completed.
    
    ## ✅ Logical Design Verification Checklist
    
    ### 1. Success Alignment
    - [ ] Success criteria directly match the skill description
    - [ ] All necessary aspects of success are captured
    - [ ] No irrelevant or unnecessary checks
    - [ ] Success criteria aligns with the reward function objectives
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ### 2. Completeness
    - [ ] Covers primary success conditions
    - [ ] Includes secondary stability conditions if needed
    - [ ] Handles edge cases appropriately
    - [ ] No missing critical requirements
    
    ### 3. Thresholds & Tolerances
    - [ ] Thresholds are reasonable and achievable
    - [ ] Tolerances allow for natural robot movement variation
    - [ ] Not too strict (preventing legitimate success)
    - [ ] Not too lenient (allowing false positives)
    
    ### 4. Physical Realism
    - [ ] Requirements are physically possible
    - [ ] Considers robot capabilities and limitations
    - [ ] Accounts for natural movement variations
    - [ ] Realistic duration requirements
    
    ## 📊 RESPONSE FORMAT
    
    ```
    LOGIC VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Logic Analysis
    [Brief assessment of the logical design]
    
    ## Logic Issues
    1. [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Success Criteria Assessment
    [Analysis of whether the criteria effectively capture skill completion]
    ```
    """
    return prompt

def verify_success_data_access_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying data access patterns in success criteria."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA DATA ACCESS VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Focus: DATA ACCESS PATTERNS
    Your task is to specifically verify that all data access in the success criteria
    follows the approved patterns and doesn't use any undefined attributes or methods.
    
    ## ✅ Data Access Verification Checklist
    
    ### 1. Object Access
    - [ ] Uses only env.scene['ObjectN'] where N is 1,2,3,4,5
    - [ ] Uses direct access patterns for all objects
    - [ ] Accesses object positions with object.data.root_pos_w
    - [ ] Does not use undefined object attributes
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    
    ### 2. Robot Access
    - [ ] Uses robot = env.scene["robot"]
    - [ ] Gets body indices with robot.body_names.index()
    - [ ] Accesses positions with robot.data.body_pos_w[:, idx]
    - [ ] Does not use undefined robot attributes
    
    ### 3. Environment Access
    - [ ] Only uses approved env attributes (num_envs, device, common_step_counter)
    - [ ] Does not store variables on env object
    - [ ] Does not use undefined env methods
    
    ## 📊 RESPONSE FORMAT
    
    ```
    DATA ACCESS VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Data Access Analysis
    [Brief assessment of data access patterns]
    
    ## Access Issues
    1. [Line X] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Line Y] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    ```
    """
    return prompt

def verify_success_calculations_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying calculations in success criteria."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA CALCULATIONS VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Focus: CALCULATIONS & NUMERICAL OPERATIONS
    Your task is to specifically verify that all calculations in the success criteria
    are mathematically correct and numerically stable.
    
    ## ✅ Calculations Verification Checklist
    
    ### 1. Distance Calculations
    - [ ] Uses correct distance formulas (Euclidean, Manhattan, etc.)
    - [ ] Handles tensor dimensions correctly
    - [ ] Uses appropriate norms for different distance types
    - [ ] Considers appropriate coordinate systems (x,y vs x,y,z)
    
    ### 2. Threshold Comparisons
    - [ ] Uses reasonable and justified thresholds
    - [ ] Threshold values are physically meaningful
    - [ ] Comparisons use correct operators (<, >, <=, >=)
    - [ ] Handles edge cases properly
    
    ### 3. Logical Operations
    - [ ] Boolean logic is correct (and, or, not)
    - [ ] Tensor boolean operations work with batched environments
    - [ ] No logical contradictions or impossible conditions
    
    ### 4. Numerical Stability
    - [ ] No potential for division by zero
    - [ ] Handles edge cases gracefully
    - [ ] Uses appropriate numerical tolerances
    - [ ] No potential for NaN or infinite values
    
    ### 5. Object Access
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    - [ ] Properly handles cases where objects might not exist
    
    ## 📊 RESPONSE FORMAT
    
    ```
    CALCULATIONS VERIFICATION RESULT: [PASS/NEEDS IMPROVEMENT]
    
    ## Calculations Analysis
    [Brief assessment of the calculations]
    
    ## Calculation Issues
    1. [Line X] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    2. [Line Y] - [Issue Description] - [Severity: High/Medium/Low] - [Fix Recommendation]
    ...
    
    ## Numerical Stability Assessment
    [Analysis of potential numerical issues]
    ```
    """
    return prompt

def verify_success_env_variables_prompt(skill_name: str, skill_description: str, success_code: str, objects_config: str = "") -> str:
    """Generate a prompt specifically focused on verifying no variables are stored on the env object."""
    prompt = f"""
    # 🔍 SUCCESS CRITERIA ENV VARIABLES VERIFICATION
    
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
    
    ## 📝 Current Success Implementation
    ```python
    {success_code}
    ```
    
    ## 🎯 Verification Focus: ENV VARIABLE USAGE
    Your task is to specifically verify that the success criteria implementation does NOT
    store any persistent variables on the env object, which would cause errors.
    
    ## ✅ Env Variable Verification Checklist
    
    ### 1. Forbidden Patterns
    - [ ] NO instances of `env.variable_name = ...`
    - [ ] NO instances of `setattr(env, 'variable_name', ...)`
    - [ ] NO instances of `if hasattr(env, 'variable_name'):`
    - [ ] NO storage of initial positions or reference values on env
    
    ### 2. Allowed Env Access
    - [ ] ONLY uses documented env attributes:
        - `env.num_envs`
        - `env.device` 
        - `env.common_step_counter`
        - `env.scene`
    
    ### 3. Stateless Implementation
    - [ ] Success criteria calculated without stored state
    - [ ] Uses only current positions/orientations
    - [ ] Does not compare to previous values stored on env
    
    ### 4. Object Access
    - [ ] Uses correct Object1...Object5 naming based on object configuration
    - [ ] Accesses objects properly without storing references on env
    
    ## 🚫 SPECIFICALLY SEARCH FOR THESE PATTERNS:
    
    ```python
    # FORBIDDEN:
    if not hasattr(env, 'initial_pos'):
        env.initial_pos = robot.data.root_pos_w.clone()
    
    # FORBIDDEN:
    env.success_timer = value
    
    # FORBIDDEN:
    setattr(env, 'skill_state', some_value)
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