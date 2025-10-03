def generate_collapse_termination_prompt(skill_name: str, skill_description: str, skill_rewards: str, objects_config: str = "", objects_mapping: str = ""):
    prompt = f"""
            You are a code generation assistant. Respond only with the exact Python code needed, no explanations or markdown.
            
            CRITICAL RULE: DO not decorate with '''python or '''
            
            Task: Generate a collapse/failure condition for the skill: {skill_name}
            The description of the skill is: {skill_description}
            
            This skill is created as a general skill which will be used across multiple tasks. 
            Therefore your collapse/failure criteria should be flexible enough to handle multiple tasks. 
            However, it should also be specific enough that the skill has been failed by a vague definition of the skill.

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

            ## 🧱 HOW TO USE OBJECT DIMENSIONS

            When you need object dimensions (radius, size, height, etc.) for collapse detection, you MUST:

            1. **Read the object configuration JSON** to understand the object properties
            2. **Hardcode the dimension values** directly in your collapse functions
            3. **NEVER try to access dimensions from the RigidObject** - they don't exist

                         ### ✅ CORRECT Examples:
             ```python
             # If object config shows: "Object1": {{"type": "sphere", "radius": 0.15}}
             football = env.scene['Object1']
             football_pos = football.data.root_pos_w
             football_radius = 0.15  # CORRECT: Get from object config

             # Check if ball is too far from robot (failure condition)
             distance_to_football = torch.norm(pelvis_pos - football_pos, dim=1)
             ball_too_far = distance_to_football > (football_radius + 2.0)  # Football radius + threshold

             # If object config shows: "Object2": {{"type": "box", "size": [0.4, 1.8, 0.2]}}
             wall = env.scene['Object2'] 
             wall_pos = wall.data.root_pos_w
             wall_height = 0.2  # CORRECT: Get from object config

             # Check if robot failed to clear wall height
             robot_too_low = (pelvis_pos[:, 2] < wall_height + 0.1) & (near_wall_condition)
             ```

            ### ❌ INCORRECT Examples:
            ```python
            # WRONG - These attributes don't exist!
            football_radius = football.radius[0]           # AttributeError!
            wall_height = wall.size[2]                     # AttributeError!
            ball_diameter = ball.dimensions[0]             # AttributeError!
            object_size = obj.data.root_size[1]           # AttributeError!
            ```

            The reward functions for the current skill are:
            
            ```python
            {skill_rewards}
            ```
            You should create a function that determines when the robot is acting in a way that isn't making progress toward the current skill's goal and is unlikley to recover in this episode.

            IMPORTANT: You do not need to check if the robot has fallen over as this is already checked elsewhere. You should only check specific skill failures.
            
            For example:
            - If the current skill is to pick up a ball after crouching, termination might happen if the robot stands up without grabbing the ball, or if the robot moves too far away from the ball.
            - If the current skill is to throw a ball after picking it up, termination might happen if the robot drops the ball
            - If the current skill is to walk to a location with a ball, termination might happen if the ball gets too far from the robot
            
            Your function should:
            1. Consider what constitutes the "starting state" based on the previous skill's success criteria
            2. Consider what constitutes progress toward the current skill's goal based on its reward functions
            3. Define a failure condition that identifies when the robot is moving away from both the starting state AND the goal state
            4. The function should return a boolean tensor indicating whether each environment should terminate
            5. You must do environment specific rewards, for instance with nearest object, it must be per environment, do not assume all envinments have the same nearest object.
            6. However, env.scene[{{object_id}}] will only ever be one object, not over all environments. So you must think about these operations.

            IGNORE objects smaller than 0.00001m³ (added for observation space consistency)

            ## CRITICAL: OBJECT ACCESS IN SCENE
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
                if f"Object{{i}}" in env.scene:  # This will cause an error
                    obj = env.scene[f"Object{{i}}"]
                ```

                ## CRITICAL: THIS IS THE ONLY WAY TO ACCESS ROBOT POSITION ROTATIONS:
                    ```python
                    # example for the head link rotation.
                        head_rot = robot.data.body_state_w[:, robot.body_names.index("head_link"), 3:7]

                    ```
            
            CRITICAL: You must not use arbitrary distance or height thresholds. We do not know the dimensions of the robot, so these are likely to cause failures. You may use relative distances, or distance / heights of known objects (from the objects config).
            
            The termination function should follow this template:
            
            ```python
            def {skill_name}_collapse(env: ManagerBasedRLEnv) -> torch.Tensor:
                \"\"\"Determine if the robot has deviated too far from the starting state without making progress.
                
                Args:
                    env: The environment instance
                    
                Returns:
                    Tensor of booleans indicating termination for each environment
                \"\"\"
                robot = env.scene["robot"]
                
                # Get relevant body positions
                # [Your code to get positions]
                
                # Define failure condition
                # [Your code to determine failure]
                
                # Check if the failure condition has persisted long enough
                collapse = check_failure_duration(env, failure_condition, "{skill_name}_collapse", duration=0.5)
                
                return collapse
            ```
            
            Make sure to implement the check_failure_duration function similar to check_success_duration, to avoid premature termination.
            
            Return ONLY the imports, the collapse function and the CollapseTerminationCfg class, just like this:
            
            ```python

            from .base_success import save_success_state, check_success_duration
            from isaaclab.managers import TerminationTermCfg as DoneTerm
            from isaaclab.utils import configclass
            from isaaclab.managers import SceneEntityCfg
            from genhrl.generation.mdp import *
            import torch
            from pathlib import Path
            from .TaskRewardsCfg import *
            from .SuccessTerminationCfg import *

            def check_failure_duration(env, failure, name, duration = 0.5):
                \"\"\"Check if failure condition has been maintained for sufficient duration.\"\"\"
                if not hasattr(env, f'{skill_name}_failure_start_time'):
                    setattr(env, f'{skill_name}_failure_start_time', torch.zeros(env.num_envs, device=env.device))
                
                # Update failure timers
                dt = 0.005  # 5ms is a common physics timestep
                current_time = env.common_step_counter * dt
                failure_start_time = getattr(env, f'{skill_name}_failure_start_time')
                failure_start_time = torch.where(
                    failure,
                    failure_start_time,  # Keep the existing time if still failing
                    current_time         # Reset to current time if no longer failing
                )
                setattr(env, f'{skill_name}_failure_start_time', failure_start_time)
                
                failure_duration = current_time - failure_start_time
                persistent_failure = failure_duration > duration
    
                # Reset timer for environments that have triggered a persistent failure
                failure_start_time = torch.where(
                    persistent_failure,
                    current_time,  # Reset to current time (or could use zeros)
                    failure_start_time
                )
                setattr(env, attr_name, failure_start_time)

                return persistent_failure
            
            def {skill_name}_collapse(env: ManagerBasedRLEnv) -> torch.Tensor:
                # Your failure detection code here
                ...
                return collapse

                # YOU MUST INCLUDE (SuccessTerminationCfg) as the inheritance.
            @configclass
            class CollapseTerminationCfg(SuccessTerminationCfg):
                collapse = DoneTerm(func={skill_name}_collapse)
            ```
            """
    return prompt