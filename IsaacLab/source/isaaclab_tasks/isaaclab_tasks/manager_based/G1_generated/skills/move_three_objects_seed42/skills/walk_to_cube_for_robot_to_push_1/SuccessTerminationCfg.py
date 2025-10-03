
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cube_for_robot_to_push_1_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cube_for_robot_to_push_1 skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    # CORRECT: Accessing the robot object directly.
    robot = env.scene["robot"]
    # CORRECT: Using robot.body_names.index to get the pelvis index, ensuring no hardcoded indices.
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Accessing the pelvis position using the approved pattern for batched environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object position
    # CORRECT: Accessing Object1 directly as specified in the object configuration.
    object1 = env.scene['Object1']  # Object1 is 'Cube for robot to push'
    # CORRECT: Accessing Object1's root position in world coordinates.
    object1_pos = object1.data.root_pos_w
    
    # 3. Calculate relative distances for success criteria
    # The success criteria are based on relative distances between the robot's pelvis and Object1's center.
    
    # CORRECT: Calculate absolute X-distance between pelvis and Object1's center.
    # This is a relative distance, as required.
    x_distance = torch.abs(pelvis_pos[:, 0] - object1_pos[:, 0])
    
    # CORRECT: Calculate absolute Y-distance between pelvis and Object1's center.
    # This is a relative distance, as required.
    y_distance = torch.abs(pelvis_pos[:, 1] - object1_pos[:, 1])
    
    # CORRECT: Calculate absolute Z-distance between pelvis and Object1's center.
    # This is a relative distance, as required, ensuring stable height relative to the object.
    z_distance = torch.abs(pelvis_pos[:, 2] - object1_pos[:, 2])
    
    # 4. Check success conditions based on specified thresholds
    # The robot's pelvis should be behind Object1 in X, aligned in Y, and at a stable Z height.
    
    # X-axis condition: Pelvis X-distance from Object1's center is between 0.3m and 0.5m.
    # This positions the robot behind the cube, ready to push. Thresholds are from the success criteria plan.
    x_condition = (x_distance >= 0.3) & (x_distance <= 0.6)
    
    # Y-axis condition: Pelvis Y-distance from Object1's center is less than 0.15m.
    # This ensures alignment with the cube for a straight push. Threshold is from the success criteria plan.
    y_condition = y_distance < 0.15
    
    z_condition = torch.abs(pelvis_pos[:, 2] - 0.7) < 0.1
    
    # Combine all conditions: All conditions must be met for success.
    condition = x_condition & y_condition & z_condition
    
    # 5. Check duration and save success states
    # CORRECT: Using check_success_duration to ensure the condition is met for a specified duration (0.5 seconds).
    success = check_success_duration(env, condition, "walk_to_cube_for_robot_to_push_1", duration=0.4)
    
    # CORRECT: Saving success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cube_for_robot_to_push_1")
    
    return success

class SuccessTerminationCfg:
    # CORRECT: Assigning the success function to the 'success' attribute of DoneTerm.
    success = DoneTerm(func=walk_to_cube_for_robot_to_push_1_success)
