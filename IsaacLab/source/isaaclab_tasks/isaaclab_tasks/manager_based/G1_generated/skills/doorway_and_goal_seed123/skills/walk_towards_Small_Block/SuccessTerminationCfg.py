
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_towards_Small_Block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_towards_Small_Block skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL REQUIREMENT: Access object positions using env.scene['ObjectName'].data.root_pos_w
    # Access the required object: Object3 (Small Block)
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # CRITICAL REQUIREMENT: Access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # Access the required robot part(s): pelvis
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL REQUIREMENT: Use only relative distances between objects and robot parts.
    # Calculate the distance components between Object3 and the robot pelvis.
    # For X and Z, we use absolute difference as proximity is key.
    # For Y, we need the robot to be *behind* the block, so a direct difference is used, not absolute.
    dist_x = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0])
    dist_y = pelvis_pos[:, 1] - object3_pos[:, 1] # Robot pelvis y relative to block y
    dist_z = torch.abs(pelvis_pos[:, 2] - object3_pos[:, 2]) # Z-distance relative to block's Z

    # Define thresholds based on the task description and reward function alignment.
    # These thresholds are relative distances, adhering to the requirements.
    # The robot should be aligned in X, slightly behind the block in Y, and at a stable Z height relative to the block.
    # Thresholds are reasonable tolerances as per requirements (e.g., 0.05-0.1m for distances).
    threshold_x = 0.2 # meters: Robot pelvis x should be within 0.2m of block's x.
    threshold_y_min = -0.8 # meters: Robot pelvis y should be at least 0.8m behind block's y.
    threshold_y_max = -0.6 # meters: Robot pelvis y should be at most 0.6m behind block's y.
    threshold_z = 0.5 # meters: Robot pelvis z should be within 0.5m of block's z.

    # Success conditions for each dimension.
    # CRITICAL REQUIREMENT: All operations must work with batched environments.
    success_x = dist_x < threshold_x
    success_y = (dist_y > threshold_y_min) & (dist_y < threshold_y_max)
    success_z = dist_z < threshold_z

    # Combine all conditions for overall success.
    success_condition = success_x & success_y & success_z

    # CRITICAL REQUIREMENT: ALWAYS use check_success_duration and save_success_state.
    # Check if the success condition has been maintained for a sufficient duration.
    # Duration of 0.5 seconds is a reasonable tolerance for a stable final position.
    success = check_success_duration(env, success_condition, "walk_towards_Small_Block", duration=0.5)
    
    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_towards_Small_Block")
    
    return success

class SuccessTerminationCfg:
    # Define the success termination using the implemented function.
    success = DoneTerm(func=walk_towards_Small_Block_success)
