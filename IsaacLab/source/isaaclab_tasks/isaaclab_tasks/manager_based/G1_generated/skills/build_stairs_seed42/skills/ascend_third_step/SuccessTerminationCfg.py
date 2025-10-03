
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def ascend_third_step_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ascend_third_step skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects - adhering to rule: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    large_block = env.scene['Object3'] # Object3 is the Large Block for robot interaction
    large_block_pos = large_block.data.root_pos_w

    # Access the required robot parts - adhering to rule: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode Object3 dimensions based on the provided configuration (x=1m y=1m z=0.9m)
    # Adhering to rule: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    large_block_height = 0.9
    large_block_half_x = 0.5 # Half of 1m x-dimension
    large_block_half_y = 0.5 # Half of 1m y-dimension

    # Calculate the Z-coordinate of the top surface of the large block
    # This is a relative position based on the block's root and its height.
    large_block_top_z = large_block_pos[:, 2] + large_block_height / 2.0

    # Calculate relative distances for left foot to large block center and top surface
    # Adhering to rule: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    dist_left_foot_x = torch.abs(large_block_pos[:, 0] - left_foot_pos[:, 0])
    dist_left_foot_y = torch.abs(large_block_pos[:, 1] - left_foot_pos[:, 1])
    dist_left_foot_z_from_top = left_foot_pos[:, 2] - large_block_top_z # Positive if foot is above top surface

    # Calculate relative distances for right foot to large block center and top surface
    dist_right_foot_x = torch.abs(large_block_pos[:, 0] - right_foot_pos[:, 0])
    dist_right_foot_y = torch.abs(large_block_pos[:, 1] - right_foot_pos[:, 1])
    dist_right_foot_z_from_top = right_foot_pos[:, 2] - large_block_top_z # Positive if foot is above top surface

    # Calculate relative distances for pelvis to large block center and top surface
    dist_pelvis_x = torch.abs(large_block_pos[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_y = torch.abs(large_block_pos[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_z_from_top = pelvis_pos[:, 2] - large_block_top_z # Positive if pelvis is above top surface

    # Success conditions for feet:
    # Feet must be horizontally within the block's surface (0.45m tolerance from center, slightly less than half_x/y to ensure stability)
    # Feet must be vertically on or slightly above the block's top surface (-0.05m to 0.15m from top_z)
    # Adhering to rule: USE LENIENT THRESHOLDS for secondary conditions and REASONABLE TOLERANCES
    left_foot_on_block = (dist_left_foot_x < 0.45) & \
                         (dist_left_foot_y < 0.45) & \
                         (dist_left_foot_z_from_top > -0.05) & \
                         (dist_left_foot_z_from_top < 0.15)

    right_foot_on_block = (dist_right_foot_x < 0.45) & \
                          (dist_right_foot_y < 0.45) & \
                          (dist_right_foot_z_from_top > -0.05) & \
                          (dist_right_foot_z_from_top < 0.15)

    # Success conditions for pelvis:
    # Pelvis must be horizontally centered over the block (0.3m tolerance from center)
    # Pelvis must be at a specific height above the block's top surface (0.6m to 0.8m from top_z)
    pelvis_on_block = (dist_pelvis_x < 0.3) & \
                      (dist_pelvis_y < 0.3) & \
                      (dist_pelvis_z_from_top > 0.6) & \
                      (dist_pelvis_z_from_top < 0.8)

    # Overall success condition: both feet and pelvis meet their respective criteria
    # Adhering to rule: HANDLE TENSOR OPERATIONS CORRECTLY for batched environments
    condition = left_foot_on_block & right_foot_on_block & pelvis_on_block

    # Check duration and save success states - adhering to rules: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "ascend_third_step", duration=0.5) # Duration of 0.5 seconds for stability
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ascend_third_step")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ascend_third_step_success)
