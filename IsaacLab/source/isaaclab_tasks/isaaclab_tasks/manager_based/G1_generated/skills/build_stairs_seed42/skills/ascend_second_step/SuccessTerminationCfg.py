
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def ascend_second_step_success(env) -> torch.Tensor:
    '''Determine if the ascend_second_step skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    # Accessing the robot object from the scene. (Requirement 3)
    robot = env.scene["robot"]
    
    # Getting indices for specific robot body parts. (Requirement 3)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')
    
    # Accessing the world positions of the robot parts. (Requirement 3)
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object position
    # Accessing the Medium Block (Object2) from the scene. (Requirement 2, 5)
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w
    
    # Hardcoding Medium Block dimensions as per object configuration. (Requirement 6, Rule: There is no way to access the SIZE of an object)
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # 3. Calculate relative positions and target heights
    # Calculate the Z position of the block's top surface. (Requirement 1, Rule: Use relative distances)
    block_top_z = medium_block_pos[:, 2] + medium_block_height / 2.0
    
    # Calculate the target Z position for the feet on the block. (Requirement 1, Rule: Use relative distances)
    # A small offset (0.05m) ensures the feet are clearly on top.
    target_foot_z_on_block = block_top_z + 0.05
    
    # Calculate the target Z position for the pelvis for stable standing. (Requirement 1, Rule: Use relative distances)
    # 0.7m above the block's top surface.
    target_pelvis_z_on_block = block_top_z + 0.7
    
    # Get the horizontal center of the block. (Requirement 1, Rule: Use relative distances)
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]

    # 4. Check success conditions
    # Horizontal bounds for feet on the block. (Requirement 1, Rule: Use relative distances)
    # 0.45m from center means 0.9m coverage on a 1.0m block, allowing a 0.05m margin on each side.
    horizontal_tolerance = 0.45 
    
    # Z-height tolerance for feet.
    foot_z_tolerance = 0.1
    
    # Z-height tolerance for pelvis.
    pelvis_z_tolerance = 0.15

    # Condition for left foot horizontal position relative to block center. (Requirement 1, Rule: Consider X, Y, Z separately)
    left_foot_x_ok = torch.abs(left_foot_pos[:, 0] - block_center_x) < horizontal_tolerance
    left_foot_y_ok = torch.abs(left_foot_pos[:, 1] - block_center_y) < horizontal_tolerance
    
    # Condition for right foot horizontal position relative to block center. (Requirement 1, Rule: Consider X, Y, Z separately)
    right_foot_x_ok = torch.abs(right_foot_pos[:, 0] - block_center_x) < horizontal_tolerance
    right_foot_y_ok = torch.abs(right_foot_pos[:, 1] - block_center_y) < horizontal_tolerance
    
    # Condition for left foot Z height relative to target foot Z. (Requirement 1, Rule: Consider X, Y, Z separately)
    left_foot_z_ok = torch.abs(left_foot_pos[:, 2] - target_foot_z_on_block) < foot_z_tolerance
    
    # Condition for right foot Z height relative to target foot Z. (Requirement 1, Rule: Consider X, Y, Z separately)
    right_foot_z_ok = torch.abs(right_foot_pos[:, 2] - target_foot_z_on_block) < foot_z_tolerance
    
    # Combined condition for both feet being correctly placed on the block.
    feet_on_block_condition = (left_foot_x_ok & left_foot_y_ok & left_foot_z_ok) & \
                              (right_foot_x_ok & right_foot_y_ok & right_foot_z_ok)
    
    # Condition for pelvis Z height relative to target pelvis Z. (Requirement 1, Rule: Consider X, Y, Z separately)
    pelvis_height_ok = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block) < pelvis_z_tolerance
    
    # Overall success condition: both feet are on the block and the pelvis is at a stable standing height.
    condition = feet_on_block_condition & pelvis_height_ok
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Using check_success_duration to ensure stability over time. (Requirement 6)
    success = check_success_duration(env, condition, "ascend_second_step", duration=0.5)
    
    # Saving success states for environments that have met the criteria. (Requirement 6)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ascend_second_step")
    
    return success

class SuccessTerminationCfg:
    # Registering the success function as a termination condition.
    success = DoneTerm(func=ascend_second_step_success)
