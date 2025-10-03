
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def ascend_first_step_success(env) -> torch.Tensor:
    '''Determine if the ascend_first_step skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL RULE: Access robot directly
    robot = env.scene["robot"]
    
    # CRITICAL RULE: Access object directly using its scene name
    small_block = env.scene['Object1'] # Small Block for robot interaction
    
    # CRITICAL RULE: Hardcode object dimensions from the object configuration
    small_block_x_size = 1.0
    small_block_y_size = 1.0
    small_block_height = 0.3

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # CRITICAL RULE: Get positions for all environments (batched operation)
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # CRITICAL RULE: Access object position using env.scene['ObjectName'].data.root_pos_w
    small_block_pos = small_block.data.root_pos_w

    # Calculate target Z positions relative to the block's top surface
    # CRITICAL RULE: Use relative distances for target Z positions
    block_top_z = small_block_pos[:, 2] + small_block_height / 2.0
    target_foot_z_min = block_top_z - 0.05 # Feet must be at or slightly above block top
    target_pelvis_z = block_top_z + 0.7 # Pelvis should be 0.7m above block top for standing

    # Define horizontal bounds for feet and pelvis on the block
    # CRITICAL RULE: Use relative distances for horizontal bounds
    foot_horizontal_tolerance_x = (small_block_x_size / 2.0) - 0.1 # 0.4m from center
    foot_horizontal_tolerance_y = (small_block_y_size / 2.0) - 0.1 # 0.4m from center
    pelvis_horizontal_tolerance_x = (small_block_x_size / 2.0) - 0.2 # 0.3m from center
    pelvis_horizontal_tolerance_y = (small_block_y_size / 2.0) - 0.2 # 0.3m from center

    # --- Left Foot Conditions ---
    # CRITICAL RULE: Use relative distances for all checks
    left_foot_x_dist = torch.abs(left_foot_pos[:, 0] - small_block_pos[:, 0])
    left_foot_y_dist = torch.abs(left_foot_pos[:, 1] - small_block_pos[:, 1])
    left_foot_z_cond = left_foot_pos[:, 2] > target_foot_z_min # Above block top surface

    left_foot_on_block = (left_foot_x_dist < foot_horizontal_tolerance_x) & \
                         (left_foot_y_dist < foot_horizontal_tolerance_y) & \
                         left_foot_z_cond

    # --- Right Foot Conditions ---
    # CRITICAL RULE: Use relative distances for all checks
    right_foot_x_dist = torch.abs(right_foot_pos[:, 0] - small_block_pos[:, 0])
    right_foot_y_dist = torch.abs(right_foot_pos[:, 1] - small_block_pos[:, 1])
    right_foot_z_cond = right_foot_pos[:, 2] > target_foot_z_min # Above block top surface

    right_foot_on_block = (right_foot_x_dist < foot_horizontal_tolerance_x) & \
                          (right_foot_y_dist < foot_horizontal_tolerance_y) & \
                          right_foot_z_cond

    # --- Pelvis Conditions ---
    # CRITICAL RULE: Use relative distances for all checks
    pelvis_x_dist = torch.abs(pelvis_pos[:, 0] - small_block_pos[:, 0])
    pelvis_y_dist = torch.abs(pelvis_pos[:, 1] - small_block_pos[:, 1])
    pelvis_z_dist = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) # At target standing height

    pelvis_centered_and_height = (pelvis_x_dist < pelvis_horizontal_tolerance_x) & \
                                 (pelvis_y_dist < pelvis_horizontal_tolerance_y) & \
                                 (pelvis_z_dist < 0.1) # Within 0.1m of target Z

    # Combine all conditions for overall success
    # Both feet must be on the block AND pelvis must be centered and at standing height
    condition = left_foot_on_block & right_foot_on_block & pelvis_centered_and_height
    
    # CRITICAL RULE: Check duration and save success states
    success = check_success_duration(env, condition, "ascend_first_step", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ascend_first_step")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ascend_first_step_success)
