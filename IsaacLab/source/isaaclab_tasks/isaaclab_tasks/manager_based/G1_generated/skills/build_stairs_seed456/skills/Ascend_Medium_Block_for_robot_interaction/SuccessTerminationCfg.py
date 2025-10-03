
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Ascend_Medium_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Ascend_Medium_Block_for_robot_interaction skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    robot = env.scene["robot"]
    
    # CORRECT: Using robot.body_names.index to get indices for feet and pelvis
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')
    
    # CORRECT: Accessing robot part positions for all environments in a batch
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object position
    # CORRECT: Direct access to Object2 (Medium Block) as per object configuration
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w
    
    # CORRECT: Hardcoding object dimensions from the skill description, as dimensions cannot be accessed from RigidObject
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0
    medium_block_height = 0.6
    
    # 3. Calculate target Z positions for feet and pelvis relative to the block's top surface
    # CORRECT: Relative target Z based on block's Z position and its height, plus a small offset for foot thickness
    target_foot_z = medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.05
    # CORRECT: Relative target Z for pelvis, representing a stable standing height above the block
    target_pelvis_z = medium_block_pos[:, 2] + medium_block_height / 2.0 + 0.7
    
    # 4. Calculate relative distances for success criteria
    # CORRECT: X-axis distance between left foot and Object2 center
    left_foot_x_dist = torch.abs(left_foot_pos[:, 0] - medium_block_pos[:, 0])
    # CORRECT: Y-axis distance between left foot and Object2 center
    left_foot_y_dist = torch.abs(left_foot_pos[:, 1] - medium_block_pos[:, 1])
    # CORRECT: Z-axis distance between left foot and Object2 top surface target
    left_foot_z_dist = torch.abs(left_foot_pos[:, 2] - target_foot_z)
    
    # CORRECT: X-axis distance between right foot and Object2 center
    right_foot_x_dist = torch.abs(right_foot_pos[:, 0] - medium_block_pos[:, 0])
    # CORRECT: Y-axis distance between right foot and Object2 center
    right_foot_y_dist = torch.abs(right_foot_pos[:, 1] - medium_block_pos[:, 1])
    # CORRECT: Z-axis distance between right foot and Object2 top surface target
    right_foot_z_dist = torch.abs(right_foot_pos[:, 2] - target_foot_z)
    
    # CORRECT: Z-axis distance between pelvis and Object2 top surface target
    pelvis_z_dist = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)
    
    # 5. Check success conditions based on calculated distances and thresholds
    # Threshold for feet X/Y position (within block bounds + 0.1m buffer)
    xy_threshold = medium_block_x_size / 2.0 + 0.1 # Since x_size and y_size are both 1.0, this is 0.5 + 0.1 = 0.6m
    
    # Condition for left foot being on the block
    left_foot_on_block_xy = (left_foot_x_dist < xy_threshold) & (left_foot_y_dist < xy_threshold)
    left_foot_at_target_z = (left_foot_z_dist < 0.1) # Z-distance threshold for feet
    left_foot_condition = left_foot_on_block_xy & left_foot_at_target_z
    
    # Condition for right foot being on the block
    right_foot_on_block_xy = (right_foot_x_dist < xy_threshold) & (right_foot_y_dist < xy_threshold)
    right_foot_at_target_z = (right_foot_z_dist < 0.1) # Z-distance threshold for feet
    right_foot_condition = right_foot_on_block_xy & right_foot_at_target_z
    
    # Condition for pelvis being at a stable height above the block
    pelvis_height_condition = (pelvis_z_dist < 0.2) # Z-distance threshold for pelvis
    
    # Combine all conditions: both feet must be on the block and pelvis at stable height
    # CORRECT: Combining conditions with proper tensor operations for batched environments
    condition = left_foot_condition & right_foot_condition & pelvis_height_condition
    
    # 6. Check duration and save success states
    # CORRECT: Using check_success_duration to ensure the condition holds for a specified duration
    success = check_success_duration(env, condition, "Ascend_Medium_Block_for_robot_interaction", duration=0.5)
    
    # CORRECT: Saving success states for environments that have met the success criteria
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Ascend_Medium_Block_for_robot_interaction")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Ascend_Medium_Block_for_robot_interaction_success)
