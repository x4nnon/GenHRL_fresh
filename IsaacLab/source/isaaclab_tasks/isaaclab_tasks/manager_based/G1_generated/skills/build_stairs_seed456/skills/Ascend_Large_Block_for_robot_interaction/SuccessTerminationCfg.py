
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Ascend_Large_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Ascend_Large_Block_for_robot_interaction skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Hardcoded dimensions for the Large Block (Object3) based on the object configuration
    # "Object3": "Large Block for robot interaction" measures x=1m, y=1m, z=0.9m
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    LARGE_BLOCK_X_SIZE = 1.0
    LARGE_BLOCK_Y_SIZE = 1.0
    LARGE_BLOCK_HEIGHT = 0.9

    # 1. Get robot parts
    # Accessing the robot object directly, as per approved patterns.
    robot = env.scene["robot"]
    
    # Access robot part indices using robot.body_names.index for compliance.
    # This adheres to the rule: "ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]"
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # Access robot part positions using robot.data.body_pos_w for compliance.
    # This adheres to the rule: "ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]"
    # All operations work with batched environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    # 2. Get object position
    # Access Object3 (Large Block) using env.scene['Object3'] for compliance.
    # This adheres to the rule: "ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w"
    object_large_block = env.scene['Object3'] 
    large_block_root_pos = object_large_block.data.root_pos_w

    # Calculate target positions relative to the Large Block's root position.
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    # Target horizontal position is the center of the large block
    target_x_block = large_block_root_pos[:, 0]
    target_y_block = large_block_root_pos[:, 1]
    # Target Z for feet on top surface of the block (relative to block's root_pos_w)
    target_z_on_block = large_block_root_pos[:, 2] + LARGE_BLOCK_HEIGHT / 2.0
    # Target Z for pelvis when standing stably on the block (approx. 0.7m above block surface)
    pelvis_stable_z_target = target_z_on_block + 0.7

    # 3. Calculate distances using relative positions only.
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    # Horizontal distance of pelvis to block center (x and y components separately)
    # This adheres to the rule: "YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY"
    distance_pelvis_x = torch.abs(pelvis_pos[:, 0] - target_x_block)
    distance_pelvis_y = torch.abs(pelvis_pos[:, 1] - target_y_block)

    # Vertical distance of feet to block top surface (z component only)
    distance_left_foot_z = torch.abs(left_foot_pos[:, 2] - target_z_on_block)
    distance_right_foot_z = torch.abs(right_foot_pos[:, 2] - target_z_on_block)

    # Vertical distance of pelvis to stable target height (z component only)
    distance_pelvis_z_stable = torch.abs(pelvis_pos[:, 2] - pelvis_stable_z_target)

    # 4. Check success conditions with reasonable tolerances.
    # This adheres to the rule: "USE LENIENT THRESHOLDS" and "REASONABLE TOLERANCES"
    # Pelvis horizontally centered over the block (tolerance 0.4m for 1m block, allowing for some sway)
    pelvis_horizontally_centered = (distance_pelvis_x < 0.4) & (distance_pelvis_y < 0.4)

    # Both feet on top surface of the block (tolerance 0.1m, allowing for slight variations in foot height)
    feet_on_block_top = (distance_left_foot_z < 0.1) & (distance_right_foot_z < 0.1)

    # Pelvis at stable height above the block (tolerance 0.15m, allowing for slight posture variations)
    pelvis_at_stable_height = (distance_pelvis_z_stable < 0.15)

    # Combine all conditions for overall success.
    # All operations work with batched environments.
    condition = pelvis_horizontally_centered & feet_on_block_top & pelvis_at_stable_height
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Using a duration of 0.5 seconds to ensure stable landing.
    # This adheres to the rule: "ALWAYS use check_success_duration and save_success_state"
    success = check_success_duration(env, condition, "Ascend_Large_Block_for_robot_interaction", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Ascend_Large_Block_for_robot_interaction")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Ascend_Large_Block_for_robot_interaction_success)
