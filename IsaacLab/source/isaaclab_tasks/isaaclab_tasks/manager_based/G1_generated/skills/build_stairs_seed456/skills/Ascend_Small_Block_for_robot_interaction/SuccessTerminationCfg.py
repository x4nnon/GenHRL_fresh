
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Ascend_Small_Block_for_robot_interaction_success(env) -> torch.Tensor:
    '''Determine if the Ascend_Small_Block_for_robot_interaction skill has been successfully completed.'''

    # CRITICAL RULE: Access objects directly using env.scene['ObjectN']
    object1 = env.scene['Object1'] # Small Block for robot interaction

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: Hardcode object dimensions from the object configuration, as there is no way to access them dynamically.
    object1_height = 0.3
    object1_half_x = 1.0 / 2.0 # Object1 is 1m x 1m x 0.3m
    object1_half_y = 1.0 / 2.0

    # Calculate target z-position for feet on top of the block.
    # CRITICAL RULE: This is a relative position calculation (block's root Z + half its height).
    object1_top_z = object1.data.root_pos_w[:, 2] + object1_height / 2.0

    # Calculate relative distances for feet to the block's center (horizontal) and top surface (vertical).
    # CRITICAL RULE: All distances are relative to object positions or other robot parts.
    dist_left_foot_x = torch.abs(object1.data.root_pos_w[:, 0] - left_foot_pos[:, 0])
    dist_left_foot_y = torch.abs(object1.data.root_pos_w[:, 1] - left_foot_pos[:, 1])
    # dist_left_foot_z is the height of the foot relative to the block's top surface.
    dist_left_foot_z = left_foot_pos[:, 2] - object1_top_z

    dist_right_foot_x = torch.abs(object1.data.root_pos_w[:, 0] - right_foot_pos[:, 0])
    dist_right_foot_y = torch.abs(object1.data.root_pos_w[:, 1] - right_foot_pos[:, 1])
    # dist_right_foot_z is the height of the foot relative to the block's top surface.
    dist_right_foot_z = right_foot_pos[:, 2] - object1_top_z

    # Calculate relative distance for pelvis height above the block's top surface.
    dist_pelvis_z = pelvis_pos[:, 2] - object1_top_z

    # Define thresholds for success. These are lenient for secondary conditions.
    # Horizontal threshold for feet: within block boundaries with a small margin (0.45m from center).
    horizontal_threshold_xy = object1_half_x - 0.05 # 0.5 - 0.05 = 0.45m

    # Vertical threshold for feet: on top surface, allowing for foot thickness and slight variations.
    foot_z_min_threshold = -0.1 # Feet can be up to 10cm below the top surface (e.g., sinking slightly or thick foot)
    foot_z_max_threshold = 0.2  # Feet can be up to 20cm above the top surface (e.g., just landed or slightly lifted)

    # Vertical threshold for pelvis: stable height above block.
    pelvis_z_min_threshold = 0.6 # Pelvis should be at least 60cm above the block's top surface
    pelvis_z_max_threshold = 0.8 # Pelvis should be at most 80cm above the block's top surface

    # Check conditions for left foot being on the block's top surface.
    # CRITICAL RULE: Conditions are based on relative distances.
    left_foot_on_block_x = dist_left_foot_x < horizontal_threshold_xy
    left_foot_on_block_y = dist_left_foot_y < horizontal_threshold_xy
    left_foot_on_block_z = (dist_left_foot_z > foot_z_min_threshold) & (dist_left_foot_z < foot_z_max_threshold)
    left_foot_on_top_surface = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z

    # Check conditions for right foot being on the block's top surface.
    # CRITICAL RULE: Conditions are based on relative distances.
    right_foot_on_block_x = dist_right_foot_x < horizontal_threshold_xy
    right_foot_on_block_y = dist_right_foot_y < horizontal_threshold_xy
    right_foot_on_block_z = (dist_right_foot_z > foot_z_min_threshold) & (dist_right_foot_z < foot_z_max_threshold)
    right_foot_on_top_surface = right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z

    # Check condition for pelvis height, ensuring the robot is standing upright on the block.
    # CRITICAL RULE: Condition is based on relative distance.
    pelvis_at_stable_height = (dist_pelvis_z > pelvis_z_min_threshold) & (dist_pelvis_z < pelvis_z_max_threshold)

    # Combine all conditions for overall success.
    # Both feet must be on the block's top surface, and the pelvis must be at a stable height.
    # CRITICAL RULE: All operations handle batched environments.
    condition = left_foot_on_top_surface & right_foot_on_top_surface & pelvis_at_stable_height

    # CRITICAL RULE: Always use check_success_duration and save_success_state.
    # Duration set to 0.5 seconds to ensure stability on the block.
    success = check_success_duration(env, condition, "Ascend_Small_Block_for_robot_interaction", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Ascend_Small_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Ascend_Small_Block_for_robot_interaction_success)
