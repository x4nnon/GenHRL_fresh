
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Medium_Block_for_robot_interaction_success(env) -> torch.Tensor:
    '''Determine if the jump_onto_Medium_Block_for_robot_interaction skill has been successfully completed.'''
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # CRITICAL RULE: Access robot part positions using robot.data.body_pos_w
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: Access object positions using env.scene['ObjectName'].data.root_pos_w
    # Object2 is the Medium Block for robot interaction as per object configuration
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w

    # CRITICAL RULE: Hardcode object dimensions from the object configuration, as there's no way to access them dynamically.
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # CRITICAL RULE: All success criteria must only use relative distances between objects and robot parts.
    # Calculate the Z-position of the top surface of the medium block.
    # The root_pos_w is the center of the block, so add half its height to get the top.
    target_block_top_z = medium_block_pos[:, 2] + (medium_block_height / 2.0)

    # Calculate the target Z-position for the pelvis for stable standing.
    # Based on reward function, typical pelvis height is 0.7m from ground.
    # Relative to block top: 0.7m (ground height) - (block_height / 2.0)
    pelvis_standing_height_relative_to_ground = 0.7
    target_pelvis_z_offset_from_block_top = pelvis_standing_height_relative_to_ground - (medium_block_height / 2.0)
    target_pelvis_z = target_block_top_z + target_pelvis_z_offset_from_block_top

    # Get the X-Y center of the block for horizontal alignment checks.
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]

    # Success Condition 1: Both feet are on the top surface of the Medium Block (Z-axis).
    # CRITICAL RULE: Consider X, Y, Z components separately.
    # Using absolute difference for distance check.
    foot_z_tolerance = 0.1 # Lenient threshold for Z-position of feet
    left_foot_on_block_z = torch.abs(left_foot_pos[:, 2] - target_block_top_z) < foot_z_tolerance
    right_foot_on_block_z = torch.abs(right_foot_pos[:, 2] - target_block_top_z) < foot_z_tolerance

    # Success Condition 2: Both feet are within the horizontal boundaries of the Medium Block (X-Y plane).
    # CRITICAL RULE: Use relative distances. Half of block size + tolerance for boundaries.
    # Half of block x/y size is 0.5m. Adding 0.1m tolerance makes it 0.6m.
    foot_xy_tolerance = (medium_block_x_size / 2.0) + 0.1 # 0.5 + 0.1 = 0.6m
    left_foot_on_block_x = torch.abs(left_foot_pos[:, 0] - block_center_x) < foot_xy_tolerance
    left_foot_on_block_y = torch.abs(left_foot_pos[:, 1] - block_center_y) < foot_xy_tolerance
    right_foot_on_block_x = torch.abs(right_foot_pos[:, 0] - block_center_x) < foot_xy_tolerance
    right_foot_on_block_y = torch.abs(right_foot_pos[:, 1] - block_center_y) < foot_xy_tolerance

    # Success Condition 3: Robot's pelvis is at a stable standing height relative to the block's top surface.
    # CRITICAL RULE: Use relative distances.
    pelvis_z_tolerance = 0.15 # Lenient threshold for pelvis Z-position
    pelvis_at_stable_height = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) < pelvis_z_tolerance

    # Combine all conditions. All must be true for success.
    # CRITICAL RULE: All operations must work with batched environments.
    condition = (left_foot_on_block_z & right_foot_on_block_z &
                 left_foot_on_block_x & left_foot_on_block_y &
                 right_foot_on_block_x & right_foot_on_block_y &
                 pelvis_at_stable_height)

    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    # Duration required: 0.5 seconds as per plan.
    success = check_success_duration(env, condition, "jump_onto_Medium_Block_for_robot_interaction", duration=0.5)

    # CRITICAL RULE: Save success states for environments that succeeded.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Medium_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Medium_Block_for_robot_interaction_success)
