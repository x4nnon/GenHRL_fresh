
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Small_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_onto_Small_Block_for_robot_interaction skill has been successfully completed.'''

    # Access the robot object using the approved pattern.
    robot = env.scene["robot"]

    # Access the Small Block (Object1) using the approved pattern.
    small_block = env.scene['Object1']

    # Hardcode dimensions for the Small Block as per requirement 6 and "HOW TO USE OBJECT DIMENSIONS".
    # Object1: "Small Block for robot interaction", x=1m y=1m z=0.3m
    small_block_height = 0.3
    small_block_half_width = 0.5  # Half of 1m width
    small_block_half_depth = 0.5  # Half of 1m depth

    # Access robot body part positions using approved patterns.
    # Requirement 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Access object position using approved pattern.
    # Requirement 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    block_pos = small_block.data.root_pos_w

    # Calculate relative distances for feet to the block's center and top surface.
    # Requirement 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # Requirement 3 (Position & Motion Rules): USE RELATIVE DISTANCES.

    # Horizontal distances for left foot to block center (X and Y components separately).
    left_foot_x_dist_to_block_center = torch.abs(left_foot_pos[:, 0] - block_pos[:, 0])
    left_foot_y_dist_to_block_center = torch.abs(left_foot_pos[:, 1] - block_pos[:, 1])

    # Horizontal distances for right foot to block center (X and Y components separately).
    right_foot_x_dist_to_block_center = torch.abs(right_foot_pos[:, 0] - block_pos[:, 0])
    right_foot_y_dist_to_block_center = torch.abs(right_foot_pos[:, 1] - block_pos[:, 1])

    # Vertical distances for feet to block's top surface.
    # The block's top surface is at block_pos[:, 2] + small_block_height.
    # The success criteria uses "at its top surface", so we check relative to block_pos[:, 2] + small_block_height.
    # The tolerance of 0.1m allows for foot thickness and slight variation.
    target_foot_z_on_block = block_pos[:, 2] + small_block_height
    left_foot_z_dist_to_block_top = torch.abs(left_foot_pos[:, 2] - target_foot_z_on_block)
    right_foot_z_dist_to_block_top = torch.abs(right_foot_pos[:, 2] - target_foot_z_on_block)

    # Vertical distance for pelvis to block's top surface.
    # Target pelvis height relative to block top is 0.7m (as per success criteria plan and reward function).
    target_pelvis_z_on_block = block_pos[:, 2] + small_block_height + 0.7
    pelvis_z_dist_to_block_top = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)

    # Define success thresholds based on the plan.
    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds (except for object dimensions and tolerances).
    # Requirement 11 (Success Criteria Rules): USE LENIENT THRESHOLDS.
    # Requirement 12 (Success Criteria Rules): REASONABLE TOLERANCES.
    horizontal_tolerance = 0.05  # Added to half-dimensions for lenient bounds as per plan (0.55m for 0.5m half-width)
    foot_vertical_tolerance = 0.1 # As per plan
    pelvis_vertical_tolerance = 0.15 # As per plan

    # Condition for both feet being horizontally within the block's bounds.
    # This uses relative distances and hardcoded block dimensions, which is allowed.
    left_foot_horizontal_on_block = (left_foot_x_dist_to_block_center < (small_block_half_width + horizontal_tolerance)) & \
                                    (left_foot_y_dist_to_block_center < (small_block_half_depth + horizontal_tolerance))
    right_foot_horizontal_on_block = (right_foot_x_dist_to_block_center < (small_block_half_width + horizontal_tolerance)) & \
                                     (right_foot_y_dist_to_block_center < (small_block_half_depth + horizontal_tolerance))
    feet_horizontal_on_block = left_foot_horizontal_on_block & right_foot_horizontal_on_block

    # Condition for both feet being vertically at the block's top surface.
    # This uses relative distances and hardcoded block height, which is allowed.
    feet_vertical_on_block = (left_foot_z_dist_to_block_top < foot_vertical_tolerance) & \
                             (right_foot_z_dist_to_block_top < foot_vertical_tolerance)

    # Condition for pelvis being at a stable standing height relative to the block's top surface.
    # This uses relative distances and hardcoded block height and target standing height, which is allowed.
    pelvis_stable_height_on_block = (pelvis_z_dist_to_block_top < pelvis_vertical_tolerance)

    # Combine all conditions for overall success.
    # All conditions must be met simultaneously.
    condition = feet_horizontal_on_block & feet_vertical_on_block & pelvis_stable_height_on_block

    # Check duration and save success states.
    # Requirement 6 (Absolute Requirements): ALWAYS use check_success_duration and save_success_state.
    # Duration is 0.5 seconds as specified in the plan.
    success = check_success_duration(env, condition, "jump_onto_Small_Block_for_robot_interaction", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Small_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Small_Block_for_robot_interaction_success)
