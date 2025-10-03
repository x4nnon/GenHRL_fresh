
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_for_robot_interaction_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block_for_robot_interaction_for_climb skill has been successfully completed.'''
    # Access the required object (Small Block for robot interaction is Object1)
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    small_block = env.scene['Object1']
    small_block_pos = small_block.data.root_pos_w # [num_envs, 3]

    # Hardcoded dimensions for Object1 (Small Block: x=1m, y=1m, z=0.3m)
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode values from object config.
    small_block_x_dim = 1.0

    # Access the required robot part(s)
    robot = env.scene["robot"]

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Calculate average feet position for precise positioning
    # CRITICAL RULE: All operations must work with batched environments
    avg_feet_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_feet_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_feet_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # Define target offsets and heights relative to the block and ground
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds for target locations.
    # Instead, define targets relative to existing objects.
    target_x_offset_from_block_center = 0.5 # 0.5m before the block's center along X
    target_pelvis_z = 0.7 # Stable standing height for the robot
    target_feet_z = 0.0 # Ground level for feet

    # Calculate target coordinates relative to Object1's position
    # CRITICAL RULE: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    target_pos_x = small_block_pos[:, 0] - target_x_offset_from_block_center
    target_pos_y = small_block_pos[:, 1]

    # Calculate distances for average feet position
    # CRITICAL RULE: ALWAYS use relative distances. Consider X, Y, Z components separately.
    dist_feet_x_to_target = torch.abs(avg_feet_pos_x - target_pos_x)
    dist_feet_y_to_target = torch.abs(avg_feet_pos_y - target_pos_y)
    dist_feet_z_to_ground = torch.abs(avg_feet_pos_z - target_feet_z) # Z is relative to ground (0.0)

    # Calculate distances for pelvis position
    dist_pelvis_x_to_target = torch.abs(pelvis_pos[:, 0] - target_pos_x)
    dist_pelvis_y_to_target = torch.abs(pelvis_pos[:, 1] - target_pos_y)
    dist_pelvis_z_to_target_height = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Check for overshooting the block
    # This ensures the robot is positioned *in front* of the block, not past it.
    overshoot_threshold_buffer = 0.1 # A small buffer past the front face of the block
    block_front_face_x = small_block_pos[:, 0] + small_block_x_dim / 2
    not_overshooting = avg_feet_pos_x < (block_front_face_x + overshoot_threshold_buffer)

    # Define success conditions with reasonable tolerances
    # CRITICAL RULE: Use lenient thresholds for secondary conditions, strict for primary.
    feet_x_aligned = dist_feet_x_to_target < 0.1 # Feet are within 10cm of target X
    feet_y_aligned = dist_feet_y_to_target < 0.1 # Feet are within 10cm of target Y
    feet_z_on_ground = dist_feet_z_to_ground < 0.05 # Feet are within 5cm of the ground

    pelvis_x_aligned = dist_pelvis_x_to_target < 0.15 # Pelvis is within 15cm of target X
    pelvis_y_aligned = dist_pelvis_y_to_target < 0.15 # Pelvis is within 15cm of target Y
    pelvis_z_at_height = dist_pelvis_z_to_target_height < 0.1 # Pelvis is within 10cm of target Z height

    # Combine all conditions for overall success
    condition = (feet_x_aligned & feet_y_aligned & feet_z_on_ground &
                 pelvis_x_aligned & pelvis_y_aligned & pelvis_z_at_height &
                 not_overshooting)

    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "walk_to_Small_Block_for_robot_interaction_for_climb", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block_for_robot_interaction_for_climb")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_for_robot_interaction_for_climb_success)
