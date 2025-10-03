
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Large_Block_for_push_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Large_Block_for_push skill has been successfully completed.'''

    # 1. Get robot parts and their positions
    # REQUIREMENT: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # 2. Get Object3 (Large Block) position
    # REQUIREMENT: Access object positions using env.scene['ObjectName'].data.root_pos_w
    large_block = env.scene['Object3']
    large_block_pos = large_block.data.root_pos_w

    # 3. Hardcode Object3 dimensions (from object configuration)
    # REQUIREMENT: There is no way to access the SIZE of an object, hardcode values from config.
    large_block_height = 0.9
    large_block_width_x = 1.0
    large_block_width_y = 1.0

    # Define target parameters for pelvis positioning
    target_x_offset = 0.2 # Desired distance from the block's face to the robot's pelvis
    desired_pelvis_z = 0.7 # Desired stable pelvis height (absolute Z is allowed for posture goals)

    # Calculate target position for pelvis relative to the large block
    # REQUIREMENT: Success criteria must only use relative distances between objects and robot parts.
    # Target X: In front of the block's negative X face, with an offset.
    target_pelvis_x = large_block_pos[:, 0] - (large_block_width_x / 2.0) - target_x_offset
    # Target Y: Aligned with the block's center Y.
    target_pelvis_y = large_block_pos[:, 1]
    # Target Z: Absolute desired standing height for pelvis. (Allowed for posture goals)

    # 4. Calculate pelvis position conditions
    # REQUIREMENT: Consider X, Y, Z components separately.
    # Pelvis X-position condition: within 0.3m of target X
    pelvis_x_condition = torch.abs(pelvis_pos[:, 0] - target_pelvis_x) < 0.3
    # Pelvis Y-position condition: within 0.3m of target Y
    pelvis_y_condition = torch.abs(pelvis_pos[:, 1] - target_pelvis_y) < 0.3
    # Pelvis Z-position condition: within 0.2m of desired height
    pelvis_z_condition = torch.abs(pelvis_pos[:, 2] - desired_pelvis_z) < 0.2

    # 5. Calculate feet collision avoidance conditions
    # Define block's bounding box in world coordinates for collision check
    # REQUIREMENT: All calculations must handle batched environments.
    block_min_x = large_block_pos[:, 0] - (large_block_width_x / 2.0)
    block_max_x = large_block_pos[:, 0] + (large_block_width_x / 2.0)
    block_min_y = large_block_pos[:, 1] - (large_block_width_y / 2.0)
    block_max_y = large_block_pos[:, 1] + (large_block_width_y / 2.0)
    block_top_z = large_block_pos[:, 2] + (large_block_height / 2.0)

    # Define buffers for feet collision check
    foot_z_buffer = 0.1 # Feet must be below block_top_z - 0.1m
    foot_xy_buffer = 0.1 # Feet must be outside block's XY projection by 0.1m

    # Check if left foot is NOT on or inside the block
    # Foot Z-position must be below the block's top surface (with buffer)
    left_foot_z_ok = left_foot_pos[:, 2] < (block_top_z - foot_z_buffer)
    # Foot X-Y position must be outside the block's horizontal bounds (with buffer)
    left_foot_x_outside = (left_foot_pos[:, 0] < block_min_x - foot_xy_buffer) | (left_foot_pos[:, 0] > block_max_x + foot_xy_buffer)
    left_foot_y_outside = (left_foot_pos[:, 1] < block_min_y - foot_xy_buffer) | (left_foot_pos[:, 1] > block_max_y + foot_xy_buffer)
    # A foot is "not on/inside" if its Z is low enough OR its XY is outside the block.
    left_foot_not_colliding = left_foot_z_ok | (left_foot_x_outside | left_foot_y_outside)

    # Check if right foot is NOT on or inside the block (similar logic)
    right_foot_z_ok = right_foot_pos[:, 2] < (block_top_z - foot_z_buffer)
    right_foot_x_outside = (right_foot_pos[:, 0] < block_min_x - foot_xy_buffer) | (right_foot_pos[:, 0] > block_max_x + foot_xy_buffer)
    right_foot_y_outside = (right_foot_pos[:, 1] < block_min_y - foot_xy_buffer) | (right_foot_pos[:, 1] > block_max_y + foot_xy_buffer)
    right_foot_not_colliding = right_foot_z_ok | (right_foot_x_outside | right_foot_y_outside)

    # Combine all conditions
    # All pelvis conditions must be met AND both feet must not be colliding
    condition = pelvis_x_condition & pelvis_y_condition & pelvis_z_condition & \
                left_foot_not_colliding & right_foot_not_colliding

    # 6. Check duration and save success states
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "walk_to_Large_Block_for_push", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Large_Block_for_push")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Large_Block_for_push_success)
