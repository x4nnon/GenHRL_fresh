
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Large_Block_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Large_Block_for_climb skill has been successfully completed.'''

    # Requirement 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds (except for object dimensions and Z height goal)
    # Requirement 5: Access objects directly - objects should always exist in the scene

    # Access the required objects
    large_block = env.scene['Object3'] # Object3 is the Large Block as per configuration

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Pelvis position for all environments

    # Requirement 6: THERE IS NO way to access the SIZE of an object. Hardcode values from object config.
    # Object3 dimensions (from description: x=1m y=1m z=0.9m)
    large_block_half_x = 0.5 # Half of 1m
    large_block_half_y = 0.5 # Half of 1m
    large_block_height = 0.9

    # Define target position relative to large_block's center
    # Target X: 0.6m in front of the block's face. Block's face is at large_block.x - large_block_half_x.
    # So, target_x_offset = large_block_half_x + 0.6 = 0.5 + 0.6 = 1.1m
    # Requirement: ALL success criteria MUST ONLY use relative distances between objects and robot parts
    # The target_pelvis_x and target_pelvis_y are relative to the block's position.
    target_pelvis_x = large_block.data.root_pos_w[:, 0] - (large_block_half_x + 0.6)
    target_pelvis_y = large_block.data.root_pos_w[:, 1]
    # Requirement: Z is the only absolute position allowed for height goals.
    target_pelvis_z = 0.7 # Stable standing height

    # Calculate distances to target
    # Requirement: Consider X, Y, and Z components separately.
    dist_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    dist_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Define collision boundaries relative to Object3's center for collision check
    # These are relative to the block's position.
    block_min_x = large_block.data.root_pos_w[:, 0] - large_block_half_x
    block_max_x = large_block.data.root_pos_w[:, 0] + large_block_half_x
    block_min_y = large_block.data.root_pos_w[:, 1] - large_block_half_y
    block_max_y = large_block.data.root_pos_w[:, 1] + large_block_half_y
    block_min_z = large_block.data.root_pos_w[:, 2] - (large_block_height / 2.0)
    block_max_z = large_block.data.root_pos_w[:, 2] + (large_block_height / 2.0)

    # Check if pelvis is NOT colliding with the block's volume
    # This ensures the robot is in front of the block, not inside or on top.
    # A small buffer (0.1m) is added to block_max_z for the Z collision check, consistent with reward.
    pelvis_inside_block_x = (pelvis_pos[:, 0] > block_min_x) & (pelvis_pos[:, 0] < block_max_x)
    pelvis_inside_block_y = (pelvis_pos[:, 1] > block_min_y) & (pelvis_pos[:, 1] < block_max_y)
    pelvis_inside_block_z = (pelvis_pos[:, 2] > block_min_z) & (pelvis_pos[:, 2] < block_max_z + 0.1)
    is_pelvis_colliding = pelvis_inside_block_x & pelvis_inside_block_y & pelvis_inside_block_z

    # Success conditions:
    # 1. Pelvis is close to the target X position (in front of the block)
    # 2. Pelvis is close to the target Y position (aligned with block's center)
    # 3. Pelvis is close to the target Z position (stable standing height)
    # 4. Pelvis is NOT colliding with the block (ensures it's not inside the block's volume)
    # Requirement: Use lenient thresholds for secondary conditions, strict for primary.
    # Using 0.2m for all position distances, which is a reasonable tolerance.
    condition = (dist_x < 0.2) & \
                (dist_y < 0.2) & \
                (dist_z < 0.2) & \
                (~is_pelvis_colliding) # Ensure pelvis is not inside the block

    # Requirement 6: ALWAYS use check_success_duration and save_success_state
    # Check duration for success (e.g., 0.5 seconds of meeting criteria)
    success = check_success_duration(env, condition, "walk_to_Large_Block_for_climb", duration=0.5)

    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Large_Block_for_climb")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Large_Block_for_climb_success)
