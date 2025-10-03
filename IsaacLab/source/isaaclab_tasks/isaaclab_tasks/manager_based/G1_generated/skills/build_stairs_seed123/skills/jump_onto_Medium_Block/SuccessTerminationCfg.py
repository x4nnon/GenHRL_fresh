
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Medium_Block_success(env) -> torch.Tensor:
    '''Determine if the jump_onto_Medium_Block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # ABSOLUTE REQUIREMENT: Access object positions using env.scene['ObjectName'].data.root_pos_w
    # ABSOLUTE REQUIREMENT: Access objects directly - objects should always exist in the scene
    medium_block = env.scene['Object2']

    # ABSOLUTE REQUIREMENT: Access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # ABSOLUTE REQUIREMENT: NEVER use hard-coded positions or arbitrary thresholds for object positions.
    # ABSOLUTE REQUIREMENT: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    # Medium Block (Object2) dimensions from skill information: x=1m, y=1m, z=0.6m
    medium_block_height = 0.6
    medium_block_half_x = 0.5
    medium_block_half_y = 0.5

    # Calculate block's center and top surface Z-coordinate.
    # ABSOLUTE REQUIREMENT: All success criteria MUST ONLY use relative distances between objects and robot parts.
    # This calculates the block's top surface Z relative to its root position.
    medium_block_pos = medium_block.data.root_pos_w
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]
    block_top_z = medium_block_pos[:, 2] + medium_block_height

    # Define target pelvis height above the block's top surface for stable standing.
    # This is a relative height to the block's top surface.
    target_pelvis_z_on_block = block_top_z + 0.7 # A typical standing height for a humanoid pelvis above a surface.

    # --- Success Conditions ---
    # 1. Both feet are horizontally within the block's boundaries (with a small tolerance).
    # ABSOLUTE REQUIREMENT: All operations must work with batched environments.
    # ABSOLUTE REQUIREMENT: Use relative distances.
    # ABSOLUTE REQUIREMENT: Use lenient thresholds for secondary conditions.
    horizontal_tolerance = 0.08 # 8 cm tolerance for feet to be considered on the block horizontally
    
    # Calculate horizontal distances of feet to block center
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - block_center_x)
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - block_center_y)
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - block_center_x)
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - block_center_y)

    # Check if feet are within block's horizontal boundaries plus tolerance
    left_foot_horizontally_on_block = (left_foot_dist_x < (medium_block_half_x + horizontal_tolerance)) & \
                                       (left_foot_dist_y < (medium_block_half_y + horizontal_tolerance))
    right_foot_horizontally_on_block = (right_foot_dist_x < (medium_block_half_x + horizontal_tolerance)) & \
                                        (right_foot_dist_y < (medium_block_half_y + horizontal_tolerance))

    feet_horizontally_on_block = left_foot_horizontally_on_block & right_foot_horizontally_on_block

    # 2. Both feet are vertically close to the block's top surface (with a small tolerance).
    # ABSOLUTE REQUIREMENT: All operations must work with batched environments.
    # ABSOLUTE REQUIREMENT: Use relative distances.
    # ABSOLUTE REQUIREMENT: Use lenient thresholds for secondary conditions.
    vertical_foot_tolerance = 0.15 # 15 cm tolerance for feet height relative to block top
    
    left_foot_vertically_on_block = torch.abs(left_foot_pos[:, 2] - block_top_z) < vertical_foot_tolerance
    right_foot_vertically_on_block = torch.abs(right_foot_pos[:, 2] - block_top_z) < vertical_foot_tolerance

    feet_vertically_on_block = left_foot_vertically_on_block & right_foot_vertically_on_block

    # 3. Pelvis is at a stable standing height above the block's top surface (with a small tolerance).
    # ABSOLUTE REQUIREMENT: All operations must work with batched environments.
    # ABSOLUTE REQUIREMENT: Use relative distances.
    # ABSOLUTE REQUIREMENT: Use lenient thresholds for secondary conditions.
    pelvis_height_tolerance = 0.2 # 20 cm tolerance for pelvis height relative to target
    
    pelvis_at_target_height = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block) < pelvis_height_tolerance

    # Combine all conditions for success.
    # ABSOLUTE REQUIREMENT: All operations must work with batched environments.
    condition = feet_horizontally_on_block & feet_vertically_on_block & pelvis_at_target_height

    # ABSOLUTE REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    # Check duration for stability (e.g., 0.5 seconds of meeting criteria)
    success = check_success_duration(env, condition, "jump_onto_Medium_Block", duration=0.5)
    
    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Medium_Block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Medium_Block_success)
