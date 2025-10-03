
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def adjust_Second_0_5m_cubed_block_on_Platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the adjust_Second_0_5m_cubed_block_on_Platform skill has been successfully completed.'''

    # Access the required objects using approved patterns
    # Object2: Second 0.5m cubed block
    object2 = env.scene['Object2']
    # Object4: Platform
    object4 = env.scene['Object4']

    # Hardcode object dimensions as per rules, obtained from the object configuration
    # This is necessary because object dimensions cannot be accessed directly from RigidObjectData.
    platform_width_x = 2.0
    platform_depth_y = 2.0
    platform_height_z = 0.001
    block_size = 0.5

    # Calculate the target position for Object2's center relative to Object4's center and top surface.
    # Object4's root_pos_w is its center.
    # The block's center should align with the platform's center in x, y.
    # The block's base should be on the platform's top surface in z.
    # The target Z-center of the block is the platform's Z-center + half platform height + half block height.
    # This calculation uses relative positions derived from object4's z-position and hardcoded dimensions.
    target_object2_z_center = object4.data.root_pos_w[:, 2] + platform_height_z / 2 + block_size / 2

    # Calculate relative distances for Object2 from its target position on Object4.
    # All distances are calculated as differences between object positions, ensuring they are relative.
    distance_x = object2.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y = object2.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_z = object2.data.root_pos_w[:, 2] - target_object2_z_center

    # Define thresholds for success.
    # Max offset for block center to be fully on platform: (platform_dimension - block_dimension) / 2.
    # This ensures the block is entirely within the platform's boundaries.
    max_horizontal_offset = (platform_width_x - block_size) / 2 # For a 2m platform and 0.5m block, this is (2.0 - 0.5) / 2 = 0.75m.
    
    # Add a small tolerance to the thresholds to allow for minor variations and make success achievable.
    # These are reasonable tolerances as per guidelines (typically 0.05-0.1m).
    tolerance_horizontal = 0.05 # 5 cm tolerance for x and y positioning.
    tolerance_vertical = 0.05   # 5 cm tolerance for z positioning.

    # Success conditions:
    # 1. Block's X-center is within the platform's X-boundaries (plus tolerance).
    # This uses a relative distance check against a calculated maximum offset.
    is_on_platform_x = torch.abs(distance_x) <= (max_horizontal_offset + tolerance_horizontal)
    # 2. Block's Y-center is within the platform's Y-boundaries (plus tolerance).
    # This uses a relative distance check against a calculated maximum offset.
    is_on_platform_y = torch.abs(distance_y) <= (max_horizontal_offset + tolerance_horizontal)
    # 3. Block's Z-center is at the correct height relative to the platform's top surface (plus tolerance).
    # This uses a relative distance check against a tolerance for the z-axis.
    is_at_correct_height_z = torch.abs(distance_z) <= tolerance_vertical

    # Combine all conditions for overall success.
    # All conditions must be true for the skill to be considered successful.
    # All operations work on batched environments.
    success_condition = is_on_platform_x & is_on_platform_y & is_at_correct_height_z

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # The duration is set to 0.5 seconds, meaning the block must be in the success state for at least this long.
    success = check_success_duration(env, success_condition, "adjust_Second_0_5m_cubed_block_on_Platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "adjust_Second_0_5m_cubed_block_on_Platform")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=adjust_Second_0_5m_cubed_block_on_Platform_success)
