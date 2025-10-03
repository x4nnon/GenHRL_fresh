
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Second_0_5m_cubed_block_towards_Platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Second_0_5m_cubed_block_towards_Platform skill has been successfully completed.'''
    # Access the required objects using approved patterns
    # Object2: Second 0.5m cubed block
    object2 = env.scene['Object2']
    # Object4: Platform
    object4 = env.scene['Object4']

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcoded object dimensions from object configuration (0.5m cubed block, platform height 0.001m, platform 2x2m)
    # This adheres to the rule of hardcoding dimensions from the config, not accessing them from the object.
    object2_half_size = 0.25 # Half size of a 0.5m cubed block
    platform_height = 0.001 # Height of the platform
    platform_half_x = 1.0 # Half width of the 2m platform
    platform_half_y = 1.0 # Half depth of the 2m platform

    # Calculate target Z-height for Object2's center when it's on the platform.
    # This is a relative calculation based on the platform's position and its height, plus the block's half-size.
    platform_top_z = object4_pos[:, 2] + platform_height / 2.0
    target_obj2_z = platform_top_z + object2_half_size

    # Calculate target X and Y bounds for Object2's center when it's on the platform.
    # These are relative to the platform's position and its dimensions, ensuring the block is on the platform.
    # The X-bounds are set to allow the block to be on or very close to the edge from which it was pushed.
    target_obj2_x_min = object4_pos[:, 0] - platform_half_x + object2_half_size - 0.1 # Slightly before the ideal edge position
    target_obj2_x_max = object4_pos[:, 0] - platform_half_x + object2_half_size + 0.3 # Slightly past the ideal edge position

    # Y bounds for Object2 to be within the platform's width.
    # These are relative to the platform's Y-center and half-width, allowing for slight overlap.
    target_obj2_y_min = object4_pos[:, 1] - platform_half_y + object2_half_size - 0.1 # Allow slight overlap
    target_obj2_y_max = object4_pos[:, 1] + platform_half_y - object2_half_size + 0.1 # Allow slight overlap

    # Check if Object2 is at the correct Z-height.
    # This uses a relative distance check (absolute difference from target Z).
    is_obj2_z_correct = torch.abs(object2_pos[:, 2] - target_obj2_z) < 0.05

    # Check if Object2's X-position is within the desired range on the platform.
    # This uses relative position checks against the calculated bounds.
    is_obj2_x_on_platform = (object2_pos[:, 0] > target_obj2_x_min) & \
                           (object2_pos[:, 0] < target_obj2_x_max)

    # Check if Object2's Y-position is within the platform's Y-bounds.
    # This uses relative position checks against the calculated bounds.
    is_obj2_y_on_platform = (object2_pos[:, 1] > target_obj2_y_min) & \
                           (object2_pos[:, 1] < target_obj2_y_max)

    # Combine all conditions for success. All conditions must be met.
    condition = is_obj2_z_correct & is_obj2_x_on_platform & is_obj2_y_on_platform

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Duration set to 0.5 seconds to ensure the block is stably on the platform.
    success = check_success_duration(env, condition, "push_Second_0_5m_cubed_block_towards_Platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Second_0_5m_cubed_block_towards_Platform")

    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=push_Second_0_5m_cubed_block_towards_Platform_success)
