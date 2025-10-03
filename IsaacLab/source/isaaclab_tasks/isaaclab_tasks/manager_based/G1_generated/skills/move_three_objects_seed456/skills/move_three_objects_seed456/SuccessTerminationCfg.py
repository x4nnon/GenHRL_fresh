
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def move_three_objects_seed456_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the move_three_objects_seed456 skill has been successfully completed.'''

    # Define object dimensions based on the object configuration.
    # Block dimensions: 0.5m cube. Half-size is 0.25m. This is hard-coded from the object configuration, as required.
    block_half_size = 0.25
    # Platform dimensions: 2m x 2m x 0.001m. Half-width/length is 1.0m, height is 0.001m.
    # These values are hard-coded from the object configuration, as required.
    platform_half_x = 1.0
    platform_half_y = 1.0
    platform_height = 0.001

    # Access the three blocks and the platform using approved patterns.
    # Object1: First 0.5m cubed block
    # Accessing object directly using env.scene['ObjectName'] as required.
    object1 = env.scene['Object1']
    # Accessing object position using .data.root_pos_w as required.
    object1_pos = object1.data.root_pos_w

    # Object2: Second 0.5m cubed block
    object2 = env.scene['Object2']
    object2_pos = object2.data.root_pos_w

    # Object3: Third 0.5m cubed block
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Object4: Platform
    platform = env.scene['Object4']
    platform_pos = platform.data.root_pos_w

    # Calculate the Z-coordinate of the platform's top surface.
    # The platform's root_pos_w is at its base, so add its height to get the top.
    platform_top_z = platform_pos[:, 2] + platform_height

    # Define thresholds for success based on the plan.
    # Horizontal tolerance: 0.85m (half platform width/length - half block width/length + 0.1m tolerance)
    # This threshold is derived from object dimensions and a reasonable tolerance, not arbitrary.
    horizontal_tolerance = 0.85
    # Vertical tolerance: 0.05m (for resting on surface)
    # This is a reasonable tolerance for resting on a surface, as required.
    vertical_tolerance = 0.05

    # Function to check if a single block is on the platform.
    def is_block_on_platform(block_pos, platform_pos, platform_top_z, block_half_size, horizontal_tolerance, vertical_tolerance):
        # Calculate relative x, y distances from block center to platform center.
        # Using relative distances between objects as required.
        x_distance = torch.abs(block_pos[:, 0] - platform_pos[:, 0])
        y_distance = torch.abs(block_pos[:, 1] - platform_pos[:, 1])

        # Calculate the Z-coordinate of the block's base.
        # Block's root_pos_w is at its center, so subtract half its height to get the base.
        block_base_z = block_pos[:, 2] - block_half_size
        # Calculate the absolute Z-distance between the block's base and the platform's top surface.
        # Using relative distances between objects as required.
        z_distance = torch.abs(block_base_z - platform_top_z)

        # Check if the block is within the horizontal bounds and resting on the platform vertically.
        # All conditions must be met for the block to be considered on the platform.
        # Conditions are checked for specific components (x, y, z) separately as required.
        on_platform_x = x_distance < horizontal_tolerance
        on_platform_y = y_distance < horizontal_tolerance
        on_platform_z = z_distance < vertical_tolerance

        return on_platform_x & on_platform_y & on_platform_z

    # Check success for each block individually.
    # This handles batch processing correctly as all operations are on tensors.
    object1_on_platform = is_block_on_platform(object1_pos, platform_pos, platform_top_z, block_half_size, horizontal_tolerance, vertical_tolerance)
    object2_on_platform = is_block_on_platform(object2_pos, platform_pos, platform_top_z, block_half_size, horizontal_tolerance, vertical_tolerance)
    object3_on_platform = is_block_on_platform(object3_pos, platform_pos, platform_top_z, block_half_size, horizontal_tolerance, vertical_tolerance)

    # Overall success condition: all three blocks must be on the platform.
    # This combines conditions using tensor logical operations.
    all_blocks_on_platform = object1_on_platform & object2_on_platform & object3_on_platform

    # Check success duration and save success states.
    # The duration is set to 0.5 seconds as per the plan.
    # Using check_success_duration as required.
    success = check_success_duration(env, all_blocks_on_platform, "move_three_objects_seed456", duration=0.5)

    # Save success states for environments that have met the success criteria.
    # Using save_success_state as required.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "move_three_objects_seed456")

    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=move_three_objects_seed456_success)
