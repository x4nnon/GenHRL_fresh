
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def move_three_objects_seed123_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the move_three_objects_seed123 skill has been successfully completed.'''

    # Hardcoded dimensions from the object configuration and skill description.
    # This follows the rule of hardcoding dimensions from the object configuration.
    block_half_size = 0.5 / 2.0 # 0.25m for a 0.5m cubed block
    platform_z_height = 0.001 # From task description: platform z=0.001
    
    # Calculate the target Z-coordinate for a block resting on the platform.
    # This is a relative height calculation based on object dimensions.
    target_block_z = platform_z_height + block_half_size

    # Access the required objects using approved patterns.
    # This directly accesses objects by their scene names.
    object1_pos = env.scene['Object1'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w
    object3_pos = env.scene['Object3'].data.root_pos_w
    platform_pos = env.scene['Object4'].data.root_pos_w

    # Initialize a tensor to track if each block is on the platform for all environments.
    # This handles batch processing correctly.
    block1_on_platform = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    block2_on_platform = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    block3_on_platform = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Define thresholds for success. These are reasonable tolerances for placement.
    # XY tolerance: 0.5m allows for the block's half-size (0.25m) plus a margin.
    # Z tolerance: 0.1m allows for slight variations in height.
    # These thresholds are hardcoded as allowed by the prompt for "reasonable tolerances".
    xy_tolerance = 0.5
    z_tolerance = 0.1

    # Check conditions for Object1 (first 0.5m cubed block)
    # Calculate relative distances in X, Y, and Z components.
    # This uses relative distances between objects, as required by rule 0.
    obj1_x_diff = torch.abs(object1_pos[:, 0] - platform_pos[:, 0])
    obj1_y_diff = torch.abs(object1_pos[:, 1] - platform_pos[:, 1])
    obj1_z_diff = torch.abs(object1_pos[:, 2] - target_block_z)

    # Combine conditions for Object1. All conditions must be met.
    block1_on_platform = (obj1_x_diff < xy_tolerance) & \
                         (obj1_y_diff < xy_tolerance) & \
                         (obj1_z_diff < z_tolerance)

    # Check conditions for Object2 (second 0.5m cubed block)
    # Calculate relative distances in X, Y, and Z components.
    obj2_x_diff = torch.abs(object2_pos[:, 0] - platform_pos[:, 0])
    obj2_y_diff = torch.abs(object2_pos[:, 1] - platform_pos[:, 1])
    obj2_z_diff = torch.abs(object2_pos[:, 2] - target_block_z)

    # Combine conditions for Object2.
    block2_on_platform = (obj2_x_diff < xy_tolerance) & \
                         (obj2_y_diff < xy_tolerance) & \
                         (obj2_z_diff < z_tolerance)

    # Check conditions for Object3 (third 0.5m cubed block)
    # Calculate relative distances in X, Y, and Z components.
    obj3_x_diff = torch.abs(object3_pos[:, 0] - platform_pos[:, 0])
    obj3_y_diff = torch.abs(object3_pos[:, 1] - platform_pos[:, 1])
    obj3_z_diff = torch.abs(object3_pos[:, 2] - target_block_z)

    # Combine conditions for Object3.
    block3_on_platform = (obj3_x_diff < xy_tolerance) & \
                         (obj3_y_diff < xy_tolerance) & \
                         (obj3_z_diff < z_tolerance)

    # Overall success condition: all three blocks must be on the platform simultaneously.
    # This combines the individual block success conditions using logical AND.
    overall_success_condition = block1_on_platform & block2_on_platform & block3_on_platform

    # Check success duration and save success states.
    # This ensures the condition is met for a continuous period before declaring success.
    # This follows rule 4 and 5.
    success = check_success_duration(env, overall_success_condition, "move_three_objects_seed123", duration=0.5)
    
    # Save success states for environments that have met the success criteria.
    # This follows rule 5.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "move_three_objects_seed123")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=move_three_objects_seed123_success)
