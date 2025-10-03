
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def adjust_Third_0_5m_cubed_block_on_Platform_success(env) -> torch.Tensor:
    '''Determine if the adjust_Third_0_5m_cubed_block_on_Platform skill has been successfully completed.'''

    # Access the required objects using approved patterns.
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    # Access the required robot parts using approved patterns.
    robot = env.scene["robot"]
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]

    # Hardcode object dimensions from the object configuration.
    # This is allowed as per requirements, as there's no way to access object dimensions dynamically.
    block_half_size = 0.5 / 2.0 # 0.25m for a 0.5m cubed block
    platform_half_x = 2.0 / 2.0 # 1.0m for a 2m wide platform
    platform_half_y = 2.0 / 2.0 # 1.0m for a 2m deep platform
    platform_height = 0.001 # Z dimension of platform

    # Calculate relative distances for Object3 placement on Object4.
    # All distances are calculated between object centers or relative to object boundaries.
    # This adheres to the requirement of using only relative distances.
    dist_x_obj3_obj4 = torch.abs(object3.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0])
    dist_y_obj3_obj4 = torch.abs(object3.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1])

    # Target Z for Object3 center when resting on Object4.
    # This is a relative target based on platform's Z and dimensions.
    target_obj3_z = object4.data.root_pos_w[:, 2] + (platform_height / 2.0) + block_half_size
    dist_z_obj3_obj4 = torch.abs(object3.data.root_pos_w[:, 2] - target_obj3_z)

    # Define thresholds for Object3 placement.
    # These are reasonable tolerances for success, slightly more lenient than reward margins.
    xy_tolerance = 0.10 # Allows for a 0.05m margin + 0.05m additional tolerance
    z_tolerance = 0.05 # Allows for a 0.05m tolerance in height

    # Condition for Object3 being fully on Object4.
    # The block's center must be within the platform's half-dimensions minus the block's half-size, plus tolerance.
    obj3_on_platform_x = dist_x_obj3_obj4 <= (platform_half_x - block_half_size + xy_tolerance)
    obj3_on_platform_y = dist_y_obj3_obj4 <= (platform_half_y - block_half_size + xy_tolerance)
    obj3_on_platform_z = dist_z_obj3_obj4 <= z_tolerance

    # Condition for robot's feet not being on the platform.
    # This ensures the robot is not interfering or standing on the target area.
    # The platform's top Z is calculated relative to its root position.
    platform_top_z = object4.data.root_pos_w[:, 2] + platform_height / 2.0
    # Feet should be below this Z or outside XY bounds of the platform.
    # A small buffer (0.05m for Z, 0.1m for XY) is added to the platform boundaries for leniency.
    foot_z_threshold = platform_top_z + 0.05

    # Check if left foot is NOT on platform.
    # This uses relative distances from the foot to the platform's center and Z height.
    left_foot_not_on_platform = (left_ankle_pos[:, 2] < foot_z_threshold) | \
                                (torch.abs(left_ankle_pos[:, 0] - object4.data.root_pos_w[:, 0]) > platform_half_x + 0.1) | \
                                (torch.abs(left_ankle_pos[:, 1] - object4.data.root_pos_w[:, 1]) > platform_half_y + 0.1)

    # Check if right foot is NOT on platform.
    # This also uses relative distances from the foot to the platform's center and Z height.
    right_foot_not_on_platform = (right_ankle_pos[:, 2] < foot_z_threshold) | \
                                 (torch.abs(right_ankle_pos[:, 0] - object4.data.root_pos_w[:, 0]) > platform_half_x + 0.1) | \
                                 (torch.abs(right_ankle_pos[:, 1] - object4.data.root_pos_w[:, 1]) > platform_half_y + 0.1)

    # Combine all conditions for success. All conditions must be met.
    condition = obj3_on_platform_x & obj3_on_platform_y & obj3_on_platform_z & \
                left_foot_not_on_platform & right_foot_not_on_platform

    # Check success duration and save success states.
    # Duration is set to 0.5 seconds to ensure stability.
    success = check_success_duration(env, condition, "adjust_Third_0_5m_cubed_block_on_Platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "adjust_Third_0_5m_cubed_block_on_Platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=adjust_Third_0_5m_cubed_block_on_Platform_success)
