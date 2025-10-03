
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_third_0_5m_cubed_block_onto_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_third_0_5m_cubed_block_onto_platform skill has been successfully completed.'''

    # Access the robot and relevant objects using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3']  # third 0.5m cubed block
    object4 = env.scene['Object4']  # platform

    # Access object positions using approved patterns.
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Access robot part position using approved patterns.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode object dimensions as per requirements (cannot access dynamically).
    # Object3 (0.5m cubed block)
    object3_half_size_x = 0.25
    object3_half_size_y = 0.25
    object3_half_size_z = 0.25

    # Object4 (platform: x=2m y=2m, z=0.001)
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_half_size_z = 0.0005 # half of 0.001m height

    # 1. Object3's XY containment on Object4.
    # Calculate relative distances between Object3 and Object4 centers in X and Y.
    # This ensures the block is horizontally centered on the platform within tolerance.
    # Using relative distances as per rule 0.
    dist_x_obj3_obj4 = torch.abs(object3_pos[:, 0] - object4_pos[:, 0])
    dist_y_obj3_obj4 = torch.abs(object3_pos[:, 1] - object4_pos[:, 1])

    # Calculate the maximum allowed distance for full containment, plus a small tolerance.
    # Thresholds are derived from object dimensions and a lenient tolerance (0.05m).
    # This ensures the block is fully on the platform.
    xy_containment_threshold_x = object4_half_size_x - object3_half_size_x + 0.05
    xy_containment_threshold_y = object4_half_size_y - object3_half_size_y + 0.05

    is_contained_x = dist_x_obj3_obj4 < xy_containment_threshold_x
    is_contained_y = dist_y_obj3_obj4 < xy_containment_threshold_y

    # 2. Object3's Z height relative to Object4.
    # Calculate the target Z position for Object3's center when resting on Object4.
    # This is Object4's center Z + Object4's half height + Object3's half height.
    # Using relative distances and hardcoded dimensions.
    target_z_obj3 = object4_pos[:, 2] + object4_half_size_z + object3_half_size_z
    
    # Check if Object3's Z position is close to the target Z, with a tolerance of 0.075m.
    # This ensures the block is resting on the platform.
    is_at_correct_height = torch.abs(object3_pos[:, 2] - target_z_obj3) < 0.075

    # 3. Robot's final position (pelvis) for stability.
    # Check robot pelvis X position relative to the platform's far edge.
    # This ensures the robot has completed the push and is not too far past the platform.
    # Using relative distances.
    platform_far_x_edge = object4_pos[:, 0] + object4_half_size_x
    # The robot's pelvis should be within 0.75m past the platform's far edge.
    # This is a relative distance check.
    is_pelvis_not_overshooting = (pelvis_pos[:, 0] - platform_far_x_edge) < 0.75

    # Check robot pelvis Z height for stability.
    # This is an absolute Z height check, which is allowed for stability.
    # A stable height of 0.7m with a tolerance of 0.15m.
    is_pelvis_stable_height = torch.abs(pelvis_pos[:, 2] - 0.7) < 0.15

    # Combine all conditions for overall success. All conditions must be true.
    condition = is_contained_x & is_contained_y & is_at_correct_height & \
                is_pelvis_not_overshooting & is_pelvis_stable_height

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    success = check_success_duration(env, condition, "push_third_0_5m_cubed_block_onto_platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_third_0_5m_cubed_block_onto_platform")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_third_0_5m_cubed_block_onto_platform_success)
