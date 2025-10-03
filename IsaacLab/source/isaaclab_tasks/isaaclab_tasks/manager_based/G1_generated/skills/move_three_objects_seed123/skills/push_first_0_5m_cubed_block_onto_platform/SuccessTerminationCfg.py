
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_first_0_5m_cubed_block_onto_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_first_0_5m_cubed_block_onto_platform skill has been successfully completed.'''
    
    # Access the robot object
    # REASONING: Accessing robot using approved pattern.
    robot = env.scene["robot"]
    
    # Access the required objects based on the object configuration.
    # REASONING: Accessing objects directly using approved pattern (Object1, Object2, Object4).
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object2 = env.scene['Object2'] # 'second 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Get positions of objects.
    # REASONING: Accessing object positions using env.scene['ObjectName'].data.root_pos_w.
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Get position of robot's pelvis.
    # REASONING: Accessing robot part using robot.body_names.index('part_name').
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode object dimensions from the skill description/object configuration.
    # REASONING: Object dimensions are hardcoded as per requirements, not accessed from RigidObject.
    object1_half_size = 0.25 # 0.5m cubed block
    platform_half_x = 1.0    # 2m platform, half size is 1.0m
    platform_half_y = 1.0    # 2m platform, half size is 1.0m
    platform_z_thickness = 0.001 # Platform thickness from description

    # --- Condition 1: Object1 is entirely on Object4 (the platform) ---
    
    # Calculate relative distances between Object1 and Object4 centers in X and Y.
    # REASONING: Using relative distances between objects.
    dist_obj1_obj4_x = torch.abs(object1_pos[:, 0] - object4_pos[:, 0])
    dist_obj1_obj4_y = torch.abs(object1_pos[:, 1] - object4_pos[:, 1])

    # Calculate the target Z-height for Object1's center when resting on Object4.
    # This is platform's root Z + platform's thickness + Object1's half height.
    # REASONING: Calculating target Z-height based on relative positions and hardcoded dimensions.
    target_obj1_z = object4_pos[:, 2] + platform_z_thickness + object1_half_size
    dist_obj1_obj4_z = torch.abs(object1_pos[:, 2] - target_obj1_z)

    # Define thresholds for Object1 being on the platform.
    # The block's center must be within the platform's half-size minus the block's half-size,
    # plus a small tolerance for leniency.
    # REASONING: Using lenient thresholds derived from object dimensions, not arbitrary values.
    on_platform_x_threshold = platform_half_x - object1_half_size + 0.05
    on_platform_y_threshold = platform_half_y - object1_half_size + 0.05
    on_platform_z_threshold = 0.05 # Tolerance for Z-height

    # Check if Object1 is within the X, Y, and Z bounds of the platform.
    # REASONING: Combining conditions using tensor operations for batched environments.
    on_platform_x_cond = dist_obj1_obj4_x <= on_platform_x_threshold
    on_platform_y_cond = dist_obj1_obj4_y <= on_platform_y_threshold
    on_platform_z_cond = dist_obj1_obj4_z <= on_platform_z_threshold
    
    object1_on_platform_condition = on_platform_x_cond & on_platform_y_cond & on_platform_z_cond

    # --- Condition 2: Robot is in a stable and ready position ---

    # Check robot's pelvis height.
    # REASONING: Using absolute Z for height, as allowed, with a reasonable tolerance.
    target_pelvis_z = 0.7 # Target stable pelvis height
    pelvis_height_threshold = 0.15 # Tolerance for pelvis height
    pelvis_height_cond = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) <= pelvis_height_threshold

    # Check horizontal distance between robot's pelvis and Object2 (next block).
    # REASONING: Using relative horizontal distance between robot part and object.
    dist_pelvis_obj2_x = pelvis_pos[:, 0] - object2_pos[:, 0]
    dist_pelvis_obj2_y = pelvis_pos[:, 1] - object2_pos[:, 1]
    pelvis_obj2_horizontal_dist = torch.sqrt(dist_pelvis_obj2_x**2 + dist_pelvis_obj2_y**2)
    
    pelvis_obj2_distance_threshold = 2.0 # Robot pelvis within 2.0m horizontal distance of Object2
    pelvis_obj2_proximity_cond = pelvis_obj2_horizontal_dist <= pelvis_obj2_distance_threshold

    # Combine all success conditions.
    # REASONING: All conditions must be met for success.
    overall_success_condition = object1_on_platform_condition & pelvis_height_cond & pelvis_obj2_proximity_cond
    
    # Check duration and save success states.
    # REASONING: Using check_success_duration and save_success_state as required.
    success = check_success_duration(env, overall_success_condition, "push_first_0_5m_cubed_block_onto_platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_first_0_5m_cubed_block_onto_platform")
    
    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=push_first_0_5m_cubed_block_onto_platform_success)
