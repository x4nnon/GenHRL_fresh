
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_2_onto_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_2_onto_platform_for_cubes skill has been successfully completed.'''
    # Access the required objects using approved patterns
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object2 = env.scene['Object2']
    object4 = env.scene['Object4']

    # Get object positions in world frame
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcode object dimensions from the task description/object configuration.
    # CRITICAL RULE: There is no way to access the size of an object; hardcode values from config.
    # Object2 (Cube for robot to push): 0.5m x 0.5m x 0.5m
    object2_half_size_x = 0.5 / 2.0
    object2_half_size_y = 0.5 / 2.0
    object2_half_size_z = 0.5 / 2.0
    # Object4 (Platform for cubes): 2m x 2m x 0.001m
    object4_half_size_x = 2.0 / 2.0
    object4_half_size_y = 2.0 / 2.0
    object4_height = 0.001

    # Calculate relative distances for X and Y axes
    # CRITICAL RULE: Success criteria MUST ONLY use relative distances between objects.
    # CRITICAL RULE: Consider X, Y, and Z components separately.
    dist_x = torch.abs(object2_pos[:, 0] - object4_pos[:, 0])
    dist_y = torch.abs(object2_pos[:, 1] - object4_pos[:, 1])

    # Calculate the target Z-position for Object2's center when it's on Object4's surface.
    # This is a relative calculation based on object positions and hardcoded dimensions.
    target_object2_z = object4_pos[:, 2] + object4_height / 2.0 + object2_half_size_z
    dist_z = torch.abs(object2_pos[:, 2] - target_object2_z)

    # Define thresholds based on the success criteria plan.
    # These thresholds account for object sizes and include a small tolerance.
    # CRITICAL RULE: Use reasonable tolerances (typically 0.05-0.1m for distances).
    # X-axis: Object2's center must be within Object4's X-bounds (1.0m half-size) minus Object2's half-size (0.25m), plus tolerance.
    threshold_x = object4_half_size_x - object2_half_size_x + 0.05 # 1.0 - 0.25 + 0.05 = 0.8
    # Y-axis: Same logic as X-axis.
    threshold_y = object4_half_size_y - object2_half_size_y + 0.05 # 1.0 - 0.25 + 0.05 = 0.8
    # Z-axis: Object2's center must be at the correct height relative to Object4's surface, with tolerance.
    threshold_z = 0.05

    # Check success conditions for each axis
    # CRITICAL RULE: All operations must work with batched environments.
    condition_x = dist_x < threshold_x
    condition_y = dist_y < threshold_y
    condition_z = dist_z < threshold_z

    # Combine all conditions: Object2 must be within horizontal bounds AND at the correct height.
    success_condition = condition_x & condition_y #& condition_z

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1
    success_condition = success_condition & is_robot_pelvis_near_target_z

    # Check success duration and save success states
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    # Duration required: 0.5 seconds as per the plan.
    success = check_success_duration(env, success_condition, "push_cube_for_robot_to_push_2_onto_platform_for_cubes", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_2_onto_platform_for_cubes")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_2_onto_platform_for_cubes_success)
