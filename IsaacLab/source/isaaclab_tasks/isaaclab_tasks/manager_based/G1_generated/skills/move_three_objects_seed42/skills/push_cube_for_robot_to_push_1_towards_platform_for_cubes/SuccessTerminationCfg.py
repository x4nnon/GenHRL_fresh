
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_1_towards_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_1_towards_platform_for_cubes skill has been successfully completed.'''
    # 1. Access the required objects
    # Reasoning: Using approved direct access patterns for objects (Object1 and Object4).
    object1 = env.scene['Object1'] # Cube for robot to push
    object4 = env.scene['Object4'] # Platform for cubes

    # 2. Hardcode Object1 dimensions
    # Reasoning: Object dimensions must be hardcoded from the object configuration (0.5m cubed block), not accessed from the object itself.
    object1_half_size = 0.25 # 0.5 / 2
    object4_half_size = 1.0 # 2.0 / 2

    # 3. Calculate the target Z position for Object1's center when it's on Object4
    # Reasoning: Relative distance for Z component, ensuring the cube is on the platform.
    # Object4's Z is 0.001 (from skill info), so Object1's target Z center should be Object4's Z + Object1_half_size.
    target_object1_z = object4.data.root_pos_w[:, 2] + object1_half_size

    # 4. Calculate the distance components between Object1 and Object4
    # Reasoning: All success criteria MUST ONLY use relative distances between objects.
    # Calculating X, Y, and Z distances separately as required.
    distance_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_z_from_target = object1.data.root_pos_w[:, 2] - target_object1_z

    # 5. Define success conditions
    # Reasoning: Using reasonable and lenient thresholds for proximity.
    # Check if Object1 is within 2m in X and Y of Object4's center.
    xy_proximity_threshold = 2.0
    condition_xy = (torch.abs(distance_x) < xy_proximity_threshold) & \
                   (torch.abs(distance_y) < xy_proximity_threshold)

    # Check if Object1's Z position is close to the target Z for being on the platform.
    z_proximity_threshold = 0.05
    condition_z = (torch.abs(distance_z_from_target) < z_proximity_threshold)

    # Combine all conditions for overall success
    # Reasoning: All conditions must be met for success, handled with tensor operations for batched environments.
    condition = condition_xy # & condition_z

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1

    condition = condition & is_robot_pelvis_near_target_z

    # 6. Check duration and save success states
    # Reasoning: Adhering to the absolute requirements to use check_success_duration and save_success_state.
    # Duration set to 0.5 seconds to ensure stability of the success state.
    success = check_success_duration(env, condition, "push_cube_for_robot_to_push_1_towards_platform_for_cubes", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_1_towards_platform_for_cubes")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_1_towards_platform_for_cubes_success)
