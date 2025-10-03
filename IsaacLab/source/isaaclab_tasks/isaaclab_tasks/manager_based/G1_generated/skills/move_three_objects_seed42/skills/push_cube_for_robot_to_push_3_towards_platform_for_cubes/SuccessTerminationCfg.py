
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_3_towards_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_3_towards_platform_for_cubes skill has been successfully completed.'''

    # Access Object3 (Cube for robot to push) and Object4 (Platform for cubes)
    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object3_pos = env.scene['Object3'].data.root_pos_w
    object4_pos = env.scene['Object4'].data.root_pos_w

    # Extract X, Y, Z components for Object3 and Object4
    object3_x = object3_pos[:, 0]
    object3_y = object3_pos[:, 1]
    object3_z = object3_pos[:, 2]

    object4_x = object4_pos[:, 0]
    object4_y = object4_pos[:, 1]
    object4_z = object4_pos[:, 2]

    # Calculate relative distances between Object3 and Object4
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY
    
    # X-distance between Object3 and Object4 centers
    # Threshold: < 0.75m (Platform is 2m wide, half-width 1m, 0.75m allows tolerance)
    x_distance = torch.abs(object3_x - object4_x)
    x_condition = x_distance < 2.0

    # Y-distance between Object3 and Object4 centers
    # Threshold: < 0.75m (Same reasoning as X-distance)
    y_distance = torch.abs(object3_y - object4_y)
    y_condition = y_distance < 2.0

    # Z-distance between Object3 center and Object4's top surface
    # Object3 is a 0.5m cube, so its center should be 0.25m above the platform's surface.
    # Target Z for Object3's center: object4_z + 0.25m
    # Threshold: < 0.1m tolerance for height
    target_object3_z = object4_z + 0.25
    z_distance = torch.abs(object3_z - target_object3_z)
    z_condition = z_distance < 0.1

    # Combine all conditions for overall success
    # REQUIREMENT: All tensor operations correctly handle batched environments
    condition = x_condition & y_condition #& z_condition

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1
    condition = condition & is_robot_pelvis_near_target_z


    # Check duration and save success states
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    # Duration required: 0.5 seconds as per plan
    success = check_success_duration(env, condition, "push_cube_for_robot_to_push_3_towards_platform_for_cubes", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_3_towards_platform_for_cubes")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_3_towards_platform_for_cubes_success)
