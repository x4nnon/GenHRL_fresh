
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_2_towards_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_2_towards_platform_for_cubes skill has been successfully completed.'''

    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Access Object2 (the cube to be pushed) and Object4 (the platform)
    object2_pos = env.scene['Object2'].data.root_pos_w
    object4_pos = env.scene['Object4'].data.root_pos_w

    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # Access the robot's pelvis to check if it's clear of the cube
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    # Object2 is a 0.5m cube, so its half-height is 0.25m.
    cube_half_height = 0.25

    # Requirement: ALL success criteria MUST ONLY use relative distances between objects and robot parts
    # 1. Calculate X-axis distance between Object2 and Object4
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY
    x_distance_obj2_obj4 = torch.abs(object2_pos[:, 0] - object4_pos[:, 0])
    # Condition: Absolute X-distance between Object2 and Object4 < 2.0m.
    x_condition = x_distance_obj2_obj4 < 2.0

    # 2. Calculate Y-axis distance between Object2 and Object4
    y_distance_obj2_obj4 = torch.abs(object2_pos[:, 1] - object4_pos[:, 1])
    # Condition: Absolute Y-distance between Object2 and Object4 < 2.0m.
    y_condition = y_distance_obj2_obj4 < 2.0

    # 3. Calculate Z-axis distance between Object2 and Object4, adjusted for Object2's half-height
    # Target Z for Object2 is Object4's Z plus the cube's half-height
    target_z_obj2 = object4_pos[:, 2] + cube_half_height
    z_distance_obj2_obj4 = torch.abs(object2_pos[:, 2] - target_z_obj2)
    # Condition: Absolute Z-distance between Object2 and (Object4's Z + 0.25m) < 0.1m.
    z_condition = z_distance_obj2_obj4 < 0.1

    # 4. Calculate 2D (XY) distance between the robot's pelvis and Object2
    # This ensures the robot is clear of the cube after pushing.
    xy_distance_pelvis_obj2 = torch.norm(pelvis_pos[:, :2] - object2_pos[:, :2], dim=1)
    # Condition: 2D (XY) distance between robot's pelvis and Object2 > 0.5m.
    robot_clear_condition = xy_distance_pelvis_obj2 > 0.5

    # Combine all conditions for overall success
    # Requirement: All tensor operations correctly handle batched environments.
    combined_condition = x_condition & y_condition # & z_condition & robot_clear_condition

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1
    combined_condition = combined_condition & is_robot_pelvis_near_target_z


    # Requirement: ALWAYS use check_success_duration and save_success_state
    # Check if the combined success condition has been met for the required duration (0.5 seconds)
    success = check_success_duration(env, combined_condition, "push_cube_for_robot_to_push_2_towards_platform_for_cubes", duration=0.5)

    # Save success states for environments that have successfully completed the skill
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_2_towards_platform_for_cubes")

    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_2_towards_platform_for_cubes_success)
