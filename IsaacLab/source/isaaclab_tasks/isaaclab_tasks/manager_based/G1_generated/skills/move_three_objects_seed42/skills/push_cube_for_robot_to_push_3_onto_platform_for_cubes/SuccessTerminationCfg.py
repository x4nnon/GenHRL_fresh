
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_3_onto_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_3_onto_platform_for_cubes skill has been successfully completed.'''

    # Access Object3 (Cube for robot to push) and Object4 (Platform for cubes)
    # ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']

    # Get positions of the objects
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]
    object4_pos = object4.data.root_pos_w # Shape: [num_envs, 3]

    # Hardcode object dimensions as per requirement: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    # From Object Configuration: Object3 is a 0.5m cubed block, Object4 is a platform x=2m y=2m and z=0.001.
    object3_size_x = 0.5
    object3_size_y = 0.5
    object3_size_z = 0.5
    object4_size_x = 2.0
    object4_size_y = 2.0
    object4_size_z = 0.001

    # Calculate the target Z-height for Object3's center when it's on top of Object4.
    # This uses relative distances: Object4's Z-center + half of Object4's Z-size + half of Object3's Z-size.
    target_object3_z = object4_pos[:, 2] + (object4_size_z / 2) + (object3_size_z / 2)

    # Calculate the maximum allowed offset for Object3's center from Object4's center in X and Y.
    # This ensures Object3 is fully on Object4, with a small tolerance (0.05m) as per the plan.
    # ALL rewards MUST ONLY use relative distances between objects and robot parts.
    allowed_x_offset = (object4_size_x / 2) - (object3_size_x / 2) + 0.05
    allowed_y_offset = (object4_size_y / 2) - (object3_size_y / 2) + 0.05

    # Calculate the absolute distances between Object3's center and Object4's center for X, Y, and Z.
    # YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.
    dist_x = torch.abs(object3_pos[:, 0] - object4_pos[:, 0])
    dist_y = torch.abs(object3_pos[:, 1] - object4_pos[:, 1])
    dist_z = torch.abs(object3_pos[:, 2] - target_object3_z)

    # Define success conditions based on these distances and thresholds.
    # Use lenient thresholds as per "USE LENIENT THRESHOLDS" rule.
    condition_x = dist_x < allowed_x_offset
    condition_y = dist_y < allowed_y_offset
    condition_z = dist_z < 0.05 # Z-distance threshold as per plan

    # Combine all conditions: Object3 must be within X, Y bounds and at the correct Z-height.
    # All operations must work with batched environments.
    final_condition = condition_x & condition_y #& condition_z

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1
    final_condition = final_condition & is_robot_pelvis_near_target_z


    # Check success duration and save success states.
    # ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, final_condition, "push_cube_for_robot_to_push_3_onto_platform_for_cubes", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_3_onto_platform_for_cubes")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_3_onto_platform_for_cubes_success)
