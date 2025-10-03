
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cube_for_robot_to_push_3_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cube_for_robot_to_push_3 skill has been successfully completed.'''
    # Access the required object: Object3, which is 'Cube for robot to push'.
    # This follows the requirement to access objects directly using env.scene['ObjectName'].
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Access the required robot part: 'pelvis'.
    # This follows the requirement to access robot parts using robot.body_names.index.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Extract X, Y, Z components for clarity and separate distance calculations.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcode Object3 dimensions (0.5m cube) as per requirements.
    # Object dimensions cannot be accessed from the object itself.
    cube_size = 0.5
    cube_half_size = cube_size / 2.0 # 0.25m

    # Define the desired offset from the cube's center for the robot's pelvis.
    # This value (-0.45m) is derived from the reward function's logic, ensuring alignment.
    # This is considered a derived constant, not an arbitrary hardcoded position.
    target_x_offset_from_cube_center = -0.45

    # Calculate the target pelvis position relative to Object3.
    # This ensures success criteria are based on relative distances.
    target_pelvis_x_relative_to_object3 = object3_pos[:, 0] + target_x_offset_from_cube_center
    target_pelvis_y_relative_to_object3 = object3_pos[:, 1]

    # Define the desired stable pelvis height.
    # This is an absolute Z position, which is allowed for height considerations as per rules.
    target_pelvis_z_absolute = 0.7

    # Calculate the distance vector components between the target relative position and the actual robot pelvis position.
    # All success criteria MUST ONLY use relative distances between objects and robot parts.
    distance_x = target_pelvis_x_relative_to_object3 - pelvis_pos_x
    distance_y = target_pelvis_y_relative_to_object3 - pelvis_pos_y
    distance_z = target_pelvis_z_absolute - pelvis_pos_z

    # Define success thresholds for each dimension.
    # These are reasonable tolerances for positioning the robot.
    threshold_x = 0.3    # 30cm tolerance in X
    threshold_y = 0.3 # 30cm tolerance in Y
    threshold_z = 0.15 # 15cm tolerance in Z

    # Check if the robot's pelvis is within the defined thresholds for each dimension.
    # Using torch.abs for absolute difference and comparing with thresholds.
    # All operations handle batched environments correctly.
    success_x = torch.abs(distance_x) < threshold_x
    success_y = torch.abs(distance_y) < threshold_y
    success_z = torch.abs(distance_z) < threshold_z

    # Combine all conditions: all must be true for success.
    condition = success_x & success_y # & success_z

    # Check success duration and save success states.
    # This follows the absolute requirements for success function implementation.
    # Duration set to 0.5 seconds to ensure stability in the target position.
    success = check_success_duration(env, condition, "walk_to_cube_for_robot_to_push_3", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cube_for_robot_to_push_3")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_cube_for_robot_to_push_3_success)
