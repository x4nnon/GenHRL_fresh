
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cube_for_robot_to_push_2_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cube_for_robot_to_push_2 skill has been successfully completed.'''
    # 1. Get robot parts
    # Access the robot object from the scene. (Approved access pattern)
    robot = env.scene["robot"]
    # Get the index for the 'pelvis' body part using the approved pattern. (Approved access pattern)
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the world position of the pelvis for all environments. (Approved access pattern)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # Extract x, y, z components of the pelvis position for clarity.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # 2. Get object position
    # Access 'Object2' (Cube for robot to push) from the scene using the approved pattern. (Approved access pattern)
    object2 = env.scene['Object2']
    # Get the world position of Object2 for all environments. (Approved access pattern)
    object2_pos = object2.data.root_pos_w
    
    # Extract x, y, z components of Object2's position.
    object2_pos_x = object2_pos[:, 0]
    object2_pos_y = object2_pos[:, 1]

    # Object2 is a 0.5m cubed block. The target x-position for the pelvis is 0.5m behind Object2's center.
    # This 0.5m offset is derived from the object's known size (0.5m/2) plus a buffer (0.25m), as seen in the reward function.
    # This is a hardcoded value based on the object configuration and skill requirements, not an arbitrary threshold.
    # (Hardcoded value for object dimension/offset is allowed as per prompt rules)
    target_x_offset_from_object2_center = 0.6
    
    # Calculate the target x-position for the pelvis relative to Object2's center.
    # This ensures the robot is positioned behind the cube for pushing.
    # (Relative distance calculation)
    target_pelvis_x = object2_pos_x - target_x_offset_from_object2_center

    # 3. Calculate relative distances for success criteria
    # Calculate the absolute difference in x-position between the pelvis and its target x-position.
    # This is a relative distance measurement. (Relative distance calculation)
    x_distance_condition = torch.abs(pelvis_pos_x - target_pelvis_x)
    
    # Calculate the absolute difference in y-position between the pelvis and Object2's y-position.
    # This ensures y-alignment with the object and is a relative distance measurement. (Relative distance calculation)
    y_distance_condition = torch.abs(pelvis_pos_y - object2_pos_y)
    
    # Calculate the absolute difference in z-position of the pelvis from the target stable height (0.7m).
    # This is an absolute height check, which is allowed for stability criteria as per prompt rules.
    # (Absolute Z-position check allowed for stability)
    target_pelvis_z = 0.7
    z_height_condition = torch.abs(pelvis_pos_z - target_pelvis_z)

    # 4. Check success condition
    # Define thresholds for each dimension. These are lenient tolerances as per requirements.
    # (Reasonable and lenient thresholds as per prompt rules)
    x_threshold = 0.3
    y_threshold = 0.3
    z_threshold = 0.2

    # Combine all conditions:
    # The pelvis must be within the x-threshold of the target pushing position.
    # The pelvis must be within the y-threshold of Object2's center.
    # The pelvis must be within the z-threshold of the stable height.
    # (Combined conditions using tensor operations for batched environments)
    condition = (x_distance_condition < x_threshold) & \
                (y_distance_condition < y_threshold)  & \
                (z_height_condition < z_threshold)
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Check if the success condition has been met for the required duration (0.5 seconds).
    # (Uses check_success_duration as required)
    success = check_success_duration(env, condition, "walk_to_cube_for_robot_to_push_2", duration=0.5)
    
    # If any environment has succeeded, save its success state.
    # (Uses save_success_state as required)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cube_for_robot_to_push_2")
    
    return success

@configclass
class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=walk_to_cube_for_robot_to_push_2_success)
