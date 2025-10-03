
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Small_Block_for_robot_interaction_to_base_position_success(env) -> torch.Tensor:
    '''Determine if the push_Small_Block_for_robot_interaction_to_base_position skill has been successfully completed.'''

    # Access the required object. Object1 is mapped to "Small Block for robot interaction" as per configuration.
    # This adheres to the requirement of accessing objects directly using env.scene['ObjectName'].
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w # Get object's world position.

    # Access the required robot part. The pelvis is used as the reference point for the robot.
    # This adheres to the requirement of accessing robot parts using robot.body_names.index.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Get robot pelvis's world position.

    # Hardcode Object1's half-height based on the provided object configuration (z=0.3m, so half height is 0.15m).
    # This adheres to the rule that object dimensions must be hardcoded from the config, not accessed dynamically.
    object1_half_height = 0.15

    # Calculate relative distances between Object1 and the robot's pelvis.
    # This adheres to the requirement of using only relative distances between objects and robot parts.
    # X-distance: Absolute difference between Object1's X and robot pelvis's X.
    distance_x_obj1_pelvis = torch.abs(object1_pos[:, 0] - robot_pelvis_pos[:, 0])
    # Y-distance: Signed difference between Object1's Y and robot pelvis's Y.
    # This is kept signed to check if Object1 is *in front* of the robot.
    distance_y_obj1_pelvis = object1_pos[:, 1] - robot_pelvis_pos[:, 1]
    # Z-distance: Absolute difference between Object1's Z and its target height (half its height from the ground).
    # The Z-component is allowed to be absolute relative to the ground for height checks.
    distance_z_obj1_ground = torch.abs(object1_pos[:, 2] - object1_half_height)

    # Define success conditions based on relative distances and hardcoded object dimensions.
    # These thresholds are reasonable tolerances for positioning.
    # 1. Object1 is aligned on the X-axis with the robot's pelvis (within 0.2m).
    success_x_alignment = distance_x_obj1_pelvis < 0.2
    # 2. Object1 is approximately 1.5m in front of the robot's pelvis on the Y-axis (between 1.4m and 1.6m).
    success_y_position = (distance_y_obj1_pelvis > 1.4) & (distance_y_obj1_pelvis < 1.6)
    # 3. Object1 is at its correct height, stable on the ground (within 0.05m of its half-height).
    success_z_height = distance_z_obj1_ground < 0.05

    # Combine all conditions for overall success. All conditions must be met.
    condition = success_x_alignment & success_y_position & success_z_height

    # Check success duration and save success states.
    # This adheres to the absolute requirements for success function implementation.
    success = check_success_duration(env, condition, "push_Small_Block_for_robot_interaction_to_base_position", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Small_Block_for_robot_interaction_to_base_position")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_Small_Block_for_robot_interaction_to_base_position_success)
