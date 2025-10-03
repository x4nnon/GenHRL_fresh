
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Medium_Block_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Medium_Block_for_climb skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object using the approved pattern.
    robot = env.scene["robot"]

    # Access the Medium Block (Object2) as per the object configuration, using the approved pattern.
    object2 = env.scene['Object2']

    # Get the pelvis index using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the pelvis position in world coordinates using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode Object2 dimensions from the object configuration (x=1m y=1m z=0.6m) as per rule 6.
    object2_size_x = 1.0
    object2_size_y = 1.0

    # Calculate the X-coordinate of Object2's front face.
    # This is relative to Object2's root position and its X-dimension, adhering to relative distance rule.
    object2_front_face_x = object2.data.root_pos_w[:, 0] + (object2_size_x / 2.0)

    # --- X-axis condition: Pelvis X-position slightly in front of Object2's front face. ---
    # The robot's pelvis X should be between 0.1m and 0.3m in front of the block's front face.
    # This means pelvis_x should be greater than (front_face_x + 0.1) and less than (front_face_x + 0.3).
    # All calculations are relative to object2's position, adhering to relative distance rule.
    # Using tensor operations for batched environments.
    x_condition_min = pelvis_pos[:, 0] > (object2_front_face_x + 0.1)
    x_condition_max = pelvis_pos[:, 0] < (object2_front_face_x + 0.3)
    x_axis_success = x_condition_min & x_condition_max

    # --- Y-axis condition: Pelvis Y-position aligned with Object2's center. ---
    # Calculate the absolute Y-distance between the pelvis and Object2's center.
    # Object2's center Y is object2.data.root_pos_w[:, 1].
    # A tolerance of +/- 0.2m is applied. Using tensor operations for batched environments.
    y_axis_distance = torch.abs(pelvis_pos[:, 1] - object2.data.root_pos_w[:, 1])
    y_axis_success = y_axis_distance < 0.2

    # --- Z-axis condition: Pelvis Z-position at a stable standing height. ---
    # The reward function uses 0.7m as the target height. A tolerance of +/- 0.1m is reasonable.
    # This is the only allowed "absolute" check, as it refers to height from the ground (z=0),
    # which is consistent with the reward function and the prompt's allowance for stable standing height.
    # Using tensor operations for batched environments.
    z_axis_distance = torch.abs(pelvis_pos[:, 2] - 0.7)
    z_axis_success = z_axis_distance < 0.1

    # Combine all conditions for overall success. All conditions must be met.
    # Using tensor operations for batched environments.
    overall_success_condition = x_axis_success & y_axis_success & z_axis_success

    # Check success duration and save success states using the approved patterns.
    # The duration is set to 0.5 seconds as per the plan.
    success = check_success_duration(env, overall_success_condition, "walk_to_Medium_Block_for_climb", duration=0.5)

    # Save success states for environments that have met the success criteria for the required duration.
    # This loop correctly handles batched environments.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Medium_Block_for_climb")

    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition using DoneTerm.
    success = DoneTerm(func=walk_to_Medium_Block_for_climb_success)
