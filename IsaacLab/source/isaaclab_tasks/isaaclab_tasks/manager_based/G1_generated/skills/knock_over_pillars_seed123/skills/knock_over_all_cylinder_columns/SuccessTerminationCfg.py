
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_all_cylinder_columns_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_all_cylinder_columns skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required object for this skill instance.
    # As per the skill description, "It should treat each object as a separate skill."
    # and the provided reward functions focus on 'Object1'.
    # Therefore, this success function will target 'Object1'.
    # CORRECT: Accessing object position using approved pattern.
    target_column = env.scene['Object1'] 

    # Access the required robot part(s).
    # CORRECT: Direct indexed access to the robot.
    robot = env.scene["robot"]
    # CORRECT: Getting the index of the required robot part using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis') 
    # CORRECT: Getting the position of the required robot part using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] 

    # Object dimensions (hardcoded from skill info, as per requirements).
    # The cylinder column has a radius of 0.3m.
    # CORRECT: Hardcoding object dimension from object configuration, as per rule 6.
    column_radius = 0.3

    # Calculate the column's current Z-position (center).
    # This is used to determine if the column is lying on the floor.
    # Z-position is allowed as an absolute value for height checks.
    # CORRECT: Accessing object Z-position using approved pattern.
    column_z_pos = target_column.data.root_pos_w[:, 2]

    # Calculate horizontal distance between robot pelvis and column.
    # This uses relative distances between robot part and object, as required.
    # CORRECT: Calculating relative distances between robot part and object.
    dist_x_pelvis_column = target_column.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_y_pelvis_column = target_column.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    horizontal_dist_pelvis_column = torch.sqrt(dist_x_pelvis_column**2 + dist_y_pelvis_column**2)

    # Success conditions:
    # 1. Column is on the floor (Z-position close to its radius).
    # A small margin (0.1m) is added to the radius to account for the column's thickness and slight variations when lying flat.
    # This is a relative check against the object's own dimension.
    # CORRECT: Using a reasonable tolerance and relative check against object dimension.
    column_is_down = column_z_pos <= (column_radius + 0.1)

    # 2. Robot pelvis is horizontally close to the knocked-over column.
    # This ensures the robot is near the object it just knocked over.
    # The threshold (0.8m) is a reasonable tolerance for proximity.
    # CORRECT: Using a reasonable tolerance for relative distance.
    robot_is_close_horizontally = horizontal_dist_pelvis_column <= 0.8

    # 3. Robot is not fallen (pelvis height is reasonable).
    # This ensures the robot maintains stability after knocking over the column.
    # Pelvis Z-position is checked against a reasonable height (0.5m), which is an allowed absolute check for stability.
    # CORRECT: Using an allowed absolute check for stability (pelvis height).
    robot_is_stable = pelvis_pos[:, 2] >= 0.5

    # Combine all conditions for success. All conditions must be met.
    # CORRECT: Combining conditions with proper tensor operations.
    condition = column_is_down & robot_is_close_horizontally & robot_is_stable

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # The duration is set to 0.5 seconds to ensure the conditions are met for a brief period.
    # CORRECT: Using check_success_duration and save_success_state as required.
    success = check_success_duration(env, condition, "knock_over_all_cylinder_columns", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_all_cylinder_columns")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_all_cylinder_columns_success)
