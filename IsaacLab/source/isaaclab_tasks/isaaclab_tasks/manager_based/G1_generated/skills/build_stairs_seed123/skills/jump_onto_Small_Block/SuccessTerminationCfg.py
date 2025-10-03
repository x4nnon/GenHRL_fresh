
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Small_Block_success(env) -> torch.Tensor:
    '''Determine if the jump_onto_Small_Block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object (Approved access pattern)
    robot = env.scene["robot"]

    # Get indices for the required robot parts using approved access patterns
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # Get positions of the robot parts in world frame (Approved access pattern)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Access Object1 (Small Block) using approved access pattern
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w

    # Hardcode Object1 dimensions as per requirements (from skill info/object config)
    # Rule: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    object1_height = 0.3
    object1_half_x = 0.5
    object1_half_y = 0.5

    # Calculate Object1's center XY and top surface Z-coordinate (Relative calculations)
    object1_center_x = object1_pos[:, 0]
    object1_center_y = object1_pos[:, 1]
    object1_top_z = object1_pos[:, 2] + object1_height

    # Define target pelvis height above Object1's top surface (from reward function context/success criteria plan)
    # This is a relative height, not an arbitrary hardcoded world position.
    target_pelvis_height_above_block = 0.7

    # Calculate relative distances for success criteria (Rule: USE RELATIVE DISTANCES)
    # Pelvis horizontal distance to Object1 center
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - object1_center_x)
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - object1_center_y)

    # Feet vertical distance to Object1's top surface
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - object1_top_z)
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - object1_top_z)

    # Feet horizontal distance to Object1 center
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - object1_center_x)
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - object1_center_y)
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - object1_center_x)
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - object1_center_y)

    # Pelvis vertical distance to target standing height above Object1's top surface
    pelvis_height_error = torch.abs(pelvis_pos[:, 2] - (object1_top_z + target_pelvis_height_above_block))

    # Define success conditions based on relative distances and thresholds (Rule: USE LENIENT THRESHOLDS, REASONABLE TOLERANCES)
    # Pelvis horizontally centered over Object1 (with tolerance)
    pelvis_horizontal_condition = (pelvis_dist_x < (object1_half_x + 0.1)) & \
                                  (pelvis_dist_y < (object1_half_y + 0.1))

    # Feet vertically close to Object1's top surface (with tolerance)
    feet_vertical_condition = (left_foot_dist_z < 0.05) & \
                              (right_foot_dist_z < 0.05)

    # Feet horizontally within Object1's bounds (with tolerance)
    feet_horizontal_condition = (left_foot_dist_x < (object1_half_x + 0.1)) & \
                                (left_foot_dist_y < (object1_half_y + 0.1)) & \
                                (right_foot_dist_x < (object1_half_x + 0.1)) & \
                                (right_foot_dist_y < (object1_half_y + 0.1))

    # Pelvis at stable standing height relative to Object1's top surface (with tolerance)
    pelvis_height_condition = (pelvis_height_error < 0.1)

    # Combine all conditions for overall success (Rule: Handle tensor operations correctly for batched environments)
    # All conditions must be met simultaneously for the robot to be considered stably standing on the block.
    condition = pelvis_horizontal_condition & \
                feet_vertical_condition & \
                feet_horizontal_condition & \
                pelvis_height_condition

    # Check success duration and save success states (Rule: ALWAYS use check_success_duration and save_success_state)
    # The duration of 0.5 seconds ensures stability on the block as per the plan.
    success = check_success_duration(env, condition, "jump_onto_Small_Block", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Small_Block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Small_Block_success)
