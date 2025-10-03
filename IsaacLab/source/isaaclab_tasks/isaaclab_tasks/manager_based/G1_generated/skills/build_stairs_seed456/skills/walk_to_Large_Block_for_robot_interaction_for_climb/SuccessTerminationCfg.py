
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Large_Block_for_robot_interaction_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Large_Block_for_robot_interaction_for_climb skill has been successfully completed.'''

    # Access robot and object using approved patterns.
    robot = env.scene["robot"]
    object_large_block = env.scene['Object3'] # Object3 is "Large Block for robot interaction"

    # Get robot pelvis position using approved pattern.
    # REQUIREMENT: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Get robot foot positions using approved patterns.
    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    robot_left_foot_pos_z = robot.data.body_pos_w[:, robot_left_foot_idx][:, 2]
    robot_right_foot_pos_z = robot.data.body_pos_w[:, robot_right_foot_idx][:, 2]

    # Get Large Block position using approved pattern.
    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object_large_block_pos = object_large_block.data.root_pos_w

    # Hardcode Large Block dimensions from the object configuration.
    # Object3: "Large Block for robot interaction" -> x=1m y=1m z=0.9m
    # REQUIREMENT: There is no way to access the SIZE of an object. Hardcode from config.
    large_block_x_dim = 1.0

    # Define target clearance and stable standing height. These are specific values defining the target state.
    # REQUIREMENT: NEVER use hard-coded positions or arbitrary thresholds for object locations, but specific target values are allowed.
    target_clearance_x = 0.5  # Desired distance from the front face of the block
    target_pelvis_z = 0.7     # Stable standing height for the robot
    ground_level_z = 0.0      # Assuming ground is at z=0

    # --- Success Condition 1: Pelvis X-position relative to Large Block's front face ---
    # Target X: block_center_x - (block_x_dim / 2) - target_clearance_x
    # This ensures the robot is positioned in front of the block's face.
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    target_x_pos = object_large_block_pos[:, 0] - (large_block_x_dim / 2.0) - target_clearance_x
    pelvis_x_condition = torch.abs(robot_pelvis_pos_x - target_x_pos) < 0.2 # Tolerance 0.2m

    # --- Success Condition 2: Pelvis Y-position aligned with Large Block's Y-center ---
    # Target Y: Center Y of the block.
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    target_y_pos = object_large_block_pos[:, 1]
    pelvis_y_condition = torch.abs(robot_pelvis_pos_y - target_y_pos) < 0.2 # Tolerance 0.2m

    # --- Success Condition 3: Pelvis Z-position at stable standing height ---
    # REQUIREMENT: Z-axis distance between robot pelvis and the ground.
    # Z is the only absolute position allowed, used sparingly for height.
    pelvis_z_condition = torch.abs(robot_pelvis_pos_z - target_pelvis_z) < 0.1 # Tolerance 0.1m

    # --- Success Condition 4: Both feet on the ground ---
    # REQUIREMENT: Z-axis distance between robot's feet and the ground.
    # Feet should be very close to the ground (z=0).
    feet_on_ground_threshold = 0.05 # Tolerance 0.05m
    left_foot_on_ground_condition = robot_left_foot_pos_z < feet_on_ground_threshold
    right_foot_on_ground_condition = robot_right_foot_pos_z < feet_on_ground_threshold
    feet_on_ground_combined_condition = left_foot_on_ground_condition & right_foot_on_ground_condition

    # Combine all conditions for overall success.
    # REQUIREMENT: All operations must work with batched environments.
    overall_success_condition = (
        pelvis_x_condition &
        pelvis_y_condition &
        pelvis_z_condition &
        feet_on_ground_combined_condition
    )

    # Check success duration and save success states.
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, overall_success_condition, "walk_to_Large_Block_for_robot_interaction_for_climb", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Large_Block_for_robot_interaction_for_climb")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Large_Block_for_robot_interaction_for_climb_success)
