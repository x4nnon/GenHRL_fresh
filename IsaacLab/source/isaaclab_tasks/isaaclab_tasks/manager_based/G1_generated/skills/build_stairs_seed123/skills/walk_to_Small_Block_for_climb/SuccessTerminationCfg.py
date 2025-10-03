
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block_for_climb skill has been successfully completed.'''
    # Access the robot object.
    robot = env.scene["robot"]
    
    # Access the Small Block (Object1) as per the object configuration.
    object1 = env.scene['Object1']
    
    # Get the world-frame positions of the required robot parts.
    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name').
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    # Get the world-frame position of Object1.
    # CRITICAL RULE: Access object positions using env.scene['ObjectName'].data.root_pos_w.
    object1_pos = object1.data.root_pos_w
    
    # Define target values based on the SUCCESS CRITERIA PLAN.
    # CRITICAL RULE: Hardcode object dimensions/offsets derived from object configuration, not from object attributes.
    # Object1 (Small Block) has x=1m, y=1m. Target x-offset is 0.75m (0.5m half-width + 0.25m buffer).
    target_x_offset_from_object1 = 0.75 
    target_pelvis_z_height = 0.7 # Target stable standing height for pelvis.
    target_foot_z_ground_threshold = 0.1 # Max Z-position for feet to be considered on the ground.
    
    # Calculate the target X and Y positions for the robot's pelvis relative to Object1.
    # CRITICAL RULE: All success criteria must only use relative distances.
    target_pelvis_x = object1_pos[:, 0] - target_x_offset_from_object1
    target_pelvis_y = object1_pos[:, 1]
    
    # Condition 1: Robot's pelvis is horizontally positioned in front of Object1 (X-axis).
    # CRITICAL RULE: Consider X, Y, Z components separately with their own thresholds.
    x_position_diff = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    x_position_condition = x_position_diff < 0.15 # Threshold from plan.
    
    # Condition 2: Robot's pelvis is horizontally aligned with Object1 (Y-axis).
    y_position_diff = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    y_position_condition = y_position_diff < 0.15 # Threshold from plan.
    
    # Condition 3: Robot's pelvis is at a stable standing height (Z-axis).
    # CRITICAL RULE: Z-position is the only absolute position allowed, used sparingly for height.
    pelvis_z_height_diff = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_height)
    pelvis_z_height_condition = pelvis_z_height_diff < 0.1 # Threshold from plan.
    
    # Condition 4: Robot's feet are on or very near the ground.
    # CRITICAL RULE: Z-position is the only absolute position allowed, used sparingly for height.
    left_foot_on_ground = left_foot_pos[:, 2] < target_foot_z_ground_threshold
    right_foot_on_ground = right_foot_pos[:, 2] < target_foot_z_ground_threshold
    feet_on_ground_condition = left_foot_on_ground & right_foot_on_ground
    
    # Combine all conditions for overall success.
    # CRITICAL RULE: All operations must work with batched environments.
    overall_success_condition = (
        x_position_condition &
        y_position_condition &
        pelvis_z_height_condition &
        feet_on_ground_condition
    )
    
    # Check success duration and save success states.
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, overall_success_condition, "walk_to_Small_Block_for_climb", duration=0.5)
    
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block_for_climb")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_for_climb_success)
