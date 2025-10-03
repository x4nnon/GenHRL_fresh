
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Large_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_onto_Large_Block_for_robot_interaction skill has been successfully completed.'''
    # CRITICAL REQUIREMENT 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # CRITICAL REQUIREMENT 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # CRITICAL REQUIREMENT 5: Access objects directly - objects should always exist in the scene
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL REQUIREMENT 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    large_block = env.scene['Object3'] # Object3 is the Large Block for robot interaction
    large_block_pos = large_block.data.root_pos_w

    # CRITICAL REQUIREMENT 6: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    # Large Block dimensions from object configuration: x=1m, y=1m, z=0.9m
    block_height = 0.9
    block_half_x = 0.5
    block_half_y = 0.5

    # Calculate the Z coordinate of the top surface of the large block (relative distance)
    block_top_z = large_block_pos[:, 2] + block_height / 2.0

    # Define tolerances for success criteria
    foot_horizontal_tolerance = 0.05 # 5cm tolerance for feet to be within block bounds
    foot_vertical_tolerance = 0.08   # 8cm tolerance for feet to be on block surface
    pelvis_min_height_above_block = 0.6 # Pelvis should be at least 0.6m above block top
    pelvis_max_height_above_block = 0.8 # Pelvis should be at most 0.8m above block top

    # --- Feet Horizontal Position Check (relative to block center) ---
    # CRITICAL REQUIREMENT 1: SUCCESS CRITERIA MUST ONLY use relative distances
    # CRITICAL REQUIREMENT 4: NEVER use hard-coded positions or arbitrary thresholds
    # CRITICAL REQUIREMENT 5: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - large_block_pos[:, 0])
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - large_block_pos[:, 1])
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - large_block_pos[:, 0])
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - large_block_pos[:, 1])

    left_foot_horizontal_ok = (left_foot_dist_x < (block_half_x + foot_horizontal_tolerance)) & \
                              (left_foot_dist_y < (block_half_y + foot_horizontal_tolerance))
    right_foot_horizontal_ok = (right_foot_dist_x < (block_half_x + foot_horizontal_tolerance)) & \
                               (right_foot_dist_y < (block_half_y + foot_horizontal_tolerance))

    # --- Feet Vertical Position Check (relative to block top surface) ---
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - block_top_z)
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - block_top_z)

    left_foot_vertical_ok = left_foot_dist_z < foot_vertical_tolerance
    right_foot_vertical_ok = right_foot_dist_z < foot_vertical_tolerance

    # --- Pelvis Vertical Position Check (relative to block top surface) ---
    pelvis_height_above_block = pelvis_pos[:, 2] - block_top_z
    pelvis_vertical_ok = (pelvis_height_above_block > pelvis_min_height_above_block) & \
                         (pelvis_height_above_block < pelvis_max_height_above_block)

    # Combine all conditions for overall success
    # All feet must be horizontally within bounds AND vertically on the surface
    feet_on_block_condition = left_foot_horizontal_ok & right_foot_horizontal_ok & \
                              left_foot_vertical_ok & right_foot_vertical_ok

    # The robot's pelvis must be at a stable standing height above the block
    overall_condition = feet_on_block_condition & pelvis_vertical_ok

    # CRITICAL REQUIREMENT 6: ALWAYS use check_success_duration and save_success_state
    # Duration required: 0.5 seconds
    success = check_success_duration(env, overall_condition, "jump_onto_Large_Block_for_robot_interaction", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Large_Block_for_robot_interaction")

    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Large_Block_for_robot_interaction_success)
