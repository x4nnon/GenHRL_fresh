
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_on_top_of_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_on_top_of_blockCube skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    robot = env.scene["robot"]

    # Access robot part indices using robot.body_names.index() as required
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Access robot part positions using robot.data.body_pos_w as required
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # Shape: [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]         # Shape: [num_envs, 3]

    # 2. Get object position
    # Access Object5 (block cube) directly as per requirements
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Hardcode object dimensions from the object configuration (0.5m cubed block)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    block_half_size = 0.25 # 0.5m / 2

    # Calculate the Z-position of the block's top surface (relative to its root)
    # This is a relative distance calculation.
    block_top_z = block_cube_pos[:, 2] + block_half_size

    # Calculate the X and Y center of the block
    # These are absolute positions, but used as reference points for relative distance calculations.
    block_center_x = block_cube_pos[:, 0]
    block_center_y = block_cube_pos[:, 1]

    # 3. Calculate relative distances for success criteria

    # Average horizontal position of the robot's feet
    avg_foot_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2

    # X-axis distance between average foot position and block center (relative distance)
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    dist_feet_x_to_block_center = torch.abs(avg_foot_x - block_center_x)

    # Y-axis distance between average foot position and block center (relative distance)
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    dist_feet_y_to_block_center = torch.abs(avg_foot_y - block_center_y)

    # Z-axis distance between left foot and block's top surface (relative distance)
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    dist_left_foot_z_to_block_top = torch.abs(left_foot_pos[:, 2] - block_top_z)

    # Z-axis distance between right foot and block's top surface (relative distance)
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    dist_right_foot_z_to_block_top = torch.abs(right_foot_pos[:, 2] - block_top_z)

    # Target pelvis height relative to the block's top surface (0.7m as per plan)
    # This is a relative target Z position.
    target_pelvis_z = block_top_z + 0.7 # Relative target Z position

    # Z-axis distance between pelvis and target standing height (relative distance)
    # This adheres to the rule: "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    dist_pelvis_z_to_target = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # 4. Check success conditions based on thresholds

    # Condition 1: Average horizontal foot position within block bounds (with tolerance)
    # Threshold: 0.3m (block half-size 0.25m + 0.05m tolerance)
    # This uses a reasonable tolerance as per "REASONABLE TOLERANCES" rule.
    cond_feet_x_on_block = dist_feet_x_to_block_center < 0.3
    cond_feet_y_on_block = dist_feet_y_to_block_center < 0.3

    # Condition 2: Both feet are at the height of the block's top surface (with tolerance)
    # Threshold: 0.1m
    # This uses a reasonable tolerance as per "REASONABLE TOLERANCES" rule.
    cond_left_foot_z_on_top = dist_left_foot_z_to_block_top < 0.1
    cond_right_foot_z_on_top = dist_right_foot_z_to_block_top < 0.1

    # Condition 3: Pelvis is at a stable standing height relative to the block's top surface (with tolerance)
    # Threshold: 0.15m
    # This uses a reasonable tolerance as per "REASONABLE TOLERANCES" rule.
    cond_pelvis_stable_height = dist_pelvis_z_to_target < 0.15

    # Combine all conditions for overall success
    # All conditions must be met simultaneously for success. All operations work with batched environments.
    overall_condition = (cond_feet_x_on_block &
                         cond_feet_y_on_block &
                         cond_left_foot_z_on_top &
                         cond_right_foot_z_on_top &
                         cond_pelvis_stable_height)

    # 5. Check duration and save success states
    # Using check_success_duration and save_success_state as required, with duration 0.5s.
    success = check_success_duration(env, overall_condition, "jump_on_top_of_blockCube", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_on_top_of_blockCube")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_on_top_of_blockCube_success)
