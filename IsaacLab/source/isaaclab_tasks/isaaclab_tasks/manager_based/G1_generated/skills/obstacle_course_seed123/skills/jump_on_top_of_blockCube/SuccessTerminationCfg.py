
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_on_top_of_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_on_top_of_blockCube skill has been successfully completed.'''

    # Access the robot object
    robot = env.scene["robot"]

    # Access the block cube object (Object5 as per configuration)
    block_cube = env.scene['Object5']

    # Get indices for the required robot body parts
    # REASONING: Using robot.body_names.index for approved access pattern.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Get positions of robot body parts (world frame)
    # REASONING: Using robot.data.body_pos_w for approved access pattern.
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get position of the block cube (world frame)
    # REASONING: Using env.scene['ObjectName'].data.root_pos_w for approved access pattern.
    block_cube_pos = block_cube.data.root_pos_w

    # Hardcode block dimensions from the object configuration (0.5m cubed)
    # REASONING: Object dimensions cannot be accessed from RigidObject, must be hardcoded from config as per rules.
    block_height = 0.5
    block_half_size = block_height / 2 # 0.25m

    # Calculate the z-coordinate of the block's top surface
    # REASONING: Relative calculation of block top surface based on its root position and half size.
    block_top_z = block_cube_pos[:, 2] + block_half_size

    # Calculate the average foot position for combined feet measurements
    # REASONING: Averaging foot positions to represent the robot's base on the block.
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # 1. Check horizontal alignment of average foot position with block center
    # REASONING: Using relative distances (abs difference) between average foot and block center.
    # Thresholds (0.15m) are from the success criteria plan, providing leniency.
    feet_horizontal_x_dist = torch.abs(avg_foot_pos_x - block_cube_pos[:, 0])
    feet_horizontal_y_dist = torch.abs(avg_foot_pos_y - block_cube_pos[:, 1])

    feet_horizontally_aligned_x = feet_horizontal_x_dist < (block_half_size + 0.15)
    feet_horizontally_aligned_y = feet_horizontal_y_dist < (block_half_size + 0.15)
    feet_horizontally_aligned = feet_horizontally_aligned_x & feet_horizontally_aligned_y

    # 2. Check vertical alignment of average foot position with block top surface
    # REASONING: Using relative distance (abs difference) between average foot z and block top z.
    # Threshold (0.15m) is from the success criteria plan, allowing for slight vertical variation.
    feet_vertical_dist = torch.abs(avg_foot_pos_z - block_top_z)
    feet_vertically_on_block = feet_vertical_dist < 0.15

    # 3. Check pelvis height for stable standing posture above the block
    # REASONING: Using relative distance (abs difference) between pelvis z and target height above block top z.
    # Target height (0.7m) is from the reward function, indicating a stable standing posture.
    # Threshold (0.2m) is from the success criteria plan, providing leniency for pelvis height.
    target_pelvis_z_on_block = block_top_z + 0.7
    pelvis_height_dist = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)
    pelvis_at_stable_height = pelvis_height_dist < 0.2

    # Combine all conditions for overall success
    # REASONING: All conditions must be met for the robot to be considered stably on the block.
    condition = feet_horizontally_aligned & feet_vertically_on_block & pelvis_at_stable_height

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # REASONING: Using check_success_duration and save_success_state as required.
    # Duration (0.5s) is from the success criteria plan to ensure stability.
    success = check_success_duration(env, condition, "jump_on_top_of_blockCube", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_on_top_of_blockCube")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_on_top_of_blockCube_success)
