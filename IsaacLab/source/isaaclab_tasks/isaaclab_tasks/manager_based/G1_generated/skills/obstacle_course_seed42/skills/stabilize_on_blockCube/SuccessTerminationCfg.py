
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def stabilize_on_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the stabilize_on_blockCube skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects.
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w  # [num_envs, 3]

    # Access the required robot parts.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]

    # Block dimensions (0.5m cubed, so half-size is 0.25)
    block_half_size_xy = 0.25+0.1
    block_half_size_z = 0.25

    # Calculate the Z-position of the top surface of the block.
    block_top_z = block_cube_pos[:, 2] + block_half_size_z -0.05

    # 1. Both feet are above the top surface of the block (Z-position).
    feet_above_block = (left_foot_pos[:, 2] > block_top_z) & (right_foot_pos[:, 2] > block_top_z)

    # 2. Both feet are within the horizontal bounds of the block (XY-position).
    feet_within_block_xy = (
        (torch.abs(left_foot_pos[:, 0] - block_cube_pos[:, 0]) < block_half_size_xy) &
        (torch.abs(left_foot_pos[:, 1] - block_cube_pos[:, 1]) < block_half_size_xy) &
        (torch.abs(right_foot_pos[:, 0] - block_cube_pos[:, 0]) < block_half_size_xy) &
        (torch.abs(right_foot_pos[:, 1] - block_cube_pos[:, 1]) < block_half_size_xy)
    )

    # Combine both conditions
    condition = feet_above_block #& feet_within_block_xy

    # Check duration and save success states.
    success = check_success_duration(env, condition, "stabilize_on_blockCube", duration=1.0)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "stabilize_on_blockCube")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=stabilize_on_blockCube_success)
