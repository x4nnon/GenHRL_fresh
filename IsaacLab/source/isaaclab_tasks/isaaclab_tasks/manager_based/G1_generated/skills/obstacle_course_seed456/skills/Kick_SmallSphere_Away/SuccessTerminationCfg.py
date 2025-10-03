
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Kick_SmallSphere_Away_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Kick_SmallSphere_Away skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL REQUIREMENT: Access robot and objects using approved patterns.
    robot = env.scene["robot"]
    small_sphere = env.scene['Object2'] # Object2 is the small sphere for robot to kick
    block_cube = env.scene['Object5']   # Object5 is the block cube for robot to jump on top of

    # CRITICAL REQUIREMENT: Access robot parts using robot.body_names.index().
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # CRITICAL REQUIREMENT: Hardcode object dimensions/initial positions from object configuration/task description.
    # The initial x-position of the small sphere (Object2) is 12m as per the task description.
    # This is an environment setup parameter, not a dynamic object attribute.
    small_sphere_initial_x_pos = 12.0

    # CRITICAL REQUIREMENT: All success criteria must only use relative distances.
    # 1. Calculate the distance Object2 (small sphere) has moved in the x-axis.
    # This is a relative distance from its initial setup position.
    sphere_moved_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_initial_x_pos

    # 2. Calculate relative distances for robot pelvis positioning near Object5 (block cube).
    # These are relative distances between the robot's pelvis and the block's center.
    dist_pelvis_block_x = torch.abs(block_cube.data.root_pos_w[:, 0] - pelvis_pos_x)
    dist_pelvis_block_y = torch.abs(block_cube.data.root_pos_w[:, 1] - pelvis_pos_y)

    # Define success conditions with reasonable tolerances.
    # Condition 1: Small sphere moved away in x by at least 1.0 meter.
    sphere_moved_condition = sphere_moved_x > 1.0

    # Condition 2: Robot pelvis positioned near the block cube (Object5).
    # Pelvis should be within 0.5m of the block's x-center.
    pelvis_x_position_condition = dist_pelvis_block_x <= 0.5
    # Pelvis should be within 0.5m of the block's y-center.
    pelvis_y_position_condition = dist_pelvis_block_y <= 0.5

    # Combine all conditions for overall success. All conditions must be met.
    success_condition = sphere_moved_condition & pelvis_x_position_condition & pelvis_y_position_condition

    # CRITICAL REQUIREMENT: Use check_success_duration and save_success_state.
    # Duration set to 0.5 seconds to ensure the state is stable.
    success = check_success_duration(env, success_condition, "Kick_SmallSphere_Away", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Kick_SmallSphere_Away")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Kick_SmallSphere_Away_success)
