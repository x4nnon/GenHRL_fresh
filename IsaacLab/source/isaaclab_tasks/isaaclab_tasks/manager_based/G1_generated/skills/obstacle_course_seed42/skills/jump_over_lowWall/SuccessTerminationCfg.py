
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def ExecuteJumpOverLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ExecuteJumpOverLowWall skill has been successfully completed.'''
    # 1. Access robot object (APPROVED PATTERN)
    robot = env.scene["robot"]

    # 2. Get pelvis index (APPROVED PATTERN)
    pelvis_idx = robot.body_names.index('pelvis')
    # 3. Get pelvis position (APPROVED PATTERN)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0] # Get x component of pelvis position

    try:
        # 4. Access low wall object (APPROVED PATTERN and try-except for error handling)
        low_wall = env.scene['Object3']
        # 5. Get low wall position (APPROVED PATTERN)
        low_wall_x = low_wall.data.root_pos_w[:, 0] # Get x component of low wall position

        # 6. Calculate relative distance in x direction (RELATIVE DISTANCE)
        distance_x_wall_pelvis = pelvis_pos_x - low_wall_x

        # 7. Define success condition based on relative distance (RELATIVE DISTANCE and REASONABLE THRESHOLD)
        success_condition = distance_x_wall_pelvis > 0.5 # Check if pelvis is 0.5m past the wall in x direction

    except KeyError:
        # 8. Handle missing object (ERROR HANDLING)
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success state (REQUIRED FUNCTIONS)
    success = check_success_duration(env, success_condition, "ExecuteJumpOverLowWall", duration=0.5) # Check if success is maintained for 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ExecuteJumpOverLowWall") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ExecuteJumpOverLowWall_success)
