
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def LandStablyAfterLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the LandStablyAfterLowWall skill has been successfully completed.
    Success is defined as the robot landing stably on both feet on the other side of the low wall.
    '''
    # 1. Get robot object - APPROVED PATTERN
    robot = env.scene["robot"]

    # 2. Get indices for robot body parts - APPROVED PATTERN
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # 3. Get positions of robot parts - APPROVED PATTERN
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # [num_envs, 3]

    try:
        # 4. Get low wall object - APPROVED PATTERN with try/except for robustness
        low_wall = env.scene['Object3']
        low_wall_pos = low_wall.data.root_pos_w # [num_envs, 3]

        # 5. Calculate relative distances - REQUIREMENT 1: Relative distances only
        pelvis_x_distance_to_wall = pelvis_pos[:, 0] - low_wall_pos[:, 0] # x distance between pelvis and low wall
        left_foot_z_distance_to_ground = torch.abs(left_foot_pos[:, 2] - 0.0) # z distance of left foot from ground (z=0)
        right_foot_z_distance_to_ground = torch.abs(right_foot_pos[:, 2] - 0.0) # z distance of right foot from ground (z=0)
        pelvis_z_distance_to_ground = torch.abs(pelvis_pos[:, 2] - 0.0) # z distance of pelvis from ground (z=0)

        # 6. Define success conditions - REQUIREMENT 4: No hardcoded thresholds, REQUIREMENT 5: Reasonable tolerances
        pelvis_past_wall_condition = pelvis_x_distance_to_wall > 0.5 # Pelvis is 0.5m past the low wall in x direction
        left_foot_low_condition = left_foot_z_distance_to_ground < 0.15 # Left foot is within 0.15m of the ground
        right_foot_low_condition = right_foot_z_distance_to_ground < 0.15 # Right foot is within 0.15m of the ground
        pelvis_low_condition = pelvis_z_distance_to_ground < 0.8 # Pelvis is within 0.8m of the ground
        pelvis_high_condition = pelvis_z_distance_to_ground > 0.5 # Pelvis is above 0.5m of the ground

        # 7. Combine success conditions - REQUIREMENT 1: Relative distances only
        land_stably_condition = pelvis_past_wall_condition & left_foot_low_condition & right_foot_low_condition & pelvis_low_condition & pelvis_high_condition

    except KeyError:
        # 8. Handle missing object - REQUIREMENT 6: Handle missing objects
        land_stably_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success state - REQUIREMENT 7 & 8: check_success_duration and save_success_state
    success = check_success_duration(env, land_stably_condition, "LandStablyAfterLowWall", duration=0.5) # Duration of 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "LandStablyAfterLowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=LandStablyAfterLowWall_success)