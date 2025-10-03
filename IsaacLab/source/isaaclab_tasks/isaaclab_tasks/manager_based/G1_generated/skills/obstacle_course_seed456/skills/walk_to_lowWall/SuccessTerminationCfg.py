
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_lowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_lowWall skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    robot = env.scene["robot"]
    # Accessing robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # 2. Get object position
    # Accessing Object3 (low wall) position using approved pattern
    low_wall = env.scene['Object3'] # Object3 is the low wall for robot to jump over
    low_wall_pos = low_wall.data.root_pos_w # Shape: [num_envs, 3]

    # Hardcoded low wall dimensions from the task description (x-axis 0.3m, y-axis 5.0m, z-axis 0.5m)
    # These values are obtained from the object configuration and task description, not from the object itself.
    low_wall_x_dim = 0.3
    low_wall_y_dim = 5.0
    low_wall_z_dim = 0.5

    # 3. Calculate relative distances for success criteria

    # X-axis condition: Pelvis is positioned in front of the wall's leading edge.
    # The wall's front face is at low_wall_pos_x + low_wall_x_dim / 2.
    # We want the pelvis to be between 0.3m and 0.7m in front of this face.
    # This means (low_wall_pos_x + low_wall_x_dim / 2) - pelvis_pos_x should be in [0.3, 0.7].
    # Using relative distances between objects and robot parts.
    relative_x_dist_to_wall_front = (low_wall_pos[:, 0] + low_wall_x_dim / 2) - pelvis_pos[:, 0]
    x_condition = (relative_x_dist_to_wall_front >= 0.3) & (relative_x_dist_to_wall_front <= 0.7)

    # Y-axis condition: Pelvis is aligned with the wall's y-extent.
    # The wall extends low_wall_y_dim / 2 in either direction from its center along y-axis.
    # We allow a tolerance of 0.2m beyond the wall's y-extent.
    # Using relative distances between objects and robot parts.
    y_distance_to_wall_center = torch.abs(pelvis_pos[:, 1] - low_wall_pos[:, 1])
    y_condition = y_distance_to_wall_center < (low_wall_y_dim / 2 + 0.2)

    # Z-axis condition: Pelvis is above the wall's top surface.
    # The wall's top surface is at low_wall_pos_z + low_wall_z_dim / 2.
    # We ensure the pelvis is at least 0.1m above this surface to prevent penetration.
    # Using relative distances between objects and robot parts.
    relative_z_dist_to_wall_top = pelvis_pos[:, 2] - (low_wall_pos[:, 2] + low_wall_z_dim / 2)
    z_condition = relative_z_dist_to_wall_top > 0.1

    # Combine all conditions for overall success
    # All conditions must be met simultaneously.
    condition = x_condition & y_condition & z_condition

    # 4. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Using check_success_duration to ensure the condition holds for a specified duration.
    # Saving success states for environments that meet the criteria.
    success = check_success_duration(env, condition, "walk_to_lowWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_lowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_lowWall_success)
