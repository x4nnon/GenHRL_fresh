
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def kick_smallSphere_away_from_wall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the kick_smallSphere_away_from_wall skill has been successfully completed.'''

    # Access the required objects using approved patterns (Rule 2, 5, 9)
    small_sphere = env.scene['Object2']
    block_cube = env.scene['Object5']
    # Note: High wall (Object4) is used in reward for initial position reference,
    # but success criteria focuses on the final state relative to the block.

    # Access the required robot part(s) using approved patterns (Rule 3)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcoded object dimensions from task description (Rule 6)
    small_sphere_radius = 0.2
    block_cube_x_dim = 0.5

    # Calculate relative positions for objects and robot (Rule 0, 3, 7)
    # Sphere positions
    small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0]
    small_sphere_pos_y = small_sphere.data.root_pos_w[:, 1]

    # Block positions
    block_cube_pos_x = block_cube.data.root_pos_w[:, 0]
    block_cube_pos_y = block_cube.data.root_pos_w[:, 1]

    # Robot pelvis positions
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is allowed for stability/height checks (DO'S AND DON'TS)

    # Sphere success conditions (Rule 0, 3, 7, 10)
    # The sphere should have moved past a certain point relative to the block.
    # Target: just before the block's front face, accounting for sphere radius and a small buffer.
    # This ensures the sphere is "away from the wall" and positioned for the next skill.
    sphere_x_target_threshold = block_cube_pos_x - (block_cube_x_dim / 2) - small_sphere_radius - 0.2
    sphere_x_moved_sufficiently = small_sphere_pos_x > sphere_x_target_threshold

    # The sphere should be roughly aligned in y with the block.
    # Using a lenient threshold (0.5m) for y-alignment.
    sphere_y_aligned = torch.abs(small_sphere_pos_y - block_cube_pos_y) < 0.5

    # Robot pelvis success conditions for next skill (Rule 0, 3, 7, 10)
    # Robot should be in front of the block, ready to jump.
    # Define a relative range for the robot's x-position with respect to the block.
    robot_x_target_min = block_cube_pos_x - (block_cube_x_dim / 2) - 1.0 # Robot should not be too far back
    robot_x_target_max = block_cube_pos_x - (block_cube_x_dim / 2) - 0.3 # Robot should not be too close or past the block
    robot_x_positioned = (pelvis_pos_x > robot_x_target_min) & (pelvis_pos_x < robot_x_target_max)

    # Robot should be roughly aligned in y with the block.
    # Using a lenient threshold (0.5m) for y-alignment.
    robot_y_aligned = torch.abs(pelvis_pos_y - block_cube_pos_y) < 0.5

    # Robot stability (pelvis z-height) (Rule 0, 3, 7, 10)
    # Pelvis z-height should be within a typical standing range (0.5m to 0.9m).
    # This is an absolute z-position check, which is allowed for height/stability.
    robot_z_stable = (pelvis_pos_z > 0.5) & (pelvis_pos_z < 0.9)

    # Combine all conditions for success (Rule 10)
    # All conditions must be met for success.
    condition = sphere_x_moved_sufficiently & sphere_y_aligned & \
                robot_x_positioned & robot_y_aligned & robot_z_stable

    # Check duration and save success states (Rule 4, 5)
    # The skill is considered successful if the conditions are met for 0.5 seconds.
    success = check_success_duration(env, condition, "kick_smallSphere_away_from_wall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "kick_smallSphere_away_from_wall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=kick_smallSphere_away_from_wall_success)
