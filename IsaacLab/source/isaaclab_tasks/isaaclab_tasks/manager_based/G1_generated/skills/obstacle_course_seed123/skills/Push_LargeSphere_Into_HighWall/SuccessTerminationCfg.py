
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Push_LargeSphere_Into_HighWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Push_LargeSphere_Into_HighWall skill has been successfully completed.'''
    # Access the required objects using approved patterns
    # Object1: large sphere for robot to push
    large_sphere = env.scene['Object1']
    # Object2: small sphere for robot to kick (used for robot positioning constraint)
    small_sphere = env.scene['Object2']
    # Object4: high wall for large sphere to push over
    high_wall = env.scene['Object4']

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    # Get pelvis index for its position
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (hardcoded from config as per requirements)
    # From skill description: "A large sphere 1m radius."
    large_sphere_radius = 1.0
    # From skill description: "A wide high wall, 5m in the y axis 1m in the z axis and 0.3m in the x axis."
    high_wall_x_dim = 0.3
    # From skill description: "A small sphere 0.2m radius."
    small_sphere_radius = 0.2

    # Calculate positions of relevant edges for relative distance checks
    # Large sphere's front edge in x-axis, relative to its center
    large_sphere_front_x = large_sphere.data.root_pos_w[:, 0] + large_sphere_radius
    # High wall's back edge in x-axis, relative to its center
    high_wall_back_x = high_wall.data.root_pos_w[:, 0] + (high_wall_x_dim / 2.0)
    # Small sphere's front edge in x-axis, relative to its center
    small_sphere_front_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_radius

    # 1. Success Condition: Large sphere has passed the high wall
    # This checks if the front of the large sphere is significantly past the back of the high wall.
    # Using a relative distance threshold of 0.5m to ensure it's clearly pushed through.
    sphere_past_wall_condition = (large_sphere_front_x > high_wall_back_x + 0.5)

    # 2. Success Condition: Robot's pelvis is past the high wall
    # This ensures the robot has followed through with the push and is past the wall.
    # Using a relative distance threshold of 0.1m.
    pelvis_past_wall_condition = (pelvis_pos_x > high_wall_back_x + 0.1)

    # 3. Success Condition: Robot's pelvis has not overshot the small sphere
    # This ensures the robot is positioned appropriately for the next skill (kicking the small sphere).
    # The pelvis should be before the small sphere's front edge by at least 0.5m.
    pelvis_before_small_sphere_condition = (pelvis_pos_x < small_sphere_front_x - 0.5)

    # 4. Success Condition: Robot's pelvis is generally aligned with the path (y-axis)
    # This prevents the robot from veering too far off the central path during or after the push.
    # It checks the absolute difference in y-coordinates between the pelvis and the large sphere's initial y-position.
    # A lenient threshold of 1.5m is used for general alignment.
    pelvis_y_alignment_condition = (torch.abs(pelvis_pos_y - large_sphere.data.root_pos_w[:, 1]) < 1.5)

    # Combine all conditions for overall success
    # All conditions must be met simultaneously for success.
    success_condition = sphere_past_wall_condition & \
                        pelvis_past_wall_condition & \
                        pelvis_before_small_sphere_condition & \
                        pelvis_y_alignment_condition

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # The skill is considered successful if the conditions are met for a duration of 0.5 seconds.
    success = check_success_duration(env, success_condition, "Push_LargeSphere_Into_HighWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Push_LargeSphere_Into_HighWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Push_LargeSphere_Into_HighWall_success)
