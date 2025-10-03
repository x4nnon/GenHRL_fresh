
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Kick_SmallSphere_Away_From_Wall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Kick_SmallSphere_Away_From_Wall skill has been successfully completed.'''

    # Access the required objects using approved patterns
    # Object2 is the small sphere for robot to kick
    object_small_sphere = env.scene['Object2']
    # Object4 is the high wall for large sphere to push over
    object_high_wall = env.scene['Object4']
    # Object5 is the block cube for robot to jump on top of
    object_block = env.scene['Object5']

    # Access the robot object
    robot = env.scene["robot"]

    # Access the required robot part (pelvis) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Hardcoded object dimensions from the object configuration and task description
    # High wall x-dimension is 0.3m (from task description)
    high_wall_x_dim = 0.3
    # Block x-dimension is 0.5m (from task description: 0.5m cubed)
    block_x_dim = 0.5
    # Small sphere radius is 0.2m (from task description)
    small_sphere_radius = 0.2

    # Calculate the initial assumed x-position of the small sphere relative to the high wall.
    # The high wall's x-end is its root_pos_w[:, 0] + half its x-dimension.
    # The small sphere starts 3m away from this point.
    # This creates a relative reference point based on existing objects, not a hardcoded absolute position.
    initial_sphere_x_ref = object_high_wall.data.root_pos_w[:, 0] + (high_wall_x_dim / 2.0) + 3.0

    # Success condition 1: Small sphere moved significantly away in x-direction.
    # This checks if the current x-position of Object2 is at least 0.5m beyond its initial reference point.
    # This is a relative distance check between the sphere's current position and its calculated initial reference.
    sphere_moved_sufficiently = object_small_sphere.data.root_pos_w[:, 0] > (initial_sphere_x_ref + 0.5)

    # Success condition 2: Robot pelvis is past the small sphere.
    # This checks if the robot's pelvis x-position is greater than the small sphere's x-position.
    # This is a relative position check.
    robot_past_sphere_x = pelvis_pos_x > object_small_sphere.data.root_pos_w[:, 0]

    # Success condition 3: Robot pelvis is before the block (Object5).
    # The block's start x-position is its root_pos_w[:, 0] - half its x-dimension.
    # We allow the robot to be slightly past this point (0.5m tolerance) but not deep into the block.
    # This is a relative position check.
    robot_before_block_x = pelvis_pos_x < (object_block.data.root_pos_w[:, 0] - (block_x_dim / 2.0) + 0.5)

    # Success condition 4: Robot pelvis is roughly aligned in y with the sphere's path.
    # This ensures the robot didn't deviate too much sideways after the kick.
    # This is a relative distance check in the y-axis.
    pelvis_y_aligned = torch.abs(pelvis_pos_y - object_small_sphere.data.root_pos_w[:, 1]) < 0.5

    # Combine all conditions for overall success. All conditions must be met.
    success_condition = sphere_moved_sufficiently & robot_past_sphere_x & robot_before_block_x & pelvis_y_aligned

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # A duration of 0.5 seconds is chosen to ensure the conditions are stable for a short period.
    success = check_success_duration(env, success_condition, "Kick_SmallSphere_Away_From_Wall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Kick_SmallSphere_Away_From_Wall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Kick_SmallSphere_Away_From_Wall_success)
