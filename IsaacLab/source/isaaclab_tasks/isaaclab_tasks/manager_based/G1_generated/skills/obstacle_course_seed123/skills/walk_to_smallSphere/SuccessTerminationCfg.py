
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_smallSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_smallSphere skill has been successfully completed.'''

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # 1. Get robot pelvis position
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL RULE: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # 2. Get small sphere (Object2) position
    small_sphere = env.scene['Object2'] # Object2 is the small sphere for robot to kick
    small_sphere_pos = small_sphere.data.root_pos_w # Shape: [num_envs, 3]

    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from task description.
    # Object dimensions from task description: "A small sphere 0.2m radius."
    small_sphere_radius = 0.2
    # Desired clearance for kicking, as per reward function and success plan
    kicking_clearance = 0.15
    # Calculate the target x-offset from the sphere's center for the robot's pelvis
    # The robot should be behind the sphere in the x-direction to kick it forward.
    target_x_offset = small_sphere_radius + kicking_clearance # 0.2 + 0.15 = 0.35m

    # CRITICAL RULE: ALL success criteria MUST ONLY use relative distances between objects and robot parts
    # 3. Calculate relative distances for success criteria
    # Success condition 1: Robot pelvis is positioned behind the small sphere in the x-axis.
    # Target X: Robot pelvis should be target_x_offset behind the sphere in x-axis.
    # abs(pelvis_pos_x - (Object2.data.root_pos_w[:, 0] - 0.35)) < 0.15m
    x_distance_diff = torch.abs(pelvis_pos[:, 0] - (small_sphere_pos[:, 0] - target_x_offset))
    x_condition = x_distance_diff < 0.15 # Using lenient threshold as per success plan

    # Success condition 2: Robot pelvis is aligned with the sphere in the y-axis.
    # Target Y: Robot pelvis should be aligned with the sphere in y-axis.
    # abs(pelvis_pos_y - Object2.data.root_pos_w[:, 1]) < 0.15m
    y_distance_diff = torch.abs(pelvis_pos[:, 1] - small_sphere_pos[:, 1])
    y_condition = y_distance_diff < 0.15 # Using lenient threshold as per success plan

    # Success condition 3: Robot pelvis is at a stable height in the z-axis.
    # CRITICAL RULE: Z-height can be an absolute value if it represents a stable posture.
    # Target Z: Stable pelvis height for the robot (0.7m as per reward function and success plan).
    # abs(pelvis_pos_z - 0.7) < 0.15m
    z_distance_diff = torch.abs(pelvis_pos[:, 2] - 0.7)
    z_condition = z_distance_diff < 0.15 # Using lenient threshold as per success plan

    # Combine all conditions: all must be true for success.
    condition = x_condition & y_condition & z_condition

    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    # 4. Check duration and save success states
    # Duration required: 0.5 seconds as per success plan
    success = check_success_duration(env, condition, "walk_to_smallSphere", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_smallSphere")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_smallSphere_success)
