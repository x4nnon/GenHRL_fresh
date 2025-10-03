
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
import torch

def walk_to_largeSphere_success(env) -> torch.Tensor:
    '''Determine if the walk_to_largeSphere skill has been successfully completed.

    Args:
        env: The environment instance
        robot: The robot instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using the approved pattern
    # Object1 is the large sphere for robot to push
    object_large_sphere = env.scene['Object1']
    # Object4 is the high wall for large sphere to push over
    object_high_wall = env.scene['Object4']

    # Access the required robot part(s) using the approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get object positions using the approved pattern
    large_sphere_pos = object_large_sphere.data.root_pos_w
    high_wall_pos = object_high_wall.data.root_pos_w

    # Hardcode object dimensions from the object configuration, as per requirements.
    # From object config: large sphere has 1m radius
    large_sphere_radius = 1.0
    # From object config: high wall has 0.3m in x-axis (thickness)
    high_wall_x_dim = 0.3

    # Define target offsets and thresholds. These are fixed values based on skill design.
    # Target X offset from sphere center: sphere radius (1.0m) + a buffer (0.2m) to be in pushing distance.
    target_x_offset_from_sphere_center = large_sphere_radius + 0.2
    # Target Z height for stable standing, an absolute height is allowed for Z-axis.
    target_pelvis_z = 0.7
    # Tolerance for X and Y alignment, a reasonable lenient threshold.
    threshold_xy = 0.15
    # Tolerance for Z height, a reasonable lenient threshold.
    threshold_z = 0.15
    # Clearance from the high wall: half wall x-dimension + a small buffer.
    high_wall_clearance_x = high_wall_x_dim / 2 + 0.1

    # Condition 1: Robot pelvis X-position relative to large sphere.
    # The robot's pelvis X should be at large_sphere_pos_x - target_x_offset_from_sphere_center.
    # This ensures the robot is positioned correctly before the sphere for pushing.
    cond_x = torch.abs(robot_pelvis_pos[:, 0] - (large_sphere_pos[:, 0] - target_x_offset_from_sphere_center)) < threshold_xy

    # Condition 2: Robot pelvis Y-position relative to large sphere.
    # The robot's pelvis Y should be aligned with the large sphere's Y.
    # This ensures the robot is centered with the sphere.
    cond_y = torch.abs(robot_pelvis_pos[:, 1] - large_sphere_pos[:, 1]) < threshold_xy

    # Condition 3: Robot pelvis Z-position (stable height).
    # The robot's pelvis Z should be at a stable standing height.
    # This ensures the robot is not falling or crouching.
    cond_z = torch.abs(robot_pelvis_pos[:, 2] - target_pelvis_z) < threshold_z

    # Condition 4: Robot pelvis is not past the high wall (Object4).
    # This ensures the robot has not overshot its target and is still before the high wall.
    # The robot's pelvis X must be less than the high wall's X position minus a clearance.
    cond_not_past_high_wall = robot_pelvis_pos[:, 0] < (high_wall_pos[:, 0] - high_wall_clearance_x)

    # Combine all conditions for success. All conditions must be met.
    success_condition = cond_x & cond_y & cond_z & cond_not_past_high_wall

    # Check success duration and save success states.
    # The duration is set to 0.5 seconds to ensure stability in the target position.
    success = check_success_duration(env, success_condition, "walk_to_largeSphere", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_largeSphere")

    return success

class SuccessTerminationCfg:
    # Provide required robot parameter for the success term
    success = DoneTerm(func=walk_to_largeSphere_success)
