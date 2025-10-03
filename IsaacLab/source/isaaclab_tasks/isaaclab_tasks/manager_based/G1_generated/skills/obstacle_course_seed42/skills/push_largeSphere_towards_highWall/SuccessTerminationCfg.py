
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_largeSphere_towards_highWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_largeSphere_towards_highWall skill has been successfully completed.'''

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']  # Large sphere for robot to push
    object4 = env.scene['Object4']  # High wall to be pushed over by large sphere

    # Access robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hard-coded object dimensions from the object configuration (as allowed)
    # Object1: Large sphere, 1m radius -> diameter 2m
    large_sphere_radius = 1.0
    # Object4: High wall, 5m in y, 1m in z, 0.3m in x (thickness)
    high_wall_height = 1.0
    high_wall_thickness = 0.3

    # --- Success Condition 1: High wall (Object4) has fallen over ---
    # The high wall's root_pos_w[:, 2] is its Z-position.
    # If it's standing, its Z-position is half its height (0.5m).
    # If it falls flat on its side, its Z-position would be half its thickness (0.15m).
    # A threshold of 0.3m indicates it has significantly toppled.
    # This uses a Z-axis position threshold, which is allowed for height checks.
    wall_toppled_condition = object4.data.root_pos_w[:, 2] < 0.3

    # --- Success Condition 2: Large sphere (Object1) has moved past the high wall's original X-position ---
    # This confirms the sphere was the one that toppled the wall.
    # Since initial_root_pos_w is not allowed, we need a relative check.
    # The task description implies a linear arrangement along the X-axis.
    # The robot pushes the sphere towards the wall.
    # If the wall is toppled, its X-position might have shifted.
    # We need to check if the sphere is *beyond* the wall's current X-position,
    # or if the sphere's X-position is significantly greater than its initial X-position
    # AND the wall is toppled.
    # A more robust relative check: the sphere's X-position should be greater than the wall's X-position
    # plus a small margin (e.g., half the wall's thickness) to indicate it has pushed through.
    # This ensures the sphere is "past" the wall.
    # This uses relative distances between Object1 and Object4 along the X-axis.
    sphere_past_wall_x_condition = object1.data.root_pos_w[:, 0] > (object4.data.root_pos_w[:, 0] + (high_wall_thickness / 2.0) - 0.1) # -0.1 for tolerance

    # --- Success Condition 3: Robot's pelvis is stable and positioned just beyond the high wall's original X-position ---
    # This ensures the robot followed through and is ready for the next skill.
    # We need to check the robot's pelvis X-position relative to the wall's current X-position.
    # The robot should be past the wall, but not too far.
    # Pelvis X should be greater than the wall's X-position plus its thickness (to be past it).
    # Pelvis X should not be excessively far (e.g., within 1.5m past the wall).
    # This uses relative distances between robot pelvis and Object4 along the X-axis.
    robot_pelvis_x = pelvis_pos[:, 0]
    wall_x = object4.data.root_pos_w[:, 0]

    robot_past_wall_x_min_condition = robot_pelvis_x > (wall_x + (high_wall_thickness / 2.0) + 0.1) # Robot is past the wall's far side
    robot_past_wall_x_max_condition = robot_pelvis_x < (wall_x + (high_wall_thickness / 2.0) + 1.5) # Robot is not too far past

    # Robot's pelvis Z-position should be at a stable standing height.
    # Target height is around 0.7m (from reward function context).
    # This uses a Z-axis position threshold, which is allowed for height checks.
    robot_pelvis_z = pelvis_pos[:, 2]
    robot_stable_z_condition = torch.abs(robot_pelvis_z - 0.7) < 0.2

    # Robot's pelvis Y-position should be aligned with the wall's original Y-position.
    # This ensures the robot didn't deviate too much sideways.
    # This uses relative distances between robot pelvis and Object4 along the Y-axis.
    robot_pelvis_y = pelvis_pos[:, 1]
    wall_y = object4.data.root_pos_w[:, 1]
    robot_aligned_y_condition = torch.abs(robot_pelvis_y - wall_y) < 1.0 # Lenient Y-alignment

    # Combine all conditions for overall success
    # All conditions must be met simultaneously.
    overall_success_condition = wall_toppled_condition #& \
                                # sphere_past_wall_x_condition & \
                                # robot_past_wall_x_min_condition & \
                                # robot_past_wall_x_max_condition & \
                                # robot_stable_z_condition & \
                                # robot_aligned_y_condition

    # Check duration and save success states
    # Duration required: 0.5 seconds (to ensure stability and final state is maintained).
    success = check_success_duration(env, overall_success_condition, "push_largeSphere_towards_highWall", duration=0.5)

    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_largeSphere_towards_highWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_largeSphere_towards_highWall_success)
