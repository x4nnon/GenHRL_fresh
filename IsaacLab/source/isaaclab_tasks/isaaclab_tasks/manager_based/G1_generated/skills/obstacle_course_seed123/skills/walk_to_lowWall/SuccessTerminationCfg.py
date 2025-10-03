
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_lowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_lowWall skill has been successfully completed.'''
    # Requirement 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Requirement 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement 5: Access objects directly - objects should always exist in the scene
    object3 = env.scene['Object3'] # Low wall
    object3_pos = object3.data.root_pos_w

    # Requirement 6: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    # Object3 dimensions from task description: 0.3m x-axis, 5m y-axis, 0.5m z-axis
    object3_x_dim = 0.3
    object3_y_dim = 5.0
    object3_z_dim = 0.5

    # Calculate target positions relative to the low wall and ground
    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds for absolute positions.
    # These are relative calculations or target heights relative to ground (z=0).
    
    # Pelvis X-position: between 0.5m and 0.8m from the wall's front face
    # Wall's front face x-coordinate
    wall_front_x = object3_pos[:, 0] - (object3_x_dim / 2.0)
    
    # Target range for pelvis x-coordinate relative to wall's front face
    # Pelvis x should be greater than (wall_front_x - 0.8)
    # Pelvis x should be less than (wall_front_x - 0.5)
    pelvis_x_min_dist_from_wall = 0.5 # meters
    pelvis_x_max_dist_from_wall = 0.8 # meters

    # Condition 1: Pelvis X-position
    # Robot's pelvis x-coordinate must be within the optimal jumping distance range from the wall's front face.
    # This ensures the robot is not too far or too close, and also implicitly handles overshooting.
    pelvis_x_condition = (pelvis_pos[:, 0] < (wall_front_x - pelvis_x_min_dist_from_wall)) & \
                         (pelvis_pos[:, 0] > (wall_front_x - pelvis_x_max_dist_from_wall))

    # Condition 2: Pelvis Y-position
    # Absolute difference between pelvis y-coordinate and Object3's y-coordinate should be less than 0.3m.
    # This ensures alignment with the wall's center.
    pelvis_y_condition = torch.abs(pelvis_pos[:, 1] - object3_pos[:, 1]) < 0.3

    # Condition 3: Pelvis Z-position
    # Absolute difference between pelvis z-coordinate and target stable height (0.7m) should be less than 0.15m.
    # This ensures the robot is at a stable height.
    target_pelvis_z = 0.7 # meters, relative to ground (z=0)
    pelvis_z_condition = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) < 0.15

    # Condition 4: Feet Z-position
    # Both feet's z-coordinates should be less than 0.1m from the ground (z=0).
    # This ensures the robot is stable and not mid-air.
    ground_z = 0.0
    feet_z_threshold = 0.1 # meters
    feet_on_ground_condition = (left_foot_pos[:, 2] < (ground_z + feet_z_threshold)) & \
                               (right_foot_pos[:, 2] < (ground_z + feet_z_threshold))

    # Combine all conditions
    # All conditions must be met for success.
    overall_condition = pelvis_x_condition & pelvis_y_condition & pelvis_z_condition & feet_on_ground_condition

    # Requirement 6: ALWAYS use check_success_duration and save_success_state
    # Check duration for 0.5 seconds as specified in the plan.
    success = check_success_duration(env, overall_condition, "walk_to_lowWall", duration=0.5)
    
    # Save success states for environments that succeeded.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_lowWall")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_lowWall_success)
