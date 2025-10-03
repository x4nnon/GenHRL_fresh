
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cylinderColumn1_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cylinderColumn1 skill has been successfully completed.'''

    # 1. Get robot parts
    # Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]
    # Getting the index for the 'pelvis' body part using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Accessing the world-frame position of the pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # 2. Get object position
    # Accessing 'Object1' (Cylinder Column 1) using the approved pattern.
    object1 = env.scene['Object1']
    # Accessing the world-frame root position of Object1 for all environments.
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]

    # 3. Check if column is still standing
    # The cylinder has a radius of 0.3m. When standing, its Z-position should be above its radius
    cylinder_fallen_z_threshold = 0.3 + 0.1  # Radius + tolerance = 0.4m
    column_standing_condition = (object1_pos[:, 2] > cylinder_fallen_z_threshold)

    # 4. Calculate relative distances
    # Requirement: Success criteria must only use relative distances.
    # Calculate the absolute difference in X-positions between Object1's center and the robot's pelvis.
    # This measures how far the robot is in front/behind the cylinder.
    abs_diff_x = torch.abs(object1_pos[:, 0] - pelvis_pos[:, 0])

    # Calculate the absolute difference in Y-positions between Object1's center and the robot's pelvis.
    # This measures how well the robot is aligned with the cylinder along its width.
    abs_diff_y = torch.abs(object1_pos[:, 1] - pelvis_pos[:, 1])

    # 5. Check success condition
    # Success criteria for X-axis: The absolute difference in X-position is between 0.4m and 0.6m.
    # This ensures the robot is at an optimal distance in front of the cylinder.
    condition_x = abs_diff_x <= 0.6

    # Success criteria for Y-axis: The absolute difference in Y-position is less than 0.2m.
    # This ensures the robot is aligned with the cylinder's center along the Y-axis.
    condition_y = (abs_diff_y < 0.5)

    # Combine all conditions: Both X and Y positioning criteria must be met for success,
    # and the column must still be standing
    success_condition = condition_x & condition_y & column_standing_condition

    # 6. Check duration and save success states
    # Requirement: ALWAYS use check_success_duration and save_success_state.
    # The duration required for success is 0.5 seconds, as specified in the plan.
    success = check_success_duration(env, success_condition, "walk_to_cylinderColumn1", duration=0.2)

    # If any environment has achieved success, save its state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cylinderColumn1")

    return success

class SuccessTerminationCfg:
    # Define the success termination using the implemented function.
    success = DoneTerm(func=walk_to_cylinderColumn1_success)
