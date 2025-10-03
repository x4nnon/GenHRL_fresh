
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_smallSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_to_smallSphere skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    # Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]
    # Getting the index for the 'pelvis' body part using the approved pattern.
    robot_pelvis_idx = robot.body_names.index('pelvis')
    # Accessing the world position of the robot's pelvis for all environments.
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # 2. Get object position
    # Accessing 'Object2' (small sphere) directly from the scene using the approved pattern.
    object_smallSphere = env.scene['Object2']
    # Accessing the world position of the small sphere for all environments.
    smallSphere_pos = object_smallSphere.data.root_pos_w
    smallSphere_pos_x = smallSphere_pos[:, 0]
    smallSphere_pos_y = smallSphere_pos[:, 1]

    # 3. Calculate relative distances and check conditions
    # The success criteria must only use relative distances between objects and robot parts.

    # X-axis condition: Pelvis should be behind the sphere (up to 0.5m) or slightly past its center (up to 0.2m).
    # This ensures the robot is positioned to kick it forward without overshooting significantly.
    # The thresholds are relative to the sphere's x-position, adhering to the relative distance rule.
    x_condition_min = robot_pelvis_pos_x > (smallSphere_pos_x - 0.5)
    x_condition_max = robot_pelvis_pos_x < (smallSphere_pos_x + 0.2)
    x_condition = x_condition_min & x_condition_max

    # Y-axis condition: Pelvis should be aligned with the sphere in the y-axis.
    # Calculating the absolute difference in Y positions, which is a relative distance.
    y_distance = torch.abs(robot_pelvis_pos_y - smallSphere_pos_y)
    # Using a reasonable tolerance for alignment.
    y_condition = y_distance < 0.2

    # Z-axis condition: Pelvis should be at a stable standing height.
    # The Z-height is the only absolute position allowed for success criteria, as it indicates stability.
    # Calculating the absolute difference from the target stable height (0.7m).
    z_distance = torch.abs(robot_pelvis_pos_z - 0.7)
    # Using a reasonable tolerance for height stability.
    z_condition = z_distance < 0.15

    # Combine all conditions. All conditions must be met for success.
    condition = x_condition & y_condition & z_condition

    # 4. Check duration and save success states
    # Using check_success_duration to ensure the condition is met for a specified duration (0.5 seconds).
    success = check_success_duration(env, condition, "walk_to_smallSphere", duration=0.5)
    # Saving success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_smallSphere")

    return success

class SuccessTerminationCfg:
    # Registering the success function as a termination condition.
    success = DoneTerm(func=walk_to_smallSphere_success)
