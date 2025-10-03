
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cylinderColumn2_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cylinderColumn2 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL: Accessing the robot object using the approved pattern
    robot = env.scene["robot"]

    # CRITICAL: Accessing the target object (Cylinder Column 2) using the approved pattern
    object2 = env.scene['Object2']

    # CRITICAL: Accessing the robot's pelvis position using the approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL: Accessing Object2's root position using the approved pattern
    object2_pos = object2.data.root_pos_w

    # Add column-standing check (cylinder not already fallen)
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_standing_condition = (object2_pos[:, 2] > cylinder_fallen_z_threshold)

    # Success band similar to column1: 0.4m–0.6m in front, lateral |y| < 0.2m
    abs_diff_x = torch.abs(object2_pos[:, 0] - robot_pelvis_pos[:, 0])
    abs_diff_y = torch.abs(object2_pos[:, 1] - robot_pelvis_pos[:, 1])
    condition_x = abs_diff_x <= 0.6
    condition_y = (abs_diff_y < 0.5)

    overall_success_condition = condition_x & condition_y & column_standing_condition

    # CRITICAL: Check success duration and save success states using approved functions.
    # The duration is set to 0.5 seconds as per the success criteria plan.
    success = check_success_duration(env, overall_success_condition, "walk_to_cylinderColumn2", duration=0.2)

    # CRITICAL: Save success states for environments that have met the success criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cylinderColumn2")

    return success

class SuccessTerminationCfg:
    # CRITICAL: Register the success function as a termination condition.
    success = DoneTerm(func=walk_to_cylinderColumn2_success)
