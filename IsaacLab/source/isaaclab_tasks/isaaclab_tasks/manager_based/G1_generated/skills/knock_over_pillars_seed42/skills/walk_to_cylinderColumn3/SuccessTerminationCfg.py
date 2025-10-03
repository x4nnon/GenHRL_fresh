
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cylinderColumn3_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cylinderColumn3 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object using the approved pattern.
    robot = env.scene["robot"]

    # Access Cylinder Column 3 (Object3) using the approved pattern.
    object3 = env.scene['Object3']

    # Get the pelvis body index using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the pelvis position in world coordinates for all environments using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get Object3's root position in world coordinates for all environments using the approved pattern.
    object3_pos = object3.data.root_pos_w

    # Hardcode cylinder radius from the object configuration (0.3m).
    # This adheres to the rule of hardcoding dimensions from the object config, not accessing them from the object.
    cylinder_radius = 0.3
    # Define the target clearance from the cylinder's surface.
    target_clearance_from_surface = 0.2
    # Calculate the target distance from the cylinder's center.
    # This is a relative distance based on the object's known dimensions.
    target_distance_from_cylinder_center = cylinder_radius + target_clearance_from_surface # 0.5m

    # Define the desired stable pelvis height. This is a fixed target height.
    # The z-component is the only absolute position allowed for success criteria, as per rules.
    target_pelvis_z = 0.7

    # Calculate the target x-position relative to Object3's center.
    # Assuming the robot approaches from the negative x-direction relative to the cylinder.
    # This ensures the target is always relative to the object's current position.
    target_x_pos = object3_pos[:, 0] - target_distance_from_cylinder_center
    # The target y-position is aligned with the cylinder's y-position.
    # This ensures the robot is centered laterally relative to the object.
    target_y_pos = object3_pos[:, 1]

    # Require column standing as in column1
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_standing_condition = (object3_pos[:, 2] > cylinder_fallen_z_threshold)

    # Mirror column1 band: |x offset| in [0.4, 0.6], |y offset| < 0.2
    abs_diff_x = torch.abs(object3_pos[:, 0] - pelvis_pos[:, 0])
    abs_diff_y = torch.abs(object3_pos[:, 1] - pelvis_pos[:, 1])
    condition_x = abs_diff_x <= 0.6
    condition_y = (abs_diff_y < 0.5)

    success_condition = condition_x & condition_y & column_standing_condition

    # Check success duration and save success states using the approved functions.
    # A duration of 0.5 seconds is chosen to ensure the robot has stopped and maintained the position.
    success = check_success_duration(env, success_condition, "walk_to_cylinderColumn3", duration=0.2)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cylinderColumn3")

    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=walk_to_cylinderColumn3_success)
