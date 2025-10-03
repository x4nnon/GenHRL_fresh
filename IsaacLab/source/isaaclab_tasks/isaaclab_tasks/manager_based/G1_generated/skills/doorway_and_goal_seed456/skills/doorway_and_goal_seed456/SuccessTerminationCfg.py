
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def doorway_and_goal_seed456_success(env) -> torch.Tensor:
    '''Determine if the doorway_and_goal_seed456 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object using the approved pattern.
    robot = env.scene["robot"]
    # Access the pelvis body part of the robot using the approved pattern for body names and positions.
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the world position of the robot's pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Access the required objects from the scene using the approved pattern (Object1, Object2, Object3).
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # small block

    # Get the root positions of the objects in world coordinates using the approved pattern.
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Calculate the y-position of the doorway.
    # This is a relative calculation based on the average y-position of the two wall objects, avoiding hard-coded positions.
    doorway_y_pos = (object1_pos[:, 1] + object2_pos[:, 1]) / 2.0

    # Condition 1: Robot pelvis is past the doorway's y-position.
    # The problem description implies movement along the y-axis.
    # A clearance of 0.5m is added to ensure the robot is fully through the doorway.
    # This uses a relative distance check between the pelvis y-position and the doorway's y-position.
    # The clearance is a reasonable tolerance for ensuring the robot has passed.
    doorway_clearance_y = 0.5
    is_past_doorway_y = pelvis_pos[:, 1] > (doorway_y_pos + doorway_clearance_y)

    # Condition 2: Robot pelvis is close to Object3 in x, y, and z.
    # This uses relative distances between the robot's pelvis and Object3's position for each axis, as required.
    # A threshold of 0.3m is a reasonable tolerance for being "close" to the small block.
    threshold_distance_xyz = 0.3 # meters, a reasonable tolerance for being "close"

    # Calculate absolute differences in x, y, and z coordinates between pelvis and Object3.
    # These are relative distance calculations.
    distance_x_to_object3 = torch.abs(object3_pos[:, 0] - pelvis_pos[:, 0])
    distance_y_to_object3 = torch.abs(object3_pos[:, 1] - pelvis_pos[:, 1])
    distance_z_to_object3 = torch.abs(object3_pos[:, 2] - pelvis_pos[:, 2])

    # Check if the pelvis is within the threshold for each axis.
    # These are based on relative distances.
    is_close_to_object3_x = distance_x_to_object3 < threshold_distance_xyz
    is_close_to_object3_y = distance_y_to_object3 < threshold_distance_xyz
    is_close_to_object3_z = distance_z_to_object3 < threshold_distance_xyz

    # Combine all conditions for overall success.
    # All conditions must be met simultaneously.
    condition = is_past_doorway_y & is_close_to_object3_x & is_close_to_object3_y & is_close_to_object3_z

    # Check success duration and save success states using the approved functions.
    # The duration is set to 0.5 seconds to ensure the robot maintains the successful state for a short period.
    success = check_success_duration(env, condition, "doorway_and_goal_seed456", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "doorway_and_goal_seed456")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=doorway_and_goal_seed456_success)
