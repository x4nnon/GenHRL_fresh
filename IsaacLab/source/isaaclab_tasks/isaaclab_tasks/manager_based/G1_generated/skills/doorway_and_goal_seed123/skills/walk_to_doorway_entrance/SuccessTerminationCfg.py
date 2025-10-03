
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_doorway_entrance_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_doorway_entrance skill has been successfully completed.'''
    # Access the required objects using approved patterns.
    # Object1 is "Heavy Cube (Wall 1) forming the left wall of the doorway"
    object1 = env.scene['Object1']
    # Object2 is "Heavy Cube (Wall 2) forming the right wall of the doorway"
    object2 = env.scene['Object2']

    # Access the required robot part(s) using approved patterns.
    robot = env.scene["robot"]
    # Get the index for the 'pelvis' body part.
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the world-frame position of the pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # Extract x, y, and z components for easier calculation.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Calculate the target x-position.
    # The target x-position is the center of the doorway, which is relative to the x-positions of Object1 and Object2.
    # This calculation uses relative distances between object positions, adhering to requirement 1.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # Calculate the target y-position.
    # The target y-position is just before the doorway, relative to the y-position of Object1 (or Object2, as they are aligned in y).
    # The 0.2m offset is a relative distance to position the robot "just before" the doorway, consistent with the reward function.
    # This adheres to requirement 1 (relative distance) and is explicitly allowed by the prompt's success criteria plan.
    target_y_pos = object1.data.root_pos_w[:, 1] - 0.2

    # Define the target z-position for the pelvis.
    # This is a hardcoded value representing a reasonable standing height for the robot's pelvis.
    # The prompt explicitly allows hardcoded Z-positions for height, adhering to requirement 1.
    target_pelvis_z = 0.7

    # Calculate relative distances for each dimension.
    # The success criteria must only use relative distances between objects and robot parts, adhering to requirement 1.
    distance_x = torch.abs(pelvis_pos_x - doorway_center_x)
    distance_y = torch.abs(pelvis_pos_y - target_y_pos)
    distance_z = torch.abs(pelvis_pos_z - target_pelvis_z)

    # Define success thresholds for each dimension.
    # These thresholds are lenient enough to allow for slight variations while ensuring the robot is in the correct general area.
    # 0.15m is a reasonable tolerance for positioning the entire robot, adhering to requirement 10 (reasonable tolerances).
    threshold_x = 0.15
    threshold_y = 0.15
    threshold_z = 0.15

    # Check individual success conditions for each dimension.
    # All operations work with batched environments, adhering to requirement 3.
    success_x = distance_x < threshold_x
    success_y = distance_y < threshold_y
    success_z = distance_z < threshold_z

    # Combine all conditions for overall success.
    # All three conditions (x, y, and z proximity to target) must be met.
    condition = success_x & success_y & success_z

    # Check success duration and save success states.
    # The robot must maintain the success condition for a duration of 0.5 seconds to be considered successful.
    # This adheres to requirements 4 and 5.
    success = check_success_duration(env, condition, "walk_to_doorway_entrance", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_doorway_entrance")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_doorway_entrance_success)
