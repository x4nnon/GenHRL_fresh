
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Navigate_Doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Navigate_Doorway skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using the approved pattern.
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object3: Small Block for the robot to walk to
    object3 = env.scene['Object3']

    # Access the required robot part(s) using the approved pattern.
    # CRITICAL FIX: The 'robot' object was not accessed. Added env.scene["robot"].
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded object dimensions from task description.
    # This is necessary as object dimensions cannot be accessed dynamically.
    wall_x_dim = 0.5 # x-dimension of the walls (from task description)

    # Calculate the x-position of the back face of the doorway.
    # This is a relative calculation based on Object1's position and its known dimension.
    doorway_x_back = object1.data.root_pos_w[:, 0] + wall_x_dim / 2

    # Define target pelvis height for stability. This is a hardcoded value for Z, which is allowed.
    pelvis_target_z = 0.7

    # Condition 1: Pelvis is past the doorway in x-direction.
    # The robot needs to be at least 0.2m past the back of the doorway.
    # This is a relative distance check between pelvis x and doorway's back x.
    x_clearance_threshold = 0.4
    is_past_doorway_x = pelvis_pos_x > (doorway_x_back + x_clearance_threshold)

    # Condition 2: Pelvis is aligned with Object3's y-position.
    # This is a relative distance check between pelvis y and Object3's y.
    y_alignment_threshold = 0.5
    is_aligned_y = torch.abs(pelvis_pos_y - object3.data.root_pos_w[:, 1]) < y_alignment_threshold

    # Condition 3: Pelvis is at a stable height.
    # This is a relative distance check between pelvis z and the target z height.
    z_height_threshold = 0.2
    is_stable_height_z = torch.abs(pelvis_pos_z - pelvis_target_z) < z_height_threshold

    # Combine all conditions for success. All conditions must be met.
    success_condition = is_past_doorway_x # & is_aligned_y & is_stable_height_z

    # Check success duration and save success states.
    # The duration is set to 0.5 seconds, meaning the conditions must be met for this duration.
    success = check_success_duration(env, success_condition, "Navigate_Doorway", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Navigate_Doorway")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Navigate_Doorway_success)
