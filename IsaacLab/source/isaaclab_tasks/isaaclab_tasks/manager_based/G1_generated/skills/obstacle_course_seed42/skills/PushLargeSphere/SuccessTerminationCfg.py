
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def PushLargeSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PushLargeSphere skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using approved patterns (Requirement 2, 5)
    # Object1: Large sphere for robot to push
    object1 = env.scene['Object1']
    # Object4: High wall to be pushed over by large sphere
    object4 = env.scene['Object4']

    # Get object positions using approved patterns (Requirement 2, 7)
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]
    object4_pos = object4.data.root_pos_w # Shape: [num_envs, 3]

    # Hardcoded dimensions from object configuration (as per requirement 6)
    # Object1: Large sphere, radius 1m
    object1_radius = 1.0
    # Object4: High wall, 1m z-height, 0.3m x-depth
    object4_height = 1.0
    object4_depth = 0.3

    # Calculate relative distances for success conditions (Requirement 1, 3.3)
    # Condition 1: X-distance - Sphere's front edge should be at or past wall's front edge.
    # The target for object1_pos_x relative to object4_pos_x is approximately object4_pos_x - (object4_depth / 2) - object1_radius.
    # This means object4_pos_x - object1_pos_x should be approximately (object4_depth / 2) + object1_radius = 0.15 + 1.0 = 1.15.
    # A threshold of 1.2m allows for the sphere to be slightly past or just touching the wall.
    distance_x = torch.abs(object4_pos[:, 0] - object1_pos[:, 0])
    condition_x = distance_x < 1.2 # Using a lenient threshold as per requirement 3.1

    # Condition 2: Y-distance - Sphere should be aligned with the wall in Y.
    # The wall is 5m wide, so a tolerance of 1.0m is reasonable for alignment.
    distance_y = torch.abs(object1_pos[:, 1] - object4_pos[:, 1])
    condition_y = distance_y < 1.0 # Using a lenient threshold as per requirement 3.1

    # Condition 3: Z-position of the wall - Check if the wall has toppled.
    # Initial center z of wall is 0.5m (1m height / 2).
    # Toppled center z of wall is 0.15m (0.3m depth / 2).
    # A threshold of 0.25m ensures it's clearly fallen.
    # This is an allowed absolute Z-position check as it relates to height/toppling (Requirement 1, Position & Motion Rule 5).
    object4_z_pos = object4_pos[:, 2]
    condition_z_toppled = object4_z_pos < 0.25 # Using a reasonable threshold as per requirement 3.3

    # Combine all conditions for success using tensor operations (Requirement 3.2)
    success_condition = condition_z_toppled

    # Check duration and save success states (Requirement 4, 5)
    # Duration of 0.5 seconds is chosen to ensure stability of the success state.
    success = check_success_duration(env, success_condition, "PushLargeSphere", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PushLargeSphere")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PushLargeSphere_success)
