
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def position_medium_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the position_medium_block skill has been successfully completed.'''
    # Access the required objects using approved patterns (Rule 2, Rule 5, Rule 9)
    # Object1: Small Block for robot interaction
    # Object2: Medium Block for robot interaction
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']

    # Access object positions using approved patterns (Rule 2)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Define target offsets for Object2 relative to Object1.
    # These values are hardcoded based on the block dimensions and desired stair configuration,
    # as object dimensions cannot be accessed dynamically (Rule 6 in Critical Implementation Rules).
    # Object1 (Small Block) dimensions: 1m x 1m x 0.3m (height)
    # Object2 (Medium Block) dimensions: 1m x 1m x 0.6m (height)
    # Target: Object2's center 1m in X, 1m in Y from Object1's center.
    # Target Z for Object2's center: 0.3m (Object1's height) + (Object2's height / 2) = 0.3 + (0.6 / 2.0) = 0.6m.
    # This aligns with the reward function's logic and task description.
    target_offset_x = 1.0 # Hardcoded from task description/reward logic (Rule 4 in Absolute Requirements, Rule 6 in Critical Implementation Rules)
    target_offset_y = 1.0 # Hardcoded from task description/reward logic (Rule 4 in Absolute Requirements, Rule 6 in Critical Implementation Rules)
    object2_height = 0.6 # Hardcoded from object configuration (Rule 6 in Critical Implementation Rules)
    target_z_object2_center = 0.3 + (object2_height / 2.0) # Derived from block heights, hardcoded dimensions (Rule 6 in Critical Implementation Rules)

    # Calculate the target position for Object2 relative to Object1 (Rule 0, Rule 3 in Critical Implementation Rules)
    # This ensures the success condition is based on relative distances (Rule 1 in Absolute Requirements).
    target_object2_pos_x = object1_pos[:, 0] + target_offset_x
    target_object2_pos_y = object1_pos[:, 1] + target_offset_y

    # Calculate the distance components between Object2's current position and its target position (Rule 0, Rule 3 in Critical Implementation Rules)
    # Using absolute differences for each dimension.
    distance_x = torch.abs(object2_pos[:, 0] - target_object2_pos_x)
    distance_y = torch.abs(object2_pos[:, 1] - target_object2_pos_y)
    # For Z, we compare to the derived absolute target Z for the center of Object2 (Rule 0, Rule 3 in Critical Implementation Rules, Rule 5 in Position & Motion Rules)
    distance_z = torch.abs(object2_pos[:, 2] - target_z_object2_center)

    # Define success thresholds (Rule 1 in Success Criteria Rules)
    # These thresholds are lenient enough for a positioning task (Rule 3 in Success Criteria Rules).
    threshold_xy = 0.15 # meters (Rule 4 in Absolute Requirements - allowed for thresholds)
    threshold_z = 0.15 # meters (Rule 4 in Absolute Requirements - allowed for thresholds)

    # Success condition: Object2 is within the specified thresholds of its target relative position (Rule 3 in Critical Implementation Rules)
    condition = (distance_x < threshold_xy) & \
                (distance_y < threshold_xy) & \
                (distance_z < threshold_z)

    # Check duration and save success states (Rule 6 in Absolute Requirements, Rule 4 in Critical Implementation Rules)
    # Duration of 0.5 seconds is reasonable for a stable positioning.
    success = check_success_duration(env, condition, "position_medium_block", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "position_medium_block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=position_medium_block_success)
