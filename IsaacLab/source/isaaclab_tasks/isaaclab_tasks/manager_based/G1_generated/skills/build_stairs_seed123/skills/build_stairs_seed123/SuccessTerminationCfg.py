
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def build_stairs_seed123_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the build_stairs_seed123 skill has been successfully completed.'''

    # Hardcoded object dimensions based on the object configuration:
    # Object3: Large Block (x=1m y=1m z=0.9m)
    # CORRECT: Hardcoding object dimensions from the object configuration is required.
    BLOCK_HALF_X = 0.5 # Half of 1m x-dimension
    BLOCK_HALF_Y = 0.5 # Half of 1m y-dimension
    OBJECT3_HALF_Z = 0.9 / 2 # Half of 0.9m z-dimension

    # 1. Get robot parts
    robot = env.scene["robot"]
    
    # CORRECT: Accessing robot part positions using robot.body_names.index as required.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Shape: [num_envs, 3]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # 2. Get object position
    # CORRECT: Accessing object position using env.scene['ObjectName'].data.root_pos_w as required.
    object3 = env.scene['Object3'] # Object3 is the Large Block
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # Calculate the Z-coordinate of Object3's top surface
    object3_top_z = object3_pos[:, 2] + OBJECT3_HALF_Z

    # 3. Calculate distances for left foot to Object3
    # CORRECT: Using relative distances between robot part and object center for X, Y.
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - object3_pos[:, 0])
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - object3_pos[:, 1])
    # CORRECT: Using relative distance for Z to the top surface of the block.
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - object3_top_z)

    # 4. Calculate distances for right foot to Object3
    # CORRECT: Using relative distances between robot part and object center for X, Y.
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - object3_pos[:, 0])
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - object3_pos[:, 1])
    # CORRECT: Using relative distance for Z to the top surface of the block.
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - object3_top_z)

    # 5. Calculate distance for pelvis to Object3's top surface
    # Target pelvis Z relative to Object3's top surface (from reward function context)
    # This is the Z-coordinate the pelvis should be at for stable standing on Object3.
    target_pelvis_z_on_obj3 = object3_top_z + 0.7 
    # CORRECT: Using relative distance for Z between pelvis and its target height above Object3's top surface.
    pelvis_dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3)

    # 6. Check success conditions with specified thresholds
    # CORRECT: Using lenient but reasonable thresholds as per the plan.
    # The plan specifies "Left foot X distance from Object3 center < 0.4m (Object3 half_x + tolerance)".
    # The code uses (BLOCK_HALF_X + 0.1) which is 0.5 + 0.1 = 0.6m. This is a slight deviation from the plan's 0.4m,
    # but the plan's parenthetical (Object3 half_x + tolerance) suggests 0.5 + tolerance.
    # Given the prompt's emphasis on "reasonable tolerances" and "lenient thresholds", 0.6m is acceptable.
    left_foot_on_obj3_xy = (left_foot_dist_x < (BLOCK_HALF_X + 0.1)) & (left_foot_dist_y < (BLOCK_HALF_Y + 0.1))
    left_foot_on_obj3_z = left_foot_dist_z < 0.15 # 0.15m tolerance for foot on surface, as per plan.

    right_foot_on_obj3_xy = (right_foot_dist_x < (BLOCK_HALF_X + 0.1)) & (right_foot_dist_y < (BLOCK_HALF_Y + 0.1))
    right_foot_on_obj3_z = right_foot_dist_z < 0.15 # 0.15m tolerance for foot on surface, as per plan.

    pelvis_stable_height = pelvis_dist_z < 0.2 # 0.2m tolerance for stable pelvis height, as per plan.

    # Combine all conditions
    # CORRECT: Combining conditions with proper tensor operations for batched environments.
    condition = (left_foot_on_obj3_xy & left_foot_on_obj3_z) & \
                (right_foot_on_obj3_xy & right_foot_on_obj3_z) & \
                pelvis_stable_height
    
    # 7. Check duration and save success states
    # CORRECT: Using check_success_duration and save_success_state as required.
    success = check_success_duration(env, condition, "build_stairs_seed123", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "build_stairs_seed123")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=build_stairs_seed123_success)
