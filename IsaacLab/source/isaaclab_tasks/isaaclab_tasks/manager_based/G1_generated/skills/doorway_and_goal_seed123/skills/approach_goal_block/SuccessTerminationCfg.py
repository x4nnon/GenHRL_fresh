
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_goal_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_goal_block skill has been successfully completed.'''
    # 1. Get robot parts
    # Access the robot object from the scene.
    robot = env.scene["robot"]
    # Get the index for the 'pelvis' body part.
    pelvis_idx = robot.body_names.index('pelvis')
    # Access the world-frame position of the pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object position
    # Access Object3, which is the Small Block, from the scene.
    object3 = env.scene['Object3']
    # Access the world-frame root position of Object3 for all environments.
    object3_pos = object3.data.root_pos_w
    
    # 3. Hardcode object dimensions as per requirements (Object3 is 0.3m cubed)
    # This value is obtained from the object configuration provided in the problem description.
    object3_half_height = 0.3 / 2.0

    # 4. Calculate target z for pelvis relative to Object3's base
    # The target z-position for the pelvis is 0.7m above the base of Object3.
    # Object3_pos[:, 2] is the center z, so Object3_pos[:, 2] - object3_half_height is the base z.
    target_pelvis_z = object3_pos[:, 2] - object3_half_height + 0.7
    
    # 5. Calculate relative distances in x, y, and z dimensions
    # Calculate the absolute difference in x-coordinates between pelvis and Object3's center.
    distance_x = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0])
    # Calculate the absolute difference in y-coordinates between pelvis and Object3's center.
    distance_y = torch.abs(pelvis_pos[:, 1] - object3_pos[:, 1])
    # Calculate the absolute difference in z-coordinates between pelvis and the target pelvis z.
    distance_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)
    
    # 6. Check success conditions based on specified thresholds
    # Condition for x-alignment: pelvis x-position is within 0.4m of Object3's x-center.
    condition_x = distance_x < 0.4
    # Condition for y-alignment: pelvis y-position is within 0.4m of Object3's y-center.
    condition_y = distance_y < 0.4
    # Condition for z-alignment: pelvis z-position is within 0.3m of the target z-height above Object3's base.
    condition_z = distance_z < 0.3
    
    # 7. Combine all conditions: all must be true for success.
    condition = condition_x & condition_y & condition_z
    
    # 8. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Check if the combined success condition has been met for the required duration (0.5 seconds).
    success = check_success_duration(env, condition, "approach_goal_block", duration=0.5)
    # If any environment has succeeded, save its success state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_goal_block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=approach_goal_block_success)
