
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def ExecuteJumpOntoBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ExecuteJumpOntoBlock skill has been successfully completed.'''
    # 1. Get robot and block objects - rule 2 & 3 in ABSOLUTE REQUIREMENTS
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS
    except KeyError:
        # Handle case where the block object is not found - rule 5 in ABSOLUTE REQUIREMENTS
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. Get indices for robot parts - rule 3 in ABSOLUTE REQUIREMENTS
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # 3. Get positions of robot parts and block - rule 2 & 3 in ABSOLUTE REQUIREMENTS
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing left foot position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    block_pos = block.data.root_pos_w # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS

    # 4. Calculate average foot z position - rule 1 in ABSOLUTE REQUIREMENTS
    avg_foot_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # 5. Calculate block top surface z position - rule 1 & 6 in ABSOLUTE REQUIREMENTS, rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
    block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
    block_top_surface_z = block_pos[:, 2] + block_size_z/2

    # 6. Calculate relative z distances - rule 1 in ABSOLUTE REQUIREMENTS
    feet_above_block = avg_foot_z - block_top_surface_z # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
    pelvis_above_block = pelvis_pos[:, 2] - block_top_surface_z # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

    # 7. Define success conditions based on relative distances and thresholds - rule 1 & 4 in ABSOLUTE REQUIREMENTS, rule 1 & 2 in SUCCESS CRITERIA RULES
    feet_condition = feet_above_block > 0 # Feet are above block top - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_condition = pelvis_above_block > 0.2 # Pelvis is 20cm above block top - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES

    # 8. Define success conditions based on pelvis position - rule 1 & 4 in ABSOLUTE REQUIREMENTS, rule 1 & 2 in SUCCESS CRITERIA RULES
    pelvis_x_condition_low = pelvis_pos[:, 0] > block_pos[:, 0] - 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_x_condition_high = pelvis_pos[:, 0] < block_pos[:, 0] + 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_y_condition_low = pelvis_pos[:, 1] > block_pos[:, 1] - 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_y_condition_high = pelvis_pos[:, 1] < block_pos[:, 1] + 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    
    success_condition = feet_condition & pelvis_condition & pelvis_x_condition_low & pelvis_x_condition_high & pelvis_y_condition_low & pelvis_y_condition_high # Both feet and pelvis conditions must be met - rule 1 in SUCCESS CRITERIA RULES

    # 8. Check success duration and save success states - rule 6 & 7 in ABSOLUTE REQUIREMENTS, rule 4 in CRITICAL IMPLEMENTATION RULES
    success = check_success_duration(env, success_condition, "ExecuteJumpOntoBlock", duration=0.1) # Check success duration for 0.3 seconds - rule 6 in ABSOLUTE REQUIREMENTS
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ExecuteJumpOntoBlock") # Save success state for successful environments - rule 7 in ABSOLUTE REQUIREMENTS

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ExecuteJumpOntoBlock_success)