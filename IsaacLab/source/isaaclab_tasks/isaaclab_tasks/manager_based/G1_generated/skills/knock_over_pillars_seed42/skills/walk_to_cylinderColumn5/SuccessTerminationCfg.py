
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cylinderColumn5_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cylinderColumn5 skill has been successfully completed.'''
    # 1. Get robot parts
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # 2. Get object position
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]
    
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    # From object configuration for Cylinder Column: radius = 0.3m
    cylinder_radius = 0.3 
    # Target offset for Y-position: cylinder_radius + 0.3m clearance = 0.6m
    target_offset_y = cylinder_radius + 0.3 
    # Target pelvis height (absolute Z-position, allowed for height checks)
    target_pelvis_z = 0.7 

    # Add column-standing check (not fallen)
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_standing_condition = (object5_pos[:, 2] > cylinder_fallen_z_threshold)

    # Mirror column1 band in XY relative to object center
    abs_diff_x = torch.abs(object5_pos[:, 0] - pelvis_pos[:, 0])
    abs_diff_y = torch.abs(object5_pos[:, 1] - pelvis_pos[:, 1])
    condition_x = abs_diff_x <= 0.6
    condition_y = (abs_diff_y < 0.5)

    condition = condition_x & condition_y & column_standing_condition
    
    # 5. Check duration and save success states
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "walk_to_cylinderColumn5", duration=0.2)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cylinderColumn5")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_cylinderColumn5_success)
