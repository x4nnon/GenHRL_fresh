
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_pillars_seed42_success(env) -> torch.Tensor:
    '''Determine if the knock_over_pillars_seed42 skill has been successfully completed.'''

    # Access the required robot part (pelvis for overall robot position/posture)
    # CRITICAL RULE: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-component for checking if robot is upright

    # Define thresholds based on object dimensions and reasonable clearances.
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds for object locations.
    # However, object dimensions must be hardcoded from the object configuration.
    # From object configuration: Cylinder Column, z dimension of 2m and a radius of 0.3m.
    # When fallen, its center Z-position should be its radius (0.3m).
    # We add a tolerance to this value.
    pillar_fallen_z_threshold = 0.8 # 0.3m (radius) + 0.5m (tolerance)

    # Define minimum Z-height for the robot's pelvis to ensure it's not fallen over.
    # This is a threshold for a relative height (pelvis Z relative to floor), which is allowed.
    pelvis_min_z_height = 0.5 # Robot pelvis must be at least 0.5m above the floor

    # Initialize success condition
    all_pillars_fallen = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    
    # Check each pillar's status
    for i in range(1, 6):  # Check all 5 pillars
        pillar = env.scene[f'Object{i}']
        pillar_z = pillar.data.root_pos_w[:, 2]
        pillar_is_fallen = (pillar_z < pillar_fallen_z_threshold)
        all_pillars_fallen &= pillar_is_fallen

    # Check if the robot is still upright
    robot_is_upright = (pelvis_pos_z > pelvis_min_z_height)

    # Combine conditions - all pillars must be fallen and robot must be upright
    condition = all_pillars_fallen & robot_is_upright

    # Check success duration and save success states.
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "knock_over_pillars_seed42", duration=0.5) # Duration of 0.5 seconds for stability
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_pillars_seed42")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_pillars_seed42_success)
