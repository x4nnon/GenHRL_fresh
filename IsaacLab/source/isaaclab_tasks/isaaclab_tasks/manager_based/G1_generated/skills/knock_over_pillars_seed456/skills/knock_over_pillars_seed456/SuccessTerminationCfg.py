
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_pillars_seed456_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_pillars_seed456 skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required object - Object1 is the cylinder column to be knocked over.
    # CRITICAL: Accessing object directly by its scene name 'Object1' as per requirements.
    object1 = env.scene['Object1'] 
    
    # Access the required robot part(s) - pelvis for stability check.
    # CRITICAL: Using approved pattern to get robot object.
    robot = env.scene["robot"]
    # CRITICAL: Getting the index of the required robot part using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis') 
    # CRITICAL: Getting the position of the required robot part using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] 

    # Object dimensions (from problem description) - CRITICAL: Hardcoded from object config.
    # The cylinder column has a radius of 0.3m.
    pillar_radius = 0.3

    # Calculate the Z-axis position of Object1's root.
    # This is a relative distance to the ground (z=0), which is allowed for height checks.
    object1_z_pos = object1.data.root_pos_w[:, 2]
    
    # Calculate the Z-axis position of the robot's pelvis.
    # This is a relative distance to the ground (z=0), allowed for stability checks.
    pelvis_z_pos = pelvis_pos[:, 2]

    # Calculate the horizontal distance (X and Y components) between the robot's pelvis and Object1's root.
    # CRITICAL: Using relative distances between object and robot part.
    object1_xy_pos = object1.data.root_pos_w[:, :2]
    pelvis_xy_pos = pelvis_pos[:, :2]
    horizontal_distance = torch.norm(object1_xy_pos - pelvis_xy_pos, dim=-1)

    # Success conditions:
    # 1. Object1 is knocked over: its Z-position should be close to its radius (lying on the floor).
    # CRITICAL: Using a lenient threshold (radius + 0.1m tolerance) for pillar fallen state.
    pillar_fallen = object1_z_pos < (pillar_radius + 0.1) 

    # 2. Robot pelvis is at a stable height: robot has not fallen.
    # CRITICAL: Using a reasonable minimum height (0.4m) for a standing robot pelvis.
    robot_stable = pelvis_z_pos > 0.4 

    # 3. Robot is horizontally clear of the fallen pillar: not on top of it.
    # CRITICAL: Using relative horizontal distance, ensuring robot is beyond pillar radius plus a clearance.
    # This prevents the robot from being stuck on the fallen pillar, facilitating transition to next skill.
    robot_clear_of_pillar = horizontal_distance > (pillar_radius + 0.2) 

    # Combine all conditions for overall success.
    # CRITICAL: All conditions are combined using tensor logical operations for batched environments.
    success_condition = pillar_fallen & robot_stable & robot_clear_of_pillar

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # CRITICAL: Using check_success_duration with the specified duration (0.5 seconds).
    success = check_success_duration(env, success_condition, "knock_over_pillars_seed456", duration=0.5)
    
    # CRITICAL: Saving success states for environments that have met the success criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_pillars_seed456")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_pillars_seed456_success)
