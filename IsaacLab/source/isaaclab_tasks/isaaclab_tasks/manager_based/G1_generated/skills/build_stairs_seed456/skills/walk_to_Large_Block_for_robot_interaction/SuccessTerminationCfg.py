
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Large_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Large_Block_for_robot_interaction skill has been successfully completed.
    Success means the robot's pelvis reaches the opposite side of the Large Block (Object3)
    relative to the Medium Block (Object2) at a reasonable horizontal proximity, and maintains
    a reasonable Z-height for interaction.'''
    
    # CRITICAL RULE: Access robot parts using their names and indices
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    
    # CRITICAL RULE: Access objects directly using their scene names (Object3 for Large Block)
    large_block = env.scene['Object3']
    large_block_pos = large_block.data.root_pos_w # CORRECT: Accessing object position using approved pattern
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w
    
    # CRITICAL RULE: Use relative distances between objects and robot parts.
    # Compute desired XY point on the opposite side of Object3 relative to Object2.
    target_horizontal_distance = 0.8  # meters (consistent with reward)
    pelvis_xy = pelvis_pos[:, :2]
    large_xy = large_block_pos[:, :2]
    medium_xy = medium_block_pos[:, :2]
    away_vec = large_xy - medium_xy
    away_norm = torch.norm(away_vec, dim=1, keepdim=True).clamp_min(1e-6)
    away_dir = away_vec / away_norm
    desired_xy = large_xy + away_dir * target_horizontal_distance

    # Lenient success: pelvis within a radius from the desired opposite-side point
    xy_error = torch.norm(pelvis_xy - desired_xy, dim=1)
    horizontal_success = xy_error < 0.5

    # Z-height check relative to a reasonable standing height
    target_pelvis_z_height = 0.7
    z_distance_diff = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_height)
    z_condition = z_distance_diff < 0.5

    # Combine all conditions for overall success
    condition = horizontal_success & z_condition
    
    # CRITICAL RULE: Always use check_success_duration and save_success_state
    # Check duration for success (0.5 seconds as per plan)
    success = check_success_duration(env, condition, "walk_to_Large_Block_for_robot_interaction", duration=0.5)
    
    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Large_Block_for_robot_interaction")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Large_Block_for_robot_interaction_success)
