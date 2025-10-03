
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Medium_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Medium_Block_for_robot_interaction skill has been successfully completed.
    Success means the robot's pelvis reaches the opposite side of the Medium Block (Object2)
    relative to the Small Block (Object1) at a reasonable horizontal proximity, and maintains
    a reasonable Z-height for interaction.'''
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w
    small_block = env.scene['Object1']
    small_block_pos = small_block.data.root_pos_w

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: ALL success criteria MUST ONLY use relative distances between objects and robot parts.
    # Compute a desired XY target point on the opposite side of the Medium Block relative to the Small Block.
    target_horizontal_distance = 0.8  # meters (consistent with reward shaping)
    pelvis_xy = pelvis_pos[:, :2]
    medium_xy = medium_block_pos[:, :2]
    small_xy = small_block_pos[:, :2]
    away_vec = medium_xy - small_xy
    away_norm = torch.norm(away_vec, dim=1, keepdim=True).clamp_min(1e-6)
    away_dir = away_vec / away_norm
    desired_xy = medium_xy + away_dir * target_horizontal_distance

    # Lenient success: pelvis within a radius from the desired opposite-side point
    xy_error = torch.norm(pelvis_xy - desired_xy, dim=1)
    # CRITICAL RULE: Use lenient thresholds.
    # Allow up to 0.5m radial error around the desired point to count as success.
    horizontal_success = xy_error < 0.4

    # Calculate the absolute Z-distance between the pelvis and the medium block's Z-position.
    # This checks if the pelvis is at a reasonable height relative to the block's base, ensuring upright posture.
    z_distance_abs = torch.abs(medium_block_pos[:, 2] - pelvis_pos[:, 2])

    # Z-distance less than 0.5m. This threshold ensures the robot's pelvis is not too far above or below
    # the block's base, indicating a stable and appropriate height for interaction.
    z_success = z_distance_abs < 0.5

    # Combine conditions for overall success. Both horizontal proximity and reasonable Z-height are required.
    condition = horizontal_success & z_success

    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    # Check if the success condition has been met for a duration of 0.5 seconds.
    success = check_success_duration(env, condition, "walk_to_Medium_Block_for_robot_interaction", duration=0.5)
    
    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Medium_Block_for_robot_interaction")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Medium_Block_for_robot_interaction_success)
