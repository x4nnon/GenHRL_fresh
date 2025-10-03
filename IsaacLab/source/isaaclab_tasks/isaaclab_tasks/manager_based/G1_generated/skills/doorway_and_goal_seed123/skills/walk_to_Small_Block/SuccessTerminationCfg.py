
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object3 = env.scene['Object3'] # Small Block for the robot to walk to
    # CRITICAL RULE: Access object positions using .data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    # CRITICAL RULE: Access robot part position using .data.body_pos_w
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object directly
    # Object3 dimensions (from task description: 0.3m cubed)
    object3_size_y = 0.3

    # CRITICAL RULE: Use relative distances for target positions.
    # Target y-position: slightly before the block's y-center to be "next to" it.
    # Assuming robot approaches from negative y towards positive y.
    # Target is block's y-position minus half its depth, plus a small buffer.
    # This ensures the robot stops *next to* the block, not on top of it.
    target_y_pos = object3_pos[:, 1] - (object3_size_y / 2.0) - 0.1 # 0.1m buffer

    # Target x-position: align with the block's x-position.
    # This ensures the robot is centered with the block horizontally.
    target_x_pos = object3_pos[:, 0]

    # Target z-position: stable standing height.
    # CRITICAL RULE: Z-height can be an absolute target if it represents a stable standing height.
    target_z_pos = 0.7

    # CRITICAL RULE: Calculate distances using torch.abs for specific dimensions.
    # This measures how close the robot's pelvis is to the target x, y, and z coordinates.
    distance_x = torch.abs(pelvis_pos[:, 0] - target_x_pos)
    distance_y = torch.abs(pelvis_pos[:, 1] - target_y_pos)
    distance_z = torch.abs(pelvis_pos[:, 2] - target_z_pos)

    # CRITICAL RULE: Use lenient thresholds for success conditions.
    # The robot needs to be within a certain range of the target position in all three dimensions.
    # Thresholds are chosen to be reasonable tolerances for "immediate vicinity".
    success_x = distance_x < 0.15 # Within 15cm in x-direction
    success_y = distance_y < 0.15 # Within 15cm in y-direction
    success_z = distance_z < 0.15 # Within 15cm in z-direction (for stable height)

    # Combine all conditions: all must be true for success.
    condition = success_x & success_y & success_z

    # CRITICAL RULE: Always use check_success_duration and save_success_state.
    # The robot must maintain the success condition for a short duration (0.5 seconds)
    # to ensure stability and not just a momentary pass-through.
    success = check_success_duration(env, condition, "walk_to_Small_Block", duration=0.5)

    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_success)
