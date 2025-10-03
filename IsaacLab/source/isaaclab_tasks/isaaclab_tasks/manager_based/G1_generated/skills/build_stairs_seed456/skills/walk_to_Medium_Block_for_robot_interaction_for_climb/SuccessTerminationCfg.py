
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Medium_Block_for_robot_interaction_for_climb_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Medium_Block_for_robot_interaction_for_climb skill has been successfully completed.'''
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName']
    object2 = env.scene['Object2'] # Medium Block for robot interaction

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Getting the index of the required robot part
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting the position of the required robot part
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # CRITICAL RULE: Hardcode object dimensions from config, DO NOT access from object attributes
    # Object2 dimensions (Medium Block: x=1m, y=1m, z=0.6m)
    object2_x_dim = 1.0

    # CRITICAL RULE: Target position relative to Object2's center, derived from reward function
    # Target x-position: Object2's x-center - (Object2_x_dim / 2) - desired_clearance (0.7m)
    # This means 1.2m from the center of the block, or 0.7m from the front face.
    target_pelvis_x = object2.data.root_pos_w[:, 0] - ((object2_x_dim / 2.0) + 0.7)
    # CRITICAL RULE: Align with Object2's y-center, using relative distance
    target_pelvis_y = object2.data.root_pos_w[:, 1]
    # CRITICAL RULE: Stable standing height, z is the only absolute position allowed for success criteria
    target_pelvis_z = 0.7

    # CRITICAL RULE: Calculate the absolute differences for each component using relative distances
    diff_x = torch.abs(pelvis_pos_x - target_pelvis_x)
    diff_y = torch.abs(pelvis_pos_y - target_pelvis_y)
    diff_z = torch.abs(pelvis_pos_z - target_pelvis_z)

    # CRITICAL RULE: Define reasonable thresholds for each component
    threshold_x = 0.15 # 15cm tolerance for x-position
    threshold_y = 0.15 # 15cm tolerance for y-position
    threshold_z = 0.15 # 15cm tolerance for z-position (height)

    # CRITICAL RULE: Check if all conditions are met for success
    condition = (diff_x < threshold_x) & \
                (diff_y < threshold_y) & \
                (diff_z < threshold_z)

    # CRITICAL RULE: Check success duration and save success states
    success = check_success_duration(env, condition, "walk_to_Medium_Block_for_robot_interaction_for_climb", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Medium_Block_for_robot_interaction_for_climb")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Medium_Block_for_robot_interaction_for_climb_success)
