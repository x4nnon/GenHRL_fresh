
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Second_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Second_0_5m_cubed_block skill has been successfully completed.'''
    # 1. Get robot parts
    # Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]
    # Getting the index of the 'pelvis' robot part using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Getting the position of the 'pelvis' robot part using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # Separating components for clarity and specific checks.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-height is checked against an absolute value for stable posture, which is allowed.

    # 2. Get object position
    # Accessing the 'Second 0.5m cubed block' (Object2) using the approved pattern.
    target_block = env.scene['Object2']
    # Getting the root position of the target block using the approved pattern.
    target_block_pos = target_block.data.root_pos_w
    # Separating components for clarity and specific checks.
    target_block_pos_x = target_block_pos[:, 0]
    target_block_pos_y = target_block_pos[:, 1]

    # 3. Define target relative distances and hardcoded dimensions (from object config)
    # Object2 is a 0.5m cubed block. Half-size is 0.25m.
    # Target distance from block center to robot pelvis for pushing.
    # We want the robot pelvis to be ~0.1m from the face of the block.
    # So, 0.25m (block half-size) + 0.1m (clearance) = 0.35m from block center.
    # This hardcoded value is derived from the object configuration and desired clearance, as per requirements.
    # Assuming the robot approaches from the negative X side of the block.
    target_distance_from_block_center_x = -0.35 # meters
    # Target Z-height for the pelvis for stable standing. This is a reasonable hardcoded value for robot posture.
    target_pelvis_z = 0.7 # meters

    # 4. Calculate relative distances between the robot pelvis and the target block.
    # All success criteria MUST ONLY use relative distances between objects and robot parts.
    # Relative X position of pelvis to target block's center.
    relative_pos_x = pelvis_pos_x - target_block_pos_x
    # Relative Y position of pelvis to target block's center.
    relative_pos_y = pelvis_pos_y - target_block_pos_y

    # 5. Check success conditions based on relative distances and target height.
    # X-alignment: Pelvis is at the target pushing distance from the block's center in X.
    # Using a reasonable tolerance of 0.1m, as suggested in the requirements.
    x_aligned = torch.abs(relative_pos_x - target_distance_from_block_center_x) < 0.1

    # Y-alignment: Pelvis is aligned with the block's center in Y.
    # Using a reasonable tolerance of 0.1m, as suggested in the requirements.
    y_aligned = torch.abs(relative_pos_y) < 0.1

    # Z-height: Pelvis is at a stable standing height.
    # Using a reasonable tolerance of 0.1m, as suggested in the requirements.
    z_height_ok = torch.abs(pelvis_pos_z - target_pelvis_z) < 0.1

    # Combine all conditions for overall success. All conditions must be met.
    condition = x_aligned & y_aligned & z_height_ok

    # 6. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Using check_success_duration to ensure the condition is met for a sustained period (0.5 seconds).
    success = check_success_duration(env, condition, "walk_to_Second_0_5m_cubed_block", duration=0.5)
    # Saving success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Second_0_5m_cubed_block")

    return success

class SuccessTerminationCfg:
    # Registering the success function as a termination condition.
    success = DoneTerm(func=walk_to_Second_0_5m_cubed_block_success)
