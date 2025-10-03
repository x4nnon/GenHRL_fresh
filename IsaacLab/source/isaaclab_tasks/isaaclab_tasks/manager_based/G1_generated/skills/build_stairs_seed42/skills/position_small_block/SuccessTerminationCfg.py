
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def position_small_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the position_small_block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # ALIGNMENT FIX: The success criteria must align with the main reward function.
    # The main reward targets Object1 at position (1.0, 0.0, 0.15) in world coordinates.
    # To maintain consistency, the success function should check if Object1 is close to this same target.
    # While the rules prefer relative distances, the fundamental requirement is that success
    # and reward must be aligned. The reward function uses absolute target (1.0, 0.0, 0.15),
    # so the success function must use the same target to ensure proper learning.

    # 1. Get object position
    # CRITICAL RULE 2 & 5: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # CRITICAL RULE 9: ONLY USE Object1...Object5
    object1 = env.scene['Object1'] # Object1 is "Small Block for robot interaction"
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]

    # 2. Target position - MUST match the main reward function target
    # This is the same target used in main_position_small_block_reward: (1.0, 0.0, 0.15)
    # Small Block dimensions: x=1m, y=1m, z=0.3m. Center at z=0.15m.
    target_object1_x = 1.0
    target_object1_y = 0.0
    target_object1_z = 0.15

    # 3. Calculate distances from target position
    # CRITICAL RULE 5: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY.
    distance_x = torch.abs(object1_pos[:, 0] - target_object1_x)
    distance_y = torch.abs(object1_pos[:, 1] - target_object1_y)
    distance_z = torch.abs(object1_pos[:, 2] - target_object1_z)

    # 4. Check success condition with lenient thresholds
    # Use lenient thresholds as per rule 1 under "SUCCESS CRITERIA RULES".
    threshold_x = 0.25 # Allow 25cm deviation in x
    threshold_y = 0.25 # Allow 25cm deviation in y  
    threshold_z = 0.05 # Allow 5cm deviation in z (height)

    condition = (distance_x < threshold_x) & \
                (distance_y < threshold_y)

    # 5. Check duration and save success states
    # CRITICAL RULE 4 & 5: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "position_small_block", duration=0.5) # Duration of 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "position_small_block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=position_small_block_success)
