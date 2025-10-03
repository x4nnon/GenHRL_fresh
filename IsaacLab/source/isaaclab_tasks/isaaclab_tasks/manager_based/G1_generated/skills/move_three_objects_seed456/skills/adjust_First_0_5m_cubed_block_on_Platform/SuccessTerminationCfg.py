
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def adjust_First_0_5m_cubed_block_on_Platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the adjust_First_0_5m_cubed_block_on_Platform skill has been successfully completed.'''

    # Access Object1 (First 0.5m cubed block) and Object4 (Platform)
    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1_pos = env.scene['Object1'].data.root_pos_w
    object4_pos = env.scene['Object4'].data.root_pos_w

    # Hardcode object dimensions as per requirements (no access to data.size or similar)
    # Object1 is a 0.5m cubed block
    object1_half_size = 0.25 # 0.5m / 2
    # Object4 is a platform x=2m y=2m and z=0.001
    object4_half_x_size = 1.0 # 2m / 2
    object4_half_y_size = 1.0 # 2m / 2
    object4_z_base = 0.001 # Platform's base Z-position

    # Define tolerances for success conditions
    # REQUIREMENT: Use reasonable tolerances (typically 0.05-0.1m for distances)
    position_tolerance = 0.05 # 5cm tolerance for positional checks

    # Calculate relative distances and check conditions for X, Y, and Z axes
    # REQUIREMENT: ALL success criteria MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT: All operations must work with batched environments

    # X-axis condition: Object1's center must be within Object4's X boundaries
    # The effective boundary for Object1's center is Object4's half-size minus Object1's half-size
    # plus a small tolerance to allow for slight variations.
    x_distance_abs = torch.abs(object1_pos[:, 0] - object4_pos[:, 0])
    x_condition = x_distance_abs < (object4_half_x_size - object1_half_size + position_tolerance)

    # Y-axis condition: Object1's center must be within Object4's Y boundaries
    y_distance_abs = torch.abs(object1_pos[:, 1] - object4_pos[:, 1])
    y_condition = y_distance_abs < (object4_half_y_size - object1_half_size + position_tolerance)

    # Z-axis condition: Object1 must be resting on Object4's surface
    # Object1's center Z should be at Object4's base Z + Object1's half-size
    target_object1_z = object4_z_base + object1_half_size
    z_distance_abs = torch.abs(object1_pos[:, 2] - target_object1_z)
    z_condition = z_distance_abs < position_tolerance

    # Combine all conditions for overall success
    # Object1 must be within X, Y, and Z boundaries simultaneously
    overall_condition = x_condition & y_condition & z_condition

    # Check success duration and save success states
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    # Duration is set to 0.5 seconds as per the plan to ensure stability
    success = check_success_duration(env, overall_condition, "adjust_First_0_5m_cubed_block_on_Platform", duration=0.5)
    
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "adjust_First_0_5m_cubed_block_on_Platform")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=adjust_First_0_5m_cubed_block_on_Platform_success)
