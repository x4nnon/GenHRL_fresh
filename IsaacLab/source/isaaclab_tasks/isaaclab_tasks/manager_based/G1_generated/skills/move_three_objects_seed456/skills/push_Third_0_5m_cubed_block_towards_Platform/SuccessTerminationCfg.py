
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Third_0_5m_cubed_block_towards_Platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Third_0_5m_cubed_block_towards_Platform skill has been successfully completed.'''

    # Access the Third 0.5m cubed block (Object3) and the Platform (Object4)
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']

    # Get the world positions of Object3 and Object4
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcode object dimensions as per rule 6: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    # Object3: "Third 0.5m cubed block" -> half size = 0.25m
    object3_half_size = 0.25
    # Object4: "Platform x=2m y=2m and z=0.001" -> half x = 1.0m, half y = 1.0m, z_pos = 0.001m
    platform_half_x = 1.0
    platform_half_y = 1.0
    platform_z_pos = 0.001

    # Calculate the target z-position for Object3's center when it's resting on the platform.
    # This is a relative target based on platform height and block size.
    target_object3_z = platform_z_pos + object3_half_size

    # Calculate relative distances between Object3's center and Object4's center for each axis.
    # CRITICAL RULE: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # CRITICAL RULE: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY.
    x_distance = torch.abs(object3_pos[:, 0] - object4_pos[:, 0])
    y_distance = torch.abs(object3_pos[:, 1] - object4_pos[:, 1])
    z_distance = torch.abs(object3_pos[:, 2] - target_object3_z)

    # Define tolerances for success conditions.
    # These tolerances are derived from the object sizes and a small additional margin (0.05m).
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds (unless justified as relative tolerances).
    # Here, the thresholds are relative to object dimensions, ensuring the block is on the platform.
    x_tolerance = platform_half_x - object3_half_size + 0.05 # 1.0 - 0.25 + 0.05 = 0.8m
    y_tolerance = platform_half_y - object3_half_size + 0.05 # 1.0 - 0.25 + 0.05 = 0.8m
    z_tolerance = 0.05 # 0.05m tolerance for vertical alignment, ensuring it's on the surface.

    # Check if Object3 is horizontally within the bounds of Object4.
    # CRITICAL RULE: All tensor operations must work with batched environments.
    x_on_platform = x_distance <= x_tolerance
    y_on_platform = y_distance <= y_tolerance

    # Check if Object3 is vertically resting on the surface of Object4.
    z_on_platform = z_distance <= z_tolerance

    # Combine all conditions for overall success.
    # The block must be within the horizontal bounds AND at the correct vertical height.
    condition = x_on_platform & y_on_platform & z_on_platform

    # Check success duration and save success states.
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    # Duration required: 0.5 seconds as per the plan.
    success = check_success_duration(env, condition, "push_Third_0_5m_cubed_block_towards_Platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Third_0_5m_cubed_block_towards_Platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_Third_0_5m_cubed_block_towards_Platform_success)
