
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_First_0_5m_cubed_block_towards_Platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_First_0_5m_cubed_block_towards_Platform skill has been successfully completed.'''
    
    # Access Object1 (First 0.5m cubed block) and Object4 (Platform)
    # CRITICAL RULE: Access objects directly using their scene names.
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']
    
    # Access object positions using the approved pattern.
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w.
    object1_pos = object1.data.root_pos_w
    object4_pos = object4.data.root_pos_w
    
    # Hardcode object dimensions from the object configuration.
    # CRITICAL RULE: There is no way to access the size of an object dynamically.
    # Object1 is 0.5m cubed.
    object1_half_size = 0.5 / 2.0  # 0.25m
    # Object4 (Platform) is x=2m, y=2m, z=0.001.
    platform_half_x = 2.0 / 2.0  # 1.0m
    platform_half_y = 2.0 / 2.0  # 1.0m
    # The Z value 0.001 is assumed to be the absolute Z of the platform's top surface.
    platform_top_z_absolute = 0.001 
    
    # Calculate the target Z position for Object1's center.
    # Object1's center should be at platform_top_z_absolute + object1_half_size.
    target_object1_z = platform_top_z_absolute + object1_half_size
    
    # Calculate relative distances between Object1's center and Object4's center.
    # CRITICAL RULE: Success criteria MUST ONLY use relative distances.
    # CRITICAL RULE: Consider X, Y, and Z components separately.
    distance_obj1_obj4_x = object1_pos[:, 0] - object4_pos[:, 0]
    distance_obj1_obj4_y = object1_pos[:, 1] - object4_pos[:, 1]
    # For Z, we compare Object1's Z to its target Z, which is relative to the platform's Z.
    distance_obj1_z = object1_pos[:, 2] - target_object1_z
    
    # Define success thresholds based on the plan.
    # The +0.1m leniency allows the block to be "very close to the edge" while still on the platform.
    # This is a reasonable tolerance as per the prompt's guidelines.
    threshold_x = platform_half_x - object1_half_size + 0.1
    threshold_y = platform_half_y - object1_half_size + 0.1
    threshold_z = 0.05 # Tolerance for vertical alignment. This is a reasonable tolerance.
    
    # Check success conditions for each axis.
    # CRITICAL RULE: All operations must work with batched environments.
    condition_x = torch.abs(distance_obj1_obj4_x) <= threshold_x
    condition_y = torch.abs(distance_obj1_obj4_y) <= threshold_y
    condition_z = torch.abs(distance_obj1_z) <= threshold_z
    
    # Combine all conditions for overall success.
    condition = condition_x & condition_y & condition_z
    
    # Check duration and save success states.
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, condition, "push_First_0_5m_cubed_block_towards_Platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_First_0_5m_cubed_block_towards_Platform")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_First_0_5m_cubed_block_towards_Platform_success)
