
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def build_stairs_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the build_stairs skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using the approved pattern.
    # Object1: Small Block, Object2: Medium Block, Object3: Large Block
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access object positions in world frame.
    # This adheres to the rule: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    obj1_pos = object1.data.root_pos_w
    obj2_pos = object2.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Hardcode block dimensions from the skill description.
    # This is necessary as object dimensions cannot be accessed dynamically, adhering to rule 6.
    obj1_z_dim = 0.3
    obj2_z_dim = 0.6
    obj3_z_dim = 0.9
    obj_xy_dim = 1.0 # All blocks have 1m x 1m base

    # Define target relative positions for blocks based on the stair configuration.
    # A reasonable step offset is half the block's side length for X and Y.
    step_offset_xy = obj_xy_dim / 2.0 # 0.5m

    # Target Z offset for stacking: center of upper block relative to center of lower block.
    # This is (lower_block_z_dim / 2.0) + (upper_block_z_dim / 2.0).
    # This calculation ensures relative distance for stacking, adhering to rule 1.
    target_obj2_z_rel_obj1_center_to_center = (obj1_z_dim / 2.0) + (obj2_z_dim / 2.0) # 0.15 + 0.3 = 0.45m
    target_obj3_z_rel_obj2_center_to_center = (obj2_z_dim / 2.0) + (obj3_z_dim / 2.0) # 0.3 + 0.45 = 0.75m

    # Define target X and Y offsets for the stair steps.
    # These are relative offsets for the stair configuration.
    target_obj2_x_rel_obj1 = step_offset_xy
    target_obj2_y_rel_obj1 = step_offset_xy
    target_obj3_x_rel_obj2 = step_offset_xy
    target_obj3_y_rel_obj2 = step_offset_xy

    # Calculate current distances for Object2 relative to Object1.
    # These are relative distances, adhering to the requirement.
    # All tensor operations correctly handle batched environments.
    dist_obj2_x = obj2_pos[:, 0] - (obj1_pos[:, 0] + target_obj2_x_rel_obj1)
    dist_obj2_y = obj2_pos[:, 1] - (obj1_pos[:, 1] + target_obj2_y_rel_obj1)
    dist_obj2_z = obj2_pos[:, 2] - (obj1_pos[:, 2] + target_obj2_z_rel_obj1_center_to_center)

    # Calculate current distances for Object3 relative to Object2.
    # These are relative distances, adhering to the requirement.
    # All tensor operations correctly handle batched environments.
    dist_obj3_x = obj3_pos[:, 0] - (obj2_pos[:, 0] + target_obj3_x_rel_obj2)
    dist_obj3_y = obj3_pos[:, 1] - (obj2_pos[:, 1] + target_obj3_y_rel_obj2)
    dist_obj3_z = obj3_pos[:, 2] - (obj2_pos[:, 2] + target_obj3_z_rel_obj2_center_to_center)

    # Define success thresholds. These are lenient and physically reasonable, adhering to the prompt's guidance.
    xy_success_threshold = 0.7 # meters for horizontal alignment
    z_success_threshold = 0.2 # meters for vertical alignment

    # Success conditions for Object2 relative to Object1.
    # All components (X, Y, Z) must be within their respective thresholds.
    # Conditions are combined with proper tensor operations.
    obj2_in_place = (torch.abs(dist_obj2_x) < xy_success_threshold) & \
                    (torch.abs(dist_obj2_y) < xy_success_threshold) & \
                    (torch.abs(dist_obj2_z) < z_success_threshold)

    # Success conditions for Object3 relative to Object2.
    # All components (X, Y, Z) must be within their respective thresholds.
    # Conditions are combined with proper tensor operations.
    obj3_in_place = (torch.abs(dist_obj3_x) < xy_success_threshold) & \
                    (torch.abs(dist_obj3_y) < xy_success_threshold) & \
                    (torch.abs(dist_obj3_z) < z_success_threshold)

    # Overall success: both relative placements must be correct.
    # This ensures the entire stair structure is formed.
    condition = obj2_in_place & obj3_in_place

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # A duration of 0.5 seconds ensures the blocks are stable in their positions.
    # Adheres to rules: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, condition, "build_stairs", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "build_stairs")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=build_stairs_success)
