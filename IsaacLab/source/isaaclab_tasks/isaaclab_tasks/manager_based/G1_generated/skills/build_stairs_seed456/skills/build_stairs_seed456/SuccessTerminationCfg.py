
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def build_stairs_seed456_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the build_stairs_seed456 skill has been successfully completed.'''

    # Hardcoded object dimensions from the task description.
    # CRITICAL RULE: Object dimensions are hardcoded from the description as per requirements.
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    obj_width_xy = 1.0 # All blocks are 1m x 1m in XY

    # 1. Get robot parts
    robot = env.scene["robot"]
    # CRITICAL RULE: Accessing robot parts using approved pattern.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # 2. Get object positions
    # CRITICAL RULE: Accessing object positions using approved pattern.
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    obj1_pos = object1.data.root_pos_w
    obj2_pos = object2.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Define target for Object1 relative to the origin (0,0,0) as per reward function.
    # This is an allowed hardcoded position as it defines the starting point of the stairs.
    # CRITICAL RULE: Hardcoded target positions are allowed for the initial block placement relative to the origin.
    target_obj1_x = 2.0
    target_obj1_y = 0.0
    target_obj1_z = obj1_height / 2.0 # Center Z for Object1

    # Define relative target offsets for Object2 and Object3 as per reward function.
    # CRITICAL RULE: Relative target offsets are hardcoded as per requirements.
    target_obj2_rel_x_offset = 0.5
    target_obj2_rel_y_offset = 0.5
    target_obj2_rel_z_offset = 0.0 # Base Z aligned

    target_obj3_rel_x_offset = 0.5
    target_obj3_rel_y_offset = 0.5
    target_obj3_rel_z_offset = 0.0 # Base Z aligned

    # --- Block Arrangement Conditions ---
    # Condition for Object1 being at its initial target location.
    # CRITICAL RULE: Success criteria must only use relative distances.
    # Here, Object1's position is relative to its hardcoded target.
    obj1_at_target_xy_dist = torch.norm(obj1_pos[:, :2] - torch.tensor([target_obj1_x, target_obj1_y], device=env.device), dim=1)
    obj1_at_target_z_dist = torch.abs(obj1_pos[:, 2] - target_obj1_z)
    # CRITICAL RULE: Using reasonable thresholds.
    obj1_in_place = (obj1_at_target_xy_dist < 0.2) & (obj1_at_target_z_dist < 0.1)

    # Condition for Object2 relative to Object1.
    # CRITICAL RULE: Success criteria must only use relative distances between objects.
    obj2_target_x = obj1_pos[:, 0] + target_obj2_rel_x_offset
    obj2_target_y = obj1_pos[:, 1] + target_obj2_rel_y_offset
    obj2_target_z = obj1_pos[:, 2] + target_obj2_rel_z_offset # Z-offset is 0 for base alignment

    obj2_rel_obj1_xy_dist = torch.norm(obj2_pos[:, :2] - torch.stack((obj2_target_x, obj2_target_y), dim=1), dim=1)
    obj2_rel_obj1_z_dist = torch.abs(obj2_pos[:, 2] - obj2_target_z)
    # CRITICAL RULE: Using reasonable thresholds.
    obj2_in_place = (obj2_rel_obj1_xy_dist < 0.3) & (obj2_rel_obj1_z_dist < 0.15)

    # Condition for Object3 relative to Object2.
    # CRITICAL RULE: Success criteria must only use relative distances between objects.
    obj3_target_x = obj2_pos[:, 0] + target_obj3_rel_x_offset
    obj3_target_y = obj2_pos[:, 1] + target_obj3_rel_y_offset
    obj3_target_z = obj2_pos[:, 2] + target_obj3_rel_z_offset # Z-offset is 0 for base alignment

    obj3_rel_obj2_xy_dist = torch.norm(obj3_pos[:, :2] - torch.stack((obj3_target_x, obj3_target_y), dim=1), dim=1)
    obj3_rel_obj2_z_dist = torch.abs(obj3_pos[:, 2] - obj3_target_z)
    # CRITICAL RULE: Using reasonable thresholds.
    obj3_in_place = (obj3_rel_obj2_xy_dist < 0.3) & (obj3_rel_obj2_z_dist < 0.15)

    # Overall block arrangement condition
    blocks_arranged = obj1_in_place & obj2_in_place & obj3_in_place

    # --- Robot Position on Object3 Conditions ---
    # Calculate Object3's top surface Z coordinate.
    obj3_top_z = obj3_pos[:, 2] + obj3_height / 2.0

    # Feet X-Y position relative to Object3's center.
    # CRITICAL RULE: Success criteria must only use relative distances between robot parts and objects.
    left_foot_xy_dist_to_obj3_center = torch.norm(left_foot_pos[:, :2] - obj3_pos[:, :2], dim=1)
    right_foot_xy_dist_to_obj3_center = torch.norm(right_foot_pos[:, :2] - obj3_pos[:, :2], dim=1)
    # CRITICAL RULE: Using reasonable thresholds. Block is 1m x 1m, so 0.5m from center covers the surface.
    feet_xy_on_obj3 = (left_foot_xy_dist_to_obj3_center < 0.5) & (right_foot_xy_dist_to_obj3_center < 0.5)

    # Feet Z position relative to Object3's top surface.
    # CRITICAL RULE: Success criteria must only use relative distances.
    left_foot_z_dist_to_obj3_top = torch.abs(left_foot_pos[:, 2] - obj3_top_z)
    right_foot_z_dist_to_obj3_top = torch.abs(right_foot_pos[:, 2] - obj3_top_z)
    # CRITICAL RULE: Using reasonable thresholds.
    feet_z_on_obj3 = (left_foot_z_dist_to_obj3_top < 0.15) & (right_foot_z_dist_to_obj3_top < 0.15)

    # Pelvis Z position relative to Object3's top surface, indicating standing upright.
    # CRITICAL RULE: Success criteria must only use relative distances.
    # Target pelvis height is obj3_top_z + 0.7m (from reward function).
    target_pelvis_z_on_obj3 = obj3_top_z + 0.7
    pelvis_z_dist_to_target = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3)
    # CRITICAL RULE: Using reasonable thresholds.
    pelvis_upright_on_obj3 = (pelvis_z_dist_to_target < 0.2)

    # Overall robot position condition
    robot_on_obj3 = feet_xy_on_obj3 & feet_z_on_obj3 & pelvis_upright_on_obj3

    # Final success condition: blocks are arranged AND robot is on the largest block.
    # CRITICAL RULE: All tensor operations correctly handle batched environments.
    condition = blocks_arranged & robot_on_obj3

    # 5. Check duration and save success states
    # CRITICAL RULE: Always use check_success_duration and save_success_state.
    success = check_success_duration(env, condition, "build_stairs_seed456", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "build_stairs_seed456")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=build_stairs_seed456_success)
