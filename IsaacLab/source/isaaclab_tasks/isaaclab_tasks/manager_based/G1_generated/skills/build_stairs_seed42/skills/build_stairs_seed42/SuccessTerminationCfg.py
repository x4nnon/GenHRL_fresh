
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def build_stairs_seed42_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the build_stairs_seed42 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access required objects based on the object configuration
    # Object1: Small Block (1m x 1m x 0.3m)
    # Object2: Medium Block (1m x 1m x 0.6m)
    # Object3: Large Block (1m x 1m x 0.9m)
    object1 = env.scene['Object1'] # Accessing object using approved pattern
    object2 = env.scene['Object2'] # Accessing object using approved pattern
    object3 = env.scene['Object3'] # Accessing object using approved pattern

    # Access required robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Hardcoded block dimensions (from object configuration) - REQUIRED as per prompt
    # Object3 is the Large Block, height 0.9m.
    obj3_height = 0.9
    # All blocks are 1m x 1m in XY, so half size is 0.5m.
    block_half_size_xy = 0.5

    # --- Block Arrangement Conditions ---
    # These conditions check the relative positions of the blocks to form stairs.
    # Target relative positions are based on the SUCCESS CRITERIA PLAN.

    # Relative position of Object2 (Medium Block) to Object1 (Small Block)
    # Object2.x - Object1.x - 0.5 < 0.2
    # Object2.y - Object1.y - 0.8 < 0.2
    # Object2.z - Object1.z < 0.1
    obj2_rel_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0] # Relative X distance
    obj2_rel_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1] # Relative Y distance
    obj2_rel_obj1_z = object2.data.root_pos_w[:, 2] - object1.data.root_pos_w[:, 2] # Relative Z distance

    # Condition for Object2 being correctly placed relative to Object1
    cond_obj2_rel_obj1_x = torch.abs(obj2_rel_obj1_x - 0.5) < 0.2 # X-offset condition
    cond_obj2_rel_obj1_y = torch.abs(obj2_rel_obj1_y - 0.8) < 0.2 # Y-offset condition
    cond_obj2_rel_obj1_z = torch.abs(obj2_rel_obj1_z) < 0.1 # Z-alignment condition (should be on same ground plane)

    # Relative position of Object3 (Large Block) to Object2 (Medium Block)
    # Object3.x - Object2.x - 0.5 < 0.2
    # Object3.y - Object2.y - 0.8 < 0.2
    # Object3.z - Object2.z < 0.1
    obj3_rel_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0] # Relative X distance
    obj3_rel_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1] # Relative Y distance
    obj3_rel_obj2_z = object3.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2] # Relative Z distance

    # Condition for Object3 being correctly placed relative to Object2
    cond_obj3_rel_obj2_x = torch.abs(obj3_rel_obj2_x - 0.5) < 0.2 # X-offset condition
    cond_obj3_rel_obj2_y = torch.abs(obj3_rel_obj2_y - 0.8) < 0.2 # Y-offset condition
    cond_obj3_rel_obj2_z = torch.abs(obj3_rel_obj2_z) < 0.1 # Z-alignment condition (should be on same ground plane)

    # Overall condition for blocks being arranged
    blocks_arranged = cond_obj2_rel_obj1_x & cond_obj2_rel_obj1_y & cond_obj2_rel_obj1_z & \
                      cond_obj3_rel_obj2_x & cond_obj3_rel_obj2_y & cond_obj3_rel_obj2_z

    # --- Robot on Object3 Conditions ---
    # These conditions check if the robot is standing on top of Object3.

    # Object3's center position
    obj3_center_x = object3.data.root_pos_w[:, 0]
    obj3_center_y = object3.data.root_pos_w[:, 1]
    obj3_center_z = object3.data.root_pos_w[:, 2]

    # Target Z-position for feet on top surface of Object3
    # Object's root_pos_w[:, 2] is its center. Top surface is center + half_height.
    target_foot_z_on_obj3 = obj3_center_z + (obj3_height / 2)

    # Left foot conditions relative to Object3
    # Left foot X distance from Object3 center < 0.5 (half block size)
    # Left foot Y distance from Object3 center < 0.5 (half block size)
    # Left foot Z position relative to Object3 top surface (Object3.z + 0.9/2) is within 0.1m
    cond_left_foot_x = torch.abs(left_foot_pos[:, 0] - obj3_center_x) < block_half_size_xy # Within X bounds
    cond_left_foot_y = torch.abs(left_foot_pos[:, 1] - obj3_center_y) < block_half_size_xy # Within Y bounds
    cond_left_foot_z = torch.abs(left_foot_pos[:, 2] - target_foot_z_on_obj3) < 0.1 # Z height condition

    # Right foot conditions relative to Object3
    # Right foot X distance from Object3 center < 0.5 (half block size)
    # Right foot Y distance from Object3 center < 0.5 (half block size)
    # Right foot Z position relative to Object3 top surface (Object3.z + 0.9/2) is within 0.1m
    cond_right_foot_x = torch.abs(right_foot_pos[:, 0] - obj3_center_x) < block_half_size_xy # Within X bounds
    cond_right_foot_y = torch.abs(right_foot_pos[:, 1] - obj3_center_y) < block_half_size_xy # Within Y bounds
    cond_right_foot_z = torch.abs(right_foot_pos[:, 2] - target_foot_z_on_obj3) < 0.1 # Z height condition

    # Pelvis Z position relative to Object3 top surface
    # Pelvis Z position relative to Object3 top surface (Object3.z + 0.9/2 + 0.7) is within 0.2m
    target_pelvis_z_on_obj3 = target_foot_z_on_obj3 + 0.7 # Approximately 0.7m above feet for standing
    cond_pelvis_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3) < 0.2 # Pelvis height condition

    # Overall condition for robot being on Object3
    robot_on_obj3 = cond_left_foot_x & cond_left_foot_y & cond_left_foot_z & \
                    cond_right_foot_x & cond_right_foot_y & cond_right_foot_z & \
                    cond_pelvis_z

    # Final success condition: blocks are arranged AND robot is on Object3
    overall_success_condition = blocks_arranged & robot_on_obj3

    # Check duration and save success states
    # Duration required: 0.5 seconds as per the plan.
    success = check_success_duration(env, overall_success_condition, "build_stairs_seed42", duration=0.5)

    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "build_stairs_seed42")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=build_stairs_seed42_success)
