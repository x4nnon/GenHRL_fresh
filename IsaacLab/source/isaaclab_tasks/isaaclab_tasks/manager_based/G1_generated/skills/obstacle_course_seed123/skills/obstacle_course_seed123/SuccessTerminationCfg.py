
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def obstacle_course_seed123_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the obstacle_course_seed123 skill has been successfully completed.'''

    # Hardcoded object dimensions from the object configuration, as per requirements.
    # Object5: block cube of 0.5m cubed.
    BLOCK_X_DIM = 0.5
    BLOCK_Y_DIM = 0.5
    BLOCK_Z_DIM = 0.5

    # 1. Get robot parts and their positions
    robot = env.scene["robot"]
    
    # Access pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # Access left foot position using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Shape: [num_envs, 3]
    
    # Access right foot position using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]

    # 2. Get Object5 (block cube) position
    object5 = env.scene['Object5'] # Access object directly using its name
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]

    # Calculate the Z-coordinate of the top surface of the block
    # This uses the object's root position and its hardcoded Z dimension.
    block_top_z = object5_pos[:, 2] + BLOCK_Z_DIM / 2.0

    # 3. Calculate foot conditions (relative distances)
    # Condition 1: At least one foot must be horizontally within the block's bounds.
    # Calculate relative X and Y distances for left foot to object5 center
    left_foot_rel_x = torch.abs(left_foot_pos[:, 0] - object5_pos[:, 0])
    left_foot_rel_y = torch.abs(left_foot_pos[:, 1] - object5_pos[:, 1])
    # Calculate relative Z distance for left foot to block's top surface
    left_foot_rel_z_to_top = torch.abs(left_foot_pos[:, 2] - block_top_z)

    # Check if left foot is on the block (within horizontal bounds and close to top surface)
    # Tolerances are slightly larger than half dimensions to allow for some overlap.
    left_foot_on_block_x = left_foot_rel_x < (BLOCK_X_DIM / 2.0 + 0.1) # Lenient threshold for X
    left_foot_on_block_y = left_foot_rel_y < (BLOCK_Y_DIM / 2.0 + 0.1) # Lenient threshold for Y
    left_foot_on_block_z = left_foot_rel_z_to_top < 0.1 # Strict threshold for Z (on surface)
    left_foot_on_block = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z

    # Calculate relative X and Y distances for right foot to object5 center
    right_foot_rel_x = torch.abs(right_foot_pos[:, 0] - object5_pos[:, 0])
    right_foot_rel_y = torch.abs(right_foot_pos[:, 1] - object5_pos[:, 1])
    # Calculate relative Z distance for right foot to block's top surface
    right_foot_rel_z_to_top = torch.abs(right_foot_pos[:, 2] - block_top_z)

    # Check if right foot is on the block
    right_foot_on_block_x = right_foot_rel_x < (BLOCK_X_DIM / 2.0 + 0.1) # Lenient threshold for X
    right_foot_on_block_y = right_foot_rel_y < (BLOCK_Y_DIM / 2.0 + 0.1) # Lenient threshold for Y
    right_foot_on_block_z = right_foot_rel_z_to_top < 0.1 # Strict threshold for Z (on surface)
    right_foot_on_block = right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z

    # Overall foot condition: at least one foot is on the block
    feet_on_block_condition = left_foot_on_block | right_foot_on_block

    # 4. Calculate pelvis conditions (relative distances)
    # Condition 2: Pelvis is stably positioned above the block, within its horizontal bounds.
    # Calculate relative X and Y distances for pelvis to object5 center
    pelvis_rel_x = torch.abs(pelvis_pos[:, 0] - object5_pos[:, 0])
    pelvis_rel_y = torch.abs(pelvis_pos[:, 1] - object5_pos[:, 1])

    # Check if pelvis is horizontally within the block's bounds (more lenient than feet)
    pelvis_on_block_x = pelvis_rel_x < (BLOCK_X_DIM / 2.0 + 0.2) # More lenient threshold for X
    pelvis_on_block_y = pelvis_rel_y < (BLOCK_Y_DIM / 2.0 + 0.2) # More lenient threshold for Y
    pelvis_horizontal_on_block = pelvis_on_block_x & pelvis_on_block_y

    # Calculate target pelvis Z height when standing on the block (block top Z + approximate standing height)
    # This uses the block's top Z and a hardcoded approximate standing height (0.7m).
    target_pelvis_z_on_block = block_top_z + 0.7
    # Calculate relative Z distance for pelvis to target standing height
    pelvis_rel_z_to_target = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)

    # Check if pelvis Z is close to the target standing height
    pelvis_stable_height = pelvis_rel_z_to_target < 0.15 # Reasonable tolerance for stable height

    # Overall pelvis condition
    pelvis_stable_on_block_condition = pelvis_horizontal_on_block & pelvis_stable_height

    # 5. Combine all conditions for overall success
    # Both feet and pelvis conditions must be met for success.
    condition = feet_on_block_condition & pelvis_stable_on_block_condition

    # 6. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Duration required: 0.5 seconds, as specified in the plan.
    success = check_success_duration(env, condition, "obstacle_course_seed123", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "obstacle_course_seed123")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=obstacle_course_seed123_success)
