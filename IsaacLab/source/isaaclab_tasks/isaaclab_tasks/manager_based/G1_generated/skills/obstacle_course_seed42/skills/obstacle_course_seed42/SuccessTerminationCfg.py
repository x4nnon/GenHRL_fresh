
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def obstacle_course_seed42_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the obstacle_course_seed42 skill has been successfully completed.'''

    # Hardcoded object dimensions from the task description and reward context
    # Object5: Block cube (0.5m cubed)
    # CORRECT: Hardcoding object dimensions from the task description/object configuration is explicitly allowed.
    BLOCK_SIZE = 0.5
    BLOCK_HEIGHT = 0.5 # Same as BLOCK_SIZE for a cube

    # 1. Get robot parts
    robot = env.scene["robot"]
    # CORRECT: Accessing robot parts using body_names.index for batched environments
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # CORRECT: Getting positions for all envs at once
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # 2. Get object position
    # CORRECT: Direct object access using Object5 name as per configuration
    block = env.scene['Object5']
    block_pos = block.data.root_pos_w

    # Calculate block's top surface Z-coordinate
    # CORRECT: Using relative distance for block's top surface based on its root position and hardcoded height
    block_top_z = block_pos[:, 2] + BLOCK_HEIGHT / 2.0

    # 3. Calculate feet conditions (relative distances to block)
    # Both feet's z-position must be above the block's top surface by at least -0.05m (allowing for slight contact).
    # CORRECT: Relative Z-position check for left foot. Using a lenient threshold of -0.05m.
    left_foot_z_on_block = left_foot_pos[:, 2] > block_top_z - 0.05
    # CORRECT: Relative Z-position check for right foot.
    right_foot_z_on_block = right_foot_pos[:, 2] > block_top_z - 0.05

    # Both feet's x-position must be within the block's x-bounds with a tolerance of 0.15m.
    # CORRECT: Relative X-position check for left foot within block bounds. Using a reasonable tolerance of 0.15m.
    left_foot_x_on_block = torch.abs(left_foot_pos[:, 0] - block_pos[:, 0]) < BLOCK_SIZE / 2.0 + 0.15
    # CORRECT: Relative X-position check for right foot within block bounds.
    right_foot_x_on_block = torch.abs(right_foot_pos[:, 0] - block_pos[:, 0]) < BLOCK_SIZE / 2.0 + 0.15

    # Both feet's y-position must be within the block's y-bounds with a tolerance of 0.15m.
    # CORRECT: Relative Y-position check for left foot within block bounds.
    left_foot_y_on_block = torch.abs(left_foot_pos[:, 1] - block_pos[:, 1]) < BLOCK_SIZE / 2.0 + 0.15
    # CORRECT: Relative Y-position check for right foot within block bounds.
    right_foot_y_on_block = torch.abs(right_foot_pos[:, 1] - block_pos[:, 1]) < BLOCK_SIZE / 2.0 + 0.15

    # Combine all feet conditions
    # CORRECT: Combining conditions with proper tensor operations.
    feet_on_block_condition = (left_foot_z_on_block & right_foot_z_on_block)

    # 4. Calculate pelvis condition (relative height to block)
    # The absolute difference between the pelvis z-position and the target pelvis height on the block (block_pos_z + BLOCK_HEIGHT/2.0 + 0.7m) must be less than 0.15m.
    # 0.7m is a reasonable standing height for the pelvis above the support surface, consistent with reward functions.
    target_pelvis_z_on_block = block_top_z + 0.7
    # CORRECT: Relative Z-position check for pelvis height. Using a reasonable tolerance of 0.15m.
    pelvis_height_condition = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block) < 0.15

    # 5. Combine all success conditions
    # All conditions must be met for success
    # CORRECT: Combining all conditions with proper tensor operations.
    condition = feet_on_block_condition 

    # 6. Check duration and save success states
    # CORRECT: Using check_success_duration with the specified duration of 0.5 seconds.
    success = check_success_duration(env, condition, "obstacle_course_seed42", duration=0.5)
    
    # CORRECT: Saving success states for successful environments.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "obstacle_course_seed42")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=obstacle_course_seed42_success)
