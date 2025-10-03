
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def obstacle_course_seed456_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the obstacle_course_seed456 skill has been successfully completed.'''

    # 1. Get robot parts: Accessing robot and specific body parts using approved patterns.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    # 2. Get object position: Accessing Object5 (block cube) using approved pattern.
    block_cube = env.scene['Object5'] # Object5 is the block cube for robot to jump on top of
    block_cube_pos = block_cube.data.root_pos_w

    # 3. Hardcode object dimensions: As per requirements, object dimensions are hardcoded from the description.
    block_cube_size = 0.5 # Block cube is 0.5m cubed

    # 4. Calculate relative distances and Z-position for each foot:
    # X-axis distance for left foot relative to block center
    left_foot_x_dist = torch.abs(left_foot_pos[:, 0] - block_cube_pos[:, 0])
    # Y-axis distance for left foot relative to block center
    left_foot_y_dist = torch.abs(left_foot_pos[:, 1] - block_cube_pos[:, 1])
    # Z-position for left foot (absolute Z is allowed for height checks relative to object top)
    left_foot_z_pos = left_foot_pos[:, 2]

    # X-axis distance for right foot relative to block center
    right_foot_x_dist = torch.abs(right_foot_pos[:, 0] - block_cube_pos[:, 0])
    # Y-axis distance for right foot relative to block center
    right_foot_y_dist = torch.abs(right_foot_pos[:, 1] - block_cube_pos[:, 1])
    # Z-position for right foot (absolute Z is allowed for height checks relative to object top)
    right_foot_z_pos = right_foot_pos[:, 2]

    # 5. Define thresholds based on the success criteria plan:
    # X/Y boundary threshold: (block_cube_size / 2) - 0.05m = (0.5 / 2) - 0.05 = 0.2m
    xy_threshold = (block_cube_size / 2.0) - 0.05 
    # Z-height threshold: Object5's z-position + Object5's z-dimension - 0.1m = block_cube_pos.z + 0.5 - 0.1 = block_cube_pos.z + 0.4m
    z_height_threshold = block_cube_pos[:, 2] + block_cube_size - 0.1

    # 6. Check success conditions for both feet:
    # Condition for left foot being on top of the block
    left_foot_on_block_condition = (
        (left_foot_x_dist < xy_threshold) & # Left foot within block's X-bounds
        (left_foot_y_dist < xy_threshold) & # Left foot within block's Y-bounds
        (left_foot_z_pos > z_height_threshold) # Left foot above block's top surface
    )

    # Condition for right foot being on top of the block
    right_foot_on_block_condition = (
        (right_foot_x_dist < xy_threshold) & # Right foot within block's X-bounds
        (right_foot_y_dist < xy_threshold) & # Right foot within block's Y-bounds
        (right_foot_z_pos > z_height_threshold) # Right foot above block's top surface
    )

    # Overall success condition: Both feet must be on the block
    condition = left_foot_on_block_condition & right_foot_on_block_condition
    
    # 7. Check duration and save success states - DO NOT MODIFY THIS SECTION
    success = check_success_duration(env, condition, "obstacle_course_seed456", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "obstacle_course_seed456")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=obstacle_course_seed456_success)
