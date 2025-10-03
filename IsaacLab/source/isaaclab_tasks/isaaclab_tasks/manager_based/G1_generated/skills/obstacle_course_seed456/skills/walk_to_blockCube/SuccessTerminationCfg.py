
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_blockCube skill has been successfully completed.'''

    # 1. Get robot parts positions
    robot = env.scene["robot"]
    
    # Access robot pelvis position using approved pattern
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    
    # Access robot foot positions using approved pattern
    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    robot_left_foot_pos = robot.data.body_pos_w[:, robot_left_foot_idx]
    robot_right_foot_pos = robot.data.body_pos_w[:, robot_right_foot_idx]

    # 2. Get object position (Object5 is the block cube)
    # Access object position using approved pattern
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w

    # 3. Calculate relative distances and check conditions
    
    # Condition 1: Robot pelvis x-position relative to block cube
    # Robot should be 0.5m in front of the block cube along the x-axis.
    # This is a relative distance check, comparing robot pelvis x to (block_cube x - 0.5m).
    # The target_x_offset is a hardcoded relative offset, which is allowed as per the prompt's guidance
    # for defining target relative positions (e.g., similar to target_pelvis_z in rewards).
    target_x_offset = -0.5 # Robot should be 0.5m *behind* the block cube's center from its perspective, so block_cube_x - 0.5
    x_pos_diff = torch.abs(robot_pelvis_pos[:, 0] - (block_cube_pos[:, 0] + target_x_offset))
    x_condition = x_pos_diff < 0.15 # Tolerance of 0.15m, reasonable threshold.

    # Condition 2: Robot pelvis y-position relative to block cube
    # Robot should be centered with the block cube along the y-axis.
    # This is a relative distance check, comparing robot pelvis y to block_cube y.
    y_pos_diff = torch.abs(robot_pelvis_pos[:, 1] - block_cube_pos[:, 1])
    y_condition = y_pos_diff < 0.15 # Tolerance of 0.15m, reasonable threshold.

    # Condition 3: Robot pelvis z-position (stable standing height)
    # This is an absolute height check, allowed for stability/standing as per prompt rules ("z is the only absolute position allowed").
    pelvis_z_condition = (robot_pelvis_pos[:, 2] > 0.6) & (robot_pelvis_pos[:, 2] < 0.8) # Reasonable range for standing.

    # Condition 4: Robot feet close to the ground for stability
    # These are absolute height checks, allowed for ground contact as per prompt rules.
    left_foot_on_ground = robot_left_foot_pos[:, 2] < 0.1 # Reasonable threshold for feet on ground.
    right_foot_on_ground = robot_right_foot_pos[:, 2] < 0.1 # Reasonable threshold for feet on ground.
    feet_on_ground_condition = left_foot_on_ground & right_foot_on_ground

    # Combine all conditions using tensor operations for batched environments.
    condition = x_condition & y_condition & pelvis_z_condition & feet_on_ground_condition

    # 4. Check duration and save success states
    # Using check_success_duration to ensure the conditions are met for a specified duration (0.5s as per plan).
    success = check_success_duration(env, condition, "walk_to_blockCube", duration=0.5)
    
    # Saving success states for environments that have met the success criteria.
    # This follows the approved pattern for saving success states.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_blockCube")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_blockCube_success)
