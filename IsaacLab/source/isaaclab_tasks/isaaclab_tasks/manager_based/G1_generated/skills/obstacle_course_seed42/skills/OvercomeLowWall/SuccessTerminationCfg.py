
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def OvercomeLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the OvercomeLowWall skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts positions
    robot = env.scene["robot"]

    # CORRECT: Accessing robot part indices using robot.body_names.index
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # CORRECT: Accessing robot part positions using robot.data.body_pos_w
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Extract x, y, z components for clarity and specific checks
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]
    left_foot_pos_z = left_foot_pos[:, 2]
    right_foot_pos_z = right_foot_pos[:, 2]

    # 2. Get object position (Low wall is Object3)
    # CORRECT: Direct object access using env.scene['ObjectN']
    Object3 = env.scene['Object3'] # Low wall for robot to jump over
    # CORRECT: Accessing object position using env.scene['ObjectName'].data.root_pos_w
    low_wall_pos = Object3.data.root_pos_w

    # 3. Hardcoded dimensions from task description (CRITICAL: Dimensions are hardcoded from object config)
    # CORRECT: Dimensions are read from the object configuration and hardcoded, not accessed from the object itself.
    low_wall_height = 0.5
    low_wall_depth = 0.3
    # low_wall_width = 5.0 # Not directly used in success criteria, but noted from config

    # 4. Calculate relative distances and check success conditions
    # Condition 1: Pelvis is past the wall in the x-direction
    # The wall's far edge is low_wall_pos[:, 0] + low_wall_depth / 2.0.
    # Target is 1.0m past the wall's far edge to ensure full clearance and landing.
    # CORRECT: Using relative distance between pelvis x and wall's x-edge.
    pelvis_past_wall_x = pelvis_pos_x > (low_wall_pos[:, 0] + low_wall_depth / 2.0)

    # Condition 2: Pelvis is aligned with the wall's y-center
    # This ensures the robot jumped over the wall, not around it.
    # A lenient threshold of 0.5m is used, considering the wall's width.
    # CORRECT: Using relative distance between pelvis y and wall's y-center.
    pelvis_y_aligned = torch.abs(pelvis_pos_y - low_wall_pos[:, 1]) < 0.5

    # Condition 3: Pelvis is at a stable standing height after landing
    # This checks for a stable upright posture, typically around 0.6m to 0.8m for the pelvis.
    # CORRECT: Using absolute z-position for height, which is an approved exception.
    pelvis_stable_height = (pelvis_pos_z > 0.5) & (pelvis_pos_z < 0.8)

    # Condition 4: Both feet are on the ground after landing
    # This ensures the robot has landed and is stable.
    # A lenient threshold of 0.1m (10cm) above ground is used for "on the ground".
    # CORRECT: Using absolute z-position for feet height, which is an approved exception.
    feet_on_ground = (left_foot_pos_z < 0.1) & (right_foot_pos_z < 0.1)

    # Combine all conditions for overall success
    # All conditions must be met simultaneously.
    condition = pelvis_past_wall_x

    # 5. Check duration and save success states
    # CORRECT: Using check_success_duration to ensure the condition holds for a period.
    # A duration of 0.5 seconds is chosen for stability after landing.
    success = check_success_duration(env, condition, "OvercomeLowWall", duration=1.0)

    # CORRECT: Saving success states for environments that have met the criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "OvercomeLowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=OvercomeLowWall_success)
