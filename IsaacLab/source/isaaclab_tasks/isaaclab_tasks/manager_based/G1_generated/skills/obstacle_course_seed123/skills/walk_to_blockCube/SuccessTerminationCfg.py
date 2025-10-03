
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_blockCube skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts and their positions
    robot = env.scene["robot"]
    # Accessing robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Accessing robot left foot position using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    # Accessing robot right foot position using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # 2. Get Object5 (block cube) position
    # Accessing object position using approved pattern
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w

    # 3. Hardcode Object5 dimensions from the task description (0.5m cubed)
    # This follows the rule to hardcode object dimensions from the config, not access them from the object directly.
    block_cube_x_dim = 0.5
    block_cube_y_dim = 0.5
    block_cube_z_dim = 0.5

    # 4. Calculate relative distances and check conditions

    # Condition 1: Pelvis X-position relative to the block's front face
    # Target X: 0.25m in front of the block's front face.
    # The front face of the block is at block_cube_pos_x - block_cube_x_dim / 2.
    # So, target_pelvis_x = (block_cube_pos_x - block_cube_x_dim / 2) - 0.25
    # This calculation uses relative distances between robot parts and object positions.
    target_pelvis_x = block_cube_pos[:, 0] - (block_cube_x_dim / 2) - 0.25
    pelvis_x_diff = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    x_position_met = pelvis_x_diff < 0.1 # Threshold 0.1m

    # Condition 2: Pelvis Y-position aligned with the block's center
    # Target Y: Aligned with the center of the block cube in the Y-axis.
    # This calculation uses relative distances between robot parts and object positions.
    target_pelvis_y = block_cube_pos[:, 1]
    pelvis_y_diff = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    y_alignment_met = pelvis_y_diff < 0.1 # Threshold 0.1m

    # Condition 3: Pelvis Z-position at a stable standing height
    # Target Z: Stable standing height (0.7m). Absolute Z is allowed for height.
    target_pelvis_z = 0.7
    pelvis_z_diff = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)
    z_height_met = pelvis_z_diff < 0.1 # Threshold 0.1m

    # Condition 4: Feet Z-positions are on the ground (stable)
    # Feet Z-positions should be between 0.0m and 0.1m.
    # This uses absolute Z positions for feet, which is allowed for ground contact.
    left_foot_on_ground = (left_foot_pos[:, 2] >= 0.0) & (left_foot_pos[:, 2] <= 0.1)
    right_foot_on_ground = (right_foot_pos[:, 2] >= 0.0) & (right_foot_pos[:, 2] <= 0.1)
    feet_on_ground_met = left_foot_on_ground & right_foot_on_ground

    # Condition 5: No collision with the block cube
    # Define collision boundaries for Object5 based on its center and dimensions.
    # These are relative to the block's center, which is itself an object position.
    min_x = block_cube_pos[:, 0] - block_cube_x_dim / 2
    max_x = block_cube_pos[:, 0] + block_cube_x_dim / 2
    min_y = block_cube_pos[:, 1] - block_cube_y_dim / 2
    max_y = block_cube_pos[:, 1] + block_cube_y_dim / 2
    min_z = block_cube_pos[:, 2] - block_cube_z_dim / 2
    max_z = block_cube_pos[:, 2] + block_cube_z_dim / 2

    # Check if pelvis is inside the block cube's bounding box
    # These checks use relative positions (pelvis_pos vs block_cube_pos and dimensions).
    pelvis_in_collision = (pelvis_pos[:, 0] > min_x) & (pelvis_pos[:, 0] < max_x) & \
                          (pelvis_pos[:, 1] > min_y) & (pelvis_pos[:, 1] < max_y) & \
                          (pelvis_pos[:, 2] > min_z) & (pelvis_pos[:, 2] < max_z)

    # Check if left foot is inside the block cube's bounding box
    left_foot_in_collision = (left_foot_pos[:, 0] > min_x) & (left_foot_pos[:, 0] < max_x) & \
                             (left_foot_pos[:, 1] > min_y) & (left_foot_pos[:, 1] < max_y) & \
                             (left_foot_pos[:, 2] > min_z) & (left_foot_pos[:, 2] < max_z)

    # Check if right foot is inside the block cube's bounding box
    right_foot_in_collision = (right_foot_pos[:, 0] > min_x) & (right_foot_pos[:, 0] < max_x) & \
                              (right_foot_pos[:, 1] > min_y) & (right_foot_pos[:, 1] < max_y) & \
                              (right_foot_pos[:, 2] > min_z) & (right_foot_pos[:, 2] < max_z)

    # No collision condition: True if no part is in collision
    no_collision = ~(pelvis_in_collision | left_foot_in_collision | right_foot_in_collision)

    # Combine all conditions for overall success
    condition = x_position_met & y_alignment_met & z_height_met & feet_on_ground_met & no_collision

    # 5. Check duration and save success states
    # Using check_success_duration and save_success_state as required.
    success = check_success_duration(env, condition, "walk_to_blockCube", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_blockCube")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_blockCube_success)
