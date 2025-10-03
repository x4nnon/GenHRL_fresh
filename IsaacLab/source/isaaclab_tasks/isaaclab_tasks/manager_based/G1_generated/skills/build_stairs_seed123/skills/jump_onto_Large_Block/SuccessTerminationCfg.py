
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_onto_Large_Block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_onto_Large_Block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # ABSOLUTE REQUIREMENT 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # ABSOLUTE REQUIREMENT 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # 1. Get robot parts
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # ABSOLUTE REQUIREMENT 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # ABSOLUTE REQUIREMENT 5: Access objects directly - objects should always exist in the scene
    # 2. Get object position (Object3 is the Large Block)
    large_block = env.scene['Object3']
    large_block_pos = large_block.data.root_pos_w

    # ABSOLUTE REQUIREMENT 6: THERE IS NO way to access the SIZE of an object - hardcode from object config
    # Hardcoded dimensions for Object3 (Large Block) from the object configuration.
    # Object3: "Large Block for robot to push and climb", measuring x=1m y=1m and z=0.9m.
    large_block_height = 0.9
    large_block_x_size = 1.0
    large_block_y_size = 1.0

    # Calculate the Z-coordinate of the top surface of the block. This is a relative position based on the block's root.
    block_top_z = large_block_pos[:, 2] + large_block_height / 2.0
    # Calculate the target pelvis Z for standing, relative to the block's top surface.
    target_pelvis_z = block_top_z + 0.7 # 0.7m is a hardcoded relative height for standing.

    # 3. Calculate distances and check conditions for feet
    # For both left and right feet:
    # Horizontal (x, y) distance from Object3's center < (Object3's x/y size / 2.0) + 0.15m
    # Vertical (z) distance from Object3's top surface < 0.15m

    # Left foot conditions
    left_foot_horizontal_dist_x = torch.abs(left_foot_pos[:, 0] - large_block_pos[:, 0])
    left_foot_horizontal_dist_y = torch.abs(left_foot_pos[:, 1] - large_block_pos[:, 1])
    left_foot_vertical_dist_z = torch.abs(left_foot_pos[:, 2] - block_top_z)

    left_foot_on_block_horizontal_x = left_foot_horizontal_dist_x < (large_block_x_size / 2.0 + 0.15)
    left_foot_on_block_horizontal_y = left_foot_horizontal_dist_y < (large_block_y_size / 2.0 + 0.15)
    left_foot_on_block_vertical = left_foot_vertical_dist_z < 0.15

    left_foot_on_block = left_foot_on_block_horizontal_x & left_foot_on_block_horizontal_y & left_foot_on_block_vertical

    # Right foot conditions
    right_foot_horizontal_dist_x = torch.abs(right_foot_pos[:, 0] - large_block_pos[:, 0])
    right_foot_horizontal_dist_y = torch.abs(right_foot_pos[:, 1] - large_block_pos[:, 1])
    right_foot_vertical_dist_z = torch.abs(right_foot_pos[:, 2] - block_top_z)

    right_foot_on_block_horizontal_x = right_foot_horizontal_dist_x < (large_block_x_size / 2.0 + 0.15)
    right_foot_on_block_horizontal_y = right_foot_horizontal_dist_y < (large_block_y_size / 2.0 + 0.15)
    right_foot_on_block_vertical = right_foot_vertical_dist_z < 0.15

    right_foot_on_block = right_foot_on_block_horizontal_x & right_foot_on_block_horizontal_y & right_foot_on_block_vertical

    # 4. Calculate distances and check conditions for pelvis
    # For pelvis:
    # Horizontal (x, y) distance from Object3's center < (Object3's x/y size / 2.0) + 0.2m
    # Vertical (z) distance from the expected standing height (Object3's top surface + 0.7m) < 0.2m

    pelvis_horizontal_dist_x = torch.abs(pelvis_pos[:, 0] - large_block_pos[:, 0])
    pelvis_horizontal_dist_y = torch.abs(pelvis_pos[:, 1] - large_block_pos[:, 1])
    pelvis_vertical_dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    pelvis_horizontally_centered = (pelvis_horizontal_dist_x < (large_block_x_size / 2.0 + 0.2)) & \
                                   (pelvis_horizontal_dist_y < (large_block_y_size / 2.0 + 0.2))
    pelvis_at_standing_height = pelvis_vertical_dist_z < 0.2

    # Combine all conditions for overall success
    condition = left_foot_on_block & right_foot_on_block & pelvis_horizontally_centered & pelvis_at_standing_height

    # ABSOLUTE REQUIREMENT 6: ALWAYS use check_success_duration and save_success_state
    # 5. Check duration and save success states
    success = check_success_duration(env, condition, "jump_onto_Large_Block", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_onto_Large_Block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_onto_Large_Block_success)
