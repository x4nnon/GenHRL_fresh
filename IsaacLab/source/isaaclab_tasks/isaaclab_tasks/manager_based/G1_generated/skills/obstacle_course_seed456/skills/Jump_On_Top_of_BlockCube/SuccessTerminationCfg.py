
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Jump_On_Top_of_BlockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Jump_On_Top_of_BlockCube skill has been successfully completed.'''

    # Access the required object (Object5: block cube) using the approved pattern.
    # This adheres to requirement 2: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    block_cube = env.scene['Object5']
    block_pos = block_cube.data.root_pos_w

    # Hardcode block dimensions from the task description (0.5m cubed).
    # This adheres to requirement 6: There is no way to access the SIZE of an object, so hardcode from config.
    block_height = 0.5
    block_half_x = 0.5 / 2.0 # 0.25m
    block_half_y = 0.5 / 2.0 # 0.25m
    
    # Calculate the z-coordinate of the block's top surface.
    # This is a relative position based on the block's center and its hardcoded height.
    block_top_z = block_pos[:, 2] + block_height / 2.0

    # Access the required robot parts (left and right feet) using the approved pattern.
    # This adheres to requirement 3: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define tolerances for success criteria.
    # These are reasonable thresholds, adhering to requirement "REASONABLE TOLERANCES".
    xy_tolerance = 0.05 # 5 cm tolerance for x and y
    z_lower_tolerance = 0.1 # 10 cm below block top
    z_upper_tolerance = 0.3 # 30 cm above block top

    # Calculate distances for left foot relative to the block's center (x, y) and top surface (z).
    # This adheres to requirement 1: SUCCESS CRITERIA MUST ONLY use relative distances.
    # It also adheres to requirement "YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY".
    dist_lx = torch.abs(left_foot_pos[:, 0] - block_pos[:, 0])
    dist_ly = torch.abs(left_foot_pos[:, 1] - block_pos[:, 1])
    dist_lz_from_top = left_foot_pos[:, 2] - block_top_z # Positive if above top, negative if below

    # Calculate distances for right foot relative to the block's center (x, y) and top surface (z).
    # This also adheres to requirement 1 and "YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY".
    dist_rx = torch.abs(right_foot_pos[:, 0] - block_pos[:, 0])
    dist_ry = torch.abs(right_foot_pos[:, 1] - block_pos[:, 1])
    dist_rz_from_top = right_foot_pos[:, 2] - block_top_z

    # Check if left foot is on the block.
    # Conditions are based on relative distances and hardcoded block dimensions.
    left_foot_on_block_x = dist_lx < (block_half_x + xy_tolerance)
    left_foot_on_block_y = dist_ly < (block_half_y + xy_tolerance)
    # Z-condition: foot must be within a range around the block's top surface.
    left_foot_on_block_z = (dist_lz_from_top > -z_lower_tolerance) & (dist_lz_from_top < z_upper_tolerance)
    left_foot_on_block = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z

    # Check if right foot is on the block.
    right_foot_on_block_x = dist_rx < (block_half_x + xy_tolerance)
    right_foot_on_block_y = dist_ry < (block_half_y + xy_tolerance)
    right_foot_on_block_z = (dist_rz_from_top > -z_lower_tolerance) & (dist_rz_from_top < z_upper_tolerance)
    right_foot_on_block = right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z

    # Overall success condition: both feet are on the block.
    # This combines conditions using tensor operations, handling batched environments.
    condition = left_foot_on_block & right_foot_on_block

    # Check duration and save success states.
    # This adheres to requirement 6: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, condition, "Jump_On_Top_of_BlockCube", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Jump_On_Top_of_BlockCube")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Jump_On_Top_of_BlockCube_success)
