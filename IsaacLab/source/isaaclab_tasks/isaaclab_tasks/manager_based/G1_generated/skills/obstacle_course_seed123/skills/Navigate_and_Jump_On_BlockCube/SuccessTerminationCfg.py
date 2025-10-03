
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Navigate_and_Jump_On_BlockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Navigate_and_Jump_On_BlockCube skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required object: Object5 is the block cube.
    # CRITICAL RULE: Access objects directly using env.scene['ObjectName'].
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]

    # Access the required robot parts: feet and pelvis.
    # CRITICAL RULE: Access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')].
    robot = env.scene["robot"]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Shape: [num_envs, 3]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Object5 dimensions (from task description: 0.5m cubed).
    # CRITICAL RULE: Hardcode object dimensions from the object configuration, do not try to access them from the object itself.
    block_height = 0.5
    block_half_x = 0.25 # 0.5 / 2
    block_half_y = 0.25 # 0.5 / 2

    # Calculate the z-coordinate of the block's top surface relative to its root position.
    # This is a relative height calculation.
    block_top_z = object5_pos[:, 2] + block_height / 2.0

    # Define the target pelvis height when standing stably on the block.
    # This is relative to the block's top surface, using a standard robot standing height (0.7m).
    target_pelvis_z_on_block = block_top_z + 0.7

    # Condition for left foot being horizontally on the block.
    # CRITICAL RULE: Use relative distances (absolute differences) for X and Y components.
    left_foot_on_block_horizontal = (torch.abs(left_foot_pos[:, 0] - object5_pos[:, 0]) < block_half_x) & \
                                    (torch.abs(left_foot_pos[:, 1] - object5_pos[:, 1]) < block_half_y)

    # Condition for left foot being vertically at or just above the block's top surface.
    # CRITICAL RULE: Use reasonable tolerances. 0.1m allows for slight penetration or being just above.
    left_foot_on_block_vertical = (left_foot_pos[:, 2] > block_top_z - 0.1)

    # Condition for right foot being horizontally on the block.
    # CRITICAL RULE: Use relative distances (absolute differences) for X and Y components.
    right_foot_on_block_horizontal = (torch.abs(right_foot_pos[:, 0] - object5_pos[:, 0]) < block_half_x) & \
                                     (torch.abs(right_foot_pos[:, 1] - object5_pos[:, 1]) < block_half_y)

    # Condition for right foot being vertically at or just above the block's top surface.
    # CRITICAL RULE: Use reasonable tolerances. 0.1m allows for slight penetration or being just above.
    right_foot_on_block_vertical = (right_foot_pos[:, 2] > block_top_z - 0.1)

    # Condition for pelvis stability on the block.
    # Checks if the pelvis is at the expected standing height relative to the block's top.
    # CRITICAL RULE: Use relative distances (absolute differences) for Z component.
    # CRITICAL RULE: Use reasonable tolerances. 0.2m allows for some height variation during standing.
    pelvis_stable_on_block = (torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block) < 0.2)

    # Combine all conditions for overall success.
    # All conditions must be met simultaneously for success.
    condition = left_foot_on_block_horizontal & left_foot_on_block_vertical & \
                right_foot_on_block_horizontal & right_foot_on_block_vertical & \
                pelvis_stable_on_block

    # Check success duration and save success states.
    # CRITICAL RULE: Always use check_success_duration and save_success_state.
    # Duration set to 1.0 seconds to ensure stable standing on the block.
    success = check_success_duration(env, condition, "Navigate_and_Jump_On_BlockCube", duration=1.0)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Navigate_and_Jump_On_BlockCube")

    return success

class SuccessTerminationCfg:
    # Define the success termination term using the implemented function.
    success = DoneTerm(func=Navigate_and_Jump_On_BlockCube_success)
