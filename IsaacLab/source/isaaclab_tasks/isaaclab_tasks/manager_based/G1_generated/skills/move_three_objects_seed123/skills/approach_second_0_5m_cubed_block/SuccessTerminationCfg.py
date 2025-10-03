
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_second_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_second_0_5m_cubed_block skill has been successfully completed.'''

    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w # Position of 'second 0.5m cubed block'
    object4_pos = env.scene['Object4'].data.root_pos_w # Position of 'platform'

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Getting the index of the required robot part
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting the position of the robot's pelvis

    # CRITICAL RULE: Hardcode dimensions from object config, DO NOT access from object.
    # Object2 dimensions (0.5m cubed block)
    object2_half_depth = 0.5 / 2.0 # Assuming 0.5m is the side length of the cube
    target_clearance = 0.1 # Desired distance from the block's face for pushing

    # CRITICAL RULE: ALL success criteria MUST ONLY use relative distances between objects and robot parts.
    # Calculate the vector from Object2 to Object4 to determine the pushing direction.
    # This ensures the robot positions itself on the side opposite Object4 dynamically.
    object2_to_object4_vec = object4_pos[:, :2] - object2_pos[:, :2] # Only consider X-Y plane for direction

    # Normalize the direction vector to get a unit vector for alignment.
    direction_magnitude = torch.norm(object2_to_object4_vec, dim=1, keepdim=True)
    # Add a small epsilon to avoid division by zero if objects are at the same spot.
    direction_magnitude = torch.where(direction_magnitude == 0, torch.tensor(1e-6, device=env.device), direction_magnitude)
    norm_dir = object2_to_object4_vec / direction_magnitude # Shape: [num_envs, 2]

    # Calculate the desired pelvis position relative to Object2 based on the pushing direction.
    # The robot should be on the opposite side of Object2 from Object4.
    # So, we move *against* the normalized direction vector from Object2's center.
    desired_pelvis_xy_w = object2_pos[:, :2] - norm_dir * (object2_half_depth + target_clearance)

    # Calculate distance components to the desired X-Y position.
    # CRITICAL RULE: Use torch.abs for absolute distances.
    dist_x = torch.abs(desired_pelvis_xy_w[:, 0] - pelvis_pos[:, 0])
    dist_y = torch.abs(desired_pelvis_xy_w[:, 1] - pelvis_pos[:, 1])

    # Target pelvis height for stability. Z is the only absolute position allowed.
    target_pelvis_z = 0.7
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Define success thresholds. CRITICAL RULE: Use lenient thresholds.
    xy_tolerance = 0.15 # meters, for X-Y position
    z_tolerance = 0.1 # meters, for Z height

    # Check if the robot's pelvis is within the desired X-Y position and Z height
    success_xy = (dist_x < xy_tolerance) & (dist_y < xy_tolerance)
    success_z = (dist_z < z_tolerance)

    # Combine all conditions for overall success
    condition = success_xy & success_z

    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, condition, "approach_second_0_5m_cubed_block", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_second_0_5m_cubed_block")

    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=approach_second_0_5m_cubed_block_success)
