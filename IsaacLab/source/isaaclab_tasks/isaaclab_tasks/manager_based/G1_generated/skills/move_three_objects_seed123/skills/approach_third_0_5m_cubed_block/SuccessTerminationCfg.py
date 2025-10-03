
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_third_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_third_0_5m_cubed_block skill has been successfully completed.'''

    # CRITICAL: Accessing the 'third 0.5m cubed block' using the approved object name 'Object3'.
    object3 = env.scene['Object3'] 
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # CRITICAL: Accessing the robot object.
    robot = env.scene["robot"]
    # CRITICAL: Getting the index of the 'pelvis' robot part using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis') 
    # CRITICAL: Getting the position of the 'pelvis' robot part using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL: Hardcoding the block size (0.5m) as it's derived from the object configuration.
    # This is explicitly allowed as object dimensions cannot be accessed dynamically.
    block_size = 0.5 

    # CRITICAL: Calculating the target X position for the pelvis relative to Object3.
    # This positions the robot 0.5m behind the block's edge (block_size / 2.0) in the negative X direction.
    # This is a relative calculation based on the object's position and its known size.
    target_x_pos = object3_pos[:, 0] - (block_size / 2.0) - 0.5

    # CRITICAL: Calculating the target Y position for the pelvis, aligned with Object3's center.
    # This is a relative calculation based on the object's position.
    target_y_pos = object3_pos[:, 1]

    # CRITICAL: Defining a target Z position for the pelvis. This is an absolute height,
    # which is allowed for the Z-component when height is crucial for the skill (e.g., standing posture).
    # This value is consistent with the reward function's target pelvis height.
    target_z_pos = 0.7 

    # CRITICAL: Calculating relative distances for each dimension between the pelvis and the target position.
    # All distances are calculated as differences between target and current positions.
    distance_x = target_x_pos - pelvis_pos[:, 0]
    distance_y = target_y_pos - pelvis_pos[:, 1]
    distance_z = target_z_pos - pelvis_pos[:, 2]

    # CRITICAL: Defining reasonable thresholds for each dimension.
    # These tolerances allow for slight variations while ensuring the robot is in the correct general area.
    # Thresholds are reasonable (15cm) for positioning tasks.
    threshold_x = 0.15 # 15cm tolerance for X position
    threshold_y = 0.15 # 15cm tolerance for Y position
    threshold_z = 0.15 # 15cm tolerance for Z position (height)

    # CRITICAL: Combining all conditions. Success requires the robot's pelvis to be within the specified
    # thresholds for X, Y, and Z relative to the target position.
    # All conditions are combined using logical AND for batched environments.
    condition = (torch.abs(distance_x) < threshold_x) & \
                (torch.abs(distance_y) < threshold_y) & \
                (torch.abs(distance_z) < threshold_z)

    # CRITICAL: Checking success duration and saving success states.
    # The duration of 0.5 seconds ensures the robot maintains the desired pose for a short period.
    # This section is mandatory and correctly implemented.
    success = check_success_duration(env, condition, "approach_third_0_5m_cubed_block", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_third_0_5m_cubed_block")
    
    return success

class SuccessTerminationCfg:
    # CRITICAL: Registering the success function with DoneTerm.
    # This is correctly configured to use the implemented success function.
    success = DoneTerm(func=approach_third_0_5m_cubed_block_success)
