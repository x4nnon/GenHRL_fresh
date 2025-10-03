
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_first_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_first_0_5m_cubed_block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Access the required objects using approved patterns
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement: Access objects directly - objects should always exist in the scene
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform' (used for conceptual alignment, not direct position in success)

    # 2. Access the required robot part(s) using approved patterns
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Extract components for clarity and separate checks
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # 3. Define object dimensions and target clearances
    # Requirement: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    block_half_size = 0.25 # 0.5m cubed block, so half size is 0.25m
    clearance_behind_block = 0.15 # Desired distance behind the block for the robot's pelvis

    # 4. Calculate target positions relative to Object1
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # The robot should be positioned directly behind Object1, on the side opposite Object4.
    # Assuming Object4 (platform) is generally in the positive X direction relative to Object1 for pushing.
    # Thus, the robot should be on the negative X side of Object1.
    # Target x-position: slightly behind Object1's edge (Object1.x - block_half_size - clearance)
    # Requirement: YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN
    target_x_pos = object1.data.root_pos_w[:, 0] - block_half_size - clearance_behind_block
    # Target y-position: Aligned with the block's y-center
    target_y_pos = object1.data.root_pos_w[:, 1]
    # Target z-position: Desired stable pelvis height. This is an absolute height, which is allowed for stability.
    # Requirement: z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.
    target_z_pos = 0.7

    # 5. Calculate distances to the target position for each component
    # Requirement: All success criteria must only be based on relative distances between objects and robot parts
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.
    distance_x = torch.abs(pelvis_pos_x - target_x_pos)
    distance_y = torch.abs(pelvis_pos_y - target_y_pos)
    distance_z = torch.abs(pelvis_pos_z - target_z_pos)

    # 6. Define thresholds for each component
    # Requirement: USE LENIENT THRESHOLDS for secondary conditions, strict for primary.
    # Requirement: REASONABLE TOLERANCES (typically 0.05-0.1m for distances)
    # X-axis is primary for approach, so slightly stricter.
    threshold_x = 0.10 # Robot's pelvis should be within 10cm of the target x-position
    threshold_y = 0.15 # Robot's pelvis should be within 15cm of the target y-position (aligned with block)
    threshold_z = 0.10 # Robot's pelvis height should be within 10cm of the desired stable height

    # 7. Check success conditions for each component
    # Requirement: All operations must work with batched environments
    success_x = distance_x < threshold_x
    success_y = distance_y < threshold_y
    success_z = distance_z < threshold_z

    # 8. Combine all conditions for overall success
    success_condition = success_x & success_y & success_z

    # 9. Check duration and save success states
    # Requirement: ALWAYS use check_success_duration and save_success_state
    # Duration set to 0.5 seconds to ensure stability in the final position.
    success = check_success_duration(env, success_condition, "approach_first_0_5m_cubed_block", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_first_0_5m_cubed_block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=approach_first_0_5m_cubed_block_success)
