
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def climb_stairs_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the climb_stairs skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # ABSOLUTE REQUIREMENT: Object dimensions must be hardcoded from the description.
    # Skill Description: A small block measuring x=1m y=1m and z=0.3m.
    # A medium block measuring x=1m y=1m and z=0.6m.
    # A large block measuring x=1m y=1m and z=0.9m.
    OBJECT3_HEIGHT = 0.9
    BLOCK_X_SIZE = 1.0
    BLOCK_Y_SIZE = 1.0
    PELVIS_STABLE_HEIGHT_ABOVE_BLOCK = 0.7 # Target pelvis height relative to block top

    # ABSOLUTE REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # CRITICAL OBJECT NAMING: Using Object3 as per the object configuration for the large block.
    object3 = env.scene['Object3']

    # ABSOLUTE REQUIREMENT: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: Handle tensor operations correctly for batched environments.
    # Calculate average foot position for checking if feet are on the block.
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # ABSOLUTE REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # CRITICAL RULE: Consider X, Y, Z components separately.
    # Calculate distances for feet to Object3 (the large block).
    # Distance in X-Y plane to the center of Object3.
    dist_feet_x_obj3 = torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0])
    dist_feet_y_obj3 = torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1])
    # Z-distance to the top surface of Object3.
    # Object's root_pos_w is its center, so add half its height to get the top surface.
    dist_feet_z_obj3 = torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2))

    # Calculate distances for pelvis to Object3.
    # Distance in X-Y plane to the center of Object3.
    dist_pelvis_x_obj3 = torch.abs(pelvis_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_pelvis_y_obj3 = torch.abs(pelvis_pos[:, 1] - object3.data.root_pos_w[:, 1])
    # Z-distance to target pelvis height above Object3's top surface.
    # Target pelvis Z is block's center Z + half block height + desired stable height above block.
    target_pelvis_z_on_obj3 = object3.data.root_pos_w[:, 2] + OBJECT3_HEIGHT / 2 + PELVIS_STABLE_HEIGHT_ABOVE_BLOCK
    dist_pelvis_z_obj3 = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3)

    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds.
    # Using thresholds relative to block size or small, reasonable tolerances.
    # Feet must be horizontally within the block's top surface area.
    # A small buffer (0.1m) is subtracted from half the block size to ensure feet are well within.
    feet_on_obj3_x = dist_feet_x_obj3 < (BLOCK_X_SIZE / 2) - 0.1 # 0.4m from center
    feet_on_obj3_y = dist_feet_y_obj3 < (BLOCK_Y_SIZE / 2) - 0.1 # 0.4m from center
    # Feet must be vertically close to the top surface of the block.
    # A tolerance of 0.2m allows for slight variations in foot placement on the surface.
    feet_on_obj3_z = dist_feet_z_obj3 < 0.2

    # Pelvis must be horizontally stable over the block.
    # A tolerance of 0.3m from the center of the block.
    pelvis_stable_x = dist_pelvis_x_obj3 < 0.3
    pelvis_stable_y = dist_pelvis_y_obj3 < 0.3
    # Pelvis must be at the target stable height above the block.
    # A tolerance of 0.2m for the vertical height.
    pelvis_stable_z = dist_pelvis_z_obj3 < 0.2

    # Combine all conditions for success. All conditions must be met.
    success_condition = feet_on_obj3_x & feet_on_obj3_y & feet_on_obj3_z & \
                        pelvis_stable_x & pelvis_stable_y & pelvis_stable_z

    # ABSOLUTE REQUIREMENT: ALWAYS use check_success_duration and save_success_state.
    # Check if the success condition is maintained for a duration of 1.0 seconds.
    success = check_success_duration(env, success_condition, "climb_stairs", duration=1.0)

    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "climb_stairs")

    return success

class SuccessTerminationCfg:
    # ABSOLUTE REQUIREMENT: Do not implement reward functions here, only success functions.
    # Define the success termination using the implemented success function.
    success = DoneTerm(func=climb_stairs_success)
