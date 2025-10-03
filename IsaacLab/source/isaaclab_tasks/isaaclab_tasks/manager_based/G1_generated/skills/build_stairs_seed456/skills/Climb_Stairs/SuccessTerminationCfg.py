
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Climb_Stairs_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Climb_Stairs skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot object using the approved pattern: env.scene["robot"]
    robot = env.scene["robot"]

    # Access the required robot parts using approved patterns: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate average foot position for robust tracking, handling batched environments
    # This is a tensor operation that works across all environments.
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2
    avg_foot_pos = torch.stack([avg_foot_pos_x, avg_foot_pos_y, avg_foot_pos_z], dim=1)

    # Access Object3 (Large Block) using approved pattern: env.scene['ObjectName'].data.root_pos_w
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # Hardcode object dimensions from the object configuration as required.
    # Requirement: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    object3_height = 0.9 # From object config: Object3 is 0.9m tall
    block_width = 1.0 # From object config: All blocks are 1m x 1m horizontally

    # Calculate target Z positions for feet and pelvis relative to Object3's top surface.
    # Object3's root_pos_w is its center. Top surface is center Z + half height.
    object3_top_surface_z = object3_pos[:, 2] + object3_height / 2

    # Target pelvis height: 0.7m above Object3's top surface. This is a relative offset.
    target_pelvis_z = object3_top_surface_z + 0.7

    # --- Success Criteria Measurements (using relative distances) ---
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.

    # 1. Horizontal distance (x and y) of average foot position to Object3's center
    # This checks if the feet are horizontally centered on Object3.
    horizontal_dist_feet_x = torch.abs(avg_foot_pos[:, 0] - object3_pos[:, 0])
    horizontal_dist_feet_y = torch.abs(avg_foot_pos[:, 1] - object3_pos[:, 1])

    # Condition for horizontal centering of feet on Object3
    # Threshold: within 0.5m (half of block width, plus a small tolerance). This is a reasonable tolerance.
    feet_horizontally_centered = (horizontal_dist_feet_x < 0.5) & (horizontal_dist_feet_y < 0.5)

    # 2. Vertical distance (z) of average foot position to Object3's top surface
    # This checks if the feet are vertically aligned with Object3's top surface.
    vertical_dist_feet_z = torch.abs(avg_foot_pos[:, 2] - object3_top_surface_z)

    # Condition for vertical alignment of feet on Object3
    # Threshold: within 0.15m. This is a reasonable tolerance.
    feet_vertically_aligned = (vertical_dist_feet_z < 0.15)

    # 3. Vertical distance (z) of pelvis position to target standing height above Object3's top surface
    # This checks if the robot's pelvis is at a stable standing height on Object3.
    vertical_dist_pelvis_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Condition for pelvis stability on Object3
    # Threshold: within 0.2m. This is a reasonable tolerance.
    pelvis_stable_on_obj3 = (vertical_dist_pelvis_z < 0.2)

    # Combine all conditions for overall success. All conditions must be met simultaneously.
    overall_success_condition = feet_horizontally_centered & feet_vertically_aligned & pelvis_stable_on_obj3

    # Check success duration and save success states.
    # Requirement: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, overall_success_condition, "Climb_Stairs", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Climb_Stairs")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Climb_Stairs_success)
