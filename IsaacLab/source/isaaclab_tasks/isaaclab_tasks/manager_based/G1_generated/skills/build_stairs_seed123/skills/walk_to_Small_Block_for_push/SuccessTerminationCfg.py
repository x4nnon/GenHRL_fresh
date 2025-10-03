
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_for_push_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block_for_push skill has been successfully completed.'''
    # Requirement 1: Access object positions using env.scene['ObjectName'].data.root_pos_w
    # Access Object1 (Small Block) position
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w

    # Requirement 2: Access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    # Access the robot and its pelvis position
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Requirement 6: There is no way to access the SIZE of an object.
    # Hardcode Object1 dimensions from the skill information (x=1m, y=1m, z=0.3m)
    object1_half_width_y = 0.5  # Half of Object1's Y dimension (1m / 2)
    object1_half_height = 0.15  # Half of Object1's Z dimension (0.3m / 2)

    # Define target relative offsets based on the success criteria plan and reward function context.
    # The robot should be adjacent to the Small Block, ready to push.
    # From reward function, robot_clearance_y = 0.2m.
    # Target Y position: Object1_center_y - (object1_half_width_y + robot_clearance_y)
    # This means (object1_pos_y - pelvis_pos_y) should be equal to (0.5 + 0.2) = 0.7m
    target_y_offset_from_block_center = object1_half_width_y + 0.2

    # Target Z position: Pelvis should be at a stable standing height relative to the block.
    # The plan suggests 0.4m above Object1's center. Object1's center is at its half height (0.15m).
    # So, target pelvis Z = object1_pos_z + 0.4m.
    target_z_offset_from_block_center = 0.4

    # Requirement 0 & 3: Success criteria MUST ONLY use relative distances between objects and robot parts.
    # Requirement 5: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPARATELY.
    # Calculate the distance components between pelvis and the target relative position to Object1.
    # X-distance: Pelvis X should align with Object1 X.
    distance_x_pelvis_to_object1_x = torch.abs(pelvis_pos[:, 0] - object1_pos[:, 0])

    # Y-distance: Pelvis Y should be at the target offset behind Object1's Y center.
    # We check if the difference (Object1_Y - Pelvis_Y) is close to the target offset.
    distance_y_pelvis_to_object1_y_target = torch.abs((object1_pos[:, 1] - pelvis_pos[:, 1]) - target_y_offset_from_block_center)

    # Z-distance: Pelvis Z should be at the target height relative to Object1's Z center.
    # We check if Pelvis_Z is close to (Object1_Z + target_z_offset_from_block_center).
    distance_z_pelvis_to_object1_z_target = torch.abs(pelvis_pos[:, 2] - (object1_pos[:, 2] + target_z_offset_from_block_center))

    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds for success.
    # Thresholds for success are set based on reasonable tolerances for being "near" the block.
    # The plan suggests 0.2m for all axes, which is a lenient and reasonable tolerance.
    threshold_x = 0.2
    threshold_y = 0.2
    threshold_z = 0.2

    # Combine conditions for overall success. All conditions must be met.
    condition = (distance_x_pelvis_to_object1_x < threshold_x) & \
                (distance_y_pelvis_to_object1_y_target < threshold_y) & \
                (distance_z_pelvis_to_object1_z_target < threshold_z)

    # Requirement 6: ALWAYS use check_success_duration and save_success_state.
    # Check if the success condition has been maintained for a duration of 0.5 seconds.
    success = check_success_duration(env, condition, "walk_to_Small_Block_for_push", duration=0.5)

    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block_for_push")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_for_push_success)
