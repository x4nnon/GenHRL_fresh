
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_first_0_5m_cubed_block_towards_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_first_0_5m_cubed_block_towards_platform skill has been successfully completed.'''

    # Access the required objects using approved patterns
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Access the required robot parts using approved patterns
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcode object dimensions from the object configuration
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Read from config and hardcode.
    # From object config: "Object1": "first 0.5m cubed block" (size 0.5m cubed) - not directly used in success criteria
    # From object config: "Object4": "platform" (x=2m, y=2m, z=0.001)
    platform_x_dim = 2.0
    platform_y_dim = 2.0
    platform_z_height = 0.001

    # Calculate 2D distance between Object1 and Object4
    # CRITICAL RULE: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # This measures how close Object1 is to Object4 in the horizontal plane.
    distance_x_obj1_obj4 = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y_obj1_obj4 = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_2d_obj1_obj4 = torch.sqrt(distance_x_obj1_obj4**2 + distance_y_obj1_obj4**2)

    # Calculate 2D distance between robot pelvis and Object1
    # This measures how close the robot is to the block it's pushing.
    distance_x_pelvis_obj1 = pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0]
    distance_y_pelvis_obj1 = pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1]
    distance_2d_pelvis_obj1 = torch.sqrt(distance_x_pelvis_obj1**2 + distance_y_pelvis_obj1**2)

    # Check if robot feet are NOT on the platform
    # This ensures the robot doesn't prematurely step onto the platform.
    # Left foot check: relative distance to platform center in XY, and absolute Z height.
    dist_left_foot_obj4_x_abs = torch.abs(left_foot_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_left_foot_obj4_y_abs = torch.abs(left_foot_pos[:, 1] - object4.data.root_pos_w[:, 1])
    # Check if foot is horizontally over the platform area (with a small buffer)
    is_left_foot_over_platform_xy = (dist_left_foot_obj4_x_abs < (platform_x_dim / 2.0 - 0.1)) & \
                                    (dist_left_foot_obj4_y_abs < (platform_y_dim / 2.0 - 0.1))
    # Check if foot is at or below the platform's surface (with a small clearance)
    is_left_foot_on_platform_z = (left_foot_pos[:, 2] < (platform_z_height + 0.1)) # 0.1m clearance above platform surface
    # Condition for left foot NOT being on the platform
    left_foot_not_on_platform = ~(is_left_foot_over_platform_xy & is_left_foot_on_platform_z)

    # Right foot check (same logic as left foot)
    dist_right_foot_obj4_x_abs = torch.abs(right_foot_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_right_foot_obj4_y_abs = torch.abs(right_foot_pos[:, 1] - object4.data.root_pos_w[:, 1])
    is_right_foot_over_platform_xy = (dist_right_foot_obj4_x_abs < (platform_x_dim / 2.0 - 0.1)) & \
                                     (dist_right_foot_obj4_y_abs < (platform_y_dim / 2.0 - 0.1))
    is_right_foot_on_platform_z = (right_foot_pos[:, 2] < (platform_z_height + 0.1))
    # Condition for right foot NOT being on the platform
    right_foot_not_on_platform = ~(is_right_foot_over_platform_xy & is_right_foot_on_platform_z)

    # Define success conditions based on the skill description:
    # 1. Block moved significantly closer but not fully on the platform.
    # Initial distance is ~4m. Target range 0.9m to 2.1m from platform center.
    # CRITICAL RULE: Use lenient thresholds for secondary conditions.
    block_in_target_range = (distance_2d_obj1_obj4 > 0.9) & (distance_2d_obj1_obj4 < 2.1)

    # 2. Robot is near the block.
    # Pelvis to block 2D distance less than 1.5m.
    robot_near_block = (distance_2d_pelvis_obj1 < 1.5)

    # 3. Robot is not on the platform.
    robot_not_on_platform = left_foot_not_on_platform & right_foot_not_on_platform

    # Combine all conditions for overall success
    condition = block_in_target_range & robot_near_block & robot_not_on_platform

    # Check duration and save success states
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state
    # Duration set to 1.0 seconds as per the success criteria plan.
    success = check_success_duration(env, condition, "push_first_0_5m_cubed_block_towards_platform", duration=1.0)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_first_0_5m_cubed_block_towards_platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_first_0_5m_cubed_block_towards_platform_success)
