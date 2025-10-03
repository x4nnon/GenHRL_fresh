
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_cube_for_robot_to_push_1_onto_platform_for_cubes_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_cube_for_robot_to_push_1_onto_platform_for_cubes skill has been successfully completed.'''
    # Access the robot and relevant objects using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # Object1 is 'Cube for robot to push'
    object4 = env.scene['Object4'] # Object4 is 'Platform for cubes'

    # Get positions of Object1 and Object4 using approved patterns
    object1_pos = object1.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Get positions of robot's feet using approved patterns (body_names.index)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded dimensions from the object configuration description, as per requirements.
    # Object1: Cube for robot to push, size = [0.5, 0.5, 0.5]
    object1_half_size_x = 0.25
    object1_half_size_y = 0.25
    object1_half_size_z = 0.25
    # Object4: Platform for cubes, size = [2.0, 2.0, 0.001]
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_height = 0.001

    # --- Success Condition 1: Object1 is fully positioned on Object4 ---

    # Calculate relative horizontal distances between Object1 and Object4 centers.
    # This uses relative distances as required by the prompt.
    dist_obj1_obj4_x = torch.abs(object1_pos[:, 0] - object4_pos[:, 0])
    dist_obj1_obj4_y = torch.abs(object1_pos[:, 1] - object4_pos[:, 1])

    # Check if Object1's center is within Object4's horizontal bounds, considering their half-dimensions.
    # A small tolerance of 0.05m is added for leniency, as per the success criteria plan.
    # This ensures the entire cube is on the platform, not just its center.
    is_obj1_on_platform_x = (dist_obj1_obj4_x <= (object4_half_size_x - object1_half_size_x + 0.05))
    is_obj1_on_platform_y = (dist_obj1_obj4_y <= (object4_half_size_y - object1_half_size_y + 0.05))
    
    # Calculate the target Z position for Object1's center when it's resting on Object4's top surface.
    # This is a relative calculation based on Object4's Z position and dimensions.
    target_obj1_z = object4_pos[:, 2] + object4_height / 2.0 + object1_half_size_z
    
    # Check if Object1's Z position is at or slightly above the target Z.
    # A small tolerance of 0.05m is used, as per the success criteria plan.
    is_obj1_on_platform_z = (torch.abs(object1_pos[:, 2] - target_obj1_z) <= 0.05)

    # Combine all conditions for Object1 being on the platform using tensor operations.
    object1_on_platform_condition = is_obj1_on_platform_x & is_obj1_on_platform_y #& is_obj1_on_platform_z

    # --- Success Condition 2: Robot's feet are NOT on Object4 ---

    # Calculate relative horizontal distances for left foot to Object4's center.
    dist_left_foot_obj4_x = torch.abs(left_foot_pos[:, 0] - object4_pos[:, 0])
    dist_left_foot_obj4_y = torch.abs(left_foot_pos[:, 1] - object4_pos[:, 1])

    # Check if left foot is horizontally within Object4's bounds (with a small tolerance).
    # The tolerance is subtracted to make the "not on platform" check more strict.
    is_left_foot_horizontally_on_platform = (dist_left_foot_obj4_x < (object4_half_size_x - 0.05)) & \
                                             (dist_left_foot_obj4_y < (object4_half_size_y - 0.05))
    
    # Check if left foot is vertically above Object4's top surface (with a small tolerance).
    is_left_foot_vertically_on_platform = (left_foot_pos[:, 2] > (object4_pos[:, 2] + object4_height / 2.0 - 0.05))

    # Left foot is considered "on platform" if it's horizontally and vertically within bounds.
    is_left_foot_on_platform = is_left_foot_horizontally_on_platform & is_left_foot_vertically_on_platform

    # Calculate relative horizontal distances for right foot to Object4's center.
    dist_right_foot_obj4_x = torch.abs(right_foot_pos[:, 0] - object4_pos[:, 0])
    dist_right_foot_obj4_y = torch.abs(right_foot_pos[:, 1] - object4_pos[:, 1])

    # Check if right foot is horizontally within Object4's bounds (with a small tolerance).
    is_right_foot_horizontally_on_platform = (dist_right_foot_obj4_x < (object4_half_size_x - 0.05)) & \
                                              (dist_right_foot_obj4_y < (object4_half_size_y - 0.05))
    
    # Check if right foot is vertically above Object4's top surface (with a small tolerance).
    is_right_foot_vertically_on_platform = (right_foot_pos[:, 2] > (object4_pos[:, 2] + object4_height / 2.0 - 0.05))

    # Right foot is considered "on platform" if it's horizontally and vertically within bounds.
    is_right_foot_on_platform = is_right_foot_horizontally_on_platform & is_right_foot_vertically_on_platform

    # Robot is NOT on the platform if NEITHER foot is on the platform.
    # This uses logical NOT and OR operations on tensors.
    robot_not_on_platform_condition = ~(is_left_foot_on_platform | is_right_foot_on_platform)

    # robot pelvis z is near 0.7
    robot = env.scene["robot"]
    robot_pelvis_z = robot.data.body_pos_w[:, robot.body_names.index('pelvis')][:, 2]
    is_robot_pelvis_near_target_z = torch.abs(robot_pelvis_z - 0.7) <= 0.1

    # --- Combine all success conditions ---
    # The overall success requires Object1 to be on the platform AND the robot not to be on the platform.
    overall_success_condition = object1_on_platform_condition & is_robot_pelvis_near_target_z # & robot_not_on_platform_condition

    # Check duration and save success states using the required functions.
    # The duration is set to 0.5 seconds as specified in the plan.
    success = check_success_duration(env, overall_success_condition, "push_cube_for_robot_to_push_1_onto_platform_for_cubes", duration=0.5)
    
    # Save success states for environments that have met the criteria for the required duration.
    # This loop handles batched environments correctly.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_cube_for_robot_to_push_1_onto_platform_for_cubes")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_cube_for_robot_to_push_1_onto_platform_for_cubes_success)
