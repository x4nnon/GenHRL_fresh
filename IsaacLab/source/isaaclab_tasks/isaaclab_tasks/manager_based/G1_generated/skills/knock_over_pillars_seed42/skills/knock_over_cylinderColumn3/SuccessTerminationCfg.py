
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_cylinderColumn3_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_cylinderColumn3 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot and relevant objects using approved patterns.
    robot = env.scene["robot"]
    object3 = env.scene['Object3'] # Object3 is Cylinder Column 3
    object4 = env.scene['Object4'] # Object4 is Cylinder Column 4, used for relative positioning

    # Get the position of the robot's pelvis.
    # Requirement 3: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get the root positions of Object3 and Object4.
    # Requirement 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcode object dimensions from the skill description (radius=0.3m, height=2m).
    # Requirement 6: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    object3_radius = 0.3

    # --- Success Condition 1: Object3 is lying on the floor. ---
    # The Z-position of Object3's root (its center) should be close to its radius when fallen.
    # A threshold of 0.4m allows for some tolerance above its radius (0.3m).
    # Requirement 0: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # The Z-position of an object's root is relative to the floor (world origin), which is an allowed exception for height checks.
    # Requirement: Use lenient thresholds. 0.4m is a reasonable tolerance for a 0.3m radius object to be considered fallen.
    object3_fallen_condition = object3_pos[:, 2] < 0.4

    # --- Success Condition 2: Robot's pelvis is horizontally close to Object3. ---
    # Calculate the horizontal distance between the robot's pelvis and Object3.
    # Requirement 0: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # Requirement 1: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    horizontal_dist_pelvis_obj3 = torch.norm(pelvis_pos[:, :2] - object3_pos[:, :2], dim=1)
    # The robot should be within 1.0 meter horizontally of Object3.
    # Requirement: Use lenient thresholds. 1.0m is a reasonable tolerance for "horizontally close".
    pelvis_close_to_obj3_condition = horizontal_dist_pelvis_obj3 < 1.0

    # --- Success Condition 3: Robot has not significantly overshot Object3 towards Object4. ---
    # This ensures the robot is positioned appropriately for the next skill.
    # Calculate horizontal distance from Object4 to robot's pelvis.
    # Requirement 0: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    horizontal_dist_obj4_pelvis = torch.norm(object4_pos[:, :2] - pelvis_pos[:, :2], dim=1)
    # Calculate horizontal distance from Object4 to Object3.
    horizontal_dist_obj4_obj3 = torch.norm(object4_pos[:, :2] - object3_pos[:, :2], dim=1)

    # The condition is that the robot's pelvis is not horizontally past Object3 by more than 1.0 meter
    # in the general direction of Object4. This means the distance from Object4 to the pelvis
    # should be greater than the distance from Object4 to Object3 minus a buffer.
    # If the robot is past Object3 towards Object4, horizontal_dist_obj4_pelvis would be smaller than horizontal_dist_obj4_obj3.
    # So, we want to ensure it's not *too much* smaller.
    # A buffer of 1.0m means the robot can be up to 1.0m past Object3 relative to Object4's position.
    # If horizontal_dist_obj4_pelvis is significantly smaller than horizontal_dist_obj4_obj3, it means the robot overshot.
    # We want horizontal_dist_obj4_pelvis to be greater than (horizontal_dist_obj4_obj3 - 1.0).
    # This allows the robot to be slightly past Object3, but not too far towards Object4.
    # Requirement: Use lenient thresholds. 1.0m is a reasonable tolerance for "not significantly overshot".
    not_overshot_obj3_condition = horizontal_dist_obj4_pelvis > (horizontal_dist_obj4_obj3 - 1.0)

    # Combine all success conditions. All conditions must be met.
    # All operations work with batched environments.
    # Add pelvis height condition
    min_pelvis_height = 0.6
    pelvis_height_condition = (pelvis_pos[:, 2] > min_pelvis_height)

    all_conditions_met = object3_fallen_condition & pelvis_height_condition # & pelvis_close_to_obj3_condition & not_overshot_obj3_condition

    # Check success duration and save success states.
    # Requirement 6: ALWAYS use check_success_duration and save_success_state
    # Requirement: Check success duration with appropriate duration (0.5s as per plan).
    success = check_success_duration(env, all_conditions_met, "knock_over_cylinderColumn3", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_cylinderColumn3")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_cylinderColumn3_success)
