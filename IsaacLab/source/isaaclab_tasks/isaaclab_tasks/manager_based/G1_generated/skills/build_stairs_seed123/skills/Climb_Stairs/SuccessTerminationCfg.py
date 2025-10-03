
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
    # 1. Get robot parts
    # Accessing the robot object from the scene. (Requirement 3)
    robot = env.scene["robot"]

    # Getting indices for specific robot body parts. (Requirement 3)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Getting the world positions of the robot parts. (Requirement 3)
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # 2. Get object position
    # Accessing Object3 (Large Block) from the scene. (Requirement 2, 5)
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w

    # 3. Hardcode object dimensions and tolerances from the success criteria plan. (Requirement 6)
    # These values are read from the object configuration and reward function context.
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    block_xy_size = 1.0
    block3_height = 0.9
    pelvis_target_offset = 0.7
    foot_on_block_z_tolerance = 0.1
    pelvis_z_tolerance = 0.15

    # Calculate the top Z-coordinate of Object3.
    object3_top_z = object3_pos[:, 2] + block3_height

    # 4. Check success conditions based on relative distances. (Requirement 1)

    # Condition for left foot being on Object3:
    # X-distance check: Left foot's X position relative to Object3's center X. (Requirement 1, Rule 5 under Position & Motion Rules)
    left_foot_on_obj3_x = torch.abs(left_foot_pos[:, 0] - object3_pos[:, 0]) < (block_xy_size / 2) + 0.1
    # Y-distance check: Left foot's Y position relative to Object3's center Y. (Requirement 1, Rule 5 under Position & Motion Rules)
    left_foot_on_obj3_y = torch.abs(left_foot_pos[:, 1] - object3_pos[:, 1]) < (block_xy_size / 2) + 0.1
    # Z-distance check: Left foot's Z position relative to Object3's top surface. (Requirement 1, Rule 5 under Position & Motion Rules)
    left_foot_on_obj3_z = left_foot_pos[:, 2] > (object3_top_z - foot_on_block_z_tolerance)
    # Combine all conditions for the left foot.
    left_foot_on_obj3 = left_foot_on_obj3_x & left_foot_on_obj3_y & left_foot_on_obj3_z

    # Condition for right foot being on Object3:
    # X-distance check: Right foot's X position relative to Object3's center X. (Requirement 1, Rule 5 under Position & Motion Rules)
    right_foot_on_obj3_x = torch.abs(right_foot_pos[:, 0] - object3_pos[:, 0]) < (block_xy_size / 2) + 0.1
    # Y-distance check: Right foot's Y position relative to Object3's center Y. (Requirement 1, Rule 5 under Position & Motion Rules)
    right_foot_on_obj3_y = torch.abs(right_foot_pos[:, 1] - object3_pos[:, 1]) < (block_xy_size / 2) + 0.1
    # Z-distance check: Right foot's Z position relative to Object3's top surface. (Requirement 1, Rule 5 under Position & Motion Rules)
    right_foot_on_obj3_z = right_foot_pos[:, 2] > (object3_top_z - foot_on_block_z_tolerance)
    # Combine all conditions for the right foot.
    right_foot_on_obj3 = right_foot_on_obj3_x & right_foot_on_obj3_y & right_foot_on_obj3_z

    # Condition for pelvis being at a stable height relative to Object3's top surface. (Requirement 1, Rule 5 under Position & Motion Rules)
    # Target pelvis Z is Object3's top Z + target_pelvis_offset.
    pelvis_target_z = object3_top_z + pelvis_target_offset
    # Check if pelvis Z is within tolerance of the target Z.
    pelvis_stable_height = torch.abs(pelvis_pos[:, 2] - pelvis_target_z) < pelvis_z_tolerance

    # Overall success condition: Both feet are on Object3 AND pelvis is at a stable height.
    condition = left_foot_on_obj3 & right_foot_on_obj3 & pelvis_stable_height

    # 5. Check duration and save success states. (Requirement 6)
    # The duration is set to 0.5 seconds as per the success criteria plan.
    success = check_success_duration(env, condition, "Climb_Stairs", duration=0.5)

    # Save success states for environments that have met the success criteria. (Requirement 6)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Climb_Stairs")

    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=Climb_Stairs_success)
