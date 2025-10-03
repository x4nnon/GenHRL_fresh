
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def navigate_through_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the navigate_through_doorway skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Object3 is the Small Block for the robot to walk to.
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w
    object3_pos_x = object3_pos[:, 0]
    object3_pos_y = object3_pos[:, 1]

    # Define the doorway exit y-position.
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # The task description states Object3 is 2m past the doorway in the y-axis.
    # Therefore, the doorway exit y-coordinate is 2m before Object3's y-coordinate.
    doorway_exit_y = object3_pos_y - 2.0

    # Condition 1: Robot's pelvis must have passed the doorway.
    # Requirement: All operations must work with batched environments.
    pelvis_past_doorway_condition = (pelvis_pos_y > doorway_exit_y)

    # Condition 2: Robot's pelvis must be close to Object3 (Small Block).
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.
    distance_x = torch.abs(object3_pos_x - pelvis_pos_x)
    distance_y = torch.abs(object3_pos_y - pelvis_pos_y)
    
    # For the z-component, we want the robot to be standing upright near the block.
    # Requirement: z is the only absolute position allowed, used sparingly.
    # A target pelvis height of 0.7m is consistent with standing and the reward function.
    pelvis_target_z = 0.7
    distance_z = torch.abs(pelvis_pos_z - pelvis_target_z)

    # Requirement: Use lenient thresholds.
    # Thresholds for proximity to Object3.
    # These thresholds are chosen to be lenient enough for successful completion while ensuring proximity.
    threshold_x = 0.4 # Within 40cm in x-direction
    threshold_y = 0.4 # Within 40cm in y-direction
    threshold_z = 0.2 # Within 20cm in z-height from target

    close_to_object3_x = (distance_x < threshold_x)
    close_to_object3_y = (distance_y < threshold_y)
    close_to_object3_z = (distance_z < threshold_z)

    # Combine all conditions: robot must pass the doorway AND be close to the block in all relevant dimensions.
    # Requirement: All operations must work with batched environments.
    condition = pelvis_past_doorway_condition & \
                close_to_object3_x & \
                close_to_object3_y & \
                close_to_object3_z

    # Requirement: ALWAYS use check_success_duration and save_success_state.
    # Duration of 0.5 seconds ensures the robot maintains the successful state for a short period.
    success = check_success_duration(env, condition, "navigate_through_doorway", duration=0.5)
    
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "navigate_through_doorway")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=navigate_through_doorway_success)
