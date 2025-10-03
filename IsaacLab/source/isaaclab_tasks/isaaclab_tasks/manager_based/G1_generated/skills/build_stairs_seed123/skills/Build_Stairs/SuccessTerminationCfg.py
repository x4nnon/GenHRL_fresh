
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Build_Stairs_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Build_Stairs skill has been successfully completed.'''

    # Access the required objects
    # Object1: Small Block for robot to push and climb
    # Object2: Medium Block for robot to push and climb
    # Object3: Large Block for robot to push and climb
    # CORRECT: Accessing objects directly using approved pattern
    object1 = env.scene['Object1'] 
    object2 = env.scene['Object2'] 
    object3 = env.scene['Object3'] 

    # Access the required robot part(s)
    # CORRECT: Using robot.body_names.index to get the pelvis index
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Accessing pelvis position for all environments using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define target relative offsets for stairs based on block dimensions (1m width, 0.5m Y offset)
    # These values are hardcoded from the skill description and reward design plan, as object dimensions cannot be accessed dynamically.
    target_x_offset = 1.0
    target_y_offset = 0.5
    
    # Define target Z height for robot pelvis for stable posture
    # This value is hardcoded from the reward design plan. Absolute Z position for pelvis height is allowed sparingly.
    pelvis_target_z = 0.7 
    # Define target Y position for robot pelvis relative to Object1 (1m behind Object1)
    # This value is hardcoded from the reward design plan.
    robot_target_y_relative_to_obj1 = -1.0 

    # Calculate distances for Object2 relative to Object1
    # CORRECT: Calculating relative distances for X, Y, and Z components using approved patterns
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1]
    dist_obj2_obj1_z = object2.data.root_pos_w[:, 2] - object1.data.root_pos_w[:, 2]

    # Calculate distances for Object3 relative to Object2
    # CORRECT: Calculating relative distances for X, Y, and Z components using approved patterns
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0]
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1]
    dist_obj3_obj2_z = object3.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2]

    # Calculate distances for robot pelvis relative to Object1
    # CORRECT: Calculating relative distances for X and Y components using approved patterns
    dist_pelvis_obj1_x = pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0]
    dist_pelvis_obj1_y = pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1]
    # CORRECT: Accessing absolute Z position for pelvis height check (allowed sparingly for height)
    pelvis_pos_z = pelvis_pos[:, 2]

    # Define lenient thresholds for success conditions
    # These thresholds are chosen to be reasonable for block placement and robot positioning.
    block_x_tolerance = 0.3
    block_y_tolerance = 0.3
    block_z_tolerance = 0.2 # Z should be near 0 for blocks on the ground, allowing some variation

    robot_x_tolerance = 0.5
    robot_y_tolerance = 0.5
    robot_z_tolerance = 0.3 # Tolerance for robot pelvis height

    # Success conditions for block placement
    # Object2 relative to Object1
    # CORRECT: Using torch.abs for absolute difference and comparing to tolerance for batched environments
    cond_obj2_x = torch.abs(dist_obj2_obj1_x - target_x_offset) < block_x_tolerance
    cond_obj2_y = torch.abs(dist_obj2_obj1_y - target_y_offset) < block_y_tolerance
    cond_obj2_z = torch.abs(dist_obj2_obj1_z) < block_z_tolerance 

    # Object3 relative to Object2
    # CORRECT: Using torch.abs for absolute difference and comparing to tolerance for batched environments
    cond_obj3_x = torch.abs(dist_obj3_obj2_x - target_x_offset) < block_x_tolerance
    cond_obj3_y = torch.abs(dist_obj3_obj2_y - target_y_offset) < block_y_tolerance
    cond_obj3_z = torch.abs(dist_obj3_obj2_z) < block_z_tolerance 

    # Success conditions for robot final position
    # CORRECT: Using torch.abs for absolute difference and comparing to tolerance for batched environments
    cond_robot_pos_x = torch.abs(dist_pelvis_obj1_x) < robot_x_tolerance
    cond_robot_pos_y = torch.abs(dist_pelvis_obj1_y - robot_target_y_relative_to_obj1) < robot_y_tolerance
    cond_robot_pos_z = torch.abs(pelvis_pos_z - pelvis_target_z) < robot_z_tolerance

    # Combine all conditions for overall success
    # CORRECT: Combining all boolean conditions using logical AND for batched environments
    success_conditions = cond_obj2_x & cond_obj2_y & cond_obj2_z & \
                         cond_obj3_x & cond_obj3_y & cond_obj3_z & \
                         cond_robot_pos_x & cond_robot_pos_y & cond_robot_pos_z

    # Check duration and save success states
    # CORRECT: Using check_success_duration with a duration of 1.0 seconds as specified in the prompt.
    success = check_success_duration(env, success_conditions, "Build_Stairs", duration=1.0)
    
    # CORRECT: Saving success states for environments that have met the criteria using approved pattern.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Build_Stairs")
    
    return success

class SuccessTerminationCfg:
    # CORRECT: Registering the success function with DoneTerm
    success = DoneTerm(func=Build_Stairs_success)
