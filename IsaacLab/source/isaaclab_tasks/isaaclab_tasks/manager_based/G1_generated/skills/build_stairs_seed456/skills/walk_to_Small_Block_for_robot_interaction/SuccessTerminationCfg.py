
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block_for_robot_interaction skill has been successfully completed.'''
    # 1. Get robot parts
    # Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]
    # Getting the index of the 'pelvis' body part using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Getting the position of the 'pelvis' body part using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object position
    # Accessing 'Object1' (Small Block for robot interaction) directly using the approved pattern.
    object1 = env.scene['Object1']
    # Getting the root position of 'Object1' using the approved pattern.
    object1_pos = object1.data.root_pos_w
    
    # 3. Calculate distances and positions based on success criteria plan
    # Calculate the horizontal (x, y) distance between the robot's pelvis and Object1's center.
    # This uses relative distances as required by rule 0 and 3 (Position & Motion Rules).
    distance_x = object1_pos[:, 0] - pelvis_pos[:, 0]
    distance_y = object1_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_distance = torch.sqrt(distance_x**2 + distance_y**2)
    
    # Get the vertical (z) position of the robot's pelvis.
    # The z-position is allowed as an absolute check when height is critical for the skill, as per rule 5 (Position & Motion Rules).
    pelvis_z = pelvis_pos[:, 2]
    
    # Define thresholds based on the success criteria plan.
    # These thresholds are hardcoded as they are part of the defined success criteria, which is allowed when specified in the plan.
    min_horizontal_dist = 0.5
    max_horizontal_dist = 1.0
    min_pelvis_z = 0.6
    max_pelvis_z = 0.8
    
    # 4. Check success condition
    # The success condition requires both horizontal distance and pelvis height to be within their respective ranges.
    # All operations are batched for multiple environments, adhering to rule 3 (Most Critical Rules).
    condition = (horizontal_distance >= min_horizontal_dist) & \
                (horizontal_distance <= max_horizontal_dist) & \
                (pelvis_z >= min_pelvis_z) & \
                (pelvis_z <= max_pelvis_z)
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Checking if the condition has been met for the required duration (0.5 seconds), as per rule 4 (Most Critical Rules).
    success = check_success_duration(env, condition, "walk_to_Small_Block_for_robot_interaction", duration=0.5)
    # Saving the success state for environments that have successfully completed the skill, as per rule 5 (Most Critical Rules).
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block_for_robot_interaction")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_for_robot_interaction_success)
