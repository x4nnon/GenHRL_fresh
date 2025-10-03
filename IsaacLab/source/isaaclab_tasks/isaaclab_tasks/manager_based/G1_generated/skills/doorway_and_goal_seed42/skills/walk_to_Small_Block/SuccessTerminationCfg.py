
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Small_Block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Small_Block skill has been successfully completed.'''
    # 1. Get robot parts
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    
    # Extract x, y, z components for separate distance checks
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]
    
    # 2. Get object position
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Object3 is the "Small Block for the robot to walk to"
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w
    
    # Extract x, y, z components for separate distance checks
    object3_pos_x = object3_pos[:, 0]
    object3_pos_y = object3_pos[:, 1]
    object3_pos_z = object3_pos[:, 2]
    
    # 3. Calculate distance (use specific components when appropriate)
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY
    distance_x = torch.abs(robot_pelvis_pos_x - object3_pos_x)
    distance_y = torch.abs(robot_pelvis_pos_y - object3_pos_y)
    distance_z = torch.abs(robot_pelvis_pos_z - object3_pos_z)
    
    # Define success thresholds based on the provided plan.
    # These are relative distance thresholds, not hard-coded absolute positions.
    # The Small Block is 0.3m cubed. Thresholds of 0.3m for x/y and 0.5m for z
    # allow the robot to be adjacent to or on top of the block.
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    # These thresholds are derived from the object's known size (0.3m) and the skill's goal (adjacent/on top).
    threshold_x = 0.3 # meters
    threshold_y = 0.3 # meters
    threshold_z = 0.5 # meters
    
    # 4. Check success condition for each dimension
    # Robot's pelvis must be within the specified thresholds of the Small Block's center.
    # All operations work with batched environments.
    condition_x = distance_x < threshold_x
    condition_y = distance_y < threshold_y
    condition_z = distance_z < threshold_z
    
    # Combine all conditions for overall success. All conditions must be met.
    condition = condition_x & condition_y # & condition_z
    
    # 5. Check duration and save success states
    # Requirement: ALWAYS use check_success_duration and save_success_state
    # Using a duration of 0.5 seconds to ensure stability at the target.
    success = check_success_duration(env, condition, "walk_to_Small_Block", duration=0.5)
    
    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Small_Block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Small_Block_success)
