
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_through_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_through_doorway skill has been successfully completed.'''
    # 1. Get robot parts
    # Accessing the robot object from the scene. (Rule: ALWAYS access objects directly)
    robot = env.scene["robot"]
    # Getting the index for the 'pelvis' body part. (Rule: ALWAYS access robot parts using robot.body_names.index)
    pelvis_idx = robot.body_names.index('pelvis')
    # Accessing the world position of the pelvis for all environments. (Rule: Handle tensor operations correctly for batched environments)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # 2. Get object positions
    # Accessing Object1 (Heavy Cube - Wall 1) from the scene. (Rule: ALWAYS access objects directly using Object1...Object5)
    object1 = env.scene['Object1']
    # Accessing the world position of Object1. (Rule: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w)
    object1_pos = object1.data.root_pos_w
    # Accessing Object2 (Heavy Cube - Wall 2) from the scene. (Rule: ALWAYS access objects directly using Object1...Object5)
    object2 = env.scene['Object2']
    # Accessing the world position of Object2. (Rule: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w)
    object2_pos = object2.data.root_pos_w

    # Hardcoded dimensions from the object configuration (as per requirements).
    # (Rule: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.)
    # Wall x-dimension is 0.5m.
    wall_x_dim = 0.5
    # Wall y-dimension is 5.0m.
    wall_y_dim = 5.0

    # 3. Calculate relative distances and conditions
    
    # Condition 1: Pelvis has passed the doorway's front plane in the y-axis.
    # The doorway's front plane is the y-center of the walls minus half their y-dimension.
    # This is a relative calculation based on object position and hardcoded dimension. (Rule: SUCCESS CRITERIA MUST ONLY use relative distances)
    doorway_front_y_plane = object1_pos[:, 1] - (wall_y_dim / 2.0)
    # Target y-position for success: 0.8m past the doorway's front plane.
    # This threshold is lenient, slightly less than the 1.0m target in the reward. (Rule: USE LENIENT THRESHOLDS, REASONABLE TOLERANCES)
    target_y_past_doorway = doorway_front_y_plane + 0.8
    # Check if pelvis y-position is greater than the target y-position. (Rule: Consider X, Y, Z components separately)
    pelvis_passed_doorway_y = pelvis_pos[:, 1] > target_y_past_doorway

    # Condition 2: Pelvis is horizontally centered within the doorway's x-bounds.
    # Calculate the inner x-boundaries of the doorway.
    # Object1's inner x-edge (right side of left wall). (Rule: SUCCESS CRITERIA MUST ONLY use relative distances)
    object1_inner_x = object1_pos[:, 0] + (wall_x_dim / 2.0)
    # Object2's inner x-edge (left side of right wall). (Rule: SUCCESS CRITERIA MUST ONLY use relative distances)
    object2_inner_x = object2_pos[:, 0] - (wall_x_dim / 2.0)
    # Calculate the center of the doorway in the x-axis.
    # This is a relative calculation based on object positions and hardcoded dimension. (Rule: SUCCESS CRITERIA MUST ONLY use relative distances)
    doorway_center_x = (object1_inner_x + object2_inner_x) / 2.0
    # Calculate the absolute x-distance between the pelvis and the doorway center. (Rule: Consider X, Y, Z components separately)
    x_distance_to_doorway_center = torch.abs(pelvis_pos[:, 0] - doorway_center_x)
    # Check if the x-distance is within a tolerance (0.3m).
    # The doorway gap is 0.5m, so 0.25m from center to wall edge. 0.3m allows for slight deviation. (Rule: USE LENIENT THRESHOLDS, REASONABLE TOLERANCES)
    pelvis_centered_x = x_distance_to_doorway_center < 0.3

    # Condition 3: Pelvis maintains an upright posture (z-height within a reasonable range).
    # Accessing the z-component of the pelvis position. This is an allowed absolute position check for height. (Rule: z is the only absolute position allowed)
    pelvis_z_pos = pelvis_pos[:, 2]
    # Check if pelvis z-position is within the desired range (0.5m to 0.9m).
    # The desired pelvis z is 0.7m in the reward, this range provides tolerance. (Rule: USE LENIENT THRESHOLDS, REASONABLE TOLERANCES)
    pelvis_upright_z = (pelvis_z_pos > 0.5) & (pelvis_z_pos < 0.9)
    
    # Combine all conditions for overall success. All conditions must be met. (Rule: Combine conditions correctly)
    condition = pelvis_passed_doorway_y & pelvis_centered_x & pelvis_upright_z
    
    # 4. Check duration and save success states
    # The skill requires the conditions to be met for 0.5 seconds. (Rule: ALWAYS use check_success_duration)
    success = check_success_duration(env, condition, "walk_through_doorway", duration=0.5)
    
    # If any environment has succeeded, save its success state. (Rule: ALWAYS use save_success_state)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_through_doorway")
    
    return success

class SuccessTerminationCfg:
    # Register the success function as a termination condition.
    success = DoneTerm(func=walk_through_doorway_success)
