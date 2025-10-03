
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def stabilize_after_jump_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the stabilize_after_jump skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts and their positions
    robot = env.scene["robot"]
    
    # Access pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # Access left and right foot positions using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Shape: [num_envs, 3]
    
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]
    
    # 2. Get object positions
    # Access Object3 (low wall) position using approved pattern
    object3 = env.scene['Object3'] # Low wall
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]
    
    # Access Object1 (large sphere) position using approved pattern
    object1 = env.scene['Object1'] # Large sphere
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]
    
    # 3. Hardcode object dimensions from the object configuration
    # Object3: wide low wall, 0.3m in x axis.
    low_wall_x_dim = 0.3
    # Object1: large sphere 1m radius.
    large_sphere_radius = 1.0
    
    # Define target values and thresholds as per the success criteria plan
    target_pelvis_z = 0.7
    target_foot_z = 0.05 
    target_pelvis_y = 0.0 # Global y-center
    
    pelvis_z_threshold = 0.15
    foot_z_threshold = 0.1
    pelvis_y_threshold = 0.2
    x_position_buffer = 0.1 # Buffer for x-position clearance
    
    # 4. Calculate success conditions based on relative distances
    
    # Condition 1: Pelvis Z-position (height) is stable
    # Uses relative distance of pelvis Z-position from a target height.
    pelvis_z_condition = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) < pelvis_z_threshold
    
    # Condition 2: Left foot Z-position (on ground)
    # Uses relative distance of left foot Z-position from the ground.
    left_foot_z_condition = torch.abs(left_foot_pos[:, 2] - target_foot_z) < foot_z_threshold
    
    # Condition 3: Right foot Z-position (on ground)
    # Uses relative distance of right foot Z-position from the ground.
    right_foot_z_condition = torch.abs(right_foot_pos[:, 2] - target_foot_z) < foot_z_threshold
    
    # Condition 4: Pelvis Y-position (centered on y-axis)
    # Uses relative distance of pelvis Y-position from the global y-center.
    pelvis_y_condition = torch.abs(pelvis_pos[:, 1] - target_pelvis_y) < pelvis_y_threshold
    
    # Condition 5: Pelvis X-position is past the low wall (Object3)
    # Calculate the x-coordinate of the far side of the low wall, relative to its root position and dimension.
    low_wall_far_x = object3_pos[:, 0] + (low_wall_x_dim / 2) + x_position_buffer
    pelvis_past_low_wall_condition = pelvis_pos[:, 0] > low_wall_far_x
    
    # Condition 6: Pelvis X-position is before the large sphere (Object1)
    # Calculate the x-coordinate of the near side of the large sphere, relative to its root position and radius.
    large_sphere_near_x = object1_pos[:, 0] - large_sphere_radius - x_position_buffer
    pelvis_before_large_sphere_condition = pelvis_pos[:, 0] < large_sphere_near_x
    
    # Combine all conditions
    all_conditions = (pelvis_z_condition & 
                      left_foot_z_condition & 
                      right_foot_z_condition & 
                      pelvis_y_condition & 
                      pelvis_past_low_wall_condition & 
                      pelvis_before_large_sphere_condition)
    
    # 5. Check duration and save success states
    success = check_success_duration(env, all_conditions, "stabilize_after_jump", duration=0.5)
    
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "stabilize_after_jump")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=stabilize_after_jump_success)
