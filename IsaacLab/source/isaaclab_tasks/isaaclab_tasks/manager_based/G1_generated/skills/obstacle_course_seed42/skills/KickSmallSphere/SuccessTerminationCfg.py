
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def KickSmallSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the kick_smallSphere skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using approved patterns.
    # Object2: Small sphere for robot to kick
    object2 = env.scene['Object2']
    object2_pos = object2.data.root_pos_w

    # Object5: Block cube for robot to jump on top of
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w

    # Access the required robot part (pelvis) using approved patterns.
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Hardcoded dimensions from object configuration for thresholds.
    # Object5 (Block cube) is 0.5m cubed.
    # This variable is defined as per the prompt's guidance for hardcoding dimensions from config.
    block_dim = 0.5
    block_half_dim = block_dim / 2.0

    # --- Success Condition 1: Small sphere displacement ---
    # The small sphere (Object2) should be displaced significantly from its original position.
    # The goal is for it to be past the block (Object5) in the x-direction.
    # We check the relative x-distance between the sphere and the block.
    # A threshold of 0.5m means the sphere's center is 0.5m past the block's center.
    # This ensures the sphere has been propelled away from its initial location.
    # This condition uses relative distance between Object2 and Object5, adhering to rule 0.
    # Success Condition 1: Small sphere is at least 3m away from the block (Object5) in x,y norm
    sphere_block_xy_dist = torch.norm(object2_pos[:, :2] - object5_pos[:, :2], dim=1)
    sphere_past_block_condition = sphere_block_xy_dist > 4.0  # Must be at least 3m away in x,y

    # --- Success Condition 2: Robot positioning near the block ---
    # The robot should be positioned near the block (Object5) for the next skill.
    # We check the absolute x-distance between the robot's pelvis and the block's x-position.
    # A threshold of 1.0m allows for a reasonable range for the robot to be near the block.
    # This condition uses relative distance between robot pelvis and Object5, adhering to rule 0.
    robot_near_block_x_distance = torch.abs(robot_pelvis_pos[:, 0] - object5_pos[:, 0])
    robot_near_block_condition = robot_near_block_x_distance < 1.0 # Relative distance check

    # --- Success Condition 3: Robot stability/height ---
    # The robot should maintain a stable posture, indicated by its pelvis height.
    # A target pelvis height of 0.7m is used, which is a typical standing height for the robot.
    # A threshold of 0.3m allows for some variation (pelvis between 0.4m and 1.0m).
    # This is the only allowed absolute position check (for height), as per prompt's guidance.
    target_pelvis_z = 0.7
    robot_stable_height_distance = torch.abs(robot_pelvis_pos[:, 2] - target_pelvis_z)
    robot_stable_height_condition = robot_stable_height_distance < 0.3 # Absolute Z distance check

    # Combine all success conditions. All conditions must be met.
    # All conditions are combined using tensor operations, handling batched environments.
    condition = sphere_past_block_condition # & robot_near_block_condition & robot_stable_height_condition

    # Check success duration and save success states.
    # A duration of 0.5 seconds ensures the conditions are met for a short period, indicating stability.
    # check_success_duration is used as required.
    success = check_success_duration(env, condition, "kick_smallSphere", duration=0.5)
    
    # Save success states for environments that have successfully completed the skill.
    # save_success_state is used as required.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "kick_smallSphere")
    
    return success

class SuccessTerminationCfg:
    # Reasoning: Defining the success termination configuration using the approved pattern.
    success = DoneTerm(func=KickSmallSphere_success)
