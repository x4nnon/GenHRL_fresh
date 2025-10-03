
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Navigate_and_Jump_Over_LowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Navigate_and_Jump_Over_LowWall skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # 1. Get robot parts
    robot = env.scene["robot"]
    # Access the pelvis position, which is used to determine the robot's overall location and height.
    # Complies with "ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]"
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # 2. Get object positions
    # Access Object3 (low wall) and Object1 (large sphere) as per the object configuration.
    # Complies with "ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w"
    # Complies with "ONLY USE Object1...Object5"
    low_wall = env.scene['Object3']
    low_wall_pos = low_wall.data.root_pos_w # Shape: [num_envs, 3]
    
    large_sphere = env.scene['Object1']
    large_sphere_pos = large_sphere.data.root_pos_w # Shape: [num_envs, 3]

    # 3. Hardcode object dimensions from the task description/object configuration.
    # This is required as there is no way to access object dimensions dynamically.
    # Complies with "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    low_wall_x_dim = 0.3 # From object config: 0.3m in x
    large_sphere_radius = 1.0 # From object config: 1m radius

    # 4. Calculate relative distances and check conditions.
    # Condition 1: Robot pelvis x-position is past the low wall.
    # This ensures the robot has successfully jumped over the wall.
    # The far side of the low wall is its center x-position + half its x-dimension.
    # A small clearance (0.1m) is added to ensure it's clearly past the wall.
    # Complies with "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    # Complies with "USE RELATIVE DISTANCES"
    # Complies with "REASONABLE TOLERANCES"
    low_wall_far_x = low_wall_pos[:, 0] + low_wall_x_dim / 2.0
    clearance_after_wall = 0.1 # Small clearance to ensure robot is clearly past the wall
    condition_past_wall_x = pelvis_pos[:, 0] > (low_wall_far_x + clearance_after_wall)

    # Condition 2: Robot pelvis x-position is before the large sphere.
    # This ensures the robot lands in the correct zone between the low wall and the large sphere,
    # preparing it for the next skill.
    # The near side of the large sphere is its center x-position - its radius.
    # A small clearance (0.5m) is subtracted to ensure it's not too close to the sphere.
    # Complies with "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts"
    # Complies with "USE RELATIVE DISTANCES"
    # Complies with "REASONABLE TOLERANCES"
    large_sphere_near_x = large_sphere_pos[:, 0] - large_sphere_radius
    clearance_before_sphere = 0.5 # Ensures robot is not too close to the sphere for the next skill
    condition_before_sphere_x = pelvis_pos[:, 0] < (large_sphere_near_x - clearance_before_sphere)

    # Condition 3: Robot pelvis is at a stable standing height.
    # This ensures the robot has landed stably after the jump.
    # The target standing height (0.7m) and tolerance (0.15m) are from the success criteria plan.
    # Complies with "SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts" (z-height is allowed as per prompt)
    # Complies with "REASONABLE TOLERANCES"
    target_standing_height = 0.7
    height_tolerance = 0.15
    condition_standing_height = torch.abs(pelvis_pos[:, 2] - target_standing_height) < height_tolerance
    
    # Combine all conditions. All must be true for success.
    # Complies with "HANDLE TENSOR OPERATIONS CORRECTLY"
    condition = condition_past_wall_x & condition_before_sphere_x & condition_standing_height
    
    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # The duration is set to 0.5 seconds as specified in the success criteria plan.
    # Complies with "CHECK SUCCESS DURATION" and "SAVE SUCCESS STATES"
    success = check_success_duration(env, condition, "Navigate_and_Jump_Over_LowWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Navigate_and_Jump_Over_LowWall")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Navigate_and_Jump_Over_LowWall_success)
