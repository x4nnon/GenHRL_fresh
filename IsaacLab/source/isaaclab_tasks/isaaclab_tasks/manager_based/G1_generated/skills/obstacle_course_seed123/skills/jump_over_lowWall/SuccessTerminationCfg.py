
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def jump_over_lowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the jump_over_lowWall skill has been successfully completed.'''

    # 1. Get robot parts positions
    robot = env.scene["robot"]
    
    # Get pelvis position for overall robot state and positioning
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Get foot positions to confirm landing
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]
    
    # 2. Get object positions
    # Object3 is the low wall for the robot to jump over
    low_wall = env.scene['Object3']
    low_wall_pos_x = low_wall.data.root_pos_w[:, 0]

    # Object1 is the large sphere, used as a boundary for landing
    large_sphere = env.scene['Object1']
    large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0]

    # 3. Hardcode object dimensions from the configuration (as per requirements)
    # Low wall dimensions: 0.3m in x-axis (depth)
    low_wall_depth = 0.3 
    # Large sphere dimensions: 1m radius
    large_sphere_radius = 1.0

    # 4. Calculate relative distances and check success conditions
    
    # Condition 1: Pelvis is past the low wall's far edge
    # The far edge of the low wall is low_wall_pos_x + low_wall_depth / 2.
    # We need the pelvis to be beyond this point by at least 0.1m.
    # Relative distance: pelvis_pos_x - (low_wall_pos_x + low_wall_depth / 2)
    pelvis_past_wall_condition = (pelvis_pos_x - (low_wall_pos_x + low_wall_depth / 2)) > 0.1
    
    # Condition 2: Pelvis is before the large sphere's near edge
    # The near edge of the large sphere is large_sphere_pos_x - large_sphere_radius.
    # We need the pelvis to be before this point by at least 0.1m.
    # Relative distance: (large_sphere_pos_x - large_sphere_radius) - pelvis_pos_x
    pelvis_before_sphere_condition = ((large_sphere_pos_x - large_sphere_radius) - pelvis_pos_x) > 0.1

    # Condition 3: Pelvis is at a stable standing height
    # Target stable height for pelvis is 0.7m. Tolerance is 0.15m.
    # Relative distance: torch.abs(pelvis_pos_z - target_pelvis_z)
    target_pelvis_z = 0.7
    pelvis_height_condition = torch.abs(pelvis_pos_z - target_pelvis_z) < 0.15

    # Condition 4: Left foot is near the ground (z=0)
    # Relative distance: left_foot_pos_z - ground_z (where ground_z is 0)
    left_foot_on_ground_condition = left_foot_pos_z < 0.1

    # Condition 5: Right foot is near the ground (z=0)
    # Relative distance: right_foot_pos_z - ground_z (where ground_z is 0)
    right_foot_on_ground_condition = right_foot_pos_z < 0.1

    # Combine all conditions for overall success
    condition = (pelvis_past_wall_condition & 
                 pelvis_before_sphere_condition & 
                 pelvis_height_condition & 
                 left_foot_on_ground_condition & 
                 right_foot_on_ground_condition)
    
    # 5. Check duration and save success states
    success = check_success_duration(env, condition, "jump_over_lowWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "jump_over_lowWall")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=jump_over_lowWall_success)
