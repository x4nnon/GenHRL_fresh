
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Push_LargeSphere_to_Topple_HighWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Push_LargeSphere_to_Topple_HighWall skill has been successfully completed.'''

    # Access the required objects using approved patterns (CRITICAL RULE: ALWAYS access objects directly)
    high_wall = env.scene['Object4'] # High wall for large sphere to push over
    small_sphere = env.scene['Object2'] # Small sphere for robot to kick

    # Access the required robot part(s) using approved patterns (CRITICAL RULE: ALWAYS access robot parts by name index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]
    
    # Extract x, y, z components for clarity and specific checks
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded object dimensions from task description (CRITICAL RULE: NO ACCESSING .size or .radius from object)
    # Object4 (high wall) is 0.3m in x-axis.
    high_wall_x_dim = 0.3 

    # 1. Condition: High wall is toppled (its Z-position is very low)
    # This uses an absolute Z-position, which is allowed sparingly for height-related success criteria.
    # A threshold of 0.2m is used, assuming the wall was 1m tall and falls flat.
    wall_toppled_condition = high_wall.data.root_pos_w[:, 2] < 0.2

    # 2. Condition: Robot pelvis is past the toppled high wall (in X-axis)
    # This is a relative distance check between the robot's pelvis X and the high wall's X position,
    # accounting for the wall's dimension and a small buffer.
    # (CRITICAL RULE: ONLY use relative distances between objects and robot parts)
    # (CRITICAL RULE: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY)
    pelvis_past_wall_x_condition = pelvis_pos_x > (high_wall.data.root_pos_w[:, 0] + high_wall_x_dim / 2.0 - 0.5)

    # 3. Condition: Robot pelvis is before the small sphere (in X-axis) for the next skill
    # This is a relative distance check between the robot's pelvis X and the small sphere's X position,
    # ensuring the robot doesn't overshoot the target for the next skill.
    # (CRITICAL RULE: ONLY use relative distances between objects and robot parts)
    # (CRITICAL RULE: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY)
    pelvis_before_small_sphere_x_condition = pelvis_pos_x < (small_sphere.data.root_pos_w[:, 0] - 0.5)

    # 4. Condition: Robot pelvis is generally aligned in Y-axis with the path
    # This is a relative distance check between the robot's pelvis Y and the high wall's Y position,
    # assuming the path is generally aligned with the objects.
    # (CRITICAL RULE: ONLY use relative distances between objects and robot parts)
    # (CRITICAL RULE: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY)
    pelvis_y_alignment_condition = torch.abs(pelvis_pos_y - high_wall.data.root_pos_w[:, 1]) < 1.0

    # Combine all conditions for success (CRITICAL RULE: All tensor operations must work with batched environments)
    success_condition = wall_toppled_condition & \
                        pelvis_past_wall_x_condition & \
                        pelvis_before_small_sphere_x_condition & \
                        pelvis_y_alignment_condition

    # Check duration and save success states (CRITICAL RULE: ALWAYS use check_success_duration and save_success_state)
    # A duration of 0.5 seconds is chosen to ensure the conditions are met stably.
    success = check_success_duration(env, success_condition, "Push_LargeSphere_to_Topple_HighWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Push_LargeSphere_to_Topple_HighWall")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Push_LargeSphere_to_Topple_HighWall_success)
