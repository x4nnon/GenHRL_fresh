
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
    robot = env.scene["robot"]
    # CORRECT: Accessing robot pelvis position using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # 2. Get object positions
    # CORRECT: Accessing objects directly using Object1 and Object2 names
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Object dimensions (hardcoded from environment setup description)
    # CORRECT: Hardcoding object dimensions as they cannot be accessed from RigidObjectData
    wall_thickness_x = 0.5 # x of 0.5 for the cubes, interpreted as thickness along robot's path (x-axis)

    # Find rightmost wall x position
    # CORRECT: Using relative positions to find rightmost wall
    rightmost_wall_x = torch.maximum(object1_pos[:, 0], object2_pos[:, 0])

    # 3. Check success conditions based on relative distances and specific components

    # Condition 1: Pelvis must be past both walls in the x-direction.
    # To be fully past the wall, the pelvis needs to be past rightmost_wall_x + (wall_thickness_x / 2.0).
    # A clearance of 0.2m past the back face ensures the robot is well clear.
    # CORRECT: Using relative distance for x-axis clearance
    x_clearance_threshold = 0.2 # 0.2m past the back face
    is_past_doorway = pelvis_pos_x > (rightmost_wall_x + (wall_thickness_x / 2.0) + x_clearance_threshold)

    # Condition 2: Pelvis height (z-axis) to ensure the robot is standing.
    # Minimum pelvis height to be considered upright, consistent with reward.
    # CORRECT: Using absolute z-position for height, which is an allowed exception for height checks
    min_pelvis_z = 0.5
    is_pelvis_upright = pelvis_pos_z > min_pelvis_z

    # Combine all conditions for success
    # CORRECT: Combining conditions with logical AND for batched environments
    condition = is_past_doorway & is_pelvis_upright

    # 4. Check duration and save success states
    # CORRECT: Using check_success_duration to ensure the condition holds for a duration
    # Duration of 0.5 seconds is reasonable for a walking skill.
    success = check_success_duration(env, condition, "walk_through_doorway", duration=0.5)
    
    # CORRECT: Saving success states for environments that have met the success criteria
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_through_doorway")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_through_doorway_success)


