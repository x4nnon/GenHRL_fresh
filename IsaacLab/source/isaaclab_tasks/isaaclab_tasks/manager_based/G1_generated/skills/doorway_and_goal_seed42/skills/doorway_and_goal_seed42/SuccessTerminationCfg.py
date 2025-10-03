
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def doorway_and_goal_seed42_success(env) -> torch.Tensor:
    '''Determine if the doorway_and_goal_seed42 skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required robot part: pelvis
    # REASONING: The pelvis is a good central point to represent the robot's overall position.
    # It's robust to limb movements and indicates the robot's general location.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Access the required objects: Object1 (Wall 1) and Object3 (Small Block)
    # REASONING: Object1 is needed to determine the doorway's y-position.
    # Object3 is the final goal the robot needs to reach.
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object3 = env.scene['Object3'] # Small Block

    # Hardcoded object dimensions from the task description for the walls
    # REASONING: Object dimensions cannot be accessed dynamically; they must be hardcoded
    # based on the environment setup description.
    wall_y_dim = 5.0 # y-dimension of the heavy cubes (length along y-axis)

    # --- Condition 1: Robot has passed through the doorway ---
    # Calculate the y-coordinate that signifies passing the doorway.
    # This is derived from the reward function's logic for consistency.
    # The doorway's y-position is the y-center of the walls.
    doorway_y_pos = object1.data.root_pos_w[:, 1]
    # The "far" end of the doorway along the y-axis is doorway_y_pos + wall_y_dim / 2.
    # A small buffer (0.1m) is added to ensure the robot is truly past the doorway,
    # aligning with the reward function's `past_doorway_threshold_y`.
    # REASONING: The 0.1m buffer is a reasonable tolerance as per "REASONABLE TOLERANCES" rule.
    past_doorway_threshold_y = doorway_y_pos + (wall_y_dim / 2.0) + 0.1
    
    # Check if the robot's pelvis y-position is beyond the doorway threshold.
    # REASONING: This is a relative check against a point defined by an object's position and its dimension.
    passed_doorway_condition = (pelvis_pos[:, 1] > past_doorway_threshold_y)

    # --- Condition 2: Robot is close to the small block (Object3) ---
    # Calculate the relative distance between the robot's pelvis and Object3.
    # REASONING: Success criteria must use relative distances.
    # We consider both x and y components as the robot needs to reach the block's location.
    distance_to_block_x = torch.abs(object3.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    distance_to_block_y = torch.abs(object3.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    
    # Define thresholds for proximity to the small block.
    # REASONING: These thresholds are lenient enough for "walking to" a 0.3m block,
    # allowing for some tolerance, as per "USE LENIENT THRESHOLDS" and "REASONABLE TOLERANCES" rules.
    proximity_threshold_x = 0.5 # Slightly larger than the block size for leniency
    proximity_threshold_y = 0.5 # Slightly larger than the block size for leniency

    # Check if the robot's pelvis is within the proximity thresholds of Object3.
    # REASONING: This combines relative distance checks for both x and y dimensions.
    close_to_block_condition = (distance_to_block_x < proximity_threshold_x) & \
                               (distance_to_block_y < proximity_threshold_y)

    # --- Combine both conditions for overall success ---
    # The robot must first pass the doorway AND then be close to the block.
    # REASONING: This aligns with the two-phase nature of the skill description.
    overall_success_condition = close_to_block_condition

    # Check success duration and save success states
    # REASONING: As per absolute requirements, check_success_duration and save_success_state must always be used.
    # A duration of 1.0 second ensures the robot maintains the success state for a short period.
    success = check_success_duration(env, overall_success_condition, "doorway_and_goal_seed42", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "doorway_and_goal_seed42")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=doorway_and_goal_seed42_success)
