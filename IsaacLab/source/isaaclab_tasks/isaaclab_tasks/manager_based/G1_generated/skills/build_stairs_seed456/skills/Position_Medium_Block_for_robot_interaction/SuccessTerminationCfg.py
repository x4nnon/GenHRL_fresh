
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Position_Medium_Block_for_robot_interaction_success(env) -> torch.Tensor:
    '''Determine if the Position_Medium_Block_for_robot_interaction skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL RULE: Access robot object directly
    robot = env.scene["robot"]

    # CRITICAL RULE: Access object positions using approved patterns
    object1 = env.scene['Object1']  # Small Block for robot interaction
    object2 = env.scene['Object2']  # Medium Block for robot interaction

    # CRITICAL RULE: Access robot parts using body_names.index
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL RULE: Hardcode object dimensions from the task description/object configuration
    # Blocks are 1m x 1m in x and y dimensions.
    block_width = 1.0

    # --- Success Condition 1: Object2 (Medium Block) is aligned with Object1 (Small Block) along the x-axis ---
    # CRITICAL RULE: Use relative distances between objects
    # Calculate absolute difference in x-positions between Object2 and Object1
    x_diff_obj2_obj1 = torch.abs(object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0])
    # CRITICAL RULE: Use reasonable thresholds. 0.2m allows for slight misalignment.
    x_alignment_condition = x_diff_obj2_obj1 < 1.5

    # --- Success Condition 2: Object2 is adjacent to Object1 along the y-axis ---
    y_diff_obj2_obj1 = torch.abs(object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1])
    y_adjacency_condition = y_diff_obj2_obj1 < 1.5

    # --- Success Condition 3: Robot's pelvis is in close proximity to Object2 (Medium Block) ---
    # CRITICAL RULE: Use relative distances between robot parts and objects
    # Calculate horizontal distance (x and y components) between robot's pelvis and Object2
    dist_pelvis_obj2_x = torch.abs(object2.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_obj2_y = torch.abs(object2.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_obj2_horizontal = torch.sqrt(dist_pelvis_obj2_x**2 + dist_pelvis_obj2_y**2)
    # CRITICAL RULE: Use reasonable thresholds. 0.7m is consistent with the reward function's positioning phase.
    robot_proximity_condition = dist_pelvis_obj2_horizontal < 0.7

    # Combine all conditions for overall success
    # CRITICAL RULE: All operations must work with batched environments
    overall_condition = x_alignment_condition & y_adjacency_condition # & robot_proximity_condition

    # CRITICAL RULE: Always use check_success_duration and save_success_state
    # Duration required: 0.5 seconds as per the plan
    success = check_success_duration(env, overall_condition, "Position_Medium_Block_for_robot_interaction", duration=0.5)

    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Position_Medium_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    # CRITICAL RULE: Define success using DoneTerm and the implemented function
    success = DoneTerm(func=Position_Medium_Block_for_robot_interaction_success)
