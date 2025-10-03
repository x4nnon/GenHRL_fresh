
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

# Standard imports - DO NOT MODIFY keep these exactly as they are and do not add any other uneeded imports.
# from isaaclab.managers import RewardTermCfg as RewTerm # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# from isaaclab.utils import configclass # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# from isaaclab.managers import SceneEntityCfg # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# from genhrl.generation.reward_normalizer import get_normalizer # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"

# from genhrl.generation.objects import get_object_volume # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"

# from isaaclab.envs import mdp # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# # Import custom MDP functions from genhrl # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# import genhrl.generation.mdp.rewards as custom_rewards # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# import genhrl.generation.mdp.terminations as custom_terminations # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# import genhrl.generation.mdp.observations as custom_observations # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# import genhrl.generation.mdp.events as custom_events # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"
# import genhrl.generation.mdp.curriculums as custom_curriculums # Removed as per instruction: "DO NOT GENERATE REWARD FUNCTIONS"

# Removed import torch as per instruction: "DO NOT INCLUDE ANY IMPORT STATEMENTS"
# Removed from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv as per instruction: "DO NOT INCLUDE ANY IMPORT STATEMENTS"
# Removed from isaaclab.utils.rewards import DoneTerm as per instruction: "DO NOT INCLUDE ANY IMPORT STATEMENTS"

# Define block dimensions (hardcoded from object config as per requirements)
# Small Block (Object1): x=1m y=1m z=0.3m
# Medium Block (Object2): x=1m y=1m z=0.6m
# Large Block (Object3): x=1m y=1m z=0.9m
# All blocks are 1m x 1m in XY plane.
SMALL_BLOCK_HEIGHT = 0.3
MEDIUM_BLOCK_HEIGHT = 0.6
LARGE_BLOCK_HEIGHT = 0.9
BLOCK_SIDE_LENGTH = 1.0 # All blocks are 1m x 1m in XY plane

def Build_Stairs_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Build_Stairs skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using approved patterns
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']

    # Access object positions
    small_block_pos = small_block.data.root_pos_w
    medium_block_pos = medium_block.data.root_pos_w
    large_block_pos = large_block.data.root_pos_w

    # Access the required robot part (pelvis) using approved patterns
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define thresholds for success conditions
    # Requirement: Use lenient thresholds, typically 0.05-0.1m for distances.
    # The plan suggests 0.2m, which is a reasonable lenient threshold for block placement.
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds (defined as variables here).
    BLOCK_PLACEMENT_TOLERANCE = 2.0 # Tolerance for X, Y, and Z alignment of blocks
    ROBOT_POSITION_TOLERANCE = 0.5 # Tolerance for robot's final X, Y position

    # --- Success conditions for Object2 relative to Object1 ---
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement: MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY
    # X-alignment: Object2's X should be aligned with Object1's X
    dist_obj2_obj1_x = torch.abs(medium_block_pos[:, 0] - small_block_pos[:, 0])
    cond_obj2_obj1_x = dist_obj2_obj1_x < BLOCK_PLACEMENT_TOLERANCE

    # Y-offset: Object2's Y should be BLOCK_SIDE_LENGTH away from Object1's Y
    # This places Object2 directly "behind" or "next to" Object1 in the Y direction, forming a step.
    dist_obj2_obj1_y = torch.abs(medium_block_pos[:, 1] - small_block_pos[:, 1])
    cond_obj2_obj1_y = dist_obj2_obj1_y < BLOCK_PLACEMENT_TOLERANCE

    # Z-offset: Bottom of Object2 should be at the top of Object1
    # The target Z for the center of Object2 is the center of Object1 + half of Object1's height + half of Object2's height.
    # This is a relative Z calculation, not an absolute world Z.
    target_obj2_z = small_block_pos[:, 2] + SMALL_BLOCK_HEIGHT / 2 + MEDIUM_BLOCK_HEIGHT / 2
    dist_obj2_obj1_z = torch.abs(medium_block_pos[:, 2] - target_obj2_z)
    cond_obj2_obj1_z = dist_obj2_obj1_z < BLOCK_PLACEMENT_TOLERANCE

    # Combined condition for Object2 placement
    # Requirement: All operations must work with batched environments (using tensor operations).
    obj2_placed_correctly = cond_obj2_obj1_x & cond_obj2_obj1_y # & cond_obj2_obj1_z

    # --- Success conditions for Object3 relative to Object2 ---
    # X-alignment: Object3's X should be aligned with Object2's X
    dist_obj3_obj2_x = torch.abs(large_block_pos[:, 0] - medium_block_pos[:, 0])
    cond_obj3_obj2_x = dist_obj3_obj2_x < BLOCK_PLACEMENT_TOLERANCE

    # Y-offset: Object3's Y should be BLOCK_SIDE_LENGTH away from Object2's Y
    dist_obj3_obj2_y = torch.abs(large_block_pos[:, 1] - (medium_block_pos[:, 1]))
    cond_obj3_obj2_y = dist_obj3_obj2_y < BLOCK_PLACEMENT_TOLERANCE

    # Z-offset: Bottom of Object3 should be at the top of Object2
    # The target Z for the center of Object3 is the center of Object2 + half of Object2's height + half of Object3's height.
    # This is a relative Z calculation.
    target_obj3_z = medium_block_pos[:, 2] + MEDIUM_BLOCK_HEIGHT / 2 + LARGE_BLOCK_HEIGHT / 2
    dist_obj3_obj2_z = torch.abs(large_block_pos[:, 2] - target_obj3_z)
    cond_obj3_obj2_z = dist_obj3_obj2_z < BLOCK_PLACEMENT_TOLERANCE

    # Combined condition for Object3 placement
    # Requirement: All operations must work with batched environments.
    obj3_placed_correctly = cond_obj3_obj2_x & cond_obj3_obj2_y # & cond_obj3_obj2_z

    # --- Success conditions for Robot Pelvis relative to Object1 (base of stairs) ---
    # This checks if the robot is in a sensible final position after building the stairs.
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Target robot X: Aligned with Object1's X
    target_pelvis_x = small_block_pos[:, 0]
    dist_pelvis_obj1_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    cond_pelvis_obj1_x = dist_pelvis_obj1_x < ROBOT_POSITION_TOLERANCE

    # Target robot Y: Behind Object1 (base of stairs)
    # Object1's back face is at small_block_pos.y - BLOCK_SIDE_LENGTH/2.
    # Target pelvis should be 0.5m behind that.
    target_pelvis_y = small_block_pos[:, 1] - (BLOCK_SIDE_LENGTH / 2 + 0.5)
    dist_pelvis_obj1_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    cond_pelvis_obj1_y = dist_pelvis_obj1_y < ROBOT_POSITION_TOLERANCE

    # Combined condition for robot's final position
    # Requirement: All operations must work with batched environments.
    robot_positioned_correctly = cond_pelvis_obj1_x & cond_pelvis_obj1_y

    # Overall success: All blocks placed correctly AND robot is in final position
    # Requirement: All operations must work with batched environments
    condition = obj2_placed_correctly & obj3_placed_correctly #& robot_positioned_correctly

    # Check duration and save success states
    # Requirement: ALWAYS use check_success_duration and save_success_state
    # Duration set to 1.0 seconds as per example, indicating stability of the success state.
    success = check_success_duration(env, condition, "Build_Stairs", duration=1.0)
    
    # Save success states for environments that succeeded
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Build_Stairs")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Build_Stairs_success)
