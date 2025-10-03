
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Small_Block_into_stair_top_position_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Small_Block_into_stair_top_position skill has been successfully completed.'''

    # Hardcoded object dimensions from the object configuration
    # Object1: Small Block (x=1m y=1m z=0.3m)
    OBJECT1_HEIGHT = 0.3
    OBJECT1_HALF_HEIGHT = OBJECT1_HEIGHT / 2.0
    OBJECT1_HALF_WIDTH_Y = 0.5 # Assuming 1m width, so 0.5m half-width

    # Object2: Medium Block (x=1m y=1m z=0.6m)
    OBJECT2_HEIGHT = 0.6
    OBJECT2_HALF_HEIGHT = OBJECT2_HEIGHT / 2.0
    OBJECT2_HALF_WIDTH_Y = 0.5 # Assuming 1m width, so 0.5m half-width

    # 1. Get robot parts and object positions
    # CORRECT: Direct indexed access to the robot
    robot = env.scene["robot"]
    # CORRECT: Direct object access using Object1...Object5 names
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block

    # CORRECT: Getting positions for all envs at once (handling batch operations)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # CORRECT: Using robot.body_names.index instead of hardcoded indices
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # 2. Calculate target position for Object1 relative to Object2
    # CORRECT: Target position is calculated relative to Object2's current position and object dimensions.
    # This ensures relative positioning based on object dimensions and Object2's current position.
    target_object1_pos_x = object2_pos[:, 0]
    target_object1_pos_y = object2_pos[:, 1] + OBJECT2_HALF_WIDTH_Y + OBJECT1_HALF_WIDTH_Y
    target_object1_pos_z = object2_pos[:, 2] + OBJECT2_HALF_HEIGHT - OBJECT1_HALF_HEIGHT

    # 3. Success conditions for Object1 placement relative to Object2
    # CORRECT: Using relative distances for X-alignment.
    object1_x_aligned = torch.abs(object1_pos[:, 0] - target_object1_pos_x) < 0.15

    # CORRECT: Using relative distances for Y-adjacency.
    object1_y_adjacent = torch.abs(object1_pos[:, 1] - target_object1_pos_y) < 0.15

    # CORRECT: Using relative distances for Z-alignment of top surfaces.
    object1_z_aligned = torch.abs(object1_pos[:, 2] - target_object1_pos_z) < 0.15

    # CORRECT: Combining conditions with proper tensor operations
    object1_placed_correctly = object1_x_aligned & object1_y_adjacent & object1_z_aligned

    # 4. Success conditions for robot's stable standing position and clearance
    # Target stable pelvis height. This is a fixed value for a stable standing posture,
    # explicitly allowed for Z-axis in "DO'S AND DON'TS" section.
    pelvis_target_z = 0.7
    # CORRECT: Using absolute Z-position for robot's pelvis height, as allowed for Z-axis.
    robot_pelvis_stable_height = torch.abs(pelvis_pos[:, 2] - pelvis_target_z) < 0.15

    # Approximate center of the combined Object1/Object2 structure for clearance calculation.
    # CORRECT: This uses relative positions of Object1 and Object2.
    combined_center_x = (object1_pos[:, 0] + object2_pos[:, 0]) / 2.0
    combined_center_y = (object1_pos[:, 1] + object2_pos[:, 1]) / 2.0

    # Calculate 2D distance (X-Y) from robot's pelvis to the combined structure's center.
    # CORRECT: This is a relative distance calculation.
    dist_pelvis_to_structure_xy = torch.sqrt(
        (pelvis_pos[:, 0] - combined_center_x)**2 +
        (pelvis_pos[:, 1] - combined_center_y)**2
    )

    # CORRECT: Using relative distance for robot clearance.
    robot_clear_of_structure = dist_pelvis_to_structure_xy > 0.8

    # Combined condition for robot stability and clearance
    # CORRECT: Combining conditions with proper tensor operations
    robot_in_final_state = robot_pelvis_stable_height & robot_clear_of_structure

    # 5. Overall success condition: Both object placement and robot state must be met
    # CORRECT: Combining all conditions with proper tensor operations
    condition = object1_placed_correctly & robot_in_final_state

    # 6. Check duration and save success states
    # CORRECT: Using check_success_duration and appropriate duration
    success = check_success_duration(env, condition, "push_Small_Block_into_stair_top_position", duration=0.5)
    # CORRECT: Saving success states for successful environments
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Small_Block_into_stair_top_position")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_Small_Block_into_stair_top_position_success)
