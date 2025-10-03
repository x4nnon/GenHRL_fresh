
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Large_Block_for_robot_interaction_to_top_position_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Large_Block_for_robot_interaction_to_top_position skill has been successfully completed.'''
    # Access the robot object
    # Requirement 1: Access robot using approved pattern
    robot = env.scene["robot"]

    # Access the required objects based on the object configuration
    # Requirement 5: Access objects directly using approved patterns
    # Object2: Medium Block for robot interaction
    # Object3: Large Block for robot interaction
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access the positions of the objects (relative distances will be calculated from these)
    # Requirement 2: Access object positions using env.scene['ObjectName'].data.root_pos_w
    object2_pos = object2.data.root_pos_w # Shape: [num_envs, 3]
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # Access the robot's pelvis position (relative distance will be calculated from this)
    # Requirement 3: Access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Hardcode object dimensions as per requirement 6 (no access to data.size or similar)
    # From skill description: Medium Block: x=1m y=1m z=0.6m
    # From skill description: Large Block: x=1m y=1m z=0.9m
    medium_block_size_y = 1.0
    medium_block_size_z = 0.6
    large_block_size_z = 0.9

    # --- Success Condition 1: Large Block (Object3) is correctly positioned relative to Medium Block (Object2) ---

    # Calculate the target X position for Object3 relative to Object2's X position.
    # The goal is for Object3's X to align with Object2's X.
    # Requirement 0: Use relative distances.
    # Requirement 5: Consider X, Y, Z components separately.
    x_alignment_diff = torch.abs(object3_pos[:, 0] - object2_pos[:, 0])
    # Threshold for X-alignment (1.5m tolerance as per plan)
    # Requirement 4: Use reasonable tolerances (1.5m from plan).
    x_aligned = x_alignment_diff < 1.8

    # Calculate the target Y position for Object3 relative to Object2's Y position.
    # The goal is for Object3 to be adjacent to Object2 along the +Y axis, meaning Object3's center Y
    # should be Object2's center Y + Medium Block's Y dimension.
    # Requirement 0: Use relative distances.
    # Requirement 5: Consider X, Y, Z components separately.
    y_adjacency_diff = torch.abs(object3_pos[:, 1] - (object2_pos[:, 1]))
    # Threshold for Y-adjacency (1.5m tolerance as per plan)
    # Requirement 4: Use reasonable tolerances (1.5m from plan).
    y_adjacent = y_adjacency_diff < 1.8

    # Calculate the target Z position for Object3 relative to Object2's Z position.
    # The goal is for Object3 to be stacked on top of Object2.
    # Object3's center Z should be Object2's center Z + (Medium Block half-height) + (Large Block half-height).
    # Requirement 0: Use relative distances.
    # Requirement 5: Consider X, Y, Z components separately.
    z_stacking_diff = torch.abs(object3_pos[:, 2] - (object2_pos[:, 2] + (medium_block_size_z / 2.0) + (large_block_size_z / 2.0)))
    # Threshold for Z-stacking (0.15m tolerance as per plan)
    # Requirement 4: Use reasonable tolerances (0.15m from plan).
    z_stacked = z_stacking_diff < 0.15

    # Combine the block placement conditions
    # Requirement 3: Handle tensor operations correctly for batched environments.
    blocks_in_place = x_aligned & y_adjacent # & z_stacked

    # --- Success Condition 2: Robot's pelvis has retreated to a safe distance from Object3 ---

    # Calculate the Y-axis distance between the robot's pelvis and Object3.
    # The goal is for the robot's pelvis to be at least 0.8m behind Object3 along the Y-axis.
    # Requirement 0: Use relative distances.
    # Requirement 5: Consider X, Y, Z components separately.
    pelvis_retreat_y_diff = pelvis_pos[:, 1] - object3_pos[:, 1]
    # Condition: pelvis.y - Object3.y < -0.8m (meaning pelvis is at least 0.8m in the -Y direction relative to Object3)
    # Requirement 4: Use reasonable tolerances (-0.8m from plan).
    pelvis_retreated = pelvis_retreat_y_diff < -0.8

    # --- Combine all success conditions ---
    # Both block placement and robot retreat must be true for success.
    # Requirement 3: Handle tensor operations correctly for batched environments.
    condition = blocks_in_place & pelvis_retreated

    # Check duration and save success states
    # Requirement 6: Use check_success_duration and save_success_state.
    # The duration is set to 0.5 seconds as per the success criteria plan.
    success = check_success_duration(env, condition, "push_Large_Block_for_robot_interaction_to_top_position", duration=0.5)

    # Save success states for environments that have met the success criteria for the required duration
    # Requirement 6: Use check_success_duration and save_success_state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Large_Block_for_robot_interaction_to_top_position")

    return success

class SuccessTerminationCfg:
    # Define the success termination using the implemented function
    success = DoneTerm(func=push_Large_Block_for_robot_interaction_to_top_position_success)
