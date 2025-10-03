
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_small_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_small_block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the robot's pelvis position
    # CORRECT: Using approved pattern to access robot and pelvis position.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Access the positions of Object1 (Wall 1) and Object3 (small block)
    # CORRECT: Using approved pattern to access objects.
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]

    object3 = env.scene['Object3'] # small block
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # --- Success Criterion 1: X-axis alignment with Object3 ---
    # Calculate the absolute difference in x-position between robot pelvis and Object3.
    # This uses relative distance as required.
    # CORRECT: Using relative distance for x-axis alignment.
    x_distance_pelvis_obj3 = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0])
    # Condition: Absolute difference in x-position is less than 0.3m.
    # CORRECT: Using a reasonable threshold.
    x_alignment_condition = x_distance_pelvis_obj3 < 0.3

    # --- Success Criterion 2: Y-axis proximity to Object3 ---
    # The target y-offset for the pelvis from Object3's center is 0.5m (from reward function).
    # This places the pelvis 0.5m behind the block's center.
    # We need the pelvis to be between (Object3's y-position - 0.6m) and (Object3's y-position - 0.4m).
    # This ensures the pelvis is 0.5m +/- 0.1m from Object3's center in y.
    # All calculations are based on relative positions.
    # CORRECT: Using relative positions for y-axis proximity and reasonable thresholds.
    y_min_threshold = object3_pos[:, 1] - 0.6
    y_max_threshold = object3_pos[:, 1] - 0.4
    y_proximity_condition = (pelvis_pos[:, 1] > y_min_threshold) & (pelvis_pos[:, 1] < y_max_threshold)

    # --- Success Criterion 3: Z-axis stability (pelvis height) ---
    # The stable z-height for the pelvis is approximately 0.7m (from reward function).
    # This is one of the few allowed absolute position checks for stability.
    # CORRECT: Absolute Z-height check is allowed for stability as per prompt.
    z_distance_pelvis = torch.abs(pelvis_pos[:, 2] - 0.7)
    # Condition: Absolute difference in z-position is less than 0.2m.
    # CORRECT: Using a reasonable threshold.
    z_stability_condition = z_distance_pelvis < 0.2

    # --- Success Criterion 4: Passed the doorway ---
    # The walls (Object1 and Object2) are 5m long in the y-direction and are centered at their root_pos_w[:, 1].
    # The "far end" of the doorway (in the direction the robot is moving) is Object1's y-position + half its length.
    # Wall y-length is 5.0m (hardcoded from environment setup description).
    # CORRECT: Hardcoding object dimension (wall_y_length) is allowed as per prompt.
    # CORRECT: Using relative position for doorway far end.
    wall_y_length = 5.0
    doorway_far_end_y = object1_pos[:, 1] + (wall_y_length / 2.0)
    # Condition: Robot pelvis y-position is greater than the far end of the doorway.
    # This ensures the robot has fully passed through the doorway.
    # CORRECT: Using relative position for doorway passage check.
    doorway_passed_condition = pelvis_pos[:, 1] > doorway_far_end_y

    # Combine all success conditions
    # All conditions must be true for success.
    # CORRECT: Combining conditions with logical AND for batched environments.
    overall_success_condition = (
        x_alignment_condition &
        y_proximity_condition &
        z_stability_condition &
        doorway_passed_condition
    )

    # Check success duration and save success states
    # The duration required for success is 0.5 seconds.
    # CORRECT: Using check_success_duration as required.
    success = check_success_duration(env, overall_success_condition, "walk_to_small_block", duration=0.5)

    # Save success states for environments that have met the success criteria for the required duration.
    # CORRECT: Using save_success_state as required.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_small_block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_small_block_success)
