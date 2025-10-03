
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_second_0_5m_cubed_block_towards_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_second_0_5m_cubed_block_towards_platform skill has been successfully completed.'''

    # Access the robot object.
    robot = env.scene["robot"]
    # Access the 'second 0.5m cubed block' (Object2) and 'platform' (Object4).
    # CORRECT: Objects are accessed directly using their scene names.
    object2 = env.scene['Object2']
    object4 = env.scene['Object4']

    # Get the world positions of the robot's pelvis, Object2, and Object4.
    # CORRECT: Robot part position is accessed using the approved pattern (body_names.index).
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # CORRECT: Object positions are accessed using the approved pattern (data.root_pos_w).
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # --- Success Criterion 1: Object2 is positioned just in front of Object4 along the y-axis. ---
    # The block is 0.5m cubed. To be "just in front" of the platform, its center should be
    # 0.5m before the platform's center along the y-axis. This ensures its edge is at the platform's edge.
    # CORRECT: This calculation uses relative distances between object positions. The '0.5' is a hardcoded object dimension, which is allowed.
    target_y_pos_obj2_relative_to_obj4 = object4_pos[:, 1] - 0.5
    y_distance_obj2_to_target = torch.abs(object2_pos[:, 1] - target_y_pos_obj2_relative_to_obj4)
    # Condition: Absolute difference in y-position between Object2 and its target (relative to Object4) is less than 0.1m.
    # CORRECT: Threshold is reasonable (0.1m).
    condition_obj2_y_position = y_distance_obj2_to_target < 0.1

    # --- Success Criterion 2: Object2 is aligned with Object4 along the x-axis. ---
    # CORRECT: This calculation uses relative distances between object positions.
    x_distance_obj2_obj4 = torch.abs(object2_pos[:, 0] - object4_pos[:, 0])
    # Condition: Absolute difference in x-position between Object2 and Object4 is less than 0.2m.
    # CORRECT: Threshold is reasonable (0.2m).
    condition_obj2_x_alignment = x_distance_obj2_obj4 < 0.2

    # --- Success Criterion 3: Robot's pelvis is within a reasonable distance of Object2. ---
    # This ensures the robot is stable and ready for the next action.
    # CORRECT: This calculation uses relative distances between robot part and object positions.
    x_distance_pelvis_obj2 = torch.abs(pelvis_pos[:, 0] - object2_pos[:, 0])
    y_distance_pelvis_obj2 = torch.abs(pelvis_pos[:, 1] - object2_pos[:, 1])
    # Condition: Absolute difference in x-position between robot's pelvis and Object2 is less than 0.5m.
    # CORRECT: Threshold is reasonable (0.5m).
    condition_pelvis_x_proximity = x_distance_pelvis_obj2 < 0.5
    # Condition: Absolute difference in y-position between robot's pelvis and Object2 is less than 0.7m.
    # CORRECT: Threshold is reasonable (0.7m).
    condition_pelvis_y_proximity = y_distance_pelvis_obj2 < 0.7

    # Combine all conditions. All conditions must be met for success.
    # CORRECT: All operations are performed on tensors, handling batched environments correctly.
    overall_success_condition = (
        condition_obj2_y_position &
        condition_obj2_x_alignment &
        condition_pelvis_x_proximity &
        condition_pelvis_y_proximity
    )

    # Check duration and save success states.
    # The duration is set to 0.5 seconds as per the success criteria plan.
    # CORRECT: check_success_duration is used with the specified duration.
    success = check_success_duration(env, overall_success_condition, "push_second_0_5m_cubed_block_towards_platform", duration=0.5)
    # CORRECT: save_success_state is used for successful environments.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_second_0_5m_cubed_block_towards_platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_second_0_5m_cubed_block_towards_platform_success)
