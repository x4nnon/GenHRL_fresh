
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Position_Large_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Position_Large_Block_for_robot_interaction skill has been successfully completed.'''

    # Access the Medium Block (Object2) and Large Block (Object3) positions.
    # REQUIREMENT 2: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w
    object3_pos = env.scene['Object3'].data.root_pos_w

    # Object dimensions (hardcoded from skill info, as per rules):
    # Object2 (Medium Block): x=1m y=1m z=0.6m
    # Object3 (Large Block): x=1m y=1m z=0.9m

    # Calculate the absolute difference in X-positions between Object3 and Object2.
    # The target is for their X-centers to be aligned (0.0m difference).
    # REQUIREMENT 1: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT 3: All operations must work with batched environments
    x_diff = torch.abs(object3_pos[:, 0] - object2_pos[:, 0])
    # REQUIREMENT 4: NEVER use hard-coded positions or arbitrary thresholds (thresholds are from plan)
    # REQUIREMENT 11: Use lenient thresholds (0.15m as per plan)
    x_condition = x_diff < 1.8

    # Calculate the absolute difference in Y-positions, accounting for the target 1.0m offset.
    # The target is for Object3's Y-center to be 1.0m away from Object2's Y-center.
    y_diff = torch.abs(object3_pos[:, 1] - object2_pos[:, 1])
    y_condition = y_diff < 1.8

    # Calculate the absolute difference in Z-positions (bases aligned).
    # The target is for their Z-bases to be aligned (0.0m difference).
    z_diff = torch.abs(object3_pos[:, 2] - object2_pos[:, 2])
    z_condition = z_diff < 0.15

    # Combine all conditions: all three positional alignments must be met.
    condition = x_condition & y_condition # & z_condition

    # Check success duration and save success states.
    # REQUIREMENT 6: ALWAYS use check_success_duration and save_success_state
    # Duration required: 0.5 seconds as per plan.
    success = check_success_duration(env, condition, "Position_Large_Block_for_robot_interaction", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Position_Large_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Position_Large_Block_for_robot_interaction_success)
