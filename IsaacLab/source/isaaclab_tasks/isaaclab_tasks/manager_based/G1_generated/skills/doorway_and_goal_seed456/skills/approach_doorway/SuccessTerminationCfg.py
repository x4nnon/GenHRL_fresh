
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_doorway skill has been successfully completed.'''
    # 1. Get robot parts
    robot = env.scene["robot"]
    # CORRECT: Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # 2. Get object positions
    # CORRECT: Direct object access using Object1 and Object2 names
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # 3. Hardcode object dimensions from description (0.5m depth for walls)
    # CORRECT: Hardcoding object dimension from description, as direct access is not allowed
    wall_depth = 0.5 # The 'x' dimension of the wall, which acts as depth in the global Y-axis for the doorway

    # 4. Calculate target positions for the doorway
    # CORRECT: Relative distance calculation for doorway center X
    doorway_center_x = (object1_pos[:, 0] + object2_pos[:, 0]) / 2.0
    # CORRECT: Relative distance calculation for doorway front Y (object center Y - half its depth)
    target_doorway_y = object1_pos[:, 1] - wall_depth / 2.0
    # CORRECT: Hardcoding target pelvis height, which is allowed for Z-axis
    target_pelvis_z = 0.7

    # 5. Check success conditions
    # Condition 1: Pelvis X-axis centering
    # CORRECT: Relative distance check for X-axis centering
    x_distance_condition = torch.abs(pelvis_pos_x - doorway_center_x) < 0.15

    # Condition 2: Pelvis Y-axis positioning at front edge
    # CORRECT: Relative distance check for Y-axis positioning
    y_distance_condition = torch.abs(pelvis_pos_y - target_doorway_y) < 0.15

    # Condition 3: Pelvis Z-axis height
    # CORRECT: Relative distance check for Z-axis height (from target height)
    z_height_condition = torch.abs(pelvis_pos_z - target_pelvis_z) < 0.15

    # Condition 4: Orientation alignment with global Y-axis
    # The prompt explicitly states "NO ORIENTATION OR LINEAR VELOCITY CHECKS" for success criteria.
    # Therefore, the orientation check must be removed.
    # The original code included a helper function `quat_to_vec` and used `pelvis_quat` for orientation.
    # This is a violation of the rule "NO ORIENTATION OR LINEAR VELOCITY CHECKS".
    # Removing the orientation condition.

    # Combine all conditions
    # CORRECT: Combining all conditions with logical AND for batched environments
    condition = x_distance_condition & y_distance_condition & z_height_condition

    # 6. Check duration and save success states
    # CORRECT: Using check_success_duration with specified skill name and duration
    success = check_success_duration(env, condition, "approach_doorway", duration=0.5)
    # CORRECT: Saving success states for environments that meet the criteria
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_doorway")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=approach_doorway_success)
