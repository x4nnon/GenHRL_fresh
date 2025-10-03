
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def navigate_through_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the navigate_through_doorway skill has been successfully completed.'''
    # Access the required objects using the approved pattern (Rule 2, 5)
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # small block

    # Access the required robot part(s) using the approved pattern (Rule 3)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded dimensions from environment description for doorway calculation (Rule 6)
    # The doorway gap is 0.5m in X, as implied by the reward function's use of doorway_center_x
    # and the task description "doorway gap of 0.5m".
    doorway_gap_x = 0.5 
    # Approximate robot radius for clearance, derived from reward function context (Rule 4)
    robot_clearance_x = 0.3 # A slightly larger clearance than 0.2 to ensure full passage

    # Calculate the center of the doorway in the X-axis (Rule 1, 4)
    # This is a relative distance calculation based on the positions of Object1 and Object2.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # Calculate the target Y-position, which is Object3's Y-position (Rule 1, 4)
    # This is a relative position to Object3, avoiding hard-coded absolute values.
    target_y_pos = object3.data.root_pos_w[:, 1]

    # Success condition 1: Robot pelvis is close to Object3 in the Y-axis. (Rule 1, 4)
    # This implies the robot has passed through the doorway and reached the target block.
    # Threshold of 0.5m is a lenient tolerance for reaching the final target.
    is_near_object3_y = torch.abs(pelvis_pos_y - target_y_pos) < 0.5

    # Success condition 2: Robot pelvis is horizontally aligned with the doorway's opening in the X-axis. (Rule 1, 4)
    # This ensures the robot passed *through* the doorway, not around it.
    # The threshold is half of the doorway gap (0.25m) plus the robot's approximate clearance (0.3m).
    # Total threshold = 0.25 + 0.3 = 0.55m. This is a lenient threshold for alignment.
    is_aligned_x = torch.abs(pelvis_pos_x - doorway_center_x) < (doorway_gap_x / 2.0) + robot_clearance_x

    # Success condition 3: Robot pelvis is at a reasonable standing height in the Z-axis. (Rule 1, 4)
    # This ensures the robot is in a stable, upright posture.
    # The target height (0.7m) and tolerance (0.15m) are derived from the reward function context.
    pelvis_z_target = 0.7
    is_standing_upright_z = torch.abs(pelvis_pos_z - pelvis_z_target) < 0.15

    # Combine all conditions for overall success (Rule 1)
    # All conditions must be met simultaneously for success.
    condition = is_near_object3_y & is_aligned_x & is_standing_upright_z

    # Check duration and save success states (Rule 6)
    # The duration is set to 0.5 seconds, meaning the conditions must be met for at least this long.
    success = check_success_duration(env, condition, "navigate_through_doorway", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "navigate_through_doorway")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=navigate_through_doorway_success)
