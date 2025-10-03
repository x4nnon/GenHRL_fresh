
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_through_doorway_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_through_doorway skill has been successfully completed.'''

    # Access the robot object
    robot = env.scene["robot"]

    # Access the required robot part positions using approved patterns
    # CRITICAL RULE: ALWAYS access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Access the required environment object (Heavy Cube (Wall 1))
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object1_pos = object1.data.root_pos_w # Shape: [num_envs, 3]

    # Hardcoded wall Y-dimension from object configuration (y of 5m)
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from config.
    wall_y_dim = 5.0

    # Calculate the Y-coordinate of the far edge of the doorway (past Object1)
    # This is a relative position derived from the object's center and its hardcoded dimension.
    # CRITICAL RULE: Success criteria MUST ONLY use relative distances.
    # The doorway exit is the far Y-edge of Object1, assuming Object1's root_pos_w is its center.
    doorway_y_exit = object1_pos[:, 1] + (wall_y_dim / 2)

    # Define target stable pelvis height and overshoot limit
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds for object positions.
    # However, target heights for robot posture are allowed if they represent a stable state.
    pelvis_target_z = 0.7 # Target Z-height for stable standing
    max_overshoot_y_buffer = 0.6 # Maximum allowed distance past the doorway exit before penalizing

    # Condition 1: Pelvis, left foot, and right foot must be past the doorway's far Y-edge.
    # CRITICAL RULE: Success criteria MUST ONLY use relative distances between objects and robot parts.
    # This checks if the robot parts' Y-positions are greater than the doorway's Y-exit, plus a small buffer.
    # A buffer of 0.1m ensures complete clearance.
    pelvis_past_doorway = pelvis_pos[:, 1] > (doorway_y_exit + 0.1)
    left_foot_past_doorway = left_foot_pos[:, 1] > (doorway_y_exit + 0.1)
    right_foot_past_doorway = right_foot_pos[:, 1] > (doorway_y_exit + 0.1)

    # Condition 2: Robot's pelvis must be at a stable standing height.
    # CRITICAL RULE: Z-axis distance of pelvis relative to a target stable height (0.7m).
    # This checks if the absolute difference between current pelvis Z and target Z is within a tolerance.
    pelvis_stable_height = torch.abs(pelvis_pos[:, 2] - pelvis_target_z) < 0.15

    # Condition 3: Robot must not have overshot the immediate area past the doorway.
    # CRITICAL RULE: Success criteria MUST ONLY use relative distances.
    # This checks if the pelvis Y-position is less than the doorway exit plus the allowed overshoot buffer.
    pelvis_not_overshot = pelvis_pos[:, 1] < (doorway_y_exit + max_overshoot_y_buffer)

    # Combine all conditions for overall success
    # CRITICAL RULE: All operations must work with batched environments.
    condition = pelvis_past_doorway & left_foot_past_doorway & right_foot_past_doorway & \
                pelvis_stable_height & pelvis_not_overshot

    # Check duration and save success states
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, condition, "walk_through_doorway", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_through_doorway")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_through_doorway_success)
