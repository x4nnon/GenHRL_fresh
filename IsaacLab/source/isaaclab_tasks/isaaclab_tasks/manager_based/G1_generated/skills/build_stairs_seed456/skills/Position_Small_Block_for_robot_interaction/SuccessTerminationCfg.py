
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Position_Small_Block_for_robot_interaction_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Position_Small_Block_for_robot_interaction skill has been successfully completed.'''
    # Access the required objects using the approved pattern.
    object1 = env.scene['Object1']  # Small Block for robot interaction
    object2 = env.scene['Object2']  # Medium Block for robot interaction

    # Access the required robot part(s) using the approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode object dimensions from the object configuration.
    # This is required as RigidObjectData does not expose size attributes.
    object1_x_dim = 1.0
    object1_y_dim = 1.0
    object1_z_dim = 0.3
    object2_y_dim = 1.0

    # Define the target position for Object1 relative to Object2.
    # This aligns with the reward function's definition, ensuring consistency.
    # X-coordinate: Object1's center X should align with Object2's center X.
    target_object1_x = object2.data.root_pos_w[:, 0]
    # Y-coordinate: Object1 should be adjacent to Object2 with a 0.1m gap.
    # This is calculated by taking Object2's Y, subtracting half of Object1's Y dimension,
    # half of Object2's Y dimension, and the 0.1m gap.
    target_object1_y = object2.data.root_pos_w[:, 1] - (object1_y_dim / 2.0 + object2_y_dim / 2.0 + 0.1)
    # Z-coordinate: Object1 should be on the ground, centered vertically.
    # This is half of Object1's height. Absolute Z is allowed for height checks.
    target_object1_z = object1_z_dim / 2.0

    # Calculate the absolute difference (distance) for Object1's position relative to its target.
    # This uses relative distances between object positions and the calculated target.
    dist_object1_to_target_x = torch.abs(object1.data.root_pos_w[:, 0] - target_object1_x)
    dist_object1_to_target_y = torch.abs(object1.data.root_pos_w[:, 1] - target_object1_y)
    dist_object1_to_target_z = torch.abs(object1.data.root_pos_w[:, 2] - target_object1_z)

    # Calculate the XY distance between the robot's pelvis and Object1.
    # This ensures the robot is near the block it's supposed to be pushing.
    dist_pelvis_to_object1_xy = torch.norm(pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2], dim=1)

    # Define success conditions with reasonable tolerances.
    # Object1's X position is aligned with Object2's X.
    object1_x_aligned = dist_object1_to_target_x < 0.2
    # Object1's Y position is adjacent to Object2's Y.
    object1_y_adjacent = dist_object1_to_target_y < 0.2
    # Object1's Z position is correct (on the ground).
    object1_z_correct = dist_object1_to_target_z < 0.1
    # Robot's pelvis is within a reasonable distance of Object1 in the XY plane.
    robot_near_object1 = dist_pelvis_to_object1_xy < 1.0

    # Combine all conditions for overall success. All conditions must be met.
    condition = object1_x_aligned & object1_y_adjacent & object1_z_correct & robot_near_object1

    # Check success duration and save success states.
    # The duration is set to 1.0 seconds, meaning the conditions must be met for this duration.
    success = check_success_duration(env, condition, "Position_Small_Block_for_robot_interaction", duration=1.0)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Position_Small_Block_for_robot_interaction")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Position_Small_Block_for_robot_interaction_success)
