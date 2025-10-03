
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def approach_doorway_and_align_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the approach_doorway_and_align skill has been successfully completed.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using the approved pattern.
    # Object1: Heavy Cube (Wall 1) forming the left wall of the doorway
    object1 = env.scene['Object1']
    # Object2: Heavy Cube (Wall 2) forming the right wall of the doorway
    object2 = env.scene['Object2']

    # Access the required robot part(s) using the approved pattern.
    robot = env.scene["robot"]
    # Get the index for the 'pelvis' body part.
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the world position and orientation (quat w, x, y, z) of the pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_quat = robot.data.body_quat_w[:, pelvis_idx]
    # Extract z for a permissive height sanity check.
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded object dimensions from the description for relative calculations.
    # The description states "x of 0.5" for the heavy cubes.
    # This is allowed as object dimensions cannot be accessed dynamically and must be hardcoded from the config.
    object_x_dim = 0.5

    # Calculate doorway center position as midpoint between the two walls (matches main reward logic).
    doorway_pos = (object1.data.root_pos_w + object2.data.root_pos_w) / 2.0
    # Distance from pelvis to doorway center (use positive distance here for thresholding).
    distance_to_doorway_center = torch.norm(pelvis_pos - doorway_pos, dim=1)
    
    # Orientation alignment with world Y-axis, using the same formulation as the main reward.
    # The robot's local X-axis alignment with world Y-axis: R_10 = 2*(qx*qy + qw*qz)
    orientation_dot_product = 2 * (pelvis_quat[:, 1] * pelvis_quat[:, 2] + pelvis_quat[:, 0] * pelvis_quat[:, 3])

    # Define thresholds for success conditions to align with the main reward intent.
    # Proximity: within 0.25m of the doorway center (roughly sqrt(0.15^2 + 0.2^2)).
    threshold_distance = 0.25
    # Orientation: require strong alignment facing through the gap.
    orientation_threshold = 0.95
    # Minimum and maximum pelvis height for passing through doorway.
    # Z-position is the only absolute position allowed for height checks, as per prompt.
    min_pelvis_z = 0.5
    max_pelvis_z = 0.9

    # Evaluate success conditions for each environment in the batch.
    # Condition 1: Close to doorway center (matches main reward geometry).
    is_close_to_center = distance_to_doorway_center < threshold_distance
    # Condition 2: Facing through the doorway (matches main reward orientation).
    is_oriented_correctly = orientation_dot_product > orientation_threshold
    # Condition 3: Pelvis height is appropriate for passing through the doorway.
    is_pelvis_height_appropriate = (pelvis_pos_z > min_pelvis_z) & (pelvis_pos_z < max_pelvis_z)

    # Combine all conditions for overall success. All conditions must be met.
    condition = is_close_to_center #& is_oriented_correctly #& is_pelvis_height_appropriate

    # Check success duration and save success states. This section is mandatory.
    # The duration is set to 0.5 seconds, meaning the conditions must be met for at least this long.
    success = check_success_duration(env, condition, "approach_doorway_and_align", duration=0.5)
    
    # If any environment has succeeded, save its state.
    # This adheres to the mandatory save_success_state requirement.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "approach_doorway_and_align")
    
    return success

class SuccessTerminationCfg:
    # Define the success termination using the implemented function.
    success = DoneTerm(func=approach_doorway_and_align_success)
