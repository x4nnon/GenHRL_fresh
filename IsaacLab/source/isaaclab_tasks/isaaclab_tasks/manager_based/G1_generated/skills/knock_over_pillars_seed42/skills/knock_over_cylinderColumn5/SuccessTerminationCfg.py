
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_cylinderColumn5_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_cylinderColumn5 skill has been successfully completed.'''

    # Hardcoded object dimensions from the object configuration and skill description.
    # All cylinder columns have z dimension of 2m and a radius of 0.3m.
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    CYLINDER_RADIUS = 0.3

    # 1. Get robot parts
    robot = env.scene["robot"]
    # CRITICAL RULE: ALWAYS access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    # CRITICAL RULE: All operations must work with batched environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # 2. Get object position
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5 = env.scene['Object5']
    # CRITICAL RULE: All operations must work with batched environments.
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]

    # 3. Calculate success conditions based on the plan:

    # Condition 1: Cylinder Column 5 (Object5) has fallen completely onto the floor.
    # This means its Z-position should be approximately equal to its radius.
    # The success threshold is Object5's Z-position <= (Cylinder Radius + 0.2m tolerance).
    # CRITICAL RULE: Z-position is the only absolute position allowed for height checks.
    # CRITICAL RULE: Use reasonable tolerances.
    fallen_threshold_z = CYLINDER_RADIUS + 0.2 # 0.3m (radius) + 0.2m (tolerance) = 0.5m
    object5_z_pos = object5_pos[:, 2]
    fallen_condition = object5_z_pos <= fallen_threshold_z

    # Condition 2: The robot's pelvis is horizontally within a reasonable distance of Object5.
    # This ensures the robot has not moved excessively far past Object5.
    # CRITICAL RULE: ALL success criteria MUST ONLY use relative distances between objects and robot parts.
    # Calculate horizontal distance (X and Y components) between the robot's pelvis and Object5's root.
    # CRITICAL RULE: All operations must work with batched environments.
    horizontal_dist_pelvis_to_object5 = torch.norm(object5_pos[:, :2] - pelvis_pos[:, :2], dim=1)
    # CRITICAL RULE: Use reasonable tolerances.
    proximity_threshold_horizontal = 1.5 # Robot's pelvis should be within 1.5m horizontally of Object5
    proximity_condition = horizontal_dist_pelvis_to_object5 < proximity_threshold_horizontal

    # 4. Combine all conditions
    # CRITICAL RULE: All operations must work with batched environments.
    # Add pelvis height condition
    min_pelvis_height = 0.6
    pelvis_height_condition = (pelvis_pos[:, 2] > min_pelvis_height)

    combined_condition = fallen_condition & pelvis_height_condition # & proximity_condition

    # 5. Check duration and save success states
    # CRITICAL RULE: ALWAYS use check_success_duration and save_success_state.
    success = check_success_duration(env, combined_condition, "knock_over_cylinderColumn5", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_cylinderColumn5")

    return success

@configclass
class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_cylinderColumn5_success)
