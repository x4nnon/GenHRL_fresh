
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_largeSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_largeSphere skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # CRITICAL IMPLEMENTATION RULE: Access robot object directly.
    robot = env.scene["robot"]

    # CRITICAL IMPLEMENTATION RULE: Access robot part index using robot.body_names.index().
    # The skill description implies the robot's base should be near the sphere, so pelvis is a good proxy.
    pelvis_idx = robot.body_names.index('pelvis')
    # CRITICAL IMPLEMENTATION RULE: Access robot part position using robot.data.body_pos_w.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL IMPLEMENTATION RULE: Access object directly using its assigned name (Object1 for large sphere).
    large_sphere = env.scene['Object1']
    # CRITICAL IMPLEMENTATION RULE: Access object position using env.scene['ObjectName'].data.root_pos_w.
    large_sphere_pos = large_sphere.data.root_pos_w

    # CRITICAL IMPLEMENTATION RULE: Hardcode object dimensions from the object configuration.
    # Object1 (large sphere) has a radius of 1.0m as per the skill description.
    large_sphere_radius = 1.0

    # SUCCESS CRITERIA RULE: Calculate target x-position for pelvis relative to the large sphere.
    # The robot needs to be positioned to push the sphere towards the high wall (Object4).
    # The high wall is located at a greater x-coordinate than the large sphere.
    # Therefore, the robot's pelvis should be on the negative x-side of the sphere.
    # A clearance of 0.3m from the sphere's surface is desired.
    # So, pelvis x-position should be sphere_center_x - (sphere_radius + clearance).
    target_pelvis_x_offset_from_sphere_center = - (large_sphere_radius + 0.3)

    # SUCCESS CRITERIA RULE: Calculate relative distances for each axis.
    # X-axis distance: Pelvis x-position relative to the target x-position near the sphere.
    # This ensures the robot is positioned correctly to initiate a push.
    distance_x = torch.abs(pelvis_pos[:, 0] - (large_sphere_pos[:, 0] + target_pelvis_x_offset_from_sphere_center))

    # Y-axis distance: Pelvis y-position relative to the large sphere's y-position.
    # This ensures the robot is aligned laterally with the sphere.
    distance_y = torch.abs(pelvis_pos[:, 1] - large_sphere_pos[:, 1])

    # Z-axis distance: Pelvis z-position relative to a stable standing height.
    # A stable pelvis height for the robot is typically around 0.7m.
    # This is one of the few cases where an absolute Z-position check is allowed for stability.
    target_pelvis_z = 0.7
    distance_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # SUCCESS CRITERIA RULE: Define success conditions with reasonable tolerances.
    # X-axis condition: Pelvis is within 0.2m of the target x-position.
    success_x = distance_x < 0.2

    # Y-axis condition: Pelvis is aligned with the sphere's y-position within 0.3m.
    success_y = distance_y < 0.3

    # Z-axis condition: Pelvis is at a stable height (0.7m) within 0.2m.
    success_z = distance_z < 0.2

    # SUCCESS CRITERIA RULE: Combine all conditions for overall success.
    # All conditions must be met for the skill to be considered successful.
    condition = success_x & success_y & success_z

    # CRITICAL IMPLEMENTATION RULE: Use check_success_duration to ensure the state is maintained.
    # A duration of 0.5 seconds is reasonable for stable standing.
    success = check_success_duration(env, condition, "walk_to_largeSphere", duration=0.5)

    # CRITICAL IMPLEMENTATION RULE: Save success states for environments that have succeeded.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_largeSphere")

    return success

class SuccessTerminationCfg:
    # CRITICAL IMPLEMENTATION RULE: Assign the success function to a DoneTerm.
    success = DoneTerm(func=walk_to_largeSphere_success)
