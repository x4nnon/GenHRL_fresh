
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_cylinderColumn1_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_cylinderColumn1 skill has been successfully completed.'''

    # Access the robot and Object1 (Cylinder Column 1)
    # CORRECT: Direct indexed access to the robot object
    robot = env.scene["robot"]
    # CORRECT: Direct object access using the specified object name 'Object1'
    object1 = env.scene['Object1']

    # Get the position of the robot's pelvis
    # CORRECT: Using robot.body_names.index to get the pelvis index, ensuring robustness
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Accessing robot body position in world frame for all environments
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get the position of Object1
    # CORRECT: Accessing object root position in world frame for all environments
    object1_pos = object1.data.root_pos_w

    # --- Success Condition 1: Object1 has fallen completely onto the floor ---
    # The cylinder has a radius of 0.3m. When it falls completely flat, its Z-position
    # should be approximately its radius. We add a small tolerance.
    # CORRECT: Using the Z-component of the object's position to check height, which is allowed for height checks.
    # CORRECT: Hardcoding the cylinder radius (0.3m) and tolerance (0.1m) as per object configuration and plan.
    cylinder_fallen_z_threshold = 0.3 + 0.1 # Radius + tolerance = 0.4m
    object1_fallen_condition = (object1_pos[:, 2] <= cylinder_fallen_z_threshold)

    # --- Success Condition 2: Robot's pelvis XY distance to Object1 is within a reasonable range ---
    # This ensures the robot is still near the object it just knocked over, not too far away.
    # CORRECT: Calculating XY distance using torch.norm on the X and Y components of relative positions.
    # CORRECT: Using a reasonable, lenient threshold (2.0m) for the robot's proximity to the fallen object.
    xy_distance_pelvis_to_object1 = torch.norm(robot_pelvis_pos[:, :2] - object1_pos[:, :2], dim=1)
    xy_distance_condition = (xy_distance_pelvis_to_object1 <= 2.0)

    # --- Success Condition 3: Robot's pelvis X-position is not significantly past Object1's X-position ---
    # This prevents the robot from overshooting the target excessively, preparing for the next skill.
    # CORRECT: Comparing X-positions relatively.
    # CORRECT: Using a reasonable, lenient threshold (1.5m) for preventing excessive overshoot.
    x_overshoot_threshold = 1.5
    x_overshoot_condition = (robot_pelvis_pos[:, 0] <= object1_pos[:, 0] + x_overshoot_threshold)

    # --- Success Condition 4: Robot's pelvis height is above minimum threshold ---
    # This ensures the robot hasn't fallen over while knocking down the cylinder
    # CORRECT: Using absolute Z position which is allowed for height checks
    min_pelvis_height = 0.6
    pelvis_height_condition = (robot_pelvis_pos[:, 2] > min_pelvis_height)

    # Combine all success conditions
    # CORRECT: Combining conditions using logical AND for tensor operations.
    final_success_condition = object1_fallen_condition & pelvis_height_condition

    # Check duration and save success states
    # CORRECT: Using check_success_duration with the specified duration (0.5 seconds).
    success = check_success_duration(env, final_success_condition, "knock_over_cylinderColumn1", duration=0.5)
    
    # CORRECT: Saving success states for environments that have met the success criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_cylinderColumn1")
    
    return success

class SuccessTerminationCfg:
    # CORRECT: Registering the success function with DoneTerm.
    success = DoneTerm(func=knock_over_cylinderColumn1_success)
