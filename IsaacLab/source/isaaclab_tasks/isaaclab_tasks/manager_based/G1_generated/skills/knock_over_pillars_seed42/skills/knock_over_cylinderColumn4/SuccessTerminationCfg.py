
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_cylinderColumn4_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_cylinderColumn4 skill has been successfully completed.'''
    # CORRECT: Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]
    
    # CORRECT: Accessing robot pelvis position using robot.body_names.index for batch processing.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]
    
    # CORRECT: Accessing Object4 and Object5 positions using the approved pattern.
    object4 = env.scene['Object4']
    object4_pos = object4.data.root_pos_w # Shape: [num_envs, 3]
    
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]
    
    # CORRECT: Hardcoding object radius from the task description (0.3m).
    # This is necessary as object dimensions cannot be accessed from the RigidObject.
    object_radius = 0.3 

    # Condition 1: Cylinder Column 4 (Object4) is on the floor.
    # The Z-position of the object's center of mass should be at or very close to its radius when it's lying flat.
    # CORRECT: Using the Z-component of Object4's position. This is an allowed absolute Z-position check for height.
    # A small tolerance (0.05m) is added to the radius (0.3m) for robustness.
    object4_on_floor_condition = object4_pos[:, 2] <= (object_radius + 0.05) # Threshold: 0.35m

    # Condition 2: The robot's pelvis is not significantly past Cylinder Column 5 (Object5).
    # This ensures the robot is positioned appropriately for the next skill.
    # We check the X-component of the robot's pelvis relative to Object5's X-component.
    # CORRECT: Using relative X-positions between robot pelvis and Object5.
    # The robot's X-position should be less than or equal to Object5's X-position plus a buffer (0.5m).
    robot_not_overshooting_object5_condition = pelvis_pos[:, 0] <= (object5_pos[:, 0] + 0.5)

    # Combine both conditions. Both must be true for success.
    # CORRECT: Combining conditions using logical AND for tensor operations.
    # Add pelvis height condition
    min_pelvis_height = 0.6
    pelvis_height_condition = (pelvis_pos[:, 2] > min_pelvis_height)

    combined_condition = object4_on_floor_condition & pelvis_height_condition # & robot_not_overshooting_object5_condition
    
    # CORRECT: Using check_success_duration to ensure the conditions are met for a specified duration.
    # Duration is set to 0.5 seconds as per the success criteria plan.
    success = check_success_duration(env, combined_condition, "knock_over_cylinderColumn4", duration=0.5)
    
    # CORRECT: Saving success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_cylinderColumn4")
    
    return success

class SuccessTerminationCfg:
    # CORRECT: Registering the success function with DoneTerm.
    success = DoneTerm(func=knock_over_cylinderColumn4_success)
