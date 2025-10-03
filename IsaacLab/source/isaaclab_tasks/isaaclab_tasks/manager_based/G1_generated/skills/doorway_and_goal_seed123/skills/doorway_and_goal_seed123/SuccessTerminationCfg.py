
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def doorway_and_goal_seed123_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the doorway_and_goal_seed123 skill has been successfully completed.'''

    # 1. Get robot parts
    # Access the robot object from the scene.
    robot = env.scene["robot"]
    # Get the index for the 'pelvis' body part.
    pelvis_idx = robot.body_names.index('pelvis') # Correct: Using robot.body_names.index for part access.
    # Get the world-frame position of the pelvis for all environments.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Correct: Using robot.data.body_pos_w for batched position access.

    # 2. Get object positions
    # Access Object1 (Heavy Cube Wall 1) from the scene.
    object1 = env.scene['Object1'] # Correct: Direct object access using ObjectN name.
    # Get the world-frame root position of Object1.
    object1_pos = object1.data.root_pos_w # Correct: Using object.data.root_pos_w for batched position access.

    # Access Object3 (Small Block) from the scene.
    object3 = env.scene['Object3'] # Correct: Direct object access using ObjectN name.
    # Get the world-frame root position of Object3.
    object3_pos = object3.data.root_pos_w # Correct: Using object.data.root_pos_w for batched position access.

    # 3. Calculate distances and conditions based on the success criteria plan.

    # Condition 1: Robot's pelvis is within a close proximity to Object3 in the x-dimension.
    # Calculate the absolute relative x-distance between the pelvis and Object3.
    x_distance_to_obj3 = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0]) # Correct: Using relative distance for x-component.
    # Define the threshold for x-distance to Object3 as per the success criteria plan.
    threshold_x_obj3 = 0.5  # meters. Correct: Threshold is taken directly from the success criteria plan.
    condition_x_obj3 = x_distance_to_obj3 < threshold_x_obj3

    # Condition 2: Robot's pelvis is within a close proximity to Object3 in the y-dimension.
    # Calculate the absolute relative y-distance between the pelvis and Object3.
    y_distance_to_obj3 = torch.abs(pelvis_pos[:, 1] - object3_pos[:, 1]) # Correct: Using relative distance for y-component.
    # Define the threshold for y-distance to Object3 as per the success criteria plan.
    threshold_y_obj3 = 0.5  # meters. Correct: Threshold is taken directly from the success criteria plan.
    condition_y_obj3 = y_distance_to_obj3 < threshold_y_obj3

    # Condition 3: Robot's pelvis has successfully passed through the doorway.
    # The doorway's y-position is defined by Object1's y-position.
    doorway_y_pos = object1_pos[:, 1] # Correct: Using object's y-position as a reference.
    # Define a buffer to ensure the robot is clearly past the doorway as per the success criteria plan.
    doorway_pass_buffer = 0.5  # meters. Correct: Buffer is taken directly from the success criteria plan.
    # Check if the pelvis y-position is greater than the doorway's y-position plus the buffer.
    # This ensures the robot has moved past the doorway.
    condition_passed_doorway = pelvis_pos[:, 1] > (doorway_y_pos + doorway_pass_buffer) # Correct: Using relative position to doorway.

    # 4. Combine all success conditions. All conditions must be met.
    overall_condition = condition_x_obj3 & condition_y_obj3 & condition_passed_doorway # Correct: Combining conditions with logical AND for batched environments.

    # 5. Check duration and save success states.
    # The duration required for success is 0.5 seconds, as per the success criteria plan.
    success = check_success_duration(env, overall_condition, "doorway_and_goal_seed123", duration=0.5) # Correct: Using check_success_duration.
    
    # If any environment has succeeded, save its state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "doorway_and_goal_seed123") # Correct: Using save_success_state.
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=doorway_and_goal_seed123_success)
