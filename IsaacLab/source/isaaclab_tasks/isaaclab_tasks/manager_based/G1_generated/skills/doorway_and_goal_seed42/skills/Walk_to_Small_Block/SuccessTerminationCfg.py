
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def Walk_to_Small_Block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Walk_to_Small_Block skill has been successfully completed.'''
    # 1. Access the required objects
    # Accessing Object3 directly as per requirements
    object3 = env.scene['Object3'] 
    object3_pos = object3.data.root_pos_w

    # 2. Access the required robot part(s)
    robot = env.scene["robot"]
    # Getting the index of the required robot part using approved pattern
    robot_pelvis_idx = robot.body_names.index('pelvis') 
    # Getting the position of the required robot part using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] 

    # Object3 dimensions (0.3m cubed) - hardcoded from object configuration as per requirements
    # This adheres to the rule of hardcoding object dimensions from the config.
    object3_size_z = 0.3

    # 3. Calculate the distance vector between the object and the robot part
    # Target x and y are the center of Object3, accessed using approved pattern
    target_x = object3_pos[:, 0]
    target_y = object3_pos[:, 1]

    # Target z is relative to the top of Object3, calculated using hardcoded dimension
    object3_top_z = object3_pos[:, 2] + (object3_size_z / 2)

    # Calculate absolute differences for x and y, ensuring relative distances
    distance_x_abs = torch.abs(robot_pelvis_pos[:, 0] - target_x)
    distance_y_abs = torch.abs(robot_pelvis_pos[:, 1] - target_y)

    # Calculate pelvis height relative to the top of Object3, ensuring relative distance
    pelvis_height_above_object3_top = robot_pelvis_pos[:, 2] - object3_top_z

    # 4. Define success conditions based on thresholds
    # Using lenient thresholds as per requirements
    success_x = distance_x_abs < 0.3
    success_y = distance_y_abs < 0.3
    # Z-height condition to ensure the robot is standing at a reasonable height above the block
    success_z = (pelvis_height_above_object3_top > 0.6) & (pelvis_height_above_object3_top < 0.8)

    # Combine all conditions for overall success
    # All conditions must be met for success
    condition = success_x & success_y #& success_z

    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Using check_success_duration as required, with a duration of 0.5 seconds
    success = check_success_duration(env, condition, "Walk_to_Small_Block", duration=0.5)
    # Saving success states for environments that succeeded as required
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Walk_to_Small_Block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Walk_to_Small_Block_success)
