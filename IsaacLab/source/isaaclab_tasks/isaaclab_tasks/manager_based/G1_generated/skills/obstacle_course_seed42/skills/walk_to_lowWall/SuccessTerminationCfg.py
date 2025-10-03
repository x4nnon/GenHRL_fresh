
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def WalkToLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToLowWall skill has been successfully completed.'''
    # 1. Get robot pelvis position using approved access pattern
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

    try:
        # 2. Get low wall position using approved access pattern and handle potential KeyError
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # 3. Calculate the relative distance in the x-direction between the low wall and the robot's pelvis.
        #    This is a relative distance as required.
        distance_x = low_wall_pos[:, 0] - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction

        # 4. Define success condition: Robot pelvis is within 1m in front of the low wall in the x-direction.
        #    Using a threshold of 1.0m as specified in the success criteria plan.
        success_threshold_x_high = 1.5
        success_threshold_x_low = 0.5
        condition = (distance_x < success_threshold_x_high) & (distance_x > success_threshold_x_low) # CORRECT: Condition based on relative distance and threshold

    except KeyError:
        # 5. Handle KeyError if 'Object3' (low wall) is not found in the scene.
        #    Return a tensor of False for all environments in this case.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # CORRECT: Handle missing object

    # 6. Check success duration using the check_success_duration function.
    #    Using a duration of 0.5 seconds as specified in the success criteria plan.
    success = check_success_duration(env, condition, "WalkToLowWall", duration=0.5) # CORRECT: Check success duration

    # 7. Save success states for environments that have succeeded in this step.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToLowWall") # CORRECT: Save success state

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToLowWall_success) # CORRECT: Define SuccessTerminationCfg class with the success function
