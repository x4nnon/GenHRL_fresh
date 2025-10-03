
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path
def WalkToLargeSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToLargeSphere skill has been successfully completed.
    Success is achieved when the robot's pelvis is within 1.5 meters horizontal distance of the large sphere.
    '''
    # 1. Get robot pelvis position using approved access pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern

    try:
        # 2. Get large sphere position using approved access pattern and try/except for robustness
        large_sphere = env.scene['Object1'] # Accessing Object1 (large sphere) using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        # 3. Calculate horizontal distance between robot pelvis and large sphere. Only using relative distances.
        distance_x = (large_sphere_pos[:, 0] -0.8) - pelvis_pos[:, 0] # Relative distance in x-direction
        distance_y = large_sphere_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        horizontal_distance = torch.sqrt(distance_x**2 + distance_y**2) # Euclidean distance in x-y plane

        # 4. Define success condition: horizontal distance is within 1.5 meters. Using a lenient threshold as requested.
        success_threshold = 0.5
        condition = horizontal_distance < success_threshold # Success condition based on relative distance and threshold

    except KeyError:
        # 5. Handle missing object (large sphere). Skill fails if the object is not found.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # Return False for all envs if object is missing

    # 6. Check success duration and save success states. Using check_success_duration as required.
    success = check_success_duration(env, condition, "WalkToLargeSphere", duration=0.5) # Check if success condition is maintained for 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToLargeSphere") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToLargeSphere_success)
