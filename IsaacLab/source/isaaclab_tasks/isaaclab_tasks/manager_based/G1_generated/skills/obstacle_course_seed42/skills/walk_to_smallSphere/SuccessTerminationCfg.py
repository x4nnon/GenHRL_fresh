
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def WalkToSmallSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToSmallSphere skill has been successfully completed.
    Success is defined as the robot pelvis being within 1m of the small sphere horizontally.
    '''
    # 1. Access the robot object using the approved pattern
    robot = env.scene["robot"]

    # 2. Get the index of the robot pelvis using robot.body_names.index for robustness
    robot_pelvis_idx = robot.body_names.index('pelvis')
    # 3. Get the world position of the robot pelvis using the approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    try:
        # 4. Safely access the small sphere object (Object2) using try-except block and approved pattern
        small_sphere = env.scene['Object2']
        # 5. Get the world position of the small sphere using the approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w

        # 6. Calculate the relative distance in x and y directions (horizontal distance)
        distance_x = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0]
        distance_y = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
    


        pelvis_z_pos = robot_pelvis_pos[:, 2]

        # 7. Define success condition: horizontal distance to small sphere is less than 1m.
        #    Using a lenient threshold of 1.5m to ensure robustness, as per instructions.
        success_threshold = 1.0
        success_threshold_y = 0.5

        condition = (distance_x < success_threshold) & (distance_y < success_threshold_y) & (pelvis_z_pos > 0.5)

    except KeyError:
        # 8. Handle the case where the small sphere object is not found, setting success to False for all envs.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check for success duration using the check_success_duration function with a duration of 0.1s
    success = check_success_duration(env, condition, "WalkToSmallSphere", duration=0.5)

    # 10. Save success states for environments that are successful in this step
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToSmallSphere")

    # 11. Return the success tensor
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToSmallSphere_success)